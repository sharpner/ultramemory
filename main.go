// memory-local — portable knowledge graph builder using SQLite + Ollama.
//
// Dependencies: Go, SQLite (embedded), Ollama (local daemon).
//
// Usage:
//
//	memory-local ingest <path>          # walk dir, queue all text files
//	memory-local search <query>         # hybrid search (FTS + vector)
//	memory-local status                 # queue + graph stats
//	memory-local worker                 # run the extraction worker (blocking)
//	memory-local run <path>             # ingest + worker in one shot
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/sharpner/ultramemory/bench"
	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/ingest"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/store"
)

// version is set at build time via -ldflags "-X main.version=..."
// Falls back to "(dev)" for go run / untagged builds.
var version = "(dev)"

const (
	defaultDB             = "memory-local.db"
	defaultOllama         = "http://localhost:11434"
	defaultExtractModel   = "gemma3:4b"
	defaultEmbeddingModel = "mxbai-embed-large"
	defaultGroup          = "default"
	pollInterval          = 200 * time.Millisecond
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	if os.Args[1] == "--version" || os.Args[1] == "-version" || os.Args[1] == "version" {
		fmt.Println("ultramemory " + version)
		return
	}

	// Config from env (overridable).
	dbPath           := envOr("MEMORY_DB", defaultDB)
	ollamaURL        := envOr("MEMORY_OLLAMA", defaultOllama)
	extractModel     := envOr("MEMORY_MODEL", defaultExtractModel)
	embedModel       := envOr("MEMORY_EMBED_MODEL", defaultEmbeddingModel)
	groupID          := envOr("MEMORY_GROUP", defaultGroup)
	resolveThreshold := 0.92
	if v := os.Getenv("MEMORY_RESOLVE_THRESHOLD"); v != "" {
		t, err := strconv.ParseFloat(v, 64)
		if err != nil || t <= 0 || t > 1 {
			fatalf("MEMORY_RESOLVE_THRESHOLD must be a float in (0, 1], got %q", v)
		}
		resolveThreshold = t
	}

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})))

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	db, err := store.Open(dbPath)
	must(err, "open db")
	defer db.Close() //nolint:errcheck

	client := llm.New(ollamaURL, extractModel, embedModel)

	switch os.Args[1] {
	case "ingest":
		fs := flag.NewFlagSet("ingest", flag.ExitOnError)
		source := fs.String("source", "", "source label (e.g. arXiv URL) instead of file path")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory ingest [-source URL] <path>")
		}
		must(client.Ping(ctx), "ping ollama")
		w := ingest.New(db, groupID).WithOCR(client)
		if *source != "" {
			w = w.WithSource(*source)
		}
		n, err := w.Walk(ctx, fs.Arg(0))
		must(err, "walk")
		fmt.Fprintf(os.Stderr, "✓ Queued %d chunks from %s\n", n, fs.Arg(0))

	case "worker":
		must(client.Ping(ctx), "ping ollama")
		fmt.Fprintln(os.Stderr, "Warming up model…")
		if err := client.Warmup(ctx); err != nil {
			slog.Warn("warmup failed", "err", err)
		}
		runWorker(ctx, db, client, resolveThreshold)

	case "run":
		fs := flag.NewFlagSet("run", flag.ExitOnError)
		source := fs.String("source", "", "source label (e.g. arXiv URL) instead of file path")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory run [-source URL] <path>")
		}
		must(client.Ping(ctx), "ping ollama")

		fmt.Fprintln(os.Stderr, "Warming up model…")
		if err := client.Warmup(ctx); err != nil {
			slog.Warn("warmup failed", "err", err)
		}

		w := ingest.New(db, groupID).WithOCR(client)
		if *source != "" {
			w = w.WithSource(*source)
		}
		n, err := w.Walk(ctx, fs.Arg(0))
		must(err, "walk")
		fmt.Fprintf(os.Stderr, "✓ Queued %d chunks — starting worker (Ctrl+C to stop)\n", n)
		runWorker(ctx, db, client, resolveThreshold)

	case "search":
		fs := flag.NewFlagSet("search", flag.ExitOnError)
		format    := fs.String("format",     "text", "output format: text|json")
		maxTokens := fs.Int("max-tokens",    0,      "token budget for output (0 = unlimited)")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory search [-format text|json] [-max-tokens N] <query>")
		}
		must(client.Ping(ctx), "ping ollama")
		query := strings.Join(fs.Args(), " ")
		results, err := graph.Search(ctx, db, client, query, groupID, 10)
		must(err, "search")
		printSearch(results, query, *format, *maxTokens)

	case "bench":
		fs := flag.NewFlagSet("bench", flag.ExitOnError)
		limit     := fs.Int("limit", 0, "max conversations to evaluate (0 = all)")
		baseline  := fs.Bool("baseline", false, "baseline mode: episode FTS only, no graph extraction")
		qaModel   := fs.String("qa-model", "", "override QA answering model: 'mistral-small-2506' etc (default: same as extraction model)")
		qaOnly    := fs.Bool("qa-only", false, "skip ingestion, run QA on existing DB (use with -qa-model for fast model swapping)")
		judgeModel := fs.String("judge", "", "LLM judge model for semantic evaluation: 'mistral-small-2506' (requires MISTRAL_API_KEY)")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory bench [-limit N] [-baseline] [-qa-model MODEL] [-qa-only] [-judge MODEL] <locomo10.json>")
		}
		if !*qaOnly {
			must(client.Ping(ctx), "ping ollama")
			fmt.Fprintln(os.Stderr, "Warming up model…")
			if err := client.Warmup(ctx); err != nil {
				slog.Warn("warmup failed", "err", err)
			}
		}

		mistralKey := os.Getenv("MISTRAL_API_KEY")

		var qaAnswerer llm.Answerer
		if *qaModel != "" {
			if mistralKey == "" {
				fatalf("MISTRAL_API_KEY not set — required for -qa-model %s", *qaModel)
			}
			qaAnswerer = llm.NewMistral(mistralKey, *qaModel)
			slog.Info("QA answerer", "model", *qaModel, "provider", "mistral")
		}

		var judge bench.Judge
		if *judgeModel != "" {
			if mistralKey == "" {
				fatalf("MISTRAL_API_KEY not set — required for -judge %s", *judgeModel)
			}
			judge = llm.NewMistral(mistralKey, *judgeModel)
			slog.Info("LLM judge", "model", *judgeModel, "provider", "mistral")
		}

		result, err := bench.RunLoCoMo(ctx, fs.Arg(0), db, client, qaAnswerer, judge, resolveThreshold, *limit, *baseline, *qaOnly)
		must(err, "bench")
		bench.PrintResult(result)

	case "resolve":
		fs := flag.NewFlagSet("resolve", flag.ExitOnError)
		dryRun    := fs.Bool("dry-run", false, "print planned merges without writing")
		threshold := fs.Float64("threshold", 0.85, "cosine similarity threshold (0–1]")
		_ = fs.Parse(os.Args[2:])
		if *threshold <= 0 || *threshold > 1 {
			fatalf("--threshold must be in (0, 1], got %g", *threshold)
		}
		result, err := db.ResolveEntities(ctx, groupID, store.ResolveConfig{
			Threshold: *threshold,
			DryRun:    *dryRun,
		})
		must(err, "resolve")
		printResolveResult(result, *dryRun)

	case "status":
		fs := flag.NewFlagSet("status", flag.ExitOnError)
		format := fs.String("format", "text", "output format: text|json")
		_ = fs.Parse(os.Args[2:])
		printStatus(ctx, db, groupID, *format)

	default:
		usage()
		os.Exit(1)
	}
}

const staleJobTimeout = 5 * time.Minute

// runWorker polls the SQLite queue and processes jobs with max 1 concurrent LLM call.
func runWorker(ctx context.Context, db *store.DB, client *llm.Client, resolveThreshold float64) {
	ext := graph.New(db, client, resolveThreshold)
	concurrency := runtime.NumCPU()
	if concurrency > 4 {
		concurrency = 4
	}

	// Recover jobs orphaned by a previous crash before starting.
	if n, err := db.RecoverStaleJobs(ctx, staleJobTimeout); err != nil {
		slog.Error("recover stale jobs", "err", err)
	} else if n > 0 {
		slog.Info("recovered stale jobs", "count", n)
	}

	// Worker pool — but LLM semaphore in Extractor limits actual LLM calls to 1.
	jobs := make(chan *store.Job, concurrency)

	for i := 0; i < concurrency; i++ {
		go func() {
			for job := range jobs {
				if err := ext.ProcessJob(ctx, job.Payload); err != nil {
					slog.Error("job failed", "id", job.ID, "err", err)
					if err := db.FailJob(context.Background(), job.ID, err.Error()); err != nil {
						slog.Error("fail job", "err", err)
					}
					continue
				}
				if err := db.CompleteJob(context.Background(), job.ID); err != nil {
					slog.Error("complete job", "err", err)
				}
			}
		}()
	}

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	lastLog := time.Now()
	lastRecover := time.Now()
	processed := 0

	for {
		select {
		case <-ctx.Done():
			close(jobs)
			fmt.Fprintf(os.Stderr, "\n✓ Worker stopped. Processed %d chunks.\n", processed)
			return
		case <-ticker.C:
			// Periodically recover stale jobs in case a worker goroutine panicked.
			if time.Since(lastRecover) > staleJobTimeout {
				if n, err := db.RecoverStaleJobs(ctx, staleJobTimeout); err != nil {
					slog.Error("recover stale jobs", "err", err)
				} else if n > 0 {
					slog.Info("recovered stale jobs", "count", n)
				}
				lastRecover = time.Now()
			}

			for {
				job, err := db.NextJob(ctx)
				if err != nil {
					slog.Error("poll queue", "err", err)
					break
				}
				if job == nil {
					break // queue empty
				}
				jobs <- job
				processed++

				// Progress log every 10s.
				if time.Since(lastLog) > 10*time.Second {
					stats, _ := db.QueueStats(ctx)
					slog.Info("progress",
						"processed", processed,
						"pending", stats["pending"],
						"done", stats["done"],
						"failed", stats["failed"],
					)
					lastLog = time.Now()
				}
			}
		}
	}
}

// searchHit is the JSON-serialisable presentation of one search result.
type searchHit struct {
	Rank   int     `json:"rank"`
	Type   string  `json:"type"`
	Title  string  `json:"title"`
	Body   string  `json:"body,omitempty"`
	Score  float64 `json:"score"`
	Source string  `json:"source,omitempty"`
}

// approxTokens estimates token count using the standard 1 token ≈ 4 chars heuristic.
func approxTokens(s string) int {
	return (len(s) + 3) / 4
}

func printSearch(results []graph.SearchResult, query, format string, maxTokens int) {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		used := 0
		for i, r := range results {
			cost := approxTokens(r.Title + r.Body)
			if maxTokens > 0 && used+cost > maxTokens {
				break
			}
			_ = enc.Encode(searchHit{i + 1, r.Type, r.Title, r.Body, r.Score, r.Source})
			used += cost
		}
		return
	}
	if len(results) == 0 {
		fmt.Println("(no results)")
		return
	}
	fmt.Printf("Results for %q:\n\n", query)
	used := 0
	for i, r := range results {
		cost := approxTokens(r.Title + r.Body)
		if maxTokens > 0 && used+cost > maxTokens {
			fmt.Printf("-- token budget reached (%d/%d tokens used, %d result(s) omitted) --\n",
				used, maxTokens, len(results)-i)
			break
		}
		src := ""
		if r.Source != "" {
			src = "   source=" + filepath.Base(r.Source) + "\n"
		}
		fmt.Printf("%d. [%s] %s\n   %s\n   score=%.4f\n%s\n",
			i+1, r.Type, r.Title, r.Body, r.Score, src)
		used += cost
	}
}

func printStatus(ctx context.Context, db *store.DB, groupID, format string) {
	stats, err := db.QueueStats(ctx)
	must(err, "queue stats")

	episodes, _ := db.CountEpisodes(ctx, groupID)
	entities, _ := db.CountEntities(ctx, groupID)
	edges, _ := db.CountEdges(ctx, groupID)

	if format == "json" {
		for _, s := range []string{"pending", "processing", "done", "failed"} {
			if _, ok := stats[s]; !ok {
				stats[s] = 0
			}
		}
		out := struct {
			Graph struct {
				Episodes int `json:"episodes"`
				Entities int `json:"entities"`
				Edges    int `json:"edges"`
			} `json:"graph"`
			Queue map[string]int `json:"queue"`
		}{}
		out.Graph.Episodes = episodes
		out.Graph.Entities = entities
		out.Graph.Edges = edges
		out.Queue = stats
		_ = json.NewEncoder(os.Stdout).Encode(out)
		return
	}

	fmt.Printf("── Graph ──────────────────\n")
	fmt.Printf("  episodes : %d\n", episodes)
	fmt.Printf("  entities : %d\n", entities)
	fmt.Printf("  edges    : %d\n", edges)
	fmt.Printf("\n── Queue ──────────────────\n")
	for _, s := range []string{"pending", "processing", "done", "failed"} {
		fmt.Printf("  %-10s : %d\n", s, stats[s])
	}
}

func printResolveResult(r store.ResolveResult, dryRun bool) {
	mode := ""
	if dryRun {
		mode = " (dry-run)"
	}
	fmt.Printf("Entity resolution%s:\n", mode)
	fmt.Printf("  clusters found    : %d\n", r.ClustersFound)
	fmt.Printf("  entities merged   : %d\n", r.EntitiesMerged)
	fmt.Printf("  edges retargeted  : %d\n", r.EdgesRetargeted)
	fmt.Printf("  episodes relinked : %d\n", r.EpisodesRelinked)
}

func usage() {
	fmt.Fprintf(os.Stderr, "ultramemory %s — local knowledge graph (SQLite + Ollama)\n\n", version)
	fmt.Fprintln(os.Stderr, `Commands:
  run     <path>   ingest directory + start worker (all-in-one)  [-source URL]
  ingest  <path>   queue all text files for processing         [-source URL]
  worker           process queued jobs (blocking)
  search  <query>  hybrid search over the graph (flags: -format text|json, -max-tokens N)
  resolve          merge near-duplicate entities (flags: -dry-run, -threshold 0.85)
  bench   <json>   evaluate against LoCoMo benchmark (flags: -limit N, -baseline)
  status           show queue and graph statistics

Environment:
  MEMORY_DB                  path to SQLite file  (default: memory-local.db)
  MEMORY_OLLAMA              Ollama base URL      (default: http://localhost:11434)
  MEMORY_MODEL               extraction model     (default: gemma3:4b)
  MEMORY_EMBED_MODEL         embedding model      (default: mxbai-embed-large)
  MEMORY_GROUP               namespace/group      (default: default)
  MEMORY_RESOLVE_THRESHOLD   resolve similarity   (default: 0.92)`)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func must(err error, msg string) {
	if err != nil {
		fatalf("%s: %v", msg, err)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
