// memory-local — portable knowledge graph builder using SQLite + Ollama.
//
// Dependencies: Go, SQLite (embedded), Ollama (local daemon).
//
// Usage:
//
//	memory-local ingest <path>          # walk dir, queue all text files
//	memory-local search <query>         # hybrid search (FTS + vector)
//	memory-local communities            # list detected communities
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
	"slices"
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
	extractProvider  := strings.ToLower(strings.TrimSpace(os.Getenv("MEMORY_EXTRACT_PROVIDER")))
	resolveThreshold := 0.92
	if v := os.Getenv("MEMORY_RESOLVE_THRESHOLD"); v != "" {
		t, err := strconv.ParseFloat(v, 64)
		if err != nil || t <= 0 || t > 1 {
			fatalf("MEMORY_RESOLVE_THRESHOLD must be a float in (0, 1], got %q", v)
		}
		resolveThreshold = t
	}
	// Default llmParallel: 1 for Ollama, 4 for Mistral API (handles concurrency server-side).
	defaultParallel := 1
	if extractProvider == "mistral" {
		defaultParallel = 4
	}
	llmParallel := defaultParallel
	if v := os.Getenv("MEMORY_LLM_PARALLEL"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			fatalf("MEMORY_LLM_PARALLEL must be a positive integer, got %q", v)
		}
		llmParallel = n
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

	// Build the extractor: Ollama (default) or Mistral API.
	var extractor llm.EntityExtractor
	switch extractProvider {
	case "", "ollama":
		extractor = client
	case "mistral":
		mistralKey := os.Getenv("MISTRAL_API_KEY")
		if mistralKey == "" {
			fatalf("MISTRAL_API_KEY not set — required when MEMORY_EXTRACT_PROVIDER=mistral")
		}
		extractor = llm.NewMistral(mistralKey, extractModel)
		slog.Info("extraction provider", "provider", "mistral", "model", extractModel)
	default:
		fatalf("unknown MEMORY_EXTRACT_PROVIDER %q — use 'ollama' or 'mistral'", extractProvider)
	}

	switch os.Args[1] {
	case "ingest":
		fs := flag.NewFlagSet("ingest", flag.ExitOnError)
		source := fs.String("source", "", "source label (e.g. arXiv URL) instead of file path")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory ingest [-source URL] <path>")
		}
		rejectTrailingFlags(fs)
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
		if extractProvider == "" || extractProvider == "ollama" {
			fmt.Fprintln(os.Stderr, "Warming up model…")
			if err := client.Warmup(ctx); err != nil {
				slog.Warn("warmup failed", "err", err)
			}
		}
		runWorker(ctx, db, extractor, client, resolveThreshold, llmParallel, groupID)

	case "run":
		fs := flag.NewFlagSet("run", flag.ExitOnError)
		source := fs.String("source", "", "source label (e.g. arXiv URL) instead of file path")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory run [-source URL] <path>")
		}
		rejectTrailingFlags(fs)
		must(client.Ping(ctx), "ping ollama")

		if extractProvider == "" || extractProvider == "ollama" {
			fmt.Fprintln(os.Stderr, "Warming up model…")
			if err := client.Warmup(ctx); err != nil {
				slog.Warn("warmup failed", "err", err)
			}
		}

		w := ingest.New(db, groupID).WithOCR(client)
		if *source != "" {
			w = w.WithSource(*source)
		}
		n, err := w.Walk(ctx, fs.Arg(0))
		must(err, "walk")
		fmt.Fprintf(os.Stderr, "✓ Queued %d chunks — starting worker (Ctrl+C to stop)\n", n)
		runWorker(ctx, db, extractor, client, resolveThreshold, llmParallel, groupID)

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
		// When extraction runs via Mistral API, QA should too (Ollama doesn't have the model).
		if qaAnswerer == nil && extractProvider == "mistral" {
			qaAnswerer = llm.NewMistral(mistralKey, extractModel)
			slog.Info("QA answerer", "model", extractModel, "provider", "mistral (auto)")
		}

		var judge bench.Judge
		if *judgeModel != "" {
			if mistralKey == "" {
				fatalf("MISTRAL_API_KEY not set — required for -judge %s", *judgeModel)
			}
			judge = llm.NewMistral(mistralKey, *judgeModel)
			slog.Info("LLM judge", "model", *judgeModel, "provider", "mistral")
		}

		result, err := bench.RunLoCoMo(ctx, fs.Arg(0), db, extractor, client, qaAnswerer, judge, resolveThreshold, *limit, *baseline, *qaOnly)
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

	case "retry":
		n, err := db.RequeueFailed(ctx)
		must(err, "requeue failed jobs")
		fmt.Fprintf(os.Stderr, "✓ Requeued %d failed jobs\n", n)

	case "communities":
		fs := flag.NewFlagSet("communities", flag.ExitOnError)
		format := fs.String("format", "text", "output format: text|json")
		minMembers := fs.Int("min", 2, "minimum members to show a community")
		detect := fs.Bool("detect", false, "run community detection (writes to DB)")
		ricci := fs.Bool("ricci", false, "use Mutual-kNN + ORC instead of Louvain")
		resolution := fs.Float64("resolution", 1.0, "Louvain resolution (higher = more, smaller communities)")
		_ = fs.Parse(os.Args[2:])
		if *detect && *ricci {
			fmt.Fprintln(os.Stderr, "Running Mutual-kNN + ORC + Louvain…")
			cr, err := db.MutualKNNCommunities(ctx, groupID, 20, *resolution)
			must(err, "mutual-knn communities")
			fmt.Fprintf(os.Stderr, "✓ %d communities across %d entities (Mutual-kNN + ORC)\n",
				cr.Communities, cr.Entities)
		}
		if *detect && !*ricci {
			fmt.Fprintln(os.Stderr, "Running Louvain community detection…")
			cr, err := db.DetectCommunities(ctx, groupID, *resolution)
			must(err, "detect communities")
			fmt.Fprintf(os.Stderr, "✓ %d communities across %d entities\n", cr.Communities, cr.Entities)
		}
		if *detect {
			if err := graph.GenerateCommunityReports(ctx, db, nil, groupID); err != nil {
				fmt.Fprintf(os.Stderr, "warning: community report generation failed: %v\n", err)
			}
		}
		if !*detect {
			printCommunities(ctx, db, groupID, *format, *minMembers)
		}

	case "curvature":
		fs := flag.NewFlagSet("curvature", flag.ExitOnError)
		persist := fs.Bool("store", false, "persist curvatures to edge_curvatures table")
		bridges := fs.Int("bridges", 0, "show top N bridge edges (most negative curvature)")
		format := fs.String("format", "text", "output format: text|json")
		_ = fs.Parse(os.Args[2:])

		// If -bridges without computation, just query stored curvatures.
		if *bridges > 0 && !*persist {
			top, err := db.TopBridges(ctx, groupID, *bridges)
			if err == nil && len(top) > 0 {
				printBridges(top, *format)
				break
			}
		}

		fmt.Fprintln(os.Stderr, "Computing Ollivier-Ricci curvatures…")
		curvatures, stats, err := db.ComputeCurvatures(ctx, groupID, 0)
		must(err, "compute curvatures")
		printCurvatureStats(stats, *format)

		if *persist {
			fmt.Fprintln(os.Stderr, "Storing curvatures…")
			must(db.StoreCurvatures(ctx, groupID, curvatures), "store curvatures")
			fmt.Fprintf(os.Stderr, "✓ %d curvatures stored\n", len(curvatures))
		}

		if *bridges > 0 {
			top, err := db.TopBridges(ctx, groupID, *bridges)
			if err == nil {
				printBridges(top, *format)
				break
			}
			// Fall back to in-memory sort.
			slices.SortFunc(curvatures, func(a, b store.EdgeCurvature) int {
				if a.Curvature < b.Curvature {
					return -1
				}
				if a.Curvature > b.Curvature {
					return 1
				}
				return 0
			})
			n := min(*bridges, len(curvatures))
			printBridges(curvatures[:n], *format)
		}

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
// extractor handles entity/edge extraction; embedder handles all embeddings (always Ollama).
func runWorker(ctx context.Context, db *store.DB, extractor llm.EntityExtractor, embedder *llm.Client, resolveThreshold float64, llmParallel int, groupID string) {
	ext := graph.New(db, extractor, embedder, resolveThreshold, llmParallel)
	concurrency := runtime.NumCPU()
	if concurrency > 4 {
		concurrency = 4
	}

	recoverStaleJobs(ctx, db)

	// Worker pool — but LLM semaphore in Extractor limits actual LLM calls to 1.
	jobs := make(chan *store.Job, concurrency)

	for i := 0; i < concurrency; i++ {
		go func() {
			for job := range jobs {
				if err := ext.ProcessJob(ctx, job.Payload, job.Attempts); err != nil {
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
	communityDirty := false // true when new jobs processed since last community detection

	for {
		select {
		case <-ctx.Done():
			close(jobs)
			fmt.Fprintf(os.Stderr, "\n✓ Worker stopped. Processed %d chunks.\n", processed)
			return
		case <-ticker.C:
			// Periodically recover stale jobs in case a worker goroutine panicked.
			if time.Since(lastRecover) > staleJobTimeout {
				recoverStaleJobs(ctx, db)
				lastRecover = time.Now()
			}

			queueEmpty := true
			for {
				job, err := db.NextJob(ctx)
				if err != nil {
					slog.Error("poll queue", "err", err)
					break
				}
				if job == nil {
					break // queue empty
				}
				queueEmpty = false
				communityDirty = true
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

			// Run community detection once when queue drains after processing.
			if queueEmpty && communityDirty {
				slog.Info("queue drained — running community detection")
				cr, err := db.DetectCommunities(ctx, groupID, 1.0)
				if err != nil {
					slog.Error("community detection failed — will retry on next drain", "err", err)
				} else {
					communityDirty = false
					slog.Info("communities detected", "communities", cr.Communities, "entities", cr.Entities)
					if err := graph.GenerateCommunityReports(ctx, db, nil, groupID); err != nil {
						slog.Error("community reports failed", "err", err)
					}
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
		curvTotal, curvBridges, curvInternal, curvMean := db.CurvatureStatus(ctx, groupID)
		out := struct {
			Graph struct {
				Episodes int `json:"episodes"`
				Entities int `json:"entities"`
				Edges    int `json:"edges"`
			} `json:"graph"`
			Curvature *struct {
				Edges    int     `json:"edges"`
				Bridges  int     `json:"bridges"`
				Internal int     `json:"internal"`
				Mean     float64 `json:"mean"`
			} `json:"curvature,omitempty"`
			Queue map[string]int `json:"queue"`
		}{}
		out.Graph.Episodes = episodes
		out.Graph.Entities = entities
		out.Graph.Edges = edges
		out.Queue = stats
		if curvTotal > 0 {
			out.Curvature = &struct {
				Edges    int     `json:"edges"`
				Bridges  int     `json:"bridges"`
				Internal int     `json:"internal"`
				Mean     float64 `json:"mean"`
			}{curvTotal, curvBridges, curvInternal, curvMean}
		}
		_ = json.NewEncoder(os.Stdout).Encode(out)
		return
	}

	fmt.Printf("── Graph ──────────────────\n")
	fmt.Printf("  episodes : %d\n", episodes)
	fmt.Printf("  entities : %d\n", entities)
	fmt.Printf("  edges    : %d\n", edges)

	curvTotal, curvBridges, curvInternal, curvMean := db.CurvatureStatus(ctx, groupID)
	if curvTotal > 0 {
		fmt.Printf("\n── Curvature ──────────────\n")
		fmt.Printf("  edges    : %d\n", curvTotal)
		fmt.Printf("  bridges  : %d (%.0f%%)\n", curvBridges, pct(curvBridges, curvTotal))
		fmt.Printf("  internal : %d (%.0f%%)\n", curvInternal, pct(curvInternal, curvTotal))
		fmt.Printf("  mean κ   : %.3f\n", curvMean)
	}

	fmt.Printf("\n── Queue ──────────────────\n")
	for _, s := range []string{"pending", "processing", "done", "failed"} {
		fmt.Printf("  %-10s : %d\n", s, stats[s])
	}
}

func printCommunities(ctx context.Context, db *store.DB, groupID, format string, minMembers int) {
	communities, err := db.ListCommunities(ctx, groupID)
	must(err, "list communities")

	// Filter by minimum member count.
	filtered := communities[:0]
	for _, c := range communities {
		if len(c.Members) < minMembers {
			continue
		}
		filtered = append(filtered, c)
	}

	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		for _, c := range filtered {
			_ = enc.Encode(c)
		}
		return
	}

	if len(filtered) == 0 {
		fmt.Printf("(no communities with >= %d members)\n", minMembers)
		return
	}

	fmt.Printf("── Communities (%d shown, %d total, min %d members) ────────\n\n",
		len(filtered), len(communities), minMembers)
	for _, c := range filtered {
		fmt.Printf("  Community %d  (%d members)\n", c.CommunityID, len(c.Members))
		fmt.Printf("    Members: %s\n", strings.Join(c.Members, ", "))
		if c.Report != "" {
			fmt.Printf("    Report:  %s\n", c.Report)
		}
		fmt.Println()
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
  retry            requeue all failed jobs for reprocessing
  resolve          merge near-duplicate entities (flags: -dry-run, -threshold 0.85)
  communities      detect + list communities                     (flags: -detect, -ricci, -format, -resolution)
  curvature        compute Ollivier-Ricci edge curvatures       (flags: -store, -bridges N, -format)
  bench   <json>   evaluate against LoCoMo benchmark (flags: -limit N, -baseline)
  status           show queue and graph statistics

Environment:
  MEMORY_DB                  path to SQLite file          (default: memory-local.db)
  MEMORY_OLLAMA              Ollama base URL              (default: http://localhost:11434)
  MEMORY_MODEL               extraction model             (default: gemma3:4b)
  MEMORY_EMBED_MODEL         embedding model              (default: mxbai-embed-large)
  MEMORY_GROUP               namespace/group              (default: default)
  MEMORY_RESOLVE_THRESHOLD   resolve similarity           (default: 0.92)
  MEMORY_LLM_PARALLEL        concurrent LLM calls         (default: 1 for ollama, 4 for mistral)
  MEMORY_EXTRACT_PROVIDER    extraction backend           (default: ollama; or: mistral)
  MISTRAL_API_KEY            Mistral API key              (required when provider=mistral)`)
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

func recoverStaleJobs(ctx context.Context, db *store.DB) {
	n, err := db.RecoverStaleJobs(ctx, staleJobTimeout)
	if err != nil {
		slog.Error("recover stale jobs", "err", err)
	}
	if n > 0 {
		slog.Info("recovered stale jobs", "count", n)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}

func printCurvatureStats(stats store.CurvatureStats, format string) {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(stats) //nolint:errcheck
		return
	}
	fmt.Printf("\nOllivier-Ricci Curvature Statistics\n")
	fmt.Printf("───────────────────────────────────\n")
	fmt.Printf("  Total edges:   %d\n", stats.TotalEdges)
	fmt.Printf("  Bridges (κ<0): %d (%.1f%%)\n", stats.Bridges, pct(stats.Bridges, stats.TotalEdges))
	fmt.Printf("  Internal (κ>0):%d (%.1f%%)\n", stats.Internal, pct(stats.Internal, stats.TotalEdges))
	fmt.Printf("  Flat (|κ|<.05):%d (%.1f%%)\n", stats.Flat, pct(stats.Flat, stats.TotalEdges))
	fmt.Printf("  Mean κ:        %.4f\n", stats.Mean)
	fmt.Printf("  Min κ:         %.4f\n", stats.Min)
	fmt.Printf("  Max κ:         %.4f\n", stats.Max)
	fmt.Printf("  Elapsed:       %s\n", stats.Elapsed)
}

func printBridges(bridges []store.EdgeCurvature, format string) {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(bridges) //nolint:errcheck
		return
	}
	fmt.Printf("\nTop Bridge Edges (most negative curvature)\n")
	fmt.Printf("───────────────────────────────────────────\n")
	for i, b := range bridges {
		src := b.SourceName
		if src == "" {
			src = b.SourceUUID[:min(8, len(b.SourceUUID))]
		}
		tgt := b.TargetName
		if tgt == "" {
			tgt = b.TargetUUID[:min(8, len(b.TargetUUID))]
		}
		fmt.Printf("  %3d. κ=%+.4f  %s ↔ %s\n", i+1, b.Curvature, src, tgt)
	}
}

func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(n) / float64(total) * 100
}

// rejectTrailingFlags detects flags placed after the positional path argument
// (e.g. "ingest ./path -source URL") which Go's flag package silently ignores.
func rejectTrailingFlags(fs *flag.FlagSet) {
	for _, arg := range fs.Args()[1:] {
		if strings.HasPrefix(arg, "-") {
			fatalf("flags must come before the path argument: %q\nusage: ultramemory %s [-flags] <path>", arg, fs.Name())
		}
	}
}
