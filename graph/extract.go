// Package graph orchestrates entity/edge extraction and graph building.
package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/store"
)

// IngestPayload is the JSON payload stored in the job queue.
type IngestPayload struct {
	Content string `json:"content"`
	Source  string `json:"source"`
	GroupID string `json:"group_id"`
}

// Extractor runs the full graph-building pipeline for a document chunk.
// A semaphore (cap 1) ensures max one gemma3:4b call at a time.
type Extractor struct {
	db               *store.DB
	llm              *llm.Client
	sem              chan struct{} // capacity 1 = max 1 concurrent LLM call
	muEntity         sync.Mutex   // serialise entity upserts to avoid duplicates under concurrency
	embedWG          sync.WaitGroup // tracks in-flight embedding goroutines
	resolveThreshold float64
}

// New creates a new Extractor. resolveThreshold is the minimum cosine similarity
// for two entities to be considered the same (e.g. 0.92).
func New(db *store.DB, client *llm.Client, resolveThreshold float64) *Extractor {
	return &Extractor{
		db:               db,
		llm:              client,
		sem:              make(chan struct{}, 1),
		resolveThreshold: resolveThreshold,
	}
}

// ProcessJob deserialises a queue job and runs the full extraction pipeline.
func (e *Extractor) ProcessJob(ctx context.Context, payload string) error {
	var p IngestPayload
	if err := json.Unmarshal([]byte(payload), &p); err != nil {
		return fmt.Errorf("decode payload: %w", err)
	}
	return e.Process(ctx, p.Content, p.Source, p.GroupID)
}

// Process runs entity extraction, edge extraction, and embedding for one text chunk.
// The LLM semaphore serialises calls; embedding is fire-and-forget per entity.
func (e *Extractor) Process(ctx context.Context, content, source, groupID string) error {
	epUUID := uuid.New().String()

	// ── 1. Store raw episode immediately ─────────────────────────────────────
	ep := store.Episode{
		UUID:    epUUID,
		Content: content,
		GroupID: groupID,
		Source:  source,
	}
	if err := e.db.UpsertEpisode(ctx, ep); err != nil {
		return fmt.Errorf("store episode: %w", err)
	}

	// ── 2. Acquire LLM semaphore (max 1 gemma3:4b at a time) ────────────────
	select {
	case e.sem <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	}

	start := time.Now()

	// ── 3. Entity extraction ─────────────────────────────────────────────────
	extracted, err := e.llm.ExtractEntities(ctx, content)
	if err != nil {
		<-e.sem
		return fmt.Errorf("extract entities: %w", err)
	}

	if len(extracted.Entities) == 0 {
		<-e.sem
		slog.Debug("no entities found", "source", source)
		return nil
	}

	// ── 4. Edge extraction ───────────────────────────────────────────────────
	edges, err := e.llm.ExtractEdges(ctx, extracted.Entities, content)
	if err != nil {
		<-e.sem
		return fmt.Errorf("extract edges: %w", err)
	}
	<-e.sem // release semaphore — LLM work done

	slog.Info("extracted",
		"source", shortPath(source),
		"entities", len(extracted.Entities),
		"edges", len(edges.Edges),
		"llm_ms", time.Since(start).Milliseconds(),
	)

	// ── 5. Batch-embed all entity descriptions + edge facts in one API call ──
	// Collecting all texts upfront avoids N×round-trips to nomic-embed-text.
	// Entities come first (indices 0..len-1), edges follow.
	batchTexts := make([]string, 0, len(extracted.Entities)+len(edges.Edges))
	for _, ent := range extracted.Entities {
		t := ent.Description
		if t == "" {
			t = entityEmbedText(ent.Name, ent.EntityType)
		}
		batchTexts = append(batchTexts, t)
	}
	for _, ex := range edges.Edges {
		if ex.Fact != "" {
			batchTexts = append(batchTexts, ex.Fact)
		} else {
			batchTexts = append(batchTexts, "") // placeholder — will yield nil embedding
		}
	}

	batchEmbs, batchErr := e.llm.EmbedBatch(ctx, batchTexts)
	if batchErr != nil {
		slog.Debug("embed batch failed, falling back to sequential", "err", batchErr)
		batchEmbs = nil // signals fallback below
	}

	embAt := func(i int) []float32 {
		if batchEmbs != nil && i < len(batchEmbs) {
			return batchEmbs[i]
		}
		return nil
	}

	// ── 6. Resolve entities: dedup → upsert → link ───────────────────────────
	entityUUIDs := make([]string, len(extracted.Entities))
	for i, ent := range extracted.Entities {
		vec := embAt(i)
		if vec == nil {
			// Batch failed — fall back to single embed.
			t := ent.Description
			if t == "" {
				t = entityEmbedText(ent.Name, ent.EntityType)
			}
			vec, err = e.llm.Embed(ctx, t)
			if err != nil {
				slog.Debug("embed entity failed", "name", ent.Name, "err", err)
				vec = nil
			}
		}

		e.muEntity.Lock()
		canonical, err := e.resolveOrCreate(ctx, ent.Name, ent.EntityType, groupID, vec, ent.Description)
		e.muEntity.Unlock()
		if err != nil {
			return fmt.Errorf("resolve entity %q: %w", ent.Name, err)
		}
		entityUUIDs[i] = canonical

		if err := e.db.LinkEntityEpisode(ctx, canonical, epUUID); err != nil {
			return fmt.Errorf("link entity-episode: %w", err)
		}
	}

	// ── 7. Store edges (embeddings from batch, fallback to sequential) ────────
	entCount := len(extracted.Entities)
	for ei, ex := range edges.Edges {
		if ex.SourceEntityID < 0 || ex.SourceEntityID >= len(entityUUIDs) {
			continue
		}
		if ex.TargetEntityID < 0 || ex.TargetEntityID >= len(entityUUIDs) {
			continue
		}
		edgeEmb := embAt(entCount + ei)
		if edgeEmb == nil && ex.Fact != "" {
			// Batch failed — fall back to single embed.
			edgeEmb, err = e.llm.Embed(ctx, ex.Fact)
			if err != nil {
				slog.Debug("embed edge fact failed", "fact", ex.Fact, "err", err)
			}
		}
		if err := e.db.UpsertEdge(ctx, store.Edge{
			UUID:       uuid.New().String(),
			SourceUUID: entityUUIDs[ex.SourceEntityID],
			TargetUUID: entityUUIDs[ex.TargetEntityID],
			Name:       ex.RelationType,
			Fact:       ex.Fact,
			GroupID:    groupID,
			ValidAt:    ex.ValidAt,
			InvalidAt:  ex.InvalidAt,
			Episodes:   fmt.Sprintf(`["%s"]`, epUUID),
			Embedding:  edgeEmb,
		}); err != nil {
			return fmt.Errorf("upsert edge: %w", err)
		}
	}

	// ── 7. Async episode embedding (best-effort, non-blocking) ──────────────
	e.embedWG.Add(1)
	go e.embedEpisode(context.Background(), epUUID, content)

	return nil
}

// Wait blocks until all pending embedding goroutines have finished.
// Useful in tests to ensure embeddings are written before searching.
func (e *Extractor) Wait() {
	e.embedWG.Wait()
}

// resolveOrCreate looks up an existing entity via FTS + cosine similarity, or
// inserts a new one. Caller must hold muEntity.
func (e *Extractor) resolveOrCreate(ctx context.Context, name, entityType, groupID string, embedding []float32, description string) (string, error) {
	if len(embedding) > 0 {
		candidates, err := e.db.SearchEntitiesFTS(ctx, name, groupID, 5)
		if err != nil {
			return "", fmt.Errorf("fts candidates: %w", err)
		}
		for _, c := range candidates {
			if c.EntityType != entityType || len(c.Embedding) == 0 {
				continue
			}
			if store.CosineSimilarity(embedding, c.Embedding) >= e.resolveThreshold {
				return c.UUID, nil
			}
		}
	}
	return e.db.UpsertEntity(ctx, store.Entity{
		UUID:        uuid.New().String(),
		Name:        name,
		EntityType:  entityType,
		GroupID:     groupID,
		Embedding:   embedding,
		Description: description,
	})
}

func (e *Extractor) embedEpisode(ctx context.Context, uuid, content string) {
	defer e.embedWG.Done()
	vec, err := e.llm.Embed(ctx, content)
	if err != nil {
		slog.Debug("embed episode failed", "err", err)
		return
	}
	if _, err := e.db.SQL().ExecContext(ctx,
		`UPDATE episodes SET embedding = ? WHERE uuid = ?`,
		store.EncodeEmbedding(vec), uuid,
	); err != nil {
		slog.Debug("store episode embedding failed", "err", err)
	}
}

// entityEmbedText generates a sentence for embedding an entity name.
// Wrapping in a typed sentence produces better embeddings than bare names.
func entityEmbedText(name, entityType string) string {
	switch entityType {
	case "Person":
		return "A person named " + name
	case "Organization":
		return "An organization called " + name
	case "Place":
		return "A place called " + name
	case "Product":
		return "A product called " + name
	case "Event":
		return "An event called " + name
	default:
		return "A concept called " + name
	}
}

func shortPath(s string) string {
	if len(s) <= 50 {
		return s
	}
	return "…" + s[len(s)-47:]
}
