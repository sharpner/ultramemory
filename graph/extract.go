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
	db       *store.DB
	llm      *llm.Client
	sem      chan struct{} // capacity 1 = max 1 concurrent LLM call
	muEntity sync.Mutex   // serialise entity upserts to avoid duplicates under concurrency
	embedWG  sync.WaitGroup // tracks in-flight embedding goroutines
}

// New creates a new Extractor.
func New(db *store.DB, client *llm.Client) *Extractor {
	return &Extractor{
		db:  db,
		llm: client,
		sem: make(chan struct{}, 1),
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

	// ── 5. Store entities + link to episode ──────────────────────────────────
	entityUUIDs := make([]string, len(extracted.Entities))
	for i, ent := range extracted.Entities {
		e.muEntity.Lock()
		canonical, err := e.db.UpsertEntity(ctx, store.Entity{
			UUID:       uuid.New().String(),
			Name:       ent.Name,
			EntityType: ent.EntityType,
			GroupID:    groupID,
		})
		e.muEntity.Unlock()
		if err != nil {
			return fmt.Errorf("upsert entity %q: %w", ent.Name, err)
		}
		entityUUIDs[i] = canonical

		if err := e.db.LinkEntityEpisode(ctx, canonical, epUUID); err != nil {
			return fmt.Errorf("link entity-episode: %w", err)
		}
	}

	// ── 6. Store edges ────────────────────────────────────────────────────────
	for _, ex := range edges.Edges {
		if ex.SourceEntityID < 0 || ex.SourceEntityID >= len(entityUUIDs) {
			continue
		}
		if ex.TargetEntityID < 0 || ex.TargetEntityID >= len(entityUUIDs) {
			continue
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
		}); err != nil {
			return fmt.Errorf("upsert edge: %w", err)
		}
	}

	// ── 7. Async embeddings (best-effort, non-blocking) ───────────────────────
	e.embedWG.Add(1 + len(extracted.Entities))
	go e.embedEpisode(context.Background(), epUUID, content)
	for i, ent := range extracted.Entities {
		go e.embedEntity(context.Background(), entityUUIDs[i], ent.Name+" ("+ent.EntityType+")")
	}

	return nil
}

// Wait blocks until all pending embedding goroutines have finished.
// Useful in tests to ensure embeddings are written before searching.
func (e *Extractor) Wait() {
	e.embedWG.Wait()
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

func (e *Extractor) embedEntity(ctx context.Context, id, name string) {
	defer e.embedWG.Done()
	vec, err := e.llm.Embed(ctx, name)
	if err != nil {
		slog.Debug("embed entity failed", "err", err)
		return
	}
	if _, err := e.db.SQL().ExecContext(ctx,
		`UPDATE entities SET embedding = ? WHERE uuid = ?`,
		store.EncodeEmbedding(vec), id,
	); err != nil {
		slog.Debug("store entity embedding failed", "err", err)
	}
}

func shortPath(s string) string {
	if len(s) <= 50 {
		return s
	}
	return "…" + s[len(s)-47:]
}
