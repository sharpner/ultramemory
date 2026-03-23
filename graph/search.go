package graph

import (
	"context"
	"fmt"
	"sort"

	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/store"
)

// SearchResult is one item returned from hybrid search.
type SearchResult struct {
	Type   string  // "entity" | "edge" | "episode"
	UUID   string
	Title  string  // entity name, edge relation_type, or episode source
	Body   string  // edge fact, entity type, or episode content snippet
	Score  float64
	Source string  // originating source file
}

// Search performs hybrid search (FTS5 + cosine similarity via RRF).
func Search(ctx context.Context, db *store.DB, client *llm.Client, query, groupID string, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 10
	}

	// ── 1. FTS search ─────────────────────────────────────────────────────────
	entFTS, err := db.SearchEntitiesFTS(ctx, query, groupID, limit*2)
	if err != nil {
		return nil, fmt.Errorf("entity FTS: %w", err)
	}
	edgeFTS, err := db.SearchEdgesFTS(ctx, query, groupID, limit*2)
	if err != nil {
		return nil, fmt.Errorf("edge FTS: %w", err)
	}
	epFTS, err := db.SearchEpisodesFTS(ctx, query, groupID, limit*2)
	if err != nil {
		return nil, fmt.Errorf("episode FTS: %w", err)
	}

	// ── 2. Vector search ──────────────────────────────────────────────────────
	qEmb, err := client.Embed(ctx, query)
	if err != nil {
		// Non-fatal: fall back to FTS-only.
		qEmb = nil
	}

	type scored struct {
		uuid  string
		score float64
	}

	// Lookup maps for building results — populated from FTS first, then vector.
	// Vector-only hits (no FTS match) must be in these maps too, otherwise they
	// are silently dropped when converting RRF scores to SearchResult values.
	entByUUID := map[string]store.Entity{}
	for _, e := range entFTS {
		entByUUID[e.UUID] = e
	}
	edgByUUID := map[string]store.Edge{}
	for _, e := range edgeFTS {
		edgByUUID[e.UUID] = e
	}
	epByUUID := map[string]store.Episode{}
	for _, e := range epFTS {
		epByUUID[e.UUID] = e
	}

	var entityVec, edgeVec []scored

	if len(qEmb) > 0 {
		entities, _ := db.AllEntitiesWithEmbeddings(ctx, groupID)
		for _, e := range entities {
			sim := store.CosineSimilarity(qEmb, e.Embedding)
			if sim > 0.3 {
				entityVec = append(entityVec, scored{e.UUID, sim})
				entByUUID[e.UUID] = e
			}
		}
		sort.Slice(entityVec, func(i, j int) bool { return entityVec[i].score > entityVec[j].score })
		if len(entityVec) > limit*2 {
			entityVec = entityVec[:limit*2]
		}

		edges, _ := db.AllEdgesWithEmbeddings(ctx, groupID)
		for _, e := range edges {
			sim := store.CosineSimilarity(qEmb, e.Embedding)
			if sim > 0.3 {
				edgeVec = append(edgeVec, scored{e.UUID, sim})
				edgByUUID[e.UUID] = e
			}
		}
		sort.Slice(edgeVec, func(i, j int) bool { return edgeVec[i].score > edgeVec[j].score })
		if len(edgeVec) > limit*2 {
			edgeVec = edgeVec[:limit*2]
		}
	}

	// ── 3. RRF fusion ─────────────────────────────────────────────────────────
	const k = 60
	rrf := map[string]float64{}

	for rank, e := range entFTS {
		rrf["ent:"+e.UUID] += 1.0 / float64(k+rank+1)
	}
	for rank, e := range edgeFTS {
		rrf["edg:"+e.UUID] += 1.0 / float64(k+rank+1)
	}
	for rank, s := range entityVec {
		rrf["ent:"+s.uuid] += 1.0 / float64(k+rank+1)
	}
	for rank, s := range edgeVec {
		rrf["edg:"+s.uuid] += 1.0 / float64(k+rank+1)
	}
	for rank, e := range epFTS {
		rrf["ep:"+e.UUID] += 1.0 / float64(k+rank+1)
	}

	type rfentry struct {
		key   string
		score float64
	}
	entries := make([]rfentry, 0, len(rrf))
	for k, v := range rrf {
		entries = append(entries, rfentry{k, v})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].score > entries[j].score })

	var results []SearchResult
	for _, en := range entries {
		if len(results) >= limit {
			break
		}
		key := en.key
		switch {
		case len(key) > 4 && key[:4] == "ent:":
			uid := key[4:]
			if e, ok := entByUUID[uid]; ok {
				results = append(results, SearchResult{
					Type:  "entity",
					UUID:  uid,
					Title: e.Name,
					Body:  e.EntityType,
					Score: en.score,
				})
			}
		case len(key) > 4 && key[:4] == "edg:":
			uid := key[4:]
			if e, ok := edgByUUID[uid]; ok {
				results = append(results, SearchResult{
					Type:  "edge",
					UUID:  uid,
					Title: e.Name,
					Body:  e.Fact,
					Score: en.score,
				})
			}
		case len(key) > 3 && key[:3] == "ep:":
			uid := key[3:]
			if e, ok := epByUUID[uid]; ok {
				results = append(results, SearchResult{
					Type:   "episode",
					UUID:   uid,
					Title:  e.Source,
					Body:   e.Content,
					Score:  en.score,
					Source: e.Source,
				})
			}
		}
	}

	// ── 4. MAGMA graph traversal ───────────────────────────────────────────────
	seeds := entitySeeds(results, 5)
	if len(seeds) > 0 {
		expanded, err := SpreadMAGMA(ctx, db, seeds, query, qEmb, groupID, DefaultMAGMAConfig())
		if err == nil {
			results = mergeMAGMA(results, expanded, limit)
		}
	}

	// ── 5. Populate source for each result ────────────────────────────────────
	for i, r := range results {
		switch r.Type {
		case "entity":
			results[i].Source = db.FirstEntitySource(ctx, r.UUID, groupID)
		case "edge":
			results[i].Source = db.FirstEdgeSource(ctx, r.UUID)
		}
	}

	return results, nil
}

// entitySeeds extracts the top-n entity results as MAGMA seed nodes.
func entitySeeds(results []SearchResult, n int) []ActivatedNode {
	seeds := make([]ActivatedNode, 0, n)
	for _, r := range results {
		if r.Type != "entity" {
			continue
		}
		seeds = append(seeds, ActivatedNode{UUID: r.UUID, Name: r.Title, EntityType: r.Body})
		if len(seeds) >= n {
			break
		}
	}
	return seeds
}

// mergeMAGMA appends graph-traversal results (not already in results) after the
// direct matches. Direct matches keep their RRF rank — MAGMA only expands the
// tail. Re-sorting would demote direct matches because MAGMA activation scores
// are on a different scale than RRF scores.
func mergeMAGMA(results []SearchResult, activated []ActivatedNode, limit int) []SearchResult {
	seen := make(map[string]bool, len(results))
	for _, r := range results {
		seen[r.UUID] = true
	}
	// activated is already sorted by SpreadMAGMA; append in that order.
	for _, a := range activated {
		if len(results) >= limit {
			break
		}
		if seen[a.UUID] {
			continue
		}
		seen[a.UUID] = true
		results = append(results, SearchResult{
			Type:  "entity",
			UUID:  a.UUID,
			Title: a.Name,
			Body:  a.EntityType,
			Score: a.Activation * 0.8,
		})
	}
	return results
}
