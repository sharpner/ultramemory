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
	Type    string  // "entity" | "edge" | "episode"
	UUID    string
	Title   string  // entity name, edge relation_type, or episode source
	Body    string  // edge fact, entity type, or episode content snippet
	Score   float64
	Source  string  // originating source file
	ValidAt string  // ISO 8601 date when edge fact became true (empty if unknown)
}

// Search performs triple-signal hybrid search (FTS + vector + MAGMA graph via RRF).
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

	var entityVec, edgeVec, episodeVec []scored

	if len(qEmb) > 0 {
		entities, _ := db.AllEntitiesWithEmbeddings(ctx, groupID)
		for _, e := range entities {
			sim := store.CosineSimilarity(qEmb, e.Embedding)
			// Threshold 0.5 (was 0.3): same rationale as edge threshold —
			// permissive 0.3 causes near-misses (grandfather≈grandmother entity confusion).
			if sim > 0.5 {
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
			// Threshold 0.5 (was 0.3): edge vector search at 0.3 introduces
			// semantic near-misses (grandfather≈grandmother) that confuse adversarial
			// questions. Stricter threshold reduces false positives.
			if sim > 0.5 {
				edgeVec = append(edgeVec, scored{e.UUID, sim})
				edgByUUID[e.UUID] = e
			}
		}
		sort.Slice(edgeVec, func(i, j int) bool { return edgeVec[i].score > edgeVec[j].score })
		if len(edgeVec) > limit*2 {
			edgeVec = edgeVec[:limit*2]
		}

		episodes, _ := db.AllEpisodesWithEmbeddings(ctx, groupID)
		for _, e := range episodes {
			sim := store.CosineSimilarity(qEmb, e.Embedding)
			if sim > 0.3 {
				episodeVec = append(episodeVec, scored{e.UUID, sim})
				epByUUID[e.UUID] = e
			}
		}
		sort.Slice(episodeVec, func(i, j int) bool { return episodeVec[i].score > episodeVec[j].score })
		if len(episodeVec) > limit*2 {
			episodeVec = episodeVec[:limit*2]
		}
	}

	// ── 3. MAGMA graph traversal (before RRF — becomes a third signal) ───────
	// Seeds from FTS entity hits (lexical anchor, most specific matches).
	seeds := ftsEntitySeeds(entFTS, 5)
	var magmaRanked []ActivatedNode
	if len(seeds) > 0 {
		expanded, err := SpreadMAGMA(ctx, db, seeds, query, qEmb, groupID, DefaultMAGMAConfig())
		if err == nil {
			magmaRanked = expanded
			// Ensure MAGMA-discovered entities are in the lookup map.
			for _, a := range magmaRanked {
				if _, ok := entByUUID[a.UUID]; !ok {
					entByUUID[a.UUID] = store.Entity{
						UUID:       a.UUID,
						Name:       a.Name,
						EntityType: a.EntityType,
						GroupID:    groupID,
					}
				}
			}
		}
	}

	// ── 4. Triple-signal RRF fusion (FTS + vector + MAGMA) ───────────────────
	// k=1 matches graphiti-core/memories reference implementation.
	// Lower k gives stronger rank differentiation (top hit gets 1.0, not 0.016).
	const k = 1
	rrf := map[string]float64{}

	// Signal 1: FTS ranks.
	for rank, e := range entFTS {
		rrf["ent:"+e.UUID] += 1.0 / float64(k+rank+1)
	}
	for rank, e := range edgeFTS {
		rrf["edg:"+e.UUID] += 1.0 / float64(k+rank+1)
	}
	// Episodes are boosted 1.5× because they contain the actual dialogue
	// context needed to answer questions (vs. bare entity names/types).
	for rank, e := range epFTS {
		rrf["ep:"+e.UUID] += 1.5 / float64(k+rank+1)
	}

	// Signal 2: Vector similarity ranks.
	for rank, s := range entityVec {
		rrf["ent:"+s.uuid] += 1.0 / float64(k+rank+1)
	}
	for rank, s := range edgeVec {
		rrf["edg:"+s.uuid] += 1.0 / float64(k+rank+1)
	}
	// Episode vector search boosted 1.5× (same as FTS episodes).
	for rank, s := range episodeVec {
		rrf["ep:"+s.uuid] += 1.5 / float64(k+rank+1)
	}

	// Signal 3: MAGMA graph activation ranks (Synapse triple-signal fusion).
	// MAGMA results are already sorted by activation score from SpreadMAGMA.
	for rank, a := range magmaRanked {
		rrf["ent:"+a.UUID] += 1.0 / float64(k+rank+1)
	}

	// Signal 4: Community affinity (Leiden §4 — community-bounded retrieval).
	// Entities and edges in the same community as seed entities get a boost.
	communityMap, _ := db.CommunityMap(ctx, groupID)
	if len(communityMap) > 0 && len(seeds) > 0 {
		seedCommunities := map[int]bool{}
		for _, s := range seeds {
			if cid, ok := communityMap[s.UUID]; ok {
				seedCommunities[cid] = true
			}
		}
		if len(seedCommunities) > 0 {
			// Boost edges (facts) in seed communities — conservative to avoid
			// promoting noise for adversarial "unknown" questions.
			for uuid, e := range edgByUUID {
				srcCid, srcOk := communityMap[e.SourceUUID]
				tgtCid, tgtOk := communityMap[e.TargetUUID]
				if (srcOk && seedCommunities[srcCid]) || (tgtOk && seedCommunities[tgtCid]) {
					rrf["edg:"+uuid] += 0.15
				}
			}
		}
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
				body := e.EntityType
				if e.Description != "" {
					body = e.EntityType + ": " + e.Description
				}
				results = append(results, SearchResult{
					Type:  "entity",
					UUID:  uid,
					Title: e.Name,
					Body:  body,
					Score: en.score,
				})
			}
		case len(key) > 4 && key[:4] == "edg:":
			uid := key[4:]
			if e, ok := edgByUUID[uid]; ok {
				validAt := ""
				if e.ValidAt != nil {
					validAt = *e.ValidAt
				}
				results = append(results, SearchResult{
					Type:    "edge",
					UUID:    uid,
					Title:   e.Name,
					Body:    e.Fact,
					Score:   en.score,
					ValidAt: validAt,
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

	// ── 5. Relevance cutoff + uncertainty gating ────────────────────────────
	// (a) Relative score cutoff: drop results scoring < 15% of the top hit.
	// This naturally adapts: single-hop queries with one strong match get
	// fewer results (less noise); multi-hop queries keep more context.
	if len(results) > 1 {
		topScore := results[0].Score
		cutoff := topScore * 0.15
		for i, r := range results[1:] {
			if r.Score < cutoff {
				results = results[:i+1]
				break
			}
		}
	}
	// (b) Uncertainty gating (Synapse §3.3): if the best result score is
	// below a threshold, the query likely asks about something not in the
	// knowledge base. Returning low-quality context causes a small LLM to
	// hallucinate; better to return nothing.
	const uncertaintyGate = 0.1 // With k=1 RRF, single hit scores 1.0; gate at 0.1
	if len(results) > 0 && results[0].Score < uncertaintyGate {
		results = nil
	}

	// ── 6. Populate source for each result ────────────────────────────────────
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

// ftsEntitySeeds extracts the top-n FTS entity hits as MAGMA seed nodes.
func ftsEntitySeeds(entities []store.Entity, n int) []ActivatedNode {
	seeds := make([]ActivatedNode, 0, n)
	for _, e := range entities {
		seeds = append(seeds, ActivatedNode{UUID: e.UUID, Name: e.Name, EntityType: e.EntityType})
		if len(seeds) >= n {
			break
		}
	}
	return seeds
}
