package graph

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

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
			// Threshold 0.3: entity vectors become MAGMA seeds. Higher threshold (0.5)
			// reduces MAGMA seeds → less graph traversal → worse adversarial (v22 test).
			if sim > 0.3 {
				entityVec = append(entityVec, scored{e.UUID, sim})
				entByUUID[e.UUID] = e
			}
		}
		sort.Slice(entityVec, func(i, j int) bool { return entityVec[i].score > entityVec[j].score })
		if len(entityVec) > limit*2 {
			entityVec = entityVec[:limit*2]
		}

		// v23 test: edge vector search disabled. v11 had no edge vector search → 52.7% adversarial.
		// v19 reintroduced it at 0.3 threshold → adversarial dropped to 43.4%. Hypothesis: edge
		// vector search introduces semantic near-misses (grandfather≈grandmother) that hurt
		// adversarial "unknown" questions. Removing entirely to confirm causation.
		// edgeVec remains nil — falls back to FTS-only for edge retrieval.

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

	// Signal 3b: MAGMA episode backfill — episodes linked to top MAGMA entities.
	// Provides rich dialogue context for multi-hop: when MAGMA discovers Bob via
	// Alice→KNOWS→Bob, we also retrieve episodes mentioning Bob so the LLM can
	// read the raw conversation context, not just extracted edge facts.
	// Boosted 1.2× (less than direct episode search 1.5× — indirect signal).
	if len(magmaRanked) > 0 {
		topN := 5
		if len(magmaRanked) < topN {
			topN = len(magmaRanked)
		}
		magmaUUIDs := make([]string, topN)
		for i := range topN {
			magmaUUIDs[i] = magmaRanked[i].UUID
		}
		magmaEps, _ := db.EpisodesForEntities(ctx, magmaUUIDs, groupID, limit*2)
		for _, ep := range magmaEps {
			epByUUID[ep.UUID] = ep
		}
		for rank, ep := range magmaEps {
			rrf["ep:"+ep.UUID] += 1.2 / float64(k+rank+1)
		}
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

	// ── 7. SYNAPSE §3.2 Temporal Decay ────────────────────────────────────────
	// Weight results by recency: facts from recent sessions score higher.
	// Normalised: session_N gets weight 1.0, session_1 gets exp(-λ × (N-1)/N).
	// λ=0.3 is mild — session_1 ≈ 0.75× vs session_N; avoids over-penalising
	// historical facts needed for multi-hop / temporal questions.
	// Only applied when at least 2 distinct sessions appear in the result set.
	{
		maxSess := 0
		for _, r := range results {
			if s := sessionFromSource(r.Source); s > maxSess {
				maxSess = s
			}
		}
		if maxSess > 1 {
			const lambdaT = 0.3
			for i, r := range results {
				s := sessionFromSource(r.Source)
				if s > 0 {
					age := float64(maxSess-s) / float64(maxSess)
					results[i].Score *= math.Exp(-lambdaT * age)
				}
			}
			sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
		}
	}

	// ── 8. Leiden §4 Community Reports ────────────────────────────────────────
	// Prepend LLM-generated community summaries for seed-entity communities.
	// Provides global context that helps multi-hop and open-domain questions.
	// Reports are generated at ingestion time and fetched here (no LLM at query time).
	if len(seeds) > 0 {
		communityMap, _ := db.CommunityMap(ctx, groupID)
		seedCommunityIDs := map[int]bool{}
		for _, s := range seeds {
			if cid, ok := communityMap[s.UUID]; ok && cid >= 0 {
				seedCommunityIDs[cid] = true
			}
		}
		if len(seedCommunityIDs) > 0 {
			cids := make([]int, 0, len(seedCommunityIDs))
			for cid := range seedCommunityIDs {
				cids = append(cids, cid)
			}
			reports, _ := db.CommunityReportsForIDs(ctx, groupID, cids)
			// Prepend community reports as high-score context items.
			// Score just above top result to ensure they appear first.
			topScore := 0.0
			if len(results) > 0 {
				topScore = results[0].Score
			}
			for i, report := range reports {
				results = append([]SearchResult{{
					Type:  "community",
					UUID:  "",
					Title: "community context",
					Body:  report,
					Score: topScore + float64(len(reports)-i)*0.01,
				}}, results...)
			}
		}
	}

	return results, nil
}

// sessionFromSource extracts the session number from a source path.
// "locomo/conv-26/session_7" → 7. Returns 0 if not a session source.
func sessionFromSource(source string) int {
	idx := strings.LastIndex(source, "session_")
	if idx < 0 {
		return 0
	}
	numStr := source[idx+8:]
	if slash := strings.IndexByte(numStr, '/'); slash >= 0 {
		numStr = numStr[:slash]
	}
	n, _ := strconv.Atoi(numStr)
	return n
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
