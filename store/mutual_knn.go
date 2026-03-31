package store

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/coder/hnsw"
	"gonum.org/v1/gonum/graph/community"
	"gonum.org/v1/gonum/graph/simple"
)

// MutualKNNGraph builds a mutual k-NN graph from entity embeddings.
// Edge (i,j) exists only if BOTH i ∈ kNN(j) AND j ∈ kNN(i).
// This eliminates hub nodes (arXiv: 1895→4) while preserving semantic clusters.
//
// Returns an adjGraph with sorted neighbor lists, ready for ORC computation.
func (d *DB) MutualKNNGraph(ctx context.Context, groupID string, k int) (*adjGraph, map[int64]string, int64, error) {
	if k <= 0 {
		k = 20
	}
	start := time.Now()

	// 1. Load entity UUIDs + embeddings.
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid, embedding FROM entities WHERE group_id = ? AND embedding IS NOT NULL`, groupID)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("load entities: %w", err)
	}
	defer rows.Close() //nolint:errcheck

	var uuids []string
	var embeddings [][]float32
	for rows.Next() {
		var uuid string
		var blob []byte
		if err := rows.Scan(&uuid, &blob); err != nil {
			return nil, nil, 0, err
		}
		emb := DecodeEmbedding(blob)
		if len(emb) == 0 {
			continue
		}
		uuids = append(uuids, uuid)
		embeddings = append(embeddings, emb)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, 0, err
	}

	n := len(uuids)
	if n < 2 {
		return nil, nil, 0, fmt.Errorf("need at least 2 entities with embeddings, got %d", n)
	}

	dim := len(embeddings[0])
	slog.Info("mutual-knn: loaded embeddings", "entities", n, "dim", dim, "k", k)

	// 2. Normalize embeddings for cosine similarity.
	for i := range embeddings {
		norm := float32(0)
		for _, v := range embeddings[i] {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 {
			for j := range embeddings[i] {
				embeddings[i][j] /= norm
			}
		}
	}

	// 3. HNSW approximate kNN (coder/hnsw, pure Go).
	// Build index, then search each point for k neighbors.
	hnswGraph := hnsw.NewGraph[int]()
	hnswGraph.M = 24
	hnswGraph.EfSearch = k * 4 // oversample for mutual filter accuracy

	// Add all entities to HNSW index.
	nodes := make([]hnsw.Node[int], n)
	for i := 0; i < n; i++ {
		nodes[i] = hnsw.MakeNode(i, embeddings[i])
	}
	hnswGraph.Add(nodes...)

	slog.Info("mutual-knn: HNSW index built", "elapsed", time.Since(start).Round(time.Millisecond))

	// Search k neighbors for each entity (parallel).
	type knnResult struct {
		neighbors []int
	}

	results := make([]knnResult, n)
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}

	chunkSize := (n + workers - 1) / workers
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		lo := w * chunkSize
		hi := lo + chunkSize
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				found := hnswGraph.Search(embeddings[i], k+1) // +1 for self
				nbs := make([]int, 0, k)
				for _, nd := range found {
					if nd.Key != i { // exclude self
						nbs = append(nbs, nd.Key)
					}
				}
				results[i] = knnResult{neighbors: nbs}

				if i%10000 == 0 && i > 0 {
					slog.Info("mutual-knn: search progress", "done", i, "total", n)
				}
			}
		}(lo, hi)
	}
	wg.Wait()

	slog.Info("mutual-knn: kNN search complete", "elapsed", time.Since(start).Round(time.Millisecond))

	// 4. Mutual filter: edge (i,j) only if both i∈kNN(j) AND j∈kNN(i).
	neighborSets := make([]map[int]bool, n)
	for i := range results {
		s := make(map[int]bool, len(results[i].neighbors))
		for _, nb := range results[i].neighbors {
			s[nb] = true
		}
		neighborSets[i] = s
	}

	g := &adjGraph{neighbors: make(map[int64][]int64, n)}
	idToUUID := make(map[int64]string, n)
	edgeCount := 0

	for i := 0; i < n; i++ {
		idToUUID[int64(i)] = uuids[i]
		for j := range neighborSets[i] {
			if j > i && neighborSets[j][i] { // mutual
				g.neighbors[int64(i)] = append(g.neighbors[int64(i)], int64(j))
				g.neighbors[int64(j)] = append(g.neighbors[int64(j)], int64(i))
				edgeCount++
			}
		}
	}

	g.sortNeighbors()

	// Stats.
	maxDeg := 0
	for id := int64(0); id < int64(n); id++ {
		d := g.degree(id)
		if d > maxDeg {
			maxDeg = d
		}
	}

	slog.Info("mutual-knn: graph built",
		"entities", n,
		"mutual_edges", edgeCount,
		"max_degree", maxDeg,
		"elapsed", time.Since(start).Round(time.Millisecond),
	)

	return g, idToUUID, int64(n), nil
}

// dot computes the inner product of two float32 vectors.
func dot(a, b []float32) float32 {
	s := float32(0)
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// MutualKNNCommunities builds a mutual-kNN graph from entity embeddings,
// computes ORC on it, then runs Louvain with curvature weights.
// This is the full pipeline: embeddings → kNN → mutual filter → ORC → Louvain → community_id.
func (d *DB) MutualKNNCommunities(ctx context.Context, groupID string, k int, resolution float64) (CommunityResult, error) {
	if resolution <= 0 {
		resolution = 1.0
	}

	// 1. Build mutual-kNN graph.
	g, idToUUID, nodeCount, err := d.MutualKNNGraph(ctx, groupID, k)
	if err != nil {
		return CommunityResult{}, err
	}

	// 2. Collect edges.
	edgeSet := map[edgePair]bool{}
	for id, nbs := range g.neighbors {
		for _, nb := range nbs {
			lo, hi := id, nb
			if lo > hi {
				lo, hi = hi, lo
			}
			edgeSet[edgePair{lo, hi}] = true
		}
	}

	edges := make([]edgePair, 0, len(edgeSet))
	for ep := range edgeSet {
		edges = append(edges, ep)
	}

	slog.Info("mutual-knn-communities: computing ORC", "edges", len(edges))

	// 3. Compute ORC in parallel.
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}

	type scoredEdge struct {
		ep edgePair
		k  float64
	}
	scored := make([]scoredEdge, len(edges))

	var wg sync.WaitGroup
	ch := make(chan int, 256)
	go func() {
		for i := range edges {
			ch <- i
		}
		close(ch)
	}()

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range ch {
				ep := edges[idx]
				scored[idx] = scoredEdge{ep, ollivierRicci(g, ep.lo, ep.hi)}
			}
		}()
	}
	wg.Wait()

	// Stats.
	minK, maxK, sumK := math.Inf(1), math.Inf(-1), 0.0
	for _, s := range scored {
		sumK += s.k
		if s.k < minK {
			minK = s.k
		}
		if s.k > maxK {
			maxK = s.k
		}
	}
	meanK := sumK / float64(len(scored))

	slog.Info("mutual-knn-communities: ORC done",
		"mean_κ", fmt.Sprintf("%.4f", meanK),
		"min_κ", fmt.Sprintf("%.4f", minK),
		"max_κ", fmt.Sprintf("%.4f", maxK),
	)

	// 4. Store curvatures (mapped back to UUIDs).
	var curvatures []EdgeCurvature
	for _, s := range scored {
		curvatures = append(curvatures, EdgeCurvature{
			SourceUUID: idToUUID[s.ep.lo],
			TargetUUID: idToUUID[s.ep.hi],
			Curvature:  s.k,
		})
	}
	if err := d.StoreCurvatures(ctx, groupID, curvatures); err != nil {
		slog.Warn("store curvatures failed", "err", err)
	}

	// 5. Louvain with curvature weights on the mutual-kNN graph.
	offset := -minK + 0.01
	louvainGraph := simple.NewWeightedUndirectedGraph(0, 0)
	for id := int64(0); id < nodeCount; id++ {
		louvainGraph.AddNode(simple.Node(id))
	}
	for _, s := range scored {
		w := s.k + offset
		if w < 0.01 {
			w = 0.01
		}
		louvainGraph.SetWeightedEdge(louvainGraph.NewWeightedEdge(
			simple.Node(s.ep.lo), simple.Node(s.ep.hi), w))
	}

	reduced := community.Modularize(louvainGraph, resolution, nil)
	communities := reduced.Communities()

	// 6. Write community_id to entities table.
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	stmt, err := tx.PrepareContext(ctx,
		`UPDATE entities SET community_id = ? WHERE uuid = ? AND group_id = ?`)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("prepare: %w", err)
	}
	defer stmt.Close() //nolint:errcheck

	for cid, members := range communities {
		for _, node := range members {
			uuid := idToUUID[node.ID()]
			if _, err := stmt.ExecContext(ctx, cid, uuid, groupID); err != nil {
				return CommunityResult{}, fmt.Errorf("update community: %w", err)
			}
		}
	}

	if err := tx.Commit(); err != nil {
		return CommunityResult{}, fmt.Errorf("commit: %w", err)
	}

	slog.Info("mutual-knn-communities: complete",
		"entities", nodeCount,
		"communities", len(communities),
	)

	return CommunityResult{
		Communities: len(communities),
		Entities:    int(nodeCount),
	}, nil
}

// Helper to sort slices (used by adjGraph.sortNeighbors, already defined in curvature.go).
func init() {
	// slices.Sort is available — no init needed.
	_ = slices.Sort[[]int64]
}
