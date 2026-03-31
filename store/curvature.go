package store

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"gonum.org/v1/gonum/graph/community"
	"gonum.org/v1/gonum/graph/simple"
)

// EdgeCurvature holds the Ollivier-Ricci curvature for one entity pair.
type EdgeCurvature struct {
	SourceUUID string  `json:"source_uuid"`
	TargetUUID string  `json:"target_uuid"`
	SourceName string  `json:"source_name,omitempty"`
	TargetName string  `json:"target_name,omitempty"`
	Curvature  float64 `json:"curvature"`
}

// CurvatureStats holds summary statistics from curvature computation.
type CurvatureStats struct {
	TotalEdges int     `json:"total_edges"`
	Bridges    int     `json:"bridges"`  // negative curvature
	Internal   int     `json:"internal"` // positive curvature
	Flat       int     `json:"flat"`     // near-zero (|κ| < 0.05)
	Mean       float64 `json:"mean"`
	Min        float64 `json:"min"`
	Max        float64 `json:"max"`
	Elapsed    string  `json:"elapsed"`
}

// adjGraph is an in-memory undirected graph with sorted neighbor lists.
type adjGraph struct {
	neighbors map[int64][]int64
}

func (g *adjGraph) degree(n int64) int { return len(g.neighbors[n]) }

// sortNeighbors sorts all neighbor lists for binary-search adjacency checks.
func (g *adjGraph) sortNeighbors() {
	for id := range g.neighbors {
		slices.Sort(g.neighbors[id])
	}
}

// edgePair is a canonical (lo, hi) edge representation.
type edgePair struct{ lo, hi int64 }

// ComputeCurvatures computes Ollivier-Ricci curvature for all edges in the group.
// Uses exact ORC via local bipartite matching on small support sets.
// alpha is unused (ORC uses uniform neighbor measures).
func (d *DB) ComputeCurvatures(ctx context.Context, groupID string, _ float64) ([]EdgeCurvature, CurvatureStats, error) {
	start := time.Now()

	// 1. Load entity UUID→ID mapping.
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid FROM entities WHERE group_id = ?`, groupID)
	if err != nil {
		return nil, CurvatureStats{}, fmt.Errorf("load entities: %w", err)
	}
	defer rows.Close() //nolint:errcheck

	uuidToID := map[string]int64{}
	idToUUID := map[int64]string{}
	var nextID int64

	for rows.Next() {
		var uuid string
		if err := rows.Scan(&uuid); err != nil {
			return nil, CurvatureStats{}, err
		}
		uuidToID[uuid] = nextID
		idToUUID[nextID] = uuid
		nextID++
	}
	if err := rows.Err(); err != nil {
		return nil, CurvatureStats{}, err
	}

	slog.Info("curvature: loaded entities", "count", nextID)

	// 2. Build adjacency list + collect unique edges.
	g := &adjGraph{neighbors: make(map[int64][]int64, nextID)}
	edgeSet := map[edgePair]bool{}

	edgeRows, err := d.sql.QueryContext(ctx,
		`SELECT source_uuid, target_uuid FROM edges WHERE group_id = ?`, groupID)
	if err != nil {
		return nil, CurvatureStats{}, fmt.Errorf("load edges: %w", err)
	}
	defer edgeRows.Close() //nolint:errcheck

	for edgeRows.Next() {
		var src, tgt string
		if err := edgeRows.Scan(&src, &tgt); err != nil {
			return nil, CurvatureStats{}, err
		}
		srcID, ok1 := uuidToID[src]
		tgtID, ok2 := uuidToID[tgt]
		if !ok1 || !ok2 || srcID == tgtID {
			continue
		}
		lo, hi := srcID, tgtID
		if lo > hi {
			lo, hi = hi, lo
		}
		ep := edgePair{lo, hi}
		if edgeSet[ep] {
			continue
		}
		edgeSet[ep] = true
		g.neighbors[lo] = append(g.neighbors[lo], hi)
		g.neighbors[hi] = append(g.neighbors[hi], lo)
	}
	if err := edgeRows.Err(); err != nil {
		return nil, CurvatureStats{}, err
	}

	edges := make([]edgePair, 0, len(edgeSet))
	for ep := range edgeSet {
		edges = append(edges, ep)
	}

	g.sortNeighbors()

	slog.Info("curvature: built graph", "nodes", nextID, "unique_edges", len(edges))

	// 3. Parallel ORC computation.
	workers := runtime.NumCPU()
	if workers > 8 {
		workers = 8
	}

	results := make([]EdgeCurvature, len(edges))
	var processed atomic.Int64

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
				k := ollivierRicci(g, ep.lo, ep.hi)
				results[idx] = EdgeCurvature{
					SourceUUID: idToUUID[ep.lo],
					TargetUUID: idToUUID[ep.hi],
					Curvature:  k,
				}
				n := processed.Add(1)
				if n%50000 == 0 {
					slog.Info("curvature: progress", "done", n, "total", len(edges))
				}
			}
		}()
	}
	wg.Wait()

	// 4. Compute stats.
	stats := CurvatureStats{
		TotalEdges: len(results),
		Min:        math.Inf(1),
		Max:        math.Inf(-1),
	}

	for _, r := range results {
		stats.Mean += r.Curvature
		if r.Curvature < stats.Min {
			stats.Min = r.Curvature
		}
		if r.Curvature > stats.Max {
			stats.Max = r.Curvature
		}
		switch {
		case r.Curvature < -0.05:
			stats.Bridges++
		case r.Curvature > 0.05:
			stats.Internal++
		default:
			stats.Flat++
		}
	}
	if len(results) > 0 {
		stats.Mean /= float64(len(results))
	}
	stats.Elapsed = time.Since(start).Round(time.Millisecond).String()

	slog.Info("curvature: done",
		"edges", stats.TotalEdges,
		"bridges", stats.Bridges,
		"internal", stats.Internal,
		"flat", stats.Flat,
		"mean", fmt.Sprintf("%.4f", stats.Mean),
		"elapsed", stats.Elapsed,
	)

	return results, stats, nil
}

// StoreCurvatures persists computed curvatures to the edge_curvatures table.
func (d *DB) StoreCurvatures(ctx context.Context, groupID string, curvatures []EdgeCurvature) error {
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	if _, err := tx.ExecContext(ctx,
		`DELETE FROM edge_curvatures WHERE group_id = ?`, groupID); err != nil {
		return fmt.Errorf("clear old curvatures: %w", err)
	}

	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO edge_curvatures (source_uuid, target_uuid, group_id, curvature) VALUES (?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("prepare: %w", err)
	}
	defer stmt.Close() //nolint:errcheck

	for _, c := range curvatures {
		if _, err := stmt.ExecContext(ctx, c.SourceUUID, c.TargetUUID, groupID, c.Curvature); err != nil {
			return fmt.Errorf("insert curvature: %w", err)
		}
	}

	return tx.Commit()
}

// TopBridges returns the N edges with the most negative curvature (cross-community bridges).
func (d *DB) TopBridges(ctx context.Context, groupID string, n int) ([]EdgeCurvature, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT ec.source_uuid, ec.target_uuid, ec.curvature,
		       COALESCE(e1.name, ''), COALESCE(e2.name, '')
		FROM edge_curvatures ec
		LEFT JOIN entities e1 ON e1.uuid = ec.source_uuid
		LEFT JOIN entities e2 ON e2.uuid = ec.target_uuid
		WHERE ec.group_id = ?
		ORDER BY ec.curvature ASC
		LIMIT ?`, groupID, n)
	if err != nil {
		return nil, fmt.Errorf("top bridges: %w", err)
	}
	defer rows.Close() //nolint:errcheck

	var out []EdgeCurvature
	for rows.Next() {
		var ec EdgeCurvature
		if err := rows.Scan(&ec.SourceUUID, &ec.TargetUUID, &ec.Curvature, &ec.SourceName, &ec.TargetName); err != nil {
			return nil, err
		}
		out = append(out, ec)
	}
	return out, rows.Err()
}

// CurvatureStatus returns quick stats about stored curvatures for the status command.
// Returns zero values if no curvatures computed yet.
func (d *DB) CurvatureStatus(ctx context.Context, groupID string) (total, bridges, internal int, mean float64) {
	row := d.sql.QueryRowContext(ctx, `
		SELECT COUNT(*),
		       SUM(CASE WHEN curvature < -0.05 THEN 1 ELSE 0 END),
		       SUM(CASE WHEN curvature > 0.05 THEN 1 ELSE 0 END),
		       COALESCE(AVG(curvature), 0)
		FROM edge_curvatures WHERE group_id = ?`, groupID)
	_ = row.Scan(&total, &bridges, &internal, &mean)
	return
}

// EdgeCurvatureMap returns a map of (source_uuid, target_uuid) → curvature
// for all edges in the group. Used by search to boost bridge edges.
// Returns nil map (not error) if no curvatures computed yet.
func (d *DB) EdgeCurvatureMap(ctx context.Context, groupID string) (map[[2]string]float64, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT source_uuid, target_uuid, curvature FROM edge_curvatures WHERE group_id = ?`, groupID)
	if err != nil {
		// Table might not exist on legacy DBs — not an error.
		if strings.Contains(err.Error(), "no such table") {
			return nil, nil
		}
		return nil, fmt.Errorf("query curvatures: %w", err)
	}
	defer rows.Close() //nolint:errcheck

	m := map[[2]string]float64{}
	for rows.Next() {
		var src, tgt string
		var k float64
		if err := rows.Scan(&src, &tgt, &k); err != nil {
			return nil, err
		}
		m[[2]string{src, tgt}] = k
		m[[2]string{tgt, src}] = k // symmetric
	}
	return m, rows.Err()
}

// CurvatureWeightedLouvain runs Louvain community detection with ORC as edge weights.
// This is the spin-glass approach: κ(u,v) acts as the coupling constant J_ij.
// Positive curvature = strong coupling (stay together), negative = weak (allow separation).
//
// Edge weight = max(0, κ + offset) where offset shifts all curvatures to non-negative.
// Edges with very negative curvature get weight ≈ 0 (effectively ignored by Louvain).
func (d *DB) CurvatureWeightedLouvain(ctx context.Context, groupID string, resolution float64) (CommunityResult, error) {
	if resolution <= 0 {
		resolution = 1.0
	}

	// 1. Compute curvatures.
	curvatures, stats, err := d.ComputeCurvatures(ctx, groupID, 0)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("compute curvatures: %w", err)
	}
	if err := d.StoreCurvatures(ctx, groupID, curvatures); err != nil {
		return CommunityResult{}, fmt.Errorf("store curvatures: %w", err)
	}

	// 2. Build UUID→curvature map.
	uuidCurvMap := map[[2]string]float64{}
	for _, cv := range curvatures {
		uuidCurvMap[[2]string{cv.SourceUUID, cv.TargetUUID}] = cv.Curvature
		uuidCurvMap[[2]string{cv.TargetUUID, cv.SourceUUID}] = cv.Curvature
	}

	// 3. Load entities.
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid FROM entities WHERE group_id = ?`, groupID)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("load entities: %w", err)
	}
	defer rows.Close() //nolint:errcheck

	uuidToID := map[string]int64{}
	idToUUID := map[int64]string{}
	var nextID int64
	for rows.Next() {
		var uuid string
		if err := rows.Scan(&uuid); err != nil {
			return CommunityResult{}, err
		}
		uuidToID[uuid] = nextID
		idToUUID[nextID] = uuid
		nextID++
	}
	if err := rows.Err(); err != nil {
		return CommunityResult{}, err
	}

	// 4. Build weighted graph with curvature as edge weight.
	g := simple.NewWeightedUndirectedGraph(0, 0)
	for id := int64(0); id < nextID; id++ {
		g.AddNode(simple.Node(id))
	}

	// Offset: shift curvatures so minimum becomes ε > 0.
	// This preserves relative ordering while making all weights positive.
	offset := -stats.Min + 0.01
	slog.Info("curvature-weighted louvain",
		"offset", fmt.Sprintf("%.2f", offset),
		"min_κ", fmt.Sprintf("%.2f", stats.Min),
		"max_κ", fmt.Sprintf("%.2f", stats.Max),
	)

	edgeWeights := map[[2]int64]float64{}
	edgeRows, err := d.sql.QueryContext(ctx,
		`SELECT source_uuid, target_uuid FROM edges WHERE group_id = ?`, groupID)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("load edges: %w", err)
	}
	defer edgeRows.Close() //nolint:errcheck

	for edgeRows.Next() {
		var src, tgt string
		if err := edgeRows.Scan(&src, &tgt); err != nil {
			return CommunityResult{}, err
		}
		srcID, ok1 := uuidToID[src]
		tgtID, ok2 := uuidToID[tgt]
		if !ok1 || !ok2 || srcID == tgtID {
			continue
		}
		key := [2]int64{srcID, tgtID}
		if srcID > tgtID {
			key = [2]int64{tgtID, srcID}
		}
		// Use curvature as weight (shifted to positive).
		k, ok := uuidCurvMap[[2]string{src, tgt}]
		if !ok {
			k = 0 // edges without curvature get neutral weight
		}
		w := k + offset
		if w < 0.01 {
			w = 0.01 // ensure positive
		}
		// Accumulate (multiple edges between same pair).
		edgeWeights[key] += w
	}
	if err := edgeRows.Err(); err != nil {
		return CommunityResult{}, err
	}

	for key, w := range edgeWeights {
		g.SetWeightedEdge(g.NewWeightedEdge(simple.Node(key[0]), simple.Node(key[1]), w))
	}

	// 5. Run Louvain with curvature-informed weights.
	reduced := community.Modularize(g, resolution, nil)
	communities := reduced.Communities()

	// 6. Write community_id.
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

	for communityID, members := range communities {
		for _, node := range members {
			uuid := idToUUID[node.ID()]
			if _, err := stmt.ExecContext(ctx, communityID, uuid, groupID); err != nil {
				return CommunityResult{}, fmt.Errorf("update community: %w", err)
			}
		}
	}

	if err := tx.Commit(); err != nil {
		return CommunityResult{}, fmt.Errorf("commit: %w", err)
	}

	slog.Info("curvature-weighted louvain complete",
		"group", groupID,
		"entities", nextID,
		"communities", len(communities),
	)

	return CommunityResult{
		Communities: len(communities),
		Entities:    int(nextID),
	}, nil
}

// RicciFlowCommunities detects communities via iterative Ricci flow:
// each iteration recomputes ORC on the current graph, removes the most
// negatively curved edges, and repeats until the giant component breaks.
//
// removePct controls how many edges to remove per iteration (fraction of remaining).
// maxIter caps the number of iterations.
func (d *DB) RicciFlowCommunities(ctx context.Context, groupID string, removePct float64) (CommunityResult, error) {
	if removePct <= 0 || removePct >= 1 {
		removePct = 0.05
	}
	const maxIter = 20

	// 1. Load graph from DB.
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid FROM entities WHERE group_id = ?`, groupID)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("load entities: %w", err)
	}
	defer rows.Close() //nolint:errcheck

	uuidToID := map[string]int64{}
	idToUUID := map[int64]string{}
	var nextID int64
	for rows.Next() {
		var uuid string
		if err := rows.Scan(&uuid); err != nil {
			return CommunityResult{}, err
		}
		uuidToID[uuid] = nextID
		idToUUID[nextID] = uuid
		nextID++
	}
	if err := rows.Err(); err != nil {
		return CommunityResult{}, err
	}

	g := &adjGraph{neighbors: make(map[int64][]int64, nextID)}
	edgeSet := map[edgePair]bool{}

	edgeRows, err := d.sql.QueryContext(ctx,
		`SELECT source_uuid, target_uuid FROM edges WHERE group_id = ?`, groupID)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("load edges: %w", err)
	}
	defer edgeRows.Close() //nolint:errcheck

	for edgeRows.Next() {
		var src, tgt string
		if err := edgeRows.Scan(&src, &tgt); err != nil {
			return CommunityResult{}, err
		}
		srcID, ok1 := uuidToID[src]
		tgtID, ok2 := uuidToID[tgt]
		if !ok1 || !ok2 || srcID == tgtID {
			continue
		}
		lo, hi := srcID, tgtID
		if lo > hi {
			lo, hi = hi, lo
		}
		ep := edgePair{lo, hi}
		if edgeSet[ep] {
			continue
		}
		edgeSet[ep] = true
		g.neighbors[lo] = append(g.neighbors[lo], hi)
		g.neighbors[hi] = append(g.neighbors[hi], lo)
	}
	if err := edgeRows.Err(); err != nil {
		return CommunityResult{}, err
	}
	g.sortNeighbors()

	totalEdges := len(edgeSet)
	totalRemoved := 0

	slog.Info("ricci flow: loaded graph", "nodes", nextID, "edges", totalEdges)

	// 2. Iterative flow: compute ORC → remove bottom edges → repeat.
	for iter := 0; iter < maxIter; iter++ {
		// Collect current edges.
		var currentEdges []edgePair
		for ep := range edgeSet {
			currentEdges = append(currentEdges, ep)
		}
		if len(currentEdges) == 0 {
			break
		}

		// Compute ORC on current graph.
		type scoredEdge struct {
			ep edgePair
			k  float64
		}
		scored := make([]scoredEdge, len(currentEdges))
		for i, ep := range currentEdges {
			scored[i] = scoredEdge{ep, ollivierRicci(g, ep.lo, ep.hi)}
		}
		slices.SortFunc(scored, func(a, b scoredEdge) int {
			if a.k < b.k {
				return -1
			}
			if a.k > b.k {
				return 1
			}
			return 0
		})

		// Stop if no more negative-curvature edges (nothing useful to cut).
		if scored[0].k >= 0 {
			slog.Info("ricci flow: no negative edges left, stopping", "iter", iter+1)
			break
		}

		// Remove bottom removePct% of remaining edges (only negative ones).
		nRemove := int(float64(len(scored)) * removePct)
		if nRemove < 1 {
			nRemove = 1
		}

		for i := 0; i < nRemove && i < len(scored); i++ {
			if scored[i].k >= 0 {
				break // don't remove positive-curvature edges
			}
			ep := scored[i].ep
			delete(edgeSet, ep)
			// Remove from adjacency.
			g.neighbors[ep.lo] = removeFromSorted(g.neighbors[ep.lo], ep.hi)
			g.neighbors[ep.hi] = removeFromSorted(g.neighbors[ep.hi], ep.lo)
			totalRemoved++
		}

		// Check giant component.
		giant := largestComponent(g, nextID)
		giantPct := float64(giant) / float64(nextID) * 100

		slog.Info("ricci flow: iteration",
			"iter", iter+1,
			"edges_remaining", len(edgeSet),
			"removed_this_iter", nRemove,
			"total_removed", totalRemoved,
			"giant_component", giant,
			"giant_pct", fmt.Sprintf("%.1f%%", giantPct),
		)

		// Stop when giant component is < 50% of nodes.
		if giantPct < 50 {
			slog.Info("ricci flow: giant component broken", "iter", iter+1)
			break
		}
	}

	// 3. Final connected components → communities.
	visited := make(map[int64]bool, nextID)
	var communities [][]int64

	for id := int64(0); id < nextID; id++ {
		if visited[id] {
			continue
		}
		var component []int64
		queue := []int64{id}
		for len(queue) > 0 {
			n := queue[0]
			queue = queue[1:]
			if visited[n] {
				continue
			}
			visited[n] = true
			component = append(component, n)
			for _, nb := range g.neighbors[n] {
				if !visited[nb] {
					queue = append(queue, nb)
				}
			}
		}
		communities = append(communities, component)
	}

	// 4. Store final curvatures for search signal.
	var finalCurvatures []EdgeCurvature
	for ep := range edgeSet {
		k := ollivierRicci(g, ep.lo, ep.hi)
		finalCurvatures = append(finalCurvatures, EdgeCurvature{
			SourceUUID: idToUUID[ep.lo],
			TargetUUID: idToUUID[ep.hi],
			Curvature:  k,
		})
	}
	if err := d.StoreCurvatures(ctx, groupID, finalCurvatures); err != nil {
		slog.Warn("store final curvatures failed", "err", err)
	}

	// 5. Write community_id to database.
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
		for _, id := range members {
			if _, err := stmt.ExecContext(ctx, cid, idToUUID[id], groupID); err != nil {
				return CommunityResult{}, fmt.Errorf("update community: %w", err)
			}
		}
	}

	if err := tx.Commit(); err != nil {
		return CommunityResult{}, fmt.Errorf("commit: %w", err)
	}

	slog.Info("ricci flow community detection complete",
		"group", groupID,
		"entities", nextID,
		"communities", len(communities),
		"total_edges_removed", totalRemoved,
		"edges_remaining", len(edgeSet),
	)

	return CommunityResult{
		Communities: len(communities),
		Entities:    int(nextID),
	}, nil
}

// removeFromSorted removes val from a sorted slice in O(log n + n).
func removeFromSorted(s []int64, val int64) []int64 {
	i, found := slices.BinarySearch(s, val)
	if !found {
		return s
	}
	return slices.Delete(s, i, i+1)
}

// largestComponent returns the size of the largest connected component.
func largestComponent(g *adjGraph, nodeCount int64) int {
	visited := make([]bool, nodeCount)
	largest := 0
	for id := int64(0); id < nodeCount; id++ {
		if visited[id] {
			continue
		}
		size := 0
		queue := []int64{id}
		for len(queue) > 0 {
			n := queue[0]
			queue = queue[1:]
			if visited[n] {
				continue
			}
			visited[n] = true
			size++
			for _, nb := range g.neighbors[n] {
				if !visited[nb] {
					queue = append(queue, nb)
				}
			}
		}
		if size > largest {
			largest = size
		}
	}
	return largest
}

// --- Ollivier-Ricci Curvature ---
//
// κ(u,v) = 1 - W₁(μ_u, μ_v)
//
// where μ_u = uniform distribution over N(u) (neighbors of u, NOT including u),
// and W₁ is the Wasserstein-1 distance on the shortest-path metric.
//
// For adjacent nodes with small neighborhoods (avg degree ~3), the optimal
// transport is computed via greedy bipartite matching on the distance matrix
// between support sets. This is exact for most practical cases.
//
// Reference: Ollivier, "Ricci curvature of Markov chains on metric spaces" (2009)

// ollivierRicci computes the Ollivier-Ricci curvature κ(u,v) for adjacent nodes.
// Uses uniform measures over neighbors (α=0, no laziness).
// Solves the optimal transport via greedy matching on sorted neighbor lists.
func ollivierRicci(g *adjGraph, u, v int64) float64 {
	du := g.degree(u)
	dv := g.degree(v)
	if du == 0 || dv == 0 {
		return 0
	}

	nu := g.neighbors[u]
	nv := g.neighbors[v]

	// Build distance matrix between support points of μ_u and μ_v.
	// μ_u is uniform on N(u), μ_v is uniform on N(v).
	// Distances between neighbors of adjacent nodes are 0, 1, 2, or 3.
	//
	// d(a,b) = 0 if a == b (shared neighbor)
	// d(a,b) = 1 if a and b are adjacent
	// d(a,b) = 2 if both are neighbors of same endpoint (a,b ∈ N(u) or a,b ∈ N(v))
	//           or if one is an endpoint (a==v → d(v, nb_of_v)=1, already covered)
	// d(a,b) = 3 otherwise (a ∈ N(u)\N(v), b ∈ N(v)\N(u), not adjacent)

	// Classify neighbors for fast distance computation.
	uSet := make(map[int64]bool, du)
	for _, w := range nu {
		uSet[w] = true
	}
	vSet := make(map[int64]bool, dv)
	for _, w := range nv {
		vSet[w] = true
	}

	// Build cost matrix C[i][j] = d(nu[i], nv[j]).
	cost := make([]float64, du*dv)
	for i, a := range nu {
		for j, b := range nv {
			switch {
			case a == b:
				cost[i*dv+j] = 0
			case isAdj(g, a, b):
				cost[i*dv+j] = 1
			case uSet[b] || vSet[a]:
				// b also in N(u): d(a,b) ≤ 2 via u
				// a also in N(v): d(a,b) ≤ 2 via v
				cost[i*dv+j] = 2
			default:
				cost[i*dv+j] = 3
			}
		}
	}

	// Compute W₁ = optimal transport cost with uniform weights.
	// Mass: 1/du at each source, 1/dv at each sink.
	w1 := transportCost(cost, du, dv)

	return 1.0 - w1
}

// isAdj checks adjacency via binary search on sorted neighbor list.
func isAdj(g *adjGraph, a, b int64) bool {
	nbs := g.neighbors[a]
	_, found := slices.BinarySearch(nbs, b)
	return found
}

// transportCost solves the optimal transport between uniform distributions
// on du sources and dv sinks with cost matrix C (flat, du×dv).
// Returns the total transport cost (W₁).
//
// Uses a greedy approach: iteratively match the cheapest (source, sink) pair,
// transferring as much mass as possible. This is exact for integer-ratio
// uniform distributions and a very good approximation otherwise.
func transportCost(C []float64, du, dv int) float64 {
	// Supply and demand: uniform masses.
	supply := make([]float64, du)
	demand := make([]float64, dv)
	for i := range supply {
		supply[i] = 1.0 / float64(du)
	}
	for j := range demand {
		demand[j] = 1.0 / float64(dv)
	}

	// Build sorted list of (cost, i, j) entries.
	type entry struct {
		cost float64
		i, j int
	}
	entries := make([]entry, 0, du*dv)
	for i := 0; i < du; i++ {
		for j := 0; j < dv; j++ {
			entries = append(entries, entry{C[i*dv+j], i, j})
		}
	}
	slices.SortFunc(entries, func(a, b entry) int {
		if a.cost < b.cost {
			return -1
		}
		if a.cost > b.cost {
			return 1
		}
		return 0
	})

	// Greedy assignment: cheapest pairs first.
	total := 0.0
	for _, e := range entries {
		if supply[e.i] < 1e-12 || demand[e.j] < 1e-12 {
			continue
		}
		flow := supply[e.i]
		if demand[e.j] < flow {
			flow = demand[e.j]
		}
		total += flow * e.cost
		supply[e.i] -= flow
		demand[e.j] -= flow
	}

	return total
}

// countCommon counts elements in both sorted slices (two-pointer merge).
func countCommon(a, b []int64) int {
	n := 0
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		if a[i] == b[j] {
			n++
			i++
			j++
		} else if a[i] < b[j] {
			i++
		} else {
			j++
		}
	}
	return n
}
