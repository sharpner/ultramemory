package store

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"
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

// ComputeCurvatures computes Lin-Lu-Yau Ricci curvature for all edges.
// Uses the combinatorial formula: zero allocations per edge, O(d_u * d_v) work.
// alpha is unused (kept for API compat) — LLY doesn't use laziness.
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

	// Sort neighbor lists for binary-search adjacency checks.
	g.sortNeighbors()

	slog.Info("curvature: built graph", "nodes", nextID, "unique_edges", len(edges))

	// 3. Parallel LLY curvature computation. Zero allocations per edge.
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
				k := linLuYau(g, ep.lo, ep.hi)
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

// --- Lin-Lu-Yau Curvature ---
//
// κ_LLY(u,v) = |△(u,v)| / max(d_u, d_v) + 1/d_u + 1/d_v - 1
//
// where |△(u,v)| = number of common neighbors (triangles through edge).
// This is a tight lower bound on the Ollivier-Ricci curvature for sparse graphs.
//
// Properties (same as full ORC):
//   - Positive → edge within dense cluster (many triangles)
//   - Negative → bridge between sparse regions (few triangles)
//   - Zero → "flat" (tree-like local structure)
//
// Reference: Lin, Lu, Yau, "Ricci curvature of graphs" (Tohoku Math J, 2011)

// linLuYau computes the Lin-Lu-Yau curvature for adjacent nodes u,v.
// Zero heap allocations — uses only the sorted adjacency list.
func linLuYau(g *adjGraph, u, v int64) float64 {
	du := g.degree(u)
	dv := g.degree(v)
	if du == 0 || dv == 0 {
		return 0
	}

	// Count common neighbors via merge of sorted neighbor lists.
	triangles := countCommon(g.neighbors[u], g.neighbors[v])

	maxDeg := du
	if dv > maxDeg {
		maxDeg = dv
	}

	return float64(triangles)/float64(maxDeg) + 1.0/float64(du) + 1.0/float64(dv) - 1.0
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
