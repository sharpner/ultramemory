package store

import (
	"math"
	"testing"
)

// makeGraph builds an adjGraph from an edge list with sorted neighbors.
func makeGraph(edges [][2]int64) *adjGraph {
	g := &adjGraph{neighbors: map[int64][]int64{}}
	for _, e := range edges {
		g.neighbors[e[0]] = append(g.neighbors[e[0]], e[1])
		g.neighbors[e[1]] = append(g.neighbors[e[1]], e[0])
	}
	g.sortNeighbors()
	return g
}

func approxEq(a, b, eps float64) bool { return math.Abs(a-b) < eps }

func TestORC_Triangle(t *testing.T) {
	// K₃: each node has degree 2, each edge has 1 common neighbor.
	// μ_u = {1/2 on each of 2 neighbors}, μ_v = {1/2 on each of 2 neighbors}.
	// For edge (0,1): N(0)={1,2}, N(1)={0,2}.
	// Transport: 1/2 at node 2 → node 2 (cost 0), 1/2 at node 1 → node 0 (cost... wait)
	// μ_0 = 1/2 on node 1, 1/2 on node 2
	// μ_1 = 1/2 on node 0, 1/2 on node 2
	// Shared: node 2 (mass 1/2 each) → cost 0
	// Remaining: 1/2 at node 1 (from μ_0) → node 0 (from μ_1), d(1,0)=1 → cost 1/2
	// W₁ = 0 + 1/2 = 1/2, κ = 1 - 1/2 = 1/2
	g := makeGraph([][2]int64{{0, 1}, {1, 2}, {0, 2}})
	k := ollivierRicci(g, 0, 1)
	t.Logf("triangle κ(0,1) = %.4f", k)
	if !approxEq(k, 0.5, 0.01) {
		t.Errorf("triangle: want κ≈0.5, got %f", k)
	}
}

func TestORC_Star(t *testing.T) {
	// Star S₄: center 0, leaves 1,2,3,4. No triangles.
	// Edge (0,1): N(0)={1,2,3,4}, N(1)={0}.
	// μ_0 = 1/4 each on {1,2,3,4}
	// μ_1 = 1 on {0}
	// Transport: 1/4 at node 1 → node 0, cost d(1,0)=1 → 1/4
	//            1/4 at node 2 → node 0, cost d(2,0)=1 → 1/4
	//            1/4 at node 3 → node 0, cost d(3,0)=1 → 1/4
	//            1/4 at node 4 → node 0, cost d(4,0)=1 → 1/4
	// W₁ = 1, κ = 1 - 1 = 0
	g := makeGraph([][2]int64{{0, 1}, {0, 2}, {0, 3}, {0, 4}})
	k := ollivierRicci(g, 0, 1)
	t.Logf("star κ(0,1) = %.4f", k)
	if !approxEq(k, 0.0, 0.01) {
		t.Errorf("star: want κ≈0, got %f", k)
	}
}

func TestORC_Path(t *testing.T) {
	// Path 0-1-2-3: edge (1,2).
	// N(1)={0,2}, N(2)={1,3}. μ_1 = 1/2 on {0,2}, μ_2 = 1/2 on {1,3}.
	// Optimal: 0→1 (cost 1, flow 1/2) + 2→3 (cost 1, flow 1/2) = W₁=1.0
	// κ = 1 - 1 = 0. Paths are flat in ORC — same as trees.
	g := makeGraph([][2]int64{{0, 1}, {1, 2}, {2, 3}})
	k := ollivierRicci(g, 1, 2)
	t.Logf("path κ(1,2) = %.4f", k)
	if !approxEq(k, 0.0, 0.01) {
		t.Errorf("path: want κ≈0.0, got %f", k)
	}
}

func TestORC_Bridge(t *testing.T) {
	// Two triangles connected by bridge 2-3.
	// Triangle 1: 0-1-2, Triangle 2: 3-4-5.
	//
	// Internal edge (0,1): d_0=2, d_1=2, 1 common neighbor (node 2).
	// Same as triangle above → κ = 0.5
	//
	// Bridge edge (2,3): d_2=3 (N={0,1,3}), d_3=3 (N={2,4,5}).
	// μ_2 = 1/3 on {0,1,3}, μ_3 = 1/3 on {2,4,5}
	// Distances: d(3,2)=1, d(0,2)=1, d(1,2)=1, d(0,4)=3, d(0,5)=3, d(1,4)=3, d(1,5)=3, d(3,4)=1, d(3,5)=1
	// Cheapest matching: 3→2 (cost 1, flow 1/3), then 0→4 or 0→5 (cost 3, flow 1/3), 1→remaining (cost 3, flow 1/3)
	// Optimal: 0→2 (cost 1), 3→4 (cost 1), 1→5 (cost 3) → W₁ = 5/3
	// κ = 1 - 5/3 = -2/3 ≈ -0.667
	g := makeGraph([][2]int64{
		{0, 1}, {1, 2}, {0, 2}, // triangle 1
		{3, 4}, {4, 5}, {3, 5}, // triangle 2
		{2, 3},                  // bridge
	})

	kInternal := ollivierRicci(g, 0, 1)
	kBridge := ollivierRicci(g, 2, 3)

	t.Logf("internal κ(0,1) = %.4f", kInternal)
	t.Logf("bridge   κ(2,3) = %.4f", kBridge)

	if !approxEq(kInternal, 0.5, 0.01) {
		t.Errorf("internal: want κ≈0.5, got %f", kInternal)
	}
	if !approxEq(kBridge, -2.0/3.0, 0.05) {
		t.Errorf("bridge: want κ≈-0.667, got %f", kBridge)
	}
	if kBridge >= kInternal {
		t.Errorf("bridge (%f) should be less than internal (%f)", kBridge, kInternal)
	}
}

func TestORC_Symmetry(t *testing.T) {
	g := makeGraph([][2]int64{{0, 1}, {1, 2}, {0, 2}, {2, 3}})
	k1 := ollivierRicci(g, 0, 1)
	k2 := ollivierRicci(g, 1, 0)
	if math.Abs(k1-k2) > 1e-10 {
		t.Errorf("not symmetric: κ(0,1)=%f, κ(1,0)=%f", k1, k2)
	}
}

func TestORC_K4(t *testing.T) {
	// Complete graph K₄: each node has degree 3, each edge has 2 common neighbors.
	// Edge (0,1): N(0)={1,2,3}, N(1)={0,2,3}
	// Shared: {2,3} → 2/3 mass matched at cost 0
	// Remaining: 1/3 at node 1 → 1/3 at node 0, cost 1
	// W₁ = 1/3, κ = 1 - 1/3 = 2/3
	g := makeGraph([][2]int64{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}})
	k := ollivierRicci(g, 0, 1)
	t.Logf("K₄ κ(0,1) = %.4f", k)
	if !approxEq(k, 2.0/3.0, 0.01) {
		t.Errorf("K₄: want κ≈0.667, got %f", k)
	}
}

func TestTransportCost_Identical(t *testing.T) {
	// Same distribution: cost should be 0.
	C := []float64{0, 1, 1, 0}
	w := transportCost(C, 2, 2)
	if w > 0.01 {
		t.Errorf("identical: want W₁≈0, got %f", w)
	}
}

func TestTransportCost_Dirac(t *testing.T) {
	// Two diracs at distance 1: W₁ = 1.
	// But uniform on 1 source and 1 sink: supply={1}, demand={1}, cost=1.
	C := []float64{1}
	w := transportCost(C, 1, 1)
	if !approxEq(w, 1.0, 0.01) {
		t.Errorf("dirac: want W₁≈1, got %f", w)
	}
}

