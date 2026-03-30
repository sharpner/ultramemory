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

func TestLLY_Triangle(t *testing.T) {
	// Triangle: 0-1-2, all connected. Each edge has 1 common neighbor.
	// κ = 1/max(2,2) + 1/2 + 1/2 - 1 = 0.5 + 0.5 + 0.5 - 1 = 0.5
	g := makeGraph([][2]int64{{0, 1}, {1, 2}, {0, 2}})
	k := linLuYau(g, 0, 1)
	if math.Abs(k-0.5) > 1e-6 {
		t.Errorf("triangle edge: want κ=0.5, got %f", k)
	}
	t.Logf("triangle κ(0,1) = %.4f", k)
}

func TestLLY_Star(t *testing.T) {
	// Star: 0 connected to 1,2,3,4. No triangles.
	// κ = 0/max(4,1) + 1/4 + 1/1 - 1 = 0 + 0.25 + 1 - 1 = 0.25
	g := makeGraph([][2]int64{{0, 1}, {0, 2}, {0, 3}, {0, 4}})
	k := linLuYau(g, 0, 1)
	if math.Abs(k-0.25) > 1e-6 {
		t.Errorf("star center-leaf: want κ=0.25, got %f", k)
	}
	t.Logf("star κ(0,1) = %.4f", k)
}

func TestLLY_Bridge(t *testing.T) {
	// Two triangles connected by bridge 2-3.
	// Internal (0,1): d_0=2, d_1=2, common=1. κ = 1/2 + 1/2 + 1/2 - 1 = 0.5
	// Bridge (2,3): d_2=3, d_3=3, common=0. κ = 0/3 + 1/3 + 1/3 - 1 = -0.333
	g := makeGraph([][2]int64{
		{0, 1}, {1, 2}, {0, 2},
		{3, 4}, {4, 5}, {3, 5},
		{2, 3},
	})

	kInternal := linLuYau(g, 0, 1)
	kBridge := linLuYau(g, 2, 3)

	t.Logf("internal κ(0,1) = %.4f", kInternal)
	t.Logf("bridge   κ(2,3) = %.4f", kBridge)

	if kBridge >= kInternal {
		t.Errorf("bridge (%f) should be less than internal (%f)", kBridge, kInternal)
	}
	if kBridge >= 0 {
		t.Errorf("bridge should be negative, got %f", kBridge)
	}
	if kInternal <= 0 {
		t.Errorf("internal should be positive, got %f", kInternal)
	}
}

func TestLLY_Symmetry(t *testing.T) {
	g := makeGraph([][2]int64{{0, 1}, {1, 2}, {0, 2}, {2, 3}})
	k1 := linLuYau(g, 0, 1)
	k2 := linLuYau(g, 1, 0)
	if math.Abs(k1-k2) > 1e-10 {
		t.Errorf("not symmetric: κ(0,1)=%f, κ(1,0)=%f", k1, k2)
	}
}

func TestCountCommon(t *testing.T) {
	tests := []struct {
		a, b []int64
		want int
	}{
		{[]int64{1, 2, 3}, []int64{2, 3, 4}, 2},
		{[]int64{1, 2, 3}, []int64{4, 5, 6}, 0},
		{[]int64{1, 2, 3}, []int64{1, 2, 3}, 3},
		{nil, []int64{1}, 0},
	}
	for _, tt := range tests {
		got := countCommon(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("countCommon(%v, %v) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}
