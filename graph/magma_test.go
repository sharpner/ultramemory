package graph

import (
	"context"
	"math"
	"testing"

	"github.com/sharpner/ultramemory/store"
)

// mockGraph implements GraphTraverser with an in-memory adjacency list.
type mockGraph map[string][]store.NeighborEntity

func (m mockGraph) GetNeighbors(_ context.Context, uuid, _ string) ([]store.NeighborEntity, error) {
	return m[uuid], nil
}

// emb creates a normalised 2-D embedding for testing cosine similarity.
func emb(x, y float32) []float32 {
	n := float32(math.Sqrt(float64(x*x + y*y)))
	if n == 0 {
		return []float32{0, 0}
	}
	return []float32{x / n, y / n}
}

func TestSpreadMAGMA_BasicHop(t *testing.T) {
	g := mockGraph{
		"alice": {{UUID: "bob", Name: "Bob", EntityType: "person", GroupID: "grp", EdgeFact: "knows alice"}},
	}
	seeds := []ActivatedNode{{UUID: "alice", Name: "Alice"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, Threshold: 0.1, MaxNodes: 10, Decay: 0.5, Lambda2: 0.5}

	// No embeddings, EdgeName="" → EdgeUnknown → φ=0 → S=exp(0)=1.0
	// newScore = 1.0 * 0.5 * 1.0 = 0.5 > 0.1
	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range results {
		if r.UUID == "bob" {
			found = true
		}
	}
	if !found {
		t.Error("expected bob in results after 1-hop traversal")
	}
}

func TestSpreadMAGMA_Accumulation(t *testing.T) {
	// shared is reachable from both seeds — activation should accumulate.
	g := mockGraph{
		"seed1": {{UUID: "shared", Name: "Shared", EntityType: "person", GroupID: "grp", EdgeFact: "connection"}},
		"seed2": {{UUID: "shared", Name: "Shared", EntityType: "person", GroupID: "grp", EdgeFact: "connection"}},
	}
	seeds := []ActivatedNode{
		{UUID: "seed1", Name: "Seed1"},
		{UUID: "seed2", Name: "Seed2"},
	}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, Threshold: 0.01, MaxNodes: 10, Decay: 0.5, Lambda2: 0.5}

	// No queryEmb, EdgeName="" → S=1.0; each seed contributes 1.0*0.5*1.0=0.5 → total 1.0
	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	var sharedAct float64
	for _, r := range results {
		if r.UUID == "shared" {
			sharedAct = r.Activation
		}
	}
	// Both seeds contribute 0.5 each → total 1.0
	if sharedAct < 0.8 {
		t.Errorf("shared should accumulate from both seeds, got %f (want >= 0.8)", sharedAct)
	}
}

func TestSpreadMAGMA_Threshold(t *testing.T) {
	// Chain a→b→c→d; deeper nodes fall below threshold.
	g := mockGraph{
		"a": {{UUID: "b", Name: "B", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
		"b": {{UUID: "c", Name: "C", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
		"c": {{UUID: "d", Name: "D", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
	}
	seeds := []ActivatedNode{{UUID: "a", Name: "A"}}
	// EdgeName="" → EdgeUnknown → φ=0 → S=1.0; Decay=0.5:
	// a=1.0, b=0.5, c=0.25, d=0.125 < Threshold=0.2 → pruned
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 4, Threshold: 0.2, MaxNodes: 10, Decay: 0.5, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range results {
		if r.Activation < cfg.Threshold {
			t.Errorf("node %s: activation %f below threshold %f", r.UUID, r.Activation, cfg.Threshold)
		}
	}
	// d (0.125) must not appear
	for _, r := range results {
		if r.UUID == "d" {
			t.Error("node d should be filtered by threshold")
		}
	}
}

func TestSpreadMAGMA_BeamWidth(t *testing.T) {
	// center → 5 neighbors; BeamWidth=2 trims the next beam.
	g := mockGraph{
		"center": {
			{UUID: "n1", Name: "N1", EntityType: "x", GroupID: "grp", EdgeFact: "link"},
			{UUID: "n2", Name: "N2", EntityType: "x", GroupID: "grp", EdgeFact: "link"},
			{UUID: "n3", Name: "N3", EntityType: "x", GroupID: "grp", EdgeFact: "link"},
			{UUID: "n4", Name: "N4", EntityType: "x", GroupID: "grp", EdgeFact: "link"},
			{UUID: "n5", Name: "N5", EntityType: "x", GroupID: "grp", EdgeFact: "link"},
		},
	}
	seeds := []ActivatedNode{{UUID: "center", Name: "Center"}}
	cfg := MAGMAConfig{BeamWidth: 2, MaxHops: 1, Threshold: 0.01, MaxNodes: 50, Decay: 0.5, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	// All 5 neighbors receive activation during hop-0 (above threshold 0.01).
	if len(results) < 2 {
		t.Errorf("expected at least 2 results, got %d", len(results))
	}
}

func TestSpreadMAGMA_SemanticAffinity(t *testing.T) {
	// Use 2-D embeddings: query points along X axis.
	// techcorp aligns with X → high cos_sim; unrelated aligns with Y → low cos_sim.
	// EdgeName="" → EdgeUnknown → φ=0 → S = exp(λ₂·cos_sim) only.
	queryEmb := emb(1, 0)
	g := mockGraph{
		"alice": {
			{UUID: "techcorp", Name: "TechCorp", EntityType: "company", GroupID: "grp",
				EdgeFact: "works at", Embedding: emb(1, 0)}, // cos_sim ≈ 1.0
			{UUID: "unrelated", Name: "Unrelated", EntityType: "misc", GroupID: "grp",
				EdgeFact: "other", Embedding: emb(0, 1)}, // cos_sim = 0.0
		},
	}
	seeds := []ActivatedNode{{UUID: "alice", Name: "Alice"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, Threshold: 0.01, MaxNodes: 10, Decay: 0.5, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", queryEmb, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	var techScore, unrelScore float64
	for _, r := range results {
		switch r.UUID {
		case "techcorp":
			techScore = r.Activation
		case "unrelated":
			unrelScore = r.Activation
		}
	}
	// techcorp: exp(0.5*1.0)≈1.65, unrelated: exp(0.5*0.0)=1.0
	if techScore <= unrelScore {
		t.Errorf("techcorp (%f) should score higher than unrelated (%f) due to embedding similarity", techScore, unrelScore)
	}
}

func TestSpreadMAGMA_IntentRouting(t *testing.T) {
	// CAUSES edge with "why" query → high φ → boosted score vs default query.
	g := mockGraph{
		"alice": {
			{UUID: "event", Name: "Event", EntityType: "event", GroupID: "grp",
				EdgeFact: "causes event", EdgeName: "CAUSES"},
		},
	}
	seeds := []ActivatedNode{{UUID: "alice", Name: "Alice"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, Threshold: 0.01, MaxNodes: 10, Decay: 0.5, Lambda1: 0.5, Lambda2: 0.5}

	// "why" query → IntentWhy → φ(EdgeCausal, IntentWhy)=1.0 → S=exp(0.5*1.0)≈1.649
	whyResults, err := SpreadMAGMA(context.Background(), g, seeds, "why did alice cause the event", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	// "tell me" query → IntentWhat → φ(EdgeCausal, IntentWhat)=0.5 → S=exp(0.5*0.5)≈1.284
	whatResults, err := SpreadMAGMA(context.Background(), g, seeds, "tell me about alice", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}

	var whyScore, whatScore float64
	for _, r := range whyResults {
		if r.UUID == "event" {
			whyScore = r.Activation
		}
	}
	for _, r := range whatResults {
		if r.UUID == "event" {
			whatScore = r.Activation
		}
	}
	if whyScore <= whatScore {
		t.Errorf("CAUSES edge should score higher with 'why' intent (%f) than 'what' intent (%f)", whyScore, whatScore)
	}
}

func TestSpreadMAGMA_MaxHops(t *testing.T) {
	// a→b→c; MaxHops=1 must not reach c.
	g := mockGraph{
		"a": {{UUID: "b", Name: "B", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
		"b": {{UUID: "c", Name: "C", EntityType: "x", GroupID: "grp", EdgeFact: "deep"}},
	}
	seeds := []ActivatedNode{{UUID: "a", Name: "A"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, Threshold: 0.01, MaxNodes: 10, Decay: 0.5, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range results {
		if r.UUID == "c" {
			t.Error("MaxHops=1 must not reach node c (2 hops from seed)")
		}
	}
}

func TestSpreadMAGMA_EmptySeeds(t *testing.T) {
	results, err := SpreadMAGMA(context.Background(), mockGraph{}, nil, "", nil, "grp", DefaultMAGMAConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for empty seeds, got %d", len(results))
	}
}

func TestSpreadMAGMA_NoCycles(t *testing.T) {
	// Bidirectional graph: alice↔bob. Without visited set, scores would inflate.
	g := mockGraph{
		"alice": {{UUID: "bob", Name: "Bob", EntityType: "person", GroupID: "grp", EdgeFact: "knows"}},
		"bob":   {{UUID: "alice", Name: "Alice", EntityType: "person", GroupID: "grp", EdgeFact: "knows"}},
	}
	seeds := []ActivatedNode{{UUID: "alice", Name: "Alice"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 3, Threshold: 0.01, MaxNodes: 10, Decay: 0.5, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Alice is a seed (score=1.0). Bob is 1-hop (score=0.5).
	// Alice must NOT re-appear as a higher-scored traversal result.
	aliceAct := 0.0
	bobAct := 0.0
	for _, r := range results {
		switch r.UUID {
		case "alice":
			aliceAct = r.Activation
		case "bob":
			bobAct = r.Activation
		}
	}
	if aliceAct != 1.0 {
		t.Errorf("alice should keep seed score 1.0, got %f", aliceAct)
	}
	if bobAct > aliceAct {
		t.Errorf("bob (%f) should not outscore seed alice (%f) via cycle inflation", bobAct, aliceAct)
	}
}

func TestDefaultMAGMAConfig(t *testing.T) {
	cfg := DefaultMAGMAConfig()
	if cfg.BeamWidth != 10 {
		t.Errorf("BeamWidth: want 10, got %d", cfg.BeamWidth)
	}
	if cfg.MaxHops != 3 {
		t.Errorf("MaxHops: want 3, got %d", cfg.MaxHops)
	}
	if cfg.Threshold != 0.1 {
		t.Errorf("Threshold: want 0.1, got %f", cfg.Threshold)
	}
	if cfg.MaxNodes != 50 {
		t.Errorf("MaxNodes: want 50, got %d", cfg.MaxNodes)
	}
	if cfg.Decay != 0.5 {
		t.Errorf("Decay: want 0.5, got %f", cfg.Decay)
	}
	if cfg.Lambda1 != 0.5 {
		t.Errorf("Lambda1: want 0.5, got %f", cfg.Lambda1)
	}
	if cfg.Lambda2 != 0.5 {
		t.Errorf("Lambda2: want 0.5, got %f", cfg.Lambda2)
	}
}
