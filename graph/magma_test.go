package graph

import (
	"context"
	"fmt"
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
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, MaxNodes: 50, Decay: 0.5, Lambda1: 1.0, Lambda2: 0.5}

	// EdgeName="" → EdgeUnknown → φ=0 → S=exp(0)=1.0 (no embeddings either)
	// Additive: score_bob = score_alice*γ + S = 1.0*0.5 + 1.0 = 1.5
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
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, MaxNodes: 50, Decay: 0.5, Lambda1: 1.0, Lambda2: 0.5}

	// EdgeName="" → S=1.0 for both seeds.
	// Additive: each seed contributes 1.0*0.5+1.0=1.5 → shared total=3.0
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
	// Both seeds contribute 1.5 each → total ≥ 2.5 (accumulation confirmed).
	if sharedAct < 2.5 {
		t.Errorf("shared should accumulate from both seeds, got %f (want >= 2.5)", sharedAct)
	}
}

func TestSpreadMAGMA_Budget(t *testing.T) {
	// Star graph: center → 10 spokes. MaxNodes=3 should cap the output.
	spokes := make([]store.NeighborEntity, 10)
	for i := range spokes {
		spokes[i] = store.NeighborEntity{
			UUID:      fmt.Sprintf("n%d", i),
			Name:      fmt.Sprintf("N%d", i),
			EntityType: "x",
			GroupID:   "grp",
			EdgeFact:  "link",
		}
	}
	g := mockGraph{"center": spokes}
	seeds := []ActivatedNode{{UUID: "center", Name: "Center"}}
	// MaxNodes=3 acts as Budget — limits output regardless of BeamWidth.
	cfg := MAGMAConfig{BeamWidth: 20, MaxHops: 5, MaxNodes: 3, Decay: 0.5, Lambda1: 1.0, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) > 3 {
		t.Errorf("MaxNodes=3 Budget should cap output, got %d results", len(results))
	}
}

func TestSpreadMAGMA_BeamWidth(t *testing.T) {
	// center → 5 neighbors; BeamWidth=2 limits frontier expansion.
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
	cfg := MAGMAConfig{BeamWidth: 2, MaxHops: 1, MaxNodes: 50, Decay: 0.5, Lambda1: 1.0, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	// At least 2 frontier survivors plus the center seed.
	if len(results) < 2 {
		t.Errorf("expected at least 2 results, got %d", len(results))
	}
}

func TestSpreadMAGMA_SemanticAffinity(t *testing.T) {
	// query → X axis. techcorp aligns with X (cos_sim≈1); unrelated aligns with Y (cos_sim=0).
	// EdgeName="" → EdgeUnknown → φ=0, so only λ₂·cos_sim contributes.
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
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, MaxNodes: 50, Decay: 0.5, Lambda1: 1.0, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", queryEmb, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	// techcorp: S=exp(0.5*1.0)≈1.649, score=1.0*0.5+1.649=2.149
	// unrelated: S=exp(0.5*0.0)=1.0,  score=1.0*0.5+1.0=1.5
	var techScore, unrelScore float64
	for _, r := range results {
		switch r.UUID {
		case "techcorp":
			techScore = r.Activation
		case "unrelated":
			unrelScore = r.Activation
		}
	}
	if techScore <= unrelScore {
		t.Errorf("techcorp (%f) should score higher than unrelated (%f)", techScore, unrelScore)
	}
}

func TestSpreadMAGMA_IntentRouting(t *testing.T) {
	// CAUSES edge + "why" query → φ(EdgeCausal, IntentWhy)=1.0 → high S.
	// CAUSES edge + "what" query → φ(EdgeCausal, IntentWhat)=0.5 → lower S.
	g := mockGraph{
		"alice": {
			{UUID: "event", Name: "Event", EntityType: "event", GroupID: "grp",
				EdgeFact: "causes event", EdgeName: "CAUSES"},
		},
	}
	seeds := []ActivatedNode{{UUID: "alice", Name: "Alice"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, MaxNodes: 50, Decay: 0.5, Lambda1: 1.0, Lambda2: 0.5}

	// "why" → IntentWhy → S=exp(1.0*1.0)=e≈2.718, score=1.0*0.5+2.718=3.218
	whyResults, err := SpreadMAGMA(context.Background(), g, seeds, "why did alice cause the event", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	// "tell me" → IntentWhat → S=exp(1.0*0.5)≈1.649, score=1.0*0.5+1.649=2.149
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
		t.Errorf("CAUSES edge: 'why' intent (%f) should outscore 'what' intent (%f)", whyScore, whatScore)
	}
}

func TestSpreadMAGMA_MaxHops(t *testing.T) {
	// a→b→c; MaxHops=1 must not reach c.
	g := mockGraph{
		"a": {{UUID: "b", Name: "B", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
		"b": {{UUID: "c", Name: "C", EntityType: "x", GroupID: "grp", EdgeFact: "deep"}},
	}
	seeds := []ActivatedNode{{UUID: "a", Name: "A"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, MaxNodes: 50, Decay: 0.5, Lambda1: 1.0, Lambda2: 0.5}

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
	// Bidirectional: alice↔bob. Without visited set, alice would be re-activated.
	g := mockGraph{
		"alice": {{UUID: "bob", Name: "Bob", EntityType: "person", GroupID: "grp", EdgeFact: "knows"}},
		"bob":   {{UUID: "alice", Name: "Alice", EntityType: "person", GroupID: "grp", EdgeFact: "knows"}},
	}
	seeds := []ActivatedNode{{UUID: "alice", Name: "Alice"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 3, MaxNodes: 50, Decay: 0.5, Lambda1: 1.0, Lambda2: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", nil, "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	// With additive formula: alice=1.0 (seed), bob=1.0*0.5+1.0=1.5 (1-hop).
	// Bob scoring higher than alice is CORRECT — that is not cycle inflation.
	// Cycle inflation would be: alice re-processed via bob→alice, raising alice above 1.0.
	aliceAct := 0.0
	bobFound := false
	for _, r := range results {
		switch r.UUID {
		case "alice":
			aliceAct = r.Activation
		case "bob":
			bobFound = true
		}
	}
	// Cycle protection: alice's base score must not be inflated above 1.0.
	// Lateral inhibition may reduce it below 1.0 (suppressed by stronger bob).
	if aliceAct > 1.0 {
		t.Errorf("alice seed score should not exceed 1.0 (cycle protection), got %f", aliceAct)
	}
	if aliceAct <= 0 {
		t.Errorf("alice should survive lateral inhibition, got %f", aliceAct)
	}
	if !bobFound {
		t.Error("bob should be reachable from alice")
	}
}

func TestDefaultMAGMAConfig(t *testing.T) {
	cfg := DefaultMAGMAConfig()
	if cfg.BeamWidth != 10 {
		t.Errorf("BeamWidth: want 10 (paper default; v18 showed BeamWidth=20 costs -11%% adversarial for +4%% multi-hop, net negative), got %d", cfg.BeamWidth)
	}
	if cfg.MaxHops != 5 {
		t.Errorf("MaxHops: want 5, got %d", cfg.MaxHops)
	}
	if cfg.Threshold != 0 {
		t.Errorf("Threshold: want 0 (paper: budget-based termination), got %f", cfg.Threshold)
	}
	if cfg.MaxNodes != 200 {
		t.Errorf("MaxNodes: want 200 (paper: Budget=200), got %d", cfg.MaxNodes)
	}
	if cfg.Decay != 0.5 {
		t.Errorf("Decay: want 0.5, got %f", cfg.Decay)
	}
	if cfg.Lambda1 != 1.0 {
		t.Errorf("Lambda1: want 1.0 (paper value), got %f", cfg.Lambda1)
	}
	if cfg.Lambda2 != 0.5 {
		t.Errorf("Lambda2: want 0.5, got %f", cfg.Lambda2)
	}
}
