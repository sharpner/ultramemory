package graph

import (
	"context"
	"testing"

	"github.com/sharpner/ultramemory/store"
)

// mockGraph implements GraphTraverser with an in-memory adjacency list.
type mockGraph map[string][]store.NeighborEntity

func (m mockGraph) GetNeighbors(_ context.Context, uuid, _ string) ([]store.NeighborEntity, error) {
	return m[uuid], nil
}

func TestSpreadMAGMA_BasicHop(t *testing.T) {
	g := mockGraph{
		"alice": {{UUID: "bob", Name: "Bob", EntityType: "person", GroupID: "grp", EdgeFact: "knows alice"}},
	}
	seeds := []ActivatedNode{{UUID: "alice", Name: "Alice"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, Threshold: 0.1, MaxNodes: 10, DecayFactor: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "test", "grp", cfg)
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
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, Threshold: 0.01, MaxNodes: 10, DecayFactor: 0.5}

	// Empty query → no keywords → affinity = 0.5 → transition = 0.5*0.5 = 0.25 per seed → total 0.50
	results, err := SpreadMAGMA(context.Background(), g, seeds, "", "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	var sharedAct float64
	for _, r := range results {
		if r.UUID == "shared" {
			sharedAct = r.Activation
		}
	}
	if sharedAct < 0.4 {
		t.Errorf("shared should accumulate from both seeds, got %f (want >= 0.4)", sharedAct)
	}
}

func TestSpreadMAGMA_Threshold(t *testing.T) {
	// Chain a→b→c→d; with high threshold early nodes get filtered out of results.
	g := mockGraph{
		"a": {{UUID: "b", Name: "B", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
		"b": {{UUID: "c", Name: "C", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
		"c": {{UUID: "d", Name: "D", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
	}
	seeds := []ActivatedNode{{UUID: "a", Name: "A"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 4, Threshold: 0.2, MaxNodes: 10, DecayFactor: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "test", "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range results {
		if r.Activation < cfg.Threshold {
			t.Errorf("node %s: activation %f below threshold %f", r.UUID, r.Activation, cfg.Threshold)
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
	cfg := MAGMAConfig{BeamWidth: 2, MaxHops: 1, Threshold: 0.01, MaxNodes: 50, DecayFactor: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "", "grp", cfg)
	if err != nil {
		t.Fatal(err)
	}
	// All 5 get activation during the single hop; BeamWidth limits traversal further, not collection.
	if len(results) < 2 {
		t.Errorf("expected at least 2 results, got %d", len(results))
	}
}

func TestSpreadMAGMA_SemanticAffinity(t *testing.T) {
	// techcorp EdgeFact contains query keywords → higher affinity → higher score.
	g := mockGraph{
		"alice": {
			{UUID: "techcorp", Name: "TechCorp", EntityType: "company", GroupID: "grp", EdgeFact: "alice works at techcorp as engineer"},
			{UUID: "unrelated", Name: "Unrelated", EntityType: "misc", GroupID: "grp", EdgeFact: "something else entirely"},
		},
	}
	seeds := []ActivatedNode{{UUID: "alice", Name: "Alice"}}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "alice engineer techcorp", "grp", DefaultMAGMAConfig())
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
	if techScore <= unrelScore {
		t.Errorf("techcorp (%f) should score higher than unrelated (%f)", techScore, unrelScore)
	}
}

func TestSpreadMAGMA_MaxHops(t *testing.T) {
	// a→b→c; MaxHops=1 must not reach c.
	g := mockGraph{
		"a": {{UUID: "b", Name: "B", EntityType: "x", GroupID: "grp", EdgeFact: "link"}},
		"b": {{UUID: "c", Name: "C", EntityType: "x", GroupID: "grp", EdgeFact: "deep"}},
	}
	seeds := []ActivatedNode{{UUID: "a", Name: "A"}}
	cfg := MAGMAConfig{BeamWidth: 10, MaxHops: 1, Threshold: 0.01, MaxNodes: 10, DecayFactor: 0.5}

	results, err := SpreadMAGMA(context.Background(), g, seeds, "test", "grp", cfg)
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
	results, err := SpreadMAGMA(context.Background(), mockGraph{}, nil, "test", "grp", DefaultMAGMAConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for empty seeds, got %d", len(results))
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
	if cfg.DecayFactor != 0.5 {
		t.Errorf("DecayFactor: want 0.5, got %f", cfg.DecayFactor)
	}
}
