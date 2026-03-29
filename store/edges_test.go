package store

import (
	"context"
	"testing"
)

func TestUpsertEdge_Dedup(t *testing.T) {
	db := openTestDB(t)
	insertEntity(t, db, "src", "Alice", "g")
	insertEntity(t, db, "tgt", "TechCorp", "g")
	ctx := context.Background()

	e := Edge{
		UUID:       "e1",
		SourceUUID: "src",
		TargetUUID: "tgt",
		Name:       "WORKS_AT",
		Fact:       "Alice works at TechCorp",
		GroupID:    "g",
	}
	if err := db.UpsertEdge(ctx, e); err != nil {
		t.Fatal(err)
	}

	// Same (source, target, name, group) → update, not duplicate.
	e2 := e
	e2.UUID = "e2"
	e2.Fact = "Alice works at TechCorp Berlin"
	if err := db.UpsertEdge(ctx, e2); err != nil {
		t.Fatal(err)
	}

	n, err := db.CountEdges(ctx, "g")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("expected 1 edge after dedup upsert, got %d", n)
	}
}

func TestUpsertEdge_DifferentRelation(t *testing.T) {
	db := openTestDB(t)
	insertEntity(t, db, "src", "Alice", "g")
	insertEntity(t, db, "tgt", "TechCorp", "g")
	ctx := context.Background()

	e1 := Edge{UUID: "e1", SourceUUID: "src", TargetUUID: "tgt", Name: "WORKS_AT", Fact: "f1", GroupID: "g"}
	e2 := Edge{UUID: "e2", SourceUUID: "src", TargetUUID: "tgt", Name: "KNOWS", Fact: "f2", GroupID: "g"}

	if err := db.UpsertEdge(ctx, e1); err != nil {
		t.Fatal(err)
	}
	if err := db.UpsertEdge(ctx, e2); err != nil {
		t.Fatal(err)
	}

	n, err := db.CountEdges(ctx, "g")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("different relation types should create 2 edges, got %d", n)
	}
}

func TestSearchEdgesFTS_EmptyQuery(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	// Single-char non-digit query sanitizes to empty → should return nil, not error.
	results, err := db.SearchEdgesFTS(ctx, "?", "g", 10)
	if err != nil {
		t.Fatalf("SearchEdgesFTS with empty query should not error: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results for empty query, got %d", len(results))
	}
}

func TestSearchEpisodesFTS_EmptyQuery(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	results, err := db.SearchEpisodesFTS(ctx, "@#$", "g", 10)
	if err != nil {
		t.Fatalf("SearchEpisodesFTS with empty query should not error: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results for empty query, got %d", len(results))
	}
}

func TestSearchEntitiesFTS_EmptyQuery(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	results, err := db.SearchEntitiesFTS(ctx, "!!!", "g", 10)
	if err != nil {
		t.Fatalf("SearchEntitiesFTS with empty query should not error: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results for empty query, got %d", len(results))
	}
}
