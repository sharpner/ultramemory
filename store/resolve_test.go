package store

import (
	"context"
	"testing"
)

// insertEntityWithEmbedding inserts an entity with a synthetic embedding.
func insertEntityWithEmbedding(t *testing.T, db *DB, uuid, name, entityType, groupID string, emb []float32) {
	t.Helper()
	if _, err := db.UpsertEntity(context.Background(), Entity{
		UUID:       uuid,
		Name:       name,
		EntityType: entityType,
		GroupID:    groupID,
		Embedding:  emb,
	}); err != nil {
		t.Fatalf("upsert entity %s: %v", name, err)
	}
}

// countEntitiesRaw counts entities directly for assertions.
func countEntitiesRaw(t *testing.T, db *DB, groupID string) int {
	t.Helper()
	n, err := db.CountEntities(context.Background(), groupID)
	if err != nil {
		t.Fatalf("count entities: %v", err)
	}
	return n
}

// countEdgesRaw counts edges for assertions.
func countEdgesRaw(t *testing.T, db *DB, groupID string) int {
	t.Helper()
	n, err := db.CountEdges(context.Background(), groupID)
	if err != nil {
		t.Fatalf("count edges: %v", err)
	}
	return n
}

// TestResolveEntities_BasicMerge verifies that two near-duplicate entities
// (cosine similarity > threshold) get merged into one.
func TestResolveEntities_BasicMerge(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	// Two nearly identical vectors — cosine similarity ≈ 1.0.
	nearVec := []float32{1, 0, 0, 0}
	insertEntityWithEmbedding(t, db, "e1", "Jonathan Harker", "person", grp, nearVec)
	insertEntityWithEmbedding(t, db, "e2", "Harker Jonathan", "person", grp, nearVec)
	// A third entity of a different type with the same vector — must NOT merge with persons.
	insertEntityWithEmbedding(t, db, "e3", "Castle Dracula", "place", grp, nearVec)

	result, err := db.ResolveEntities(ctx, grp, ResolveConfig{Threshold: 0.85})
	if err != nil {
		t.Fatal(err)
	}

	if result.ClustersFound != 1 {
		t.Errorf("expected 1 cluster, got %d", result.ClustersFound)
	}
	if result.EntitiesMerged != 1 {
		t.Errorf("expected 1 entity merged, got %d", result.EntitiesMerged)
	}

	// 3 entities minus 1 merged duplicate = 2 remaining.
	if got := countEntitiesRaw(t, db, grp); got != 2 {
		t.Errorf("expected 2 entities after merge, got %d", got)
	}
}

// TestResolveEntities_DryRun verifies that DryRun=true reports planned merges
// without touching the database.
func TestResolveEntities_DryRun(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	vec := []float32{1, 0, 0, 0}
	insertEntityWithEmbedding(t, db, "e1", "Jonathan Harker", "person", grp, vec)
	insertEntityWithEmbedding(t, db, "e2", "Harker Jonathan", "person", grp, vec)

	result, err := db.ResolveEntities(ctx, grp, ResolveConfig{Threshold: 0.85, DryRun: true})
	if err != nil {
		t.Fatal(err)
	}

	if result.ClustersFound < 1 {
		t.Errorf("expected ClustersFound > 0 in dry-run, got %d", result.ClustersFound)
	}
	// DryRun must not write anything.
	if result.EntitiesMerged != 0 {
		t.Errorf("dry-run must not merge, got EntitiesMerged=%d", result.EntitiesMerged)
	}

	// Both entities must still exist.
	if got := countEntitiesRaw(t, db, grp); got != 2 {
		t.Errorf("dry-run must not delete entities, got %d", got)
	}
}

// TestResolveEntities_CanonicalByEdgeCount verifies that the entity with more
// edges is selected as canonical regardless of insertion order.
func TestResolveEntities_CanonicalByEdgeCount(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	// e1 and e2 share nearly identical vectors → will merge.
	// e3 ("Mina") has an orthogonal vector → isolated, used only as edge target.
	dupVec  := []float32{1, 0, 0, 0}
	otherVec := []float32{0, 1, 0, 0}

	insertEntityWithEmbedding(t, db, "e1", "Harker", "person", grp, dupVec)
	insertEntityWithEmbedding(t, db, "e2", "Jonathan Harker", "person", grp, dupVec)
	insertEntityWithEmbedding(t, db, "e3", "Mina", "person", grp, otherVec)

	// e1: 1 edge total; e2: 2 edges total → e2 should be canonical.
	insertEdge(t, db, "ed1", "e1", "e3", grp)  // only edge for e1
	insertEdge(t, db, "ed2", "e2", "e3", grp)  // first edge for e2
	insertEdge(t, db, "ed3", "e3", "e2", grp)  // second edge for e2 (incoming)

	result, err := db.ResolveEntities(ctx, grp, ResolveConfig{Threshold: 0.85})
	if err != nil {
		t.Fatal(err)
	}

	// Only e1 and e2 should merge; e3 stays separate.
	if result.EntitiesMerged == 0 {
		t.Fatal("expected a merge to happen")
	}

	// Canonical (e2) must still exist; e1 must be gone.
	entities, err := db.AllEntitiesWithEmbeddings(ctx, grp)
	if err != nil {
		t.Fatal(err)
	}
	foundE2 := false
	for _, e := range entities {
		if e.UUID == "e2" {
			foundE2 = true
		}
		if e.UUID == "e1" {
			t.Errorf("e1 (fewer edges) should have been deleted as duplicate")
		}
	}
	if !foundE2 {
		t.Errorf("canonical entity e2 should still exist after merge")
	}
}

// TestResolveEntities_SelfLoopCleanup verifies that an edge between two merged
// entities (which becomes a self-loop A→A) is deleted.
func TestResolveEntities_SelfLoopCleanup(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	vec := []float32{1, 0, 0, 0}
	insertEntityWithEmbedding(t, db, "e1", "Jonathan Harker", "person", grp, vec)
	insertEntityWithEmbedding(t, db, "e2", "Harker Jonathan", "person", grp, vec)

	// Direct edge between the two duplicates — will become self-loop after merge.
	insertEdge(t, db, "ed1", "e1", "e2", grp)

	edgesBefore := countEdgesRaw(t, db, grp)
	if edgesBefore != 1 {
		t.Fatalf("expected 1 edge before merge, got %d", edgesBefore)
	}

	_, err := db.ResolveEntities(ctx, grp, ResolveConfig{Threshold: 0.85})
	if err != nil {
		t.Fatal(err)
	}

	edgesAfter := countEdgesRaw(t, db, grp)
	if edgesAfter != 0 {
		t.Errorf("self-loop edge should be deleted, got %d edges remaining", edgesAfter)
	}
}

// TestResolveEntities_NoEmbeddingSkipped verifies that entities without
// embeddings are never merged regardless of name similarity.
func TestResolveEntities_NoEmbeddingSkipped(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	// Insert two entities without embeddings.
	insertEntity(t, db, "e1", "Jonathan Harker", grp)
	insertEntity(t, db, "e2", "Harker Jonathan", grp)

	result, err := db.ResolveEntities(ctx, grp, ResolveConfig{Threshold: 0.85})
	if err != nil {
		t.Fatal(err)
	}

	if result.ClustersFound != 0 {
		t.Errorf("entities without embeddings must not be clustered, got ClustersFound=%d", result.ClustersFound)
	}
	if result.EntitiesMerged != 0 {
		t.Errorf("entities without embeddings must not be merged, got EntitiesMerged=%d", result.EntitiesMerged)
	}

	// Both entities must still exist.
	if got := countEntitiesRaw(t, db, grp); got != 2 {
		t.Errorf("expected 2 entities untouched, got %d", got)
	}
}
