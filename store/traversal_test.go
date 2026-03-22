package store

import (
	"context"
	"os"
	"testing"
)

func openTestDB(t *testing.T) *DB {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "traversal-test-*.db")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	f.Close()
	db, err := Open(f.Name())
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func insertEntity(t *testing.T, db *DB, uuid, name, groupID string) {
	t.Helper()
	if _, err := db.UpsertEntity(context.Background(), Entity{
		UUID:       uuid,
		Name:       name,
		EntityType: "person",
		GroupID:    groupID,
	}); err != nil {
		t.Fatalf("upsert entity %s: %v", name, err)
	}
}

func insertEdge(t *testing.T, db *DB, uuid, src, tgt, groupID string) {
	t.Helper()
	if err := db.UpsertEdge(context.Background(), Edge{
		UUID:       uuid,
		SourceUUID: src,
		TargetUUID: tgt,
		Name:       "KNOWS",
		Fact:       src + " knows " + tgt,
		GroupID:    groupID,
		Episodes:   "[]",
	}); err != nil {
		t.Fatalf("upsert edge %s→%s: %v", src, tgt, err)
	}
}

func TestGetNeighbors_Outgoing(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	insertEntity(t, db, "alice", "Alice", grp)
	insertEntity(t, db, "bob", "Bob", grp)
	insertEdge(t, db, "e1", "alice", "bob", grp)

	neighbors, err := db.GetNeighbors(ctx, "alice", grp)
	if err != nil {
		t.Fatal(err)
	}
	if len(neighbors) != 1 || neighbors[0].UUID != "bob" {
		t.Errorf("expected [bob], got %v", neighbors)
	}
}

func TestGetNeighbors_Incoming(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	grp := "grp"

	insertEntity(t, db, "alice", "Alice", grp)
	insertEntity(t, db, "bob", "Bob", grp)
	insertEdge(t, db, "e1", "alice", "bob", grp)

	// target should find source (bidirectional)
	neighbors, err := db.GetNeighbors(ctx, "bob", grp)
	if err != nil {
		t.Fatal(err)
	}
	if len(neighbors) != 1 || neighbors[0].UUID != "alice" {
		t.Errorf("expected [alice], got %v", neighbors)
	}
}

func TestGetNeighbors_GroupIsolation(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	insertEntity(t, db, "alice", "Alice", "grpA")
	insertEntity(t, db, "bob", "Bob", "grpA")
	insertEdge(t, db, "e1", "alice", "bob", "grpA")

	// Query from a different group — should return nothing.
	neighbors, err := db.GetNeighbors(ctx, "alice", "grpB")
	if err != nil {
		t.Fatal(err)
	}
	if len(neighbors) != 0 {
		t.Errorf("expected 0 neighbors for wrong group, got %v", neighbors)
	}
}

func TestGetNeighbors_Empty(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	insertEntity(t, db, "alice", "Alice", "grp")

	neighbors, err := db.GetNeighbors(ctx, "alice", "grp")
	if err != nil {
		t.Fatal(err)
	}
	if len(neighbors) != 0 {
		t.Errorf("expected 0 neighbors for isolated node, got %v", neighbors)
	}
}
