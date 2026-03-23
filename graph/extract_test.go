package graph

import (
	"context"
	"os"
	"testing"

	"github.com/sharpner/ultramemory/store"
)

func openExtractTestDB(t *testing.T) *store.DB {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "extract-test-*.db")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	f.Close()
	db, err := store.Open(f.Name())
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func newTestExtractor(db *store.DB, threshold float64) *Extractor {
	return &Extractor{db: db, sem: make(chan struct{}, 1), resolveThreshold: threshold}
}

func TestResolveOrCreate_SemanticDedup(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	ext := newTestExtractor(db, 0.80)

	// Seed "Jonathan Harker" with embedding [1,0,0,0].
	uuidA, err := db.UpsertEntity(ctx, store.Entity{
		UUID: "a", Name: "Jonathan Harker", EntityType: "person", GroupID: "g",
		Embedding: []float32{1, 0, 0, 0},
	})
	if err != nil {
		t.Fatal(err)
	}

	// "Harker Jonathan" with cosine ~0.99 against A — must resolve to A.
	uuidB, err := ext.resolveOrCreate(ctx, "Harker Jonathan", "person", "g", []float32{0.99, 0.14, 0, 0})
	if err != nil {
		t.Fatal(err)
	}

	if uuidA != uuidB {
		t.Errorf("semantic dedup: expected %q, got %q", uuidA, uuidB)
	}
	n, _ := db.CountEntities(ctx, "g")
	if n != 1 {
		t.Errorf("expected 1 entity after dedup, got %d", n)
	}
}

func TestResolveOrCreate_TypeGuard(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	ext := newTestExtractor(db, 0.80)

	uuidA, err := db.UpsertEntity(ctx, store.Entity{
		UUID: "a", Name: "Jonathan Harker", EntityType: "person", GroupID: "g",
		Embedding: []float32{1, 0, 0, 0},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Same name pattern + similar embedding but different entity_type — must not merge.
	uuidB, err := ext.resolveOrCreate(ctx, "Harker Jonathan", "tool", "g", []float32{0.99, 0.14, 0, 0})
	if err != nil {
		t.Fatal(err)
	}

	if uuidA == uuidB {
		t.Errorf("type guard: different entity_type must not merge, but both got %q", uuidA)
	}
}

func TestResolveOrCreate_NoEmbeddingFallsThrough(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	ext := newTestExtractor(db, 0.92)

	uuidA, err := db.UpsertEntity(ctx, store.Entity{
		UUID: "a", Name: "Jonathan Harker", EntityType: "person", GroupID: "g",
		Embedding: []float32{1, 0, 0, 0},
	})
	if err != nil {
		t.Fatal(err)
	}

	// nil embedding → FTS resolution skipped → new entity.
	uuidB, err := ext.resolveOrCreate(ctx, "Harker Jonathan", "person", "g", nil)
	if err != nil {
		t.Fatal(err)
	}

	if uuidA == uuidB {
		t.Errorf("nil embedding must not trigger semantic merge")
	}
}
