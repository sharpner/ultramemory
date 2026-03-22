package store

import (
	"context"
	"testing"
)

func TestUpsertEntity_ExactNameDedup(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	uuidA, err := db.UpsertEntity(ctx, Entity{UUID: "a", Name: "Jonathan Harker", EntityType: "person", GroupID: "g"})
	if err != nil {
		t.Fatal(err)
	}
	uuidB, err := db.UpsertEntity(ctx, Entity{UUID: "b", Name: "JONATHAN HARKER", EntityType: "person", GroupID: "g"})
	if err != nil {
		t.Fatal(err)
	}
	if uuidA != uuidB {
		t.Errorf("exact name (different case): expected same UUID, got %q vs %q", uuidA, uuidB)
	}
}

func TestUpsertEntity_DifferentNamesWithoutEmbedding(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	uuidA, err := db.UpsertEntity(ctx, Entity{UUID: "a", Name: "Jonathan Harker", EntityType: "person", GroupID: "g"})
	if err != nil {
		t.Fatal(err)
	}
	uuidB, err := db.UpsertEntity(ctx, Entity{UUID: "b", Name: "Harker Jonathan", EntityType: "person", GroupID: "g"})
	if err != nil {
		t.Fatal(err)
	}
	if uuidA == uuidB {
		t.Errorf("without embeddings, different names must produce distinct entities")
	}
}

func TestUpsertEntity_EmbeddingStoredOnInsert(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	emb := []float32{1, 0, 0, 0}
	uuid, err := db.UpsertEntity(ctx, Entity{UUID: "a", Name: "Alice", EntityType: "person", GroupID: "g", Embedding: emb})
	if err != nil {
		t.Fatal(err)
	}

	entities, err := db.AllEntitiesWithEmbeddings(ctx, "g")
	if err != nil {
		t.Fatal(err)
	}
	if len(entities) != 1 || entities[0].UUID != uuid {
		t.Fatalf("expected 1 entity with embedding, got %d", len(entities))
	}
	if CosineSimilarity(entities[0].Embedding, emb) < 0.999 {
		t.Errorf("stored embedding does not match inserted embedding")
	}
}
