package graph

import (
	"context"
	"strings"
	"testing"

	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/store"
)

// noopClient returns a client whose Embed call returns zero embeddings.
// Sufficient for FTS-only search tests where extraction is never triggered.
func noopClient(t *testing.T) *llm.Client {
	t.Helper()
	return newMockClient(t, `{"extracted_entities":[]}`, `{"edges":[]}`)
}

// TestSearch_FTSEntityHit verifies that an entity FTS hit seeds MAGMA and
// surfaces the entity's connected edge (not the entity itself — entities are
// not rendered in context output, so they are skipped in the results list).
func TestSearch_FTSEntityHit(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()

	if _, err := db.UpsertEntity(ctx, store.Entity{
		UUID: "ent1", Name: "Dracula", EntityType: "vampire", GroupID: "g",
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.UpsertEntity(ctx, store.Entity{
		UUID: "ent2", Name: "Harker", EntityType: "person", GroupID: "g",
	}); err != nil {
		t.Fatal(err)
	}
	if err := db.UpsertEdge(ctx, store.Edge{
		UUID: "edg1", SourceUUID: "ent1", TargetUUID: "ent2",
		Name: "HAUNTS", Fact: "Dracula haunts Harker at Castle Dracula",
		GroupID: "g", Episodes: "[]",
	}); err != nil {
		t.Fatal(err)
	}

	results, err := Search(ctx, db, noopClient(t), "Dracula", "g", 10)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range results {
		if r.Type == "edge" && strings.Contains(r.Body, "Dracula") {
			found = true
		}
		// Entities must NOT appear in results — they provide no context output.
		if r.Type == "entity" {
			t.Errorf("entity result leaked into results: %v", r)
		}
	}
	if !found {
		t.Errorf("expected edge result containing Dracula, got %v", results)
	}
}

func TestSearch_FTSEdgeHit(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()

	if _, err := db.UpsertEntity(ctx, store.Entity{UUID: "e1", Name: "Dracula", EntityType: "vampire", GroupID: "g"}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.UpsertEntity(ctx, store.Entity{UUID: "e2", Name: "Harker", EntityType: "person", GroupID: "g"}); err != nil {
		t.Fatal(err)
	}
	if err := db.UpsertEdge(ctx, store.Edge{
		UUID:       "edg1",
		SourceUUID: "e1",
		TargetUUID: "e2",
		Name:       "BITES",
		Fact:       "Dracula exsanguinates Harker at midnight",
		GroupID:    "g",
		Episodes:   "[]",
	}); err != nil {
		t.Fatal(err)
	}

	results, err := Search(ctx, db, noopClient(t), "exsanguinates", "g", 10)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range results {
		if r.Type == "edge" && strings.Contains(r.Body, "exsanguinates") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected edge result containing %q, got %v", "exsanguinates", results)
	}
}

func TestSearch_SourcePopulated(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()

	if _, err := db.UpsertEntity(ctx, store.Entity{UUID: "mina", Name: "Mina Harker", EntityType: "person", GroupID: "g"}); err != nil {
		t.Fatal(err)
	}
	if err := db.UpsertEpisode(ctx, store.Episode{
		UUID:    "ep1",
		Content: "Mina Harker wrote in her diary.",
		GroupID: "g",
		Source:  "dracula/mina_diary.txt",
	}); err != nil {
		t.Fatal(err)
	}
	if err := db.LinkEntityEpisode(ctx, "mina", "ep1"); err != nil {
		t.Fatal(err)
	}

	results, err := Search(ctx, db, noopClient(t), "Mina", "g", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least 1 result, got 0")
	}
	// Entity results are skipped (not rendered in context). The episode linked
	// to the entity must appear with its source populated.
	found := false
	for _, r := range results {
		if r.Type == "episode" && r.Source == "dracula/mina_diary.txt" {
			found = true
			break
		}
		if r.Type == "entity" {
			t.Errorf("entity result leaked into results: %v", r)
		}
	}
	if !found {
		t.Errorf("expected episode result with source %q, got %v", "dracula/mina_diary.txt", results)
	}
}

func TestSearch_NoResults(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()

	results, err := Search(ctx, db, noopClient(t), "xyzzyquuxblarg", "g", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results for nonsense query, got %d", len(results))
	}
}

func TestSearch_GroupIsolation(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()

	if _, err := db.UpsertEntity(ctx, store.Entity{UUID: "ent1", Name: "Dracula", EntityType: "vampire", GroupID: "groupA"}); err != nil {
		t.Fatal(err)
	}

	results, err := Search(ctx, db, noopClient(t), "Dracula", "groupB", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Errorf("entity from groupA must not appear in groupB search, got %d results", len(results))
	}
}
