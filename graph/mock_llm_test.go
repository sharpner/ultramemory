package graph

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sharpner/ultramemory/llm"
)

type mockLLM struct {
	entitiesJSON string
	edgesJSON    string
}

func newMockClient(t *testing.T, entitiesJSON, edgesJSON string) *mockLLM {
	t.Helper()
	return &mockLLM{
		entitiesJSON: entitiesJSON,
		edgesJSON:    edgesJSON,
	}
}

func (m *mockLLM) ExtractEntities(context.Context, string) (*llm.ExtractedEntities, error) {
	var result llm.ExtractedEntities
	if err := json.Unmarshal([]byte(m.entitiesJSON), &result); err == nil {
		return &result, nil
	}

	var direct []llm.ExtractedEntity
	if err := json.Unmarshal([]byte(m.entitiesJSON), &direct); err != nil {
		return nil, err
	}
	return &llm.ExtractedEntities{Entities: direct}, nil
}

func (m *mockLLM) ExtractEdges(context.Context, []llm.ExtractedEntity, string) (*llm.ExtractedEdges, error) {
	var result llm.ExtractedEdges
	if err := json.Unmarshal([]byte(m.edgesJSON), &result); err == nil {
		return &result, nil
	}

	var direct []llm.ExtractedEdge
	if err := json.Unmarshal([]byte(m.edgesJSON), &direct); err != nil {
		return nil, err
	}
	return &llm.ExtractedEdges{Edges: direct}, nil
}

func (m *mockLLM) Embed(context.Context, string) ([]float32, error) {
	return make([]float32, 4), nil
}

func (m *mockLLM) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vectors[i] = make([]float32, 4)
	}
	return vectors, nil
}

func entityJSON(names ...string) string {
	type ent struct {
		Name       string `json:"name"`
		EntityType string `json:"entity_type"`
	}
	entities := make([]ent, len(names))
	for i, n := range names {
		entities[i] = ent{Name: n, EntityType: "Person"}
	}
	b, _ := json.Marshal(map[string]any{"extracted_entities": entities})
	return string(b)
}

func edgeJSON(relation, fact string) string {
	b, _ := json.Marshal(map[string]any{"edges": []map[string]any{{
		"relation_type":    relation,
		"source_entity_id": 0,
		"target_entity_id": 1,
		"fact":             fact,
		"valid_at":         nil,
		"invalid_at":       nil,
	}}})
	return string(b)
}
