package graph

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sharpner/ultramemory/llm"
)

// mockServer spins up a fake Ollama HTTP server for tests.
// /api/chat distinguishes entity vs edge extraction by checking the system prompt.
// /api/embeddings always returns a zero vector (cosine = 0 → no semantic merging).
type mockServer struct {
	entitiesJSON string // JSON string for extracted_entities response
	edgesJSON    string // JSON string for edges response
}

func newMockClient(t *testing.T, entitiesJSON, edgesJSON string) *llm.Client {
	t.Helper()
	m := &mockServer{entitiesJSON: entitiesJSON, edgesJSON: edgesJSON}
	srv := httptest.NewServer(m.handler())
	t.Cleanup(srv.Close)
	return llm.New(srv.URL, "mock-model", "mock-embed")
}

func (m *mockServer) handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
	})

	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)

		// Edge extraction prompts contain "relation_type" in the system message.
		content := m.entitiesJSON
		for _, msg := range body.Messages {
			if msg.Role == "system" && strings.Contains(msg.Content, "relation_type") {
				content = m.edgesJSON
				break
			}
		}

		json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]string{"role": "assistant", "content": content},
			"done":    true,
		})
	})

	mux.HandleFunc("/api/embeddings", func(w http.ResponseWriter, _ *http.Request) {
		// Zero vector → cosine similarity = 0 → no semantic merging in tests.
		json.NewEncoder(w).Encode(map[string]any{"embedding": make([]float32, 4)})
	})

	mux.HandleFunc("/api/generate", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"response": "mock OCR text"})
	})

	return mux
}

// entityJSON builds a valid extracted_entities JSON string.
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

// edgeJSON builds a valid edges JSON string for a single edge between entity 0 and 1.
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
