package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestMistralEmbedBatch_ReordersByIndex(t *testing.T) {
	client := NewMistral("test-key", "mistral-embed")
	client.http = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.String() != "https://api.mistral.ai/v1/embeddings" {
				t.Fatalf("unexpected URL: %s", req.URL.String())
			}

			var body struct {
				Model string   `json:"model"`
				Input []string `json:"input"`
			}
			if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if body.Model != "mistral-embed" {
				t.Fatalf("model = %q, want mistral-embed", body.Model)
			}
			if len(body.Input) != 2 || body.Input[0] != "alpha" || body.Input[1] != "beta" {
				t.Fatalf("input = %#v, want [alpha beta]", body.Input)
			}

			return jsonResponse(`{
				"data": [
					{"index": 1, "embedding": [0.2, 0.3]},
					{"index": 0, "embedding": [0.0, 0.1]}
				]
			}`), nil
		}),
	}

	vectors, err := client.EmbedBatch(context.Background(), []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(vectors) != 2 {
		t.Fatalf("got %d vectors, want 2", len(vectors))
	}
	if vectors[0][0] != 0.0 || vectors[0][1] != 0.1 {
		t.Fatalf("vector[0] = %#v, want [0 0.1]", vectors[0])
	}
	if vectors[1][0] != 0.2 || vectors[1][1] != 0.3 {
		t.Fatalf("vector[1] = %#v, want [0.2 0.3]", vectors[1])
	}
}

func TestMistralOCR_EncodesImageURLAndJoinsPages(t *testing.T) {
	client := NewMistral("test-key", "mistral-ocr-latest")
	imageBytes := []byte("fake-image")
	client.http = &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.String() != "https://api.mistral.ai/v1/ocr" {
				t.Fatalf("unexpected URL: %s", req.URL.String())
			}

			var body struct {
				Model    string `json:"model"`
				Document struct {
					Type     string `json:"type"`
					ImageURL string `json:"image_url"`
				} `json:"document"`
			}
			if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
				t.Fatalf("decode request: %v", err)
			}
			if body.Model != "mistral-ocr-latest" {
				t.Fatalf("model = %q, want mistral-ocr-latest", body.Model)
			}
			if body.Document.Type != "image_url" {
				t.Fatalf("document.type = %q, want image_url", body.Document.Type)
			}

			wantPrefix := "data:image/jpeg;base64,"
			if !strings.HasPrefix(body.Document.ImageURL, wantPrefix) {
				t.Fatalf("image_url prefix = %q, want prefix %q", body.Document.ImageURL, wantPrefix)
			}

			encoded := strings.TrimPrefix(body.Document.ImageURL, wantPrefix)
			decoded, err := base64.StdEncoding.DecodeString(encoded)
			if err != nil {
				t.Fatalf("decode base64: %v", err)
			}
			if string(decoded) != string(imageBytes) {
				t.Fatalf("decoded image = %q, want %q", decoded, imageBytes)
			}

			return jsonResponse(`{
				"pages": [
					{"markdown": "first page"},
					{"markdown": "second page"}
				]
			}`), nil
		}),
	}

	text, err := client.OCR(context.Background(), imageBytes)
	if err != nil {
		t.Fatalf("OCR: %v", err)
	}
	if text != "first page\n\nsecond page" {
		t.Fatalf("OCR text = %q, want joined markdown", text)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func jsonResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
