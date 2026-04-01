//go:build !mistral

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// EmbeddingResponse is the Ollama /api/embeddings response.
type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// EmbedBatchResponse is the Ollama /api/embed batch response.
type EmbedBatchResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// Client is an Ollama HTTP client for entity/edge extraction and embeddings.
type Client struct {
	baseURL        string
	extractModel   string
	embeddingModel string
	http           *http.Client
	temperature    float64 // 0 = deterministic (default), >0 for retries
}

// SetTemperature sets the LLM temperature for subsequent calls.
// Use 0 for deterministic output, >0 to vary output on retries.
func (c *Client) SetTemperature(t float64) { c.temperature = t }

// Temperature returns the current temperature setting.
func (c *Client) Temperature() float64 { return c.temperature }

// New creates a new Ollama client.
func New(baseURL, extractModel, embeddingModel string) *Client {
	return &Client{
		baseURL:        strings.TrimRight(baseURL, "/"),
		extractModel:   extractModel,
		embeddingModel: embeddingModel,
		http:           &http.Client{Timeout: 3 * time.Minute},
	}
}

// Ping checks connectivity to Ollama.
func (c *Client) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/tags", nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("ollama not reachable at %s: %w", c.baseURL, err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama ping: HTTP %d", resp.StatusCode)
	}
	return nil
}

// Warmup loads the extraction model into memory (eliminates cold-start latency).
func (c *Client) Warmup(ctx context.Context) error {
	_, _, err := c.chat(ctx, c.extractModel, "Say hi.", "hi")
	return err
}

// ExtractEntities calls gemma3:4b to extract entities from content.
func (c *Client) ExtractEntities(ctx context.Context, content string) (*ExtractedEntities, error) {
	raw, _, err := c.chat(ctx, c.extractModel, entitySystem,
		"Extract entities from the following text:\n\n"+content,
	)
	if err != nil {
		return nil, err
	}

	var result ExtractedEntities
	cleaned := cleanStructuredContent(raw)
	if err := json.Unmarshal([]byte(cleaned), &result); err != nil {
		// LLM may return a bare array instead of {"extracted_entities":[...]}
		var direct []ExtractedEntity
		if err2 := json.Unmarshal([]byte(cleaned), &direct); err2 != nil {
			return nil, fmt.Errorf("entity JSON: %w (raw: %s)", err, truncate(raw, 120))
		}
		slog.Debug("ollama returned bare entity array", "count", len(direct))
		result.Entities = direct
	}
	return parseAndFilterEntities(raw, result)
}

// ExtractEdges calls gemma3:4b to extract edges between known entities.
func (c *Client) ExtractEdges(ctx context.Context, entities []ExtractedEntity, content string) (*ExtractedEdges, error) {
	if len(entities) < 2 {
		return &ExtractedEdges{}, nil
	}

	entityList := "ENTITIES:\n"
	for i, e := range entities {
		entityList += fmt.Sprintf("[%d] %s (%s)\n", i, e.Name, e.EntityType)
	}

	raw, _, err := c.chat(ctx, c.extractModel, edgeSystem,
		entityList+"\nTEXT:\n"+content,
	)
	if err != nil {
		return nil, err
	}

	clean := cleanStructuredContent(raw)
	var result ExtractedEdges
	if err := json.Unmarshal([]byte(clean), &result); err != nil {
		// gemma3:4b sometimes returns a bare array instead of {"edges":[...]}
		var direct []ExtractedEdge
		if err2 := json.Unmarshal([]byte(clean), &direct); err2 != nil {
			return nil, fmt.Errorf("edge JSON: %w (raw: %s)", err, truncate(raw, 120))
		}
		result.Edges = direct
	}
	return &result, nil
}

// Embed generates an embedding vector for the given text using the configured Ollama embedding model.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	body := map[string]any{
		"model":      c.embeddingModel,
		"prompt":     text,
		"keep_alive": -1,
	}
	raw, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/api/embeddings", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed HTTP %d: %s", resp.StatusCode, truncate(string(data), 80))
	}

	var er EmbeddingResponse
	if err := json.Unmarshal(data, &er); err != nil {
		return nil, fmt.Errorf("embed unmarshal: %w", err)
	}
	return er.Embedding, nil
}

// EmbedBatch generates embeddings for multiple texts in a single Ollama /api/embed call.
// Returns one embedding per input text in the same order. On error, returns nil.
// Falls back to sequential Embed calls if the batch endpoint is unavailable.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body := map[string]any{
		"model":      c.embeddingModel,
		"input":      texts,
		"keep_alive": -1,
	}
	raw, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/api/embed", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed batch request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20)) // 64MB max for large batches
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed batch HTTP %d: %s", resp.StatusCode, truncate(string(data), 80))
	}

	var er EmbedBatchResponse
	if err := json.Unmarshal(data, &er); err != nil {
		return nil, fmt.Errorf("embed batch unmarshal: %w", err)
	}
	if len(er.Embeddings) != len(texts) {
		return nil, fmt.Errorf("embed batch: got %d embeddings for %d texts", len(er.Embeddings), len(texts))
	}
	return er.Embeddings, nil
}

// chat sends a chat completion to Ollama and returns cleaned JSON content + latency.
// Uses /api/chat (native) instead of /v1/chat/completions to access keep_alive,
// cache_prompt and num_ctx — Ollama-native params not available via OpenAI-compat layer.
func (c *Client) chat(ctx context.Context, model, system, user string) (string, time.Duration, error) {
	temp := c.Temperature()
	body := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"format":       "json", // Ollama-native JSON mode
		"stream":       false,
		"think":        false, // qwen3.5 should answer directly here; robust JSON extraction handles stray markup
		"keep_alive":   -1,    // model stays loaded forever — no reload between jobs
		"cache_prompt": true,  // reuse KV-cache for system prompt across calls
		"options": map[string]any{
			"num_ctx":     8192, // must match Answer's num_ctx — Ollama pins KV cache at first load
			"num_predict": 4096, // max output tokens
			"temperature": temp,
		},
	}
	if temp > 0 {
		// Disable prompt caching on retries — different temperature means different output.
		body["cache_prompt"] = false
	}

	raw, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/api/chat", bytes.NewReader(raw))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := c.http.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("ollama request: %w", err)
	}
	latency := time.Since(start)
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", latency, err
	}
	if resp.StatusCode != http.StatusOK {
		return "", latency, fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, truncate(string(data), 80))
	}

	// /api/chat response: {"message": {"role": "assistant", "content": "..."}, "done": true}
	var cr struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		Done bool `json:"done"`
	}
	if err := json.Unmarshal(data, &cr); err != nil {
		return "", latency, fmt.Errorf("unmarshal: %w", err)
	}

	return strings.TrimSpace(cleanStructuredContent(cr.Message.Content)), latency, nil
}

// Answer sends a free-text prompt to the LLM and returns the response.
// Unlike extraction methods, this does not force JSON output format.
// maxTokens controls num_predict (0 defaults to 128).
func (c *Client) Answer(ctx context.Context, system, user string, maxTokens int) (string, error) {
	if maxTokens <= 0 {
		maxTokens = 128
	}
	body := map[string]any{
		"model": c.extractModel,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"stream":       false,
		"think":        false, // disable thinking for qwen3.5/deepseek — short factual answers, no reasoning
		"keep_alive":   -1,
		"cache_prompt": true,
		"options": map[string]any{
			"num_ctx":     8192, // large context for QA with many search results
			"num_predict": maxTokens,
			"temperature": 0,
		},
	}

	raw, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/api/chat", bytes.NewReader(raw))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, truncate(string(data), 80))
	}

	var cr struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal(data, &cr); err != nil {
		return "", fmt.Errorf("unmarshal: %w", err)
	}

	// Use content (final answer), not thinking (reasoning trace).
	content := cr.Message.Content
	content = thinkRe.ReplaceAllString(content, "")
	content = thinkUnclosedRe.ReplaceAllString(content, "")
	return strings.TrimSpace(content), nil
}
