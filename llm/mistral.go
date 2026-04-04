package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// MistralClient calls the Mistral API for extraction, QA, judging, embeddings and OCR.
type MistralClient struct {
	model       string
	apiKey      string
	http        *http.Client
	temperature float64
	retryBackoffs []time.Duration // overridable for tests; nil uses defaults
}

// SetTemperature sets the LLM temperature for subsequent calls.
func (m *MistralClient) SetTemperature(t float64) { m.temperature = t }

// Temperature returns the current temperature setting.
func (m *MistralClient) Temperature() float64 { return m.temperature }

// Ping is a lightweight readiness hook for provider-agnostic startup paths.
func (m *MistralClient) Ping(context.Context) error { return nil }

// Warmup is a no-op for hosted Mistral models.
func (m *MistralClient) Warmup(context.Context) error { return nil }

// NewMistral creates a Mistral client for the provided model.
func NewMistral(apiKey, model string) *MistralClient {
	return &MistralClient{
		model:  model,
		apiKey: apiKey,
		http:   &http.Client{Timeout: 60 * time.Second},
	}
}

// postJSON sends a POST with retry on transient errors (429, 503, 5xx).
// Returns response body, HTTP status code, and any transport error.
func (m *MistralClient) postJSON(ctx context.Context, url string, payload []byte, maxRespBytes int64) ([]byte, int, error) {
	backoffs := m.retryBackoffs
	if backoffs == nil {
		backoffs = []time.Duration{2 * time.Second, 5 * time.Second, 15 * time.Second}
	}

	for attempt := range len(backoffs) + 1 {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return nil, 0, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+m.apiKey)

		resp, err := m.http.Do(req)
		if err != nil {
			return nil, 0, fmt.Errorf("request: %w", err)
		}

		data, err := io.ReadAll(io.LimitReader(resp.Body, maxRespBytes))
		resp.Body.Close() //nolint:errcheck
		if err != nil {
			return nil, 0, err
		}

		if resp.StatusCode < 500 && resp.StatusCode != http.StatusTooManyRequests {
			return data, resp.StatusCode, nil
		}

		if attempt >= len(backoffs) {
			return data, resp.StatusCode, nil
		}

		slog.Warn("mistral transient error, retrying",
			"status", resp.StatusCode,
			"attempt", attempt+1,
			"backoff", backoffs[attempt],
		)

		select {
		case <-time.After(backoffs[attempt]):
		case <-ctx.Done():
			return nil, 0, ctx.Err()
		}
	}
	panic("unreachable")
}

// Judge evaluates whether a predicted answer is semantically correct given the gold answer.
// Returns true if the prediction is judged correct, false otherwise.
//
// Judge prompt: two-shot (yes/no examples), reply must be exactly "yes" or "no".
func (m *MistralClient) Judge(ctx context.Context, question, gold, prediction string) (bool, error) {
	const judgeSystem = `You are an answer evaluation judge. Given a question, a reference answer, and a predicted answer, determine if the predicted answer is semantically correct.

Rules:
- A prediction is correct if it conveys the same meaning as the reference, even with different wording.
- Partial answers that contain the key information are correct.
- Extra context or explanation is acceptable if the core answer is right.
- Reply with exactly one word: "yes" or "no".`

	user := fmt.Sprintf("Question: %s\nReference answer: %s\nPredicted answer: %s\nIs the predicted answer correct?",
		question, gold, prediction)

	body := map[string]any{
		"model": m.model,
		"messages": []map[string]string{
			{"role": "system", "content": judgeSystem},
			{"role": "user", "content": user},
		},
		"max_tokens":  5,
		"temperature": 0,
	}

	raw, _ := json.Marshal(body)
	data, status, err := m.postJSON(ctx, "https://api.mistral.ai/v1/chat/completions", raw, 1<<20)
	if err != nil {
		return false, fmt.Errorf("mistral judge: %w", err)
	}
	if status != http.StatusOK {
		return false, fmt.Errorf("mistral judge HTTP %d: %s", status, truncate(string(data), 120))
	}

	var cr struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &cr); err != nil {
		return false, fmt.Errorf("mistral judge unmarshal: %w", err)
	}
	if len(cr.Choices) == 0 {
		return false, fmt.Errorf("mistral judge: no choices")
	}

	verdict := strings.ToLower(strings.TrimSpace(cr.Choices[0].Message.Content))
	return strings.HasPrefix(verdict, "yes"), nil
}

// Answer sends a question to the Mistral API and returns the response.
// Default max_tokens is 32 — Mistral-small is verbose and tokenF1 rewards brevity.
// Unlike gemma3:4b, it does not self-terminate after a short answer.
func (m *MistralClient) Answer(ctx context.Context, system, user string, maxTokens int) (string, error) {
	if maxTokens <= 0 {
		maxTokens = 32
	}

	body := map[string]any{
		"model": m.model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"max_tokens":  maxTokens,
		"temperature": 0,
	}

	raw, _ := json.Marshal(body)
	data, status, err := m.postJSON(ctx, "https://api.mistral.ai/v1/chat/completions", raw, 1<<20)
	if err != nil {
		return "", fmt.Errorf("mistral request: %w", err)
	}
	if status != http.StatusOK {
		return "", fmt.Errorf("mistral HTTP %d: %s", status, truncate(string(data), 120))
	}

	var cr struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &cr); err != nil {
		return "", fmt.Errorf("mistral unmarshal: %w", err)
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("mistral: no choices in response")
	}
	return cr.Choices[0].Message.Content, nil
}

// Embed generates an embedding vector for the given text.
func (m *MistralClient) Embed(ctx context.Context, text string) ([]float32, error) {
	vectors, err := m.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("mistral embed: got %d vectors for 1 text", len(vectors))
	}
	return vectors[0], nil
}

// EmbedBatch generates embeddings for multiple texts in a single request.
func (m *MistralClient) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body := map[string]any{
		"model": m.model,
		"input": texts,
	}

	raw, _ := json.Marshal(body)
	data, status, err := m.postJSON(ctx, "https://api.mistral.ai/v1/embeddings", raw, 8<<20)
	if err != nil {
		return nil, fmt.Errorf("mistral embed request: %w", err)
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("mistral embed HTTP %d: %s", status, truncate(string(data), 120))
	}

	var result struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("mistral embed unmarshal: %w", err)
	}
	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("mistral embed: got %d vectors for %d texts", len(result.Data), len(texts))
	}

	vectors := make([][]float32, len(result.Data))
	for _, item := range result.Data {
		if item.Index < 0 || item.Index >= len(vectors) {
			return nil, fmt.Errorf("mistral embed: invalid index %d", item.Index)
		}
		vectors[item.Index] = item.Embedding
	}
	return vectors, nil
}

// OCR extracts text from an image using Mistral OCR.
func (m *MistralClient) OCR(ctx context.Context, imageBytes []byte) (string, error) {
	if len(imageBytes) == 0 {
		return "", fmt.Errorf("mistral OCR: empty image")
	}

	contentType := http.DetectContentType(imageBytes)
	if !strings.HasPrefix(contentType, "image/") {
		contentType = "image/jpeg"
	}

	body := map[string]any{
		"model": m.model,
		"document": map[string]string{
			"type":      "image_url",
			"image_url": "data:" + contentType + ";base64," + base64.StdEncoding.EncodeToString(imageBytes),
		},
	}

	raw, _ := json.Marshal(body)
	data, status, err := m.postJSON(ctx, "https://api.mistral.ai/v1/ocr", raw, 8<<20)
	if err != nil {
		return "", fmt.Errorf("mistral OCR request: %w", err)
	}
	if status != http.StatusOK {
		return "", fmt.Errorf("mistral OCR HTTP %d: %s", status, truncate(string(data), 120))
	}

	var result struct {
		Pages []struct {
			Markdown string `json:"markdown"`
		} `json:"pages"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("mistral OCR unmarshal: %w", err)
	}

	var pages []string
	for _, page := range result.Pages {
		text := strings.TrimSpace(page.Markdown)
		if text == "" {
			continue
		}
		pages = append(pages, text)
	}
	if len(pages) == 0 {
		return "", fmt.Errorf("mistral OCR produced no output")
	}
	return strings.Join(pages, "\n\n"), nil
}

// mistralChat sends a chat request to the Mistral API and returns the raw content string.
func (m *MistralClient) mistralChat(ctx context.Context, system, user string, jsonMode bool) (string, error) {
	body := map[string]any{
		"model": m.model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"max_tokens":  4096,
		"temperature": m.temperature,
	}
	if jsonMode {
		body["response_format"] = map[string]string{"type": "json_object"}
	}

	raw, _ := json.Marshal(body)
	data, status, err := m.postJSON(ctx, "https://api.mistral.ai/v1/chat/completions", raw, 2<<20)
	if err != nil {
		return "", fmt.Errorf("mistral request: %w", err)
	}
	if status != http.StatusOK {
		return "", fmt.Errorf("mistral HTTP %d: %s", status, truncate(string(data), 120))
	}

	var cr struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &cr); err != nil {
		return "", fmt.Errorf("mistral unmarshal: %w", err)
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("mistral: no choices in response")
	}
	return strings.TrimSpace(cleanStructuredContent(cr.Choices[0].Message.Content)), nil
}

// ExtractEntities calls the Mistral API to extract named entities from content.
func (m *MistralClient) ExtractEntities(ctx context.Context, content string) (*ExtractedEntities, error) {
	raw, err := m.mistralChat(ctx, entitySystem,
		"Extract entities from the following text:\n\n"+content,
		true,
	)
	if err != nil {
		return nil, err
	}

	var result ExtractedEntities
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		// Mistral may return a bare array instead of {"extracted_entities":[...]}
		var direct []ExtractedEntity
		if err2 := json.Unmarshal([]byte(raw), &direct); err2 != nil {
			return nil, fmt.Errorf("mistral entity JSON: %w (raw: %s)", err, truncate(raw, 120))
		}
		slog.Debug("mistral returned bare entity array", "count", len(direct))
		result.Entities = direct
	}
	return parseAndFilterEntities(raw, result)
}

// ExtractEdges calls the Mistral API to extract relationships between known entities.
func (m *MistralClient) ExtractEdges(ctx context.Context, entities []ExtractedEntity, content string) (*ExtractedEdges, error) {
	if len(entities) < 2 {
		return &ExtractedEdges{}, nil
	}

	entityList := "ENTITIES:\n"
	for i, e := range entities {
		entityList += fmt.Sprintf("[%d] %s (%s)\n", i, e.Name, e.EntityType)
	}

	raw, err := m.mistralChat(ctx, edgeSystem,
		entityList+"\nTEXT:\n"+content,
		true,
	)
	if err != nil {
		return nil, err
	}

	var result ExtractedEdges
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		// Mistral may return a bare array instead of {"edges":[...]}
		var direct []ExtractedEdge
		if err2 := json.Unmarshal([]byte(raw), &direct); err2 != nil {
			return nil, fmt.Errorf("mistral edge JSON: %w (raw: %s)", err, truncate(raw, 120))
		}
		result.Edges = direct
	}
	return &result, nil
}
