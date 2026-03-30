package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Answerer is satisfied by any LLM that can answer free-text questions.
// This allows swapping QA models independently of the extraction model.
type Answerer interface {
	Answer(ctx context.Context, system, user string, maxTokens int) (string, error)
}

// EntityExtractor is satisfied by any LLM that can extract entities and edges.
// Both *Client (Ollama) and *MistralClient implement this interface.
type EntityExtractor interface {
	ExtractEntities(ctx context.Context, content string) (*ExtractedEntities, error)
	ExtractEdges(ctx context.Context, entities []ExtractedEntity, content string) (*ExtractedEdges, error)
}

// MistralClient calls the Mistral API (OpenAI-compatible) for QA answering or judging.
type MistralClient struct {
	model  string
	apiKey string
	http   *http.Client
}

// NewMistral creates a Mistral client (QA answering or judge).
func NewMistral(apiKey, model string) *MistralClient {
	return &MistralClient{
		model:  model,
		apiKey: apiKey,
		http:   &http.Client{Timeout: 60 * time.Second},
	}
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.mistral.ai/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return false, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.http.Do(req)
	if err != nil {
		return false, fmt.Errorf("mistral judge: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return false, err
	}
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("mistral judge HTTP %d: %s", resp.StatusCode, truncate(string(data), 120))
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.mistral.ai/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("mistral request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("mistral HTTP %d: %s", resp.StatusCode, truncate(string(data), 120))
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

// mistralChat sends a chat request to the Mistral API and returns the raw content string.
func (m *MistralClient) mistralChat(ctx context.Context, system, user string, jsonMode bool) (string, error) {
	body := map[string]any{
		"model": m.model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"max_tokens":  4096,
		"temperature": 0,
	}
	if jsonMode {
		body["response_format"] = map[string]string{"type": "json_object"}
	}

	raw, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.mistral.ai/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("mistral request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("mistral HTTP %d: %s", resp.StatusCode, truncate(string(data), 120))
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
		return nil, fmt.Errorf("mistral entity JSON: %w (raw: %s)", err, truncate(raw, 120))
	}
	return &result, nil
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
