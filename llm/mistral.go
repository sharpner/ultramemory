package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Answerer is satisfied by any LLM that can answer free-text questions.
// This allows swapping QA models independently of the extraction model.
type Answerer interface {
	Answer(ctx context.Context, system, user string, maxTokens int) (string, error)
}

// MistralClient calls the Mistral API (OpenAI-compatible) for QA answering.
type MistralClient struct {
	model  string
	apiKey string
	http   *http.Client
}

// NewMistral creates a Mistral QA client.
func NewMistral(apiKey, model string) *MistralClient {
	return &MistralClient{
		model:  model,
		apiKey: apiKey,
		http:   &http.Client{Timeout: 60 * time.Second},
	}
}

// Answer sends a question to the Mistral API and returns the response.
func (m *MistralClient) Answer(ctx context.Context, system, user string, maxTokens int) (string, error) {
	if maxTokens <= 0 {
		maxTokens = 128
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
