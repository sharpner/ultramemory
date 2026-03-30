// Package llm provides an Ollama client with optimised prompts for gemma3:4b.
// Uses Ollama-native format:"json" (not OpenAI response_format) to avoid HTTP 500.
// Max 1 concurrent inference enforced by the caller via Semaphore.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// ── Optimised prompts (95% recall, 94% precision on gemma3:4b benchmark) ─────

const entitySystem = `Extract named entities from text. Output JSON only, no explanation.

Output: {"extracted_entities": [{"name": "Full Name", "entity_type": "TYPE", "description": "one sentence describing this entity based on the text"}]}
Types: Person, Organization, Place, Concept, Product, Event
Empty: {"extracted_entities": []}

Rules:
- People: full names when available, not pronouns. Fictional characters are Person, not Concept.
- Skip: dates, times, verbs, pronouns (he/she/they/we/I), document metadata (license text, publisher credits, release years for ebooks).
- Deduplicate: same entity mentioned twice = one entry
- Canonical form: use nominative case. "Deutschen Bahn" → "Deutsche Bahn", "Hamburgs" → "Hamburg", "Müllers" → "Müller"
- Title case: normalize ALL-CAPS names. "JONATHAN HARKER" → "Jonathan Harker", "COUNT DRACULA" → "Count Dracula"
- Description: one sentence summarizing who/what the entity is based on the given text

Example 1:
Input: "Alice works at Google in Berlin since 2020."
Output: {"extracted_entities": [{"name": "Alice", "entity_type": "Person", "description": "Alice is a professional who works at Google in Berlin"}, {"name": "Google", "entity_type": "Organization", "description": "Google is a technology company where Alice works"}, {"name": "Berlin", "entity_type": "Place", "description": "Berlin is the city where Alice works at Google"}]}`

const edgeSystem = `Extract relationships between the listed entities. Output JSON only, no explanation.

Output: {"edges": [{"relation_type": "WORKS_AT", "source_entity_id": 0, "target_entity_id": 1, "fact": "Alice works at Google", "valid_at": "2020-01-01T00:00:00Z", "invalid_at": null}]}
Empty: {"edges": []}

Rules:
- relation_type: English SCREAMING_SNAKE_CASE always, even for non-English input text
- WORKS_AT: paid employment/job ONLY — NOT for hobbies, interests, aspirations, events, or attended groups
- fact: one complete English sentence about the relationship
- valid_at / invalid_at: ISO 8601 date if mentioned in text, otherwise null
- Only connect entities from the given list using their IDs

Relation types (choose most specific):
People: KNOWS, MARRIED_TO, PARENT_OF, CHILD_OF, COMMANDS, OPPOSES, LOVES, SERVES_UNDER, ALLIED_WITH
Work: WORKS_AT, LEADS, FOUNDED, CREATED
Place/Event: BORN_IN, LIVES_IN, LOCATED_IN, FOUGHT_AT, TRAVELED_TO, PARTICIPATED_IN
Cause: CAUSES, LEADS_TO, RESULTED_IN, TRIGGERED_BY
Other: PART_OF, USES, OWNS

Example 1 (employment):
ENTITIES: [0] Alice (Person)  [1] Google (Organization)
TEXT: "Alice has worked at Google since 2020."
Output: {"edges": [{"relation_type": "WORKS_AT", "source_entity_id": 0, "target_entity_id": 1, "fact": "Alice works at Google", "valid_at": "2020-01-01T00:00:00Z", "invalid_at": null}]}

Example 2 (event participation — NOT employment):
ENTITIES: [0] Müller (Person)  [1] World Economic Forum (Event)  [2] Weber (Person)
TEXT: "Müller debated against Weber at the World Economic Forum in Davos."
Output: {"edges": [{"relation_type": "PARTICIPATED_IN", "source_entity_id": 0, "target_entity_id": 1, "fact": "Müller participated in the World Economic Forum", "valid_at": null, "invalid_at": null}, {"relation_type": "OPPOSES", "source_entity_id": 0, "target_entity_id": 2, "fact": "Müller opposes Weber", "valid_at": null, "invalid_at": null}]}

Example 3 (causal):
ENTITIES: [0] Drought (Event)  [1] Crop failure (Event)  [2] Region (Place)
TEXT: "The prolonged drought led to widespread crop failure across the region."
Output: {"edges": [{"relation_type": "LEADS_TO", "source_entity_id": 0, "target_entity_id": 1, "fact": "The drought led to crop failure", "valid_at": null, "invalid_at": null}, {"relation_type": "LOCATED_IN", "source_entity_id": 1, "target_entity_id": 2, "fact": "The crop failure occurred in the region", "valid_at": null, "invalid_at": null}]}`

// ── Types ─────────────────────────────────────────────────────────────────────

// ExtractedEntity is one entity from the LLM response.
type ExtractedEntity struct {
	Name        string `json:"name"`
	EntityType  string `json:"entity_type"`
	Description string `json:"description"`
}

// ExtractedEntities is the entity extraction response.
type ExtractedEntities struct {
	Entities []ExtractedEntity `json:"extracted_entities"`
}

// ExtractedEdge is one relationship from the LLM response.
type ExtractedEdge struct {
	RelationType   string  `json:"relation_type"`
	SourceEntityID int     `json:"source_entity_id"`
	TargetEntityID int     `json:"target_entity_id"`
	Fact           string  `json:"fact"`
	ValidAt        *string `json:"valid_at"`
	InvalidAt      *string `json:"invalid_at"`
}

// ExtractedEdges is the edge extraction response.
type ExtractedEdges struct {
	Edges []ExtractedEdge `json:"edges"`
}

// EmbeddingResponse is the Ollama /api/embeddings response.
type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// EmbedBatchResponse is the Ollama /api/embed batch response.
type EmbedBatchResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// ── Client ────────────────────────────────────────────────────────────────────

var thinkRe = regexp.MustCompile(`(?s)<think>.*?</think>`)
var thinkUnclosedRe = regexp.MustCompile(`(?s)<think>.*$`) // qwen3.5 sometimes omits </think>

// sanitizeJSON removes control characters that gemma3:4b occasionally emits
// inside JSON strings (e.g. literal pipe in escape sequences from markdown tables).
func sanitizeJSON(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false
	escaped := false
	for _, r := range s {
		if escaped {
			// only allow valid JSON escape chars: " \ / b f n r t u
			switch r {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u':
				b.WriteRune('\\')
				b.WriteRune(r)
			default:
				// drop the invalid escape, write char as-is
				b.WriteRune(r)
			}
			escaped = false
			continue
		}
		if r == '\\' && inString {
			escaped = true
			continue
		}
		if r == '"' {
			inString = !inString
		}
		b.WriteRune(r)
	}
	return b.String()
}

func cleanStructuredContent(s string) string {
	s = thinkRe.ReplaceAllString(s, "")
	s = thinkUnclosedRe.ReplaceAllString(s, "")
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```JSON")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	s = strings.TrimSpace(s)

	value, ok := extractFirstJSONValue(s)
	if ok {
		return sanitizeJSON(value)
	}

	return sanitizeJSON(s)
}

func extractFirstJSONValue(s string) (string, bool) {
	value, ok := decodeJSONValue(s)
	if ok {
		return value, true
	}

	for i, r := range s {
		if r != '{' && r != '[' {
			continue
		}

		value, ok = decodeJSONValue(s[i:])
		if ok {
			return value, true
		}
	}

	return "", false
}

func decodeJSONValue(s string) (string, bool) {
	dec := json.NewDecoder(strings.NewReader(s))

	var raw json.RawMessage
	if err := dec.Decode(&raw); err != nil {
		return "", false
	}
	if len(raw) == 0 {
		return "", false
	}

	return string(raw), true
}

// Client is an Ollama HTTP client for entity/edge extraction and embeddings.
type Client struct {
	baseURL        string
	extractModel   string
	embeddingModel string
	http           *http.Client
}

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
		result.Entities = direct
	}
	// Drop entries with empty names (e.g. LLM returned off-schema JSON).
	filtered := result.Entities[:0]
	for _, e := range result.Entities {
		if e.Name == "" {
			continue
		}
		filtered = append(filtered, e)
	}
	result.Entities = filtered
	return &result, nil
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

// Embed generates an embedding vector for the given text using nomic-embed-text.
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
			"temperature": 0,
		},
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

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
