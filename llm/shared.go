package llm

import (
	"context"
	"encoding/json"
	"log/slog"
	"regexp"
	"strings"
)

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

type Answerer interface {
	Answer(ctx context.Context, system, user string, maxTokens int) (string, error)
}

type EntityExtractor interface {
	ExtractEntities(ctx context.Context, content string) (*ExtractedEntities, error)
	ExtractEdges(ctx context.Context, entities []ExtractedEntity, content string) (*ExtractedEdges, error)
}

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

type OCR interface {
	OCR(ctx context.Context, imageBytes []byte) (string, error)
}

type HealthChecker interface {
	Ping(ctx context.Context) error
}

type Warmer interface {
	Warmup(ctx context.Context) error
}

type ExtractedEntity struct {
	Name        string `json:"name"`
	EntityType  string `json:"entity_type"`
	Description string `json:"description"`
}

type ExtractedEntities struct {
	Entities []ExtractedEntity `json:"extracted_entities"`
}

func parseAndFilterEntities(raw string, result ExtractedEntities) (*ExtractedEntities, error) {
	filtered := result.Entities[:0]
	dropped := 0
	for _, e := range result.Entities {
		if e.Name == "" {
			dropped++
			continue
		}
		filtered = append(filtered, e)
	}
	if dropped > 0 {
		slog.Warn("dropped entities with empty names",
			"dropped", dropped,
			"kept", len(filtered),
			"raw", truncate(raw, 200))
	}
	result.Entities = filtered
	return &result, nil
}

type ExtractedEdge struct {
	RelationType   string  `json:"relation_type"`
	SourceEntityID int     `json:"source_entity_id"`
	TargetEntityID int     `json:"target_entity_id"`
	Fact           string  `json:"fact"`
	ValidAt        *string `json:"valid_at"`
	InvalidAt      *string `json:"invalid_at"`
}

type ExtractedEdges struct {
	Edges []ExtractedEdge `json:"edges"`
}

var thinkRe = regexp.MustCompile(`(?s)<think>.*?</think>`)
var thinkUnclosedRe = regexp.MustCompile(`(?s)<think>.*$`)

func sanitizeJSON(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false
	escaped := false
	for _, r := range s {
		if escaped {
			switch r {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u':
				b.WriteRune('\\')
				b.WriteRune(r)
			default:
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

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
