//go:build !mistral

// Extraction quality tests — require Ollama with gemma3:4b.
// Run with: go test ./llm/ -v -timeout 5m -run TestExtractionQuality
//
// These tests document the *expected* extraction behaviour and serve as
// a regression suite when tuning prompts. Each test is focused on one
// failure mode observed during manual testing.
package llm

import (
	"context"
	"strings"
	"testing"
	"time"
)

func newTestClient() *Client {
	return New("http://localhost:11434", "gemma3:4b", "mxbai-embed-large")
}

func skipIfNoOllama(t *testing.T, c *Client) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := c.Ping(ctx); err != nil {
		t.Skipf("Ollama not available: %v", err)
	}
}

func ctx2m() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 2*time.Minute)
}

// ── Entity extraction ─────────────────────────────────────────────────────────

// TestExtractionQuality_CanonicalForm verifies that German inflected forms are
// normalised to the nominative: "Napoleonischen Kriege" → "Napoleonische Kriege".
func TestExtractionQuality_CanonicalForm(t *testing.T) {
	c := newTestClient()
	skipIfNoOllama(t, c)
	ctx, cancel := ctx2m()
	defer cancel()

	text := "Pierre sprach über die Napoleonischen Kriege und die Folgen für Russlands Zukunft."
	got, err := c.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	t.Logf("entities: %+v", got.Entities)

	for _, e := range got.Entities {
		name := e.Name
		if strings.Contains(name, "Napoleonischen") {
			t.Errorf("inflected form %q found — expected canonical nominative %q", name, "Napoleonische Kriege")
		}
		if strings.Contains(name, "Russlands") {
			t.Errorf("inflected form %q found — expected canonical %q", name, "Russland")
		}
	}

	// Must find the canonical form
	found := false
	for _, e := range got.Entities {
		if strings.Contains(strings.ToLower(e.Name), "napoleonische kriege") {
			found = true
		}
	}
	if !found {
		t.Errorf("canonical %q not found, got: %v", "Napoleonische Kriege", entityNames(got.Entities))
	}
}

// TestExtractionQuality_NoWorksAtForBattle verifies that battle/combat context
// does NOT produce WORKS_AT edges — it should produce FOUGHT_AT or PARTICIPATED_IN.
func TestExtractionQuality_NoWorksAtForBattle(t *testing.T) {
	c := newTestClient()
	skipIfNoOllama(t, c)
	ctx, cancel := ctx2m()
	defer cancel()

	text := "Napoleon kämpfte bei Borodino gegen Kutuzov und seine russische Armee."
	entities, err := c.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract entities: %v", err)
	}
	edges, err := c.ExtractEdges(ctx, entities.Entities, text)
	if err != nil {
		t.Fatalf("extract edges: %v", err)
	}
	t.Logf("edges: %+v", edges.Edges)

	for _, e := range edges.Edges {
		if e.RelationType == "WORKS_AT" {
			t.Errorf("WORKS_AT used in battle context — fact: %q (expected FOUGHT_AT, PARTICIPATED_IN, or similar)", e.Fact)
		}
	}

	// Must find at least one battle-specific relation
	battleRelations := map[string]bool{
		"FOUGHT_AT": true, "PARTICIPATED_IN": true, "COMMANDED_AT": true,
		"BATTLED_AT": true, "DEFEATED_AT": true, "OPPOSED": true, "OPPOSES": true,
	}
	found := false
	for _, e := range edges.Edges {
		if battleRelations[e.RelationType] {
			found = true
		}
	}
	if !found {
		t.Errorf("no battle-specific relation found, got types: %v", relationTypes(edges.Edges))
	}
}

// TestExtractionQuality_EnglishRelationType verifies that relation_type is always
// English SCREAMING_SNAKE_CASE even when the input text is German.
func TestExtractionQuality_EnglishRelationType(t *testing.T) {
	c := newTestClient()
	skipIfNoOllama(t, c)
	ctx, cancel := ctx2m()
	defer cancel()

	text := "Fürst Andrej diente unter Kutuzov und verliebte sich in Natascha Rostowa."
	entities, err := c.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract entities: %v", err)
	}
	edges, err := c.ExtractEdges(ctx, entities.Entities, text)
	if err != nil {
		t.Fatalf("extract edges: %v", err)
	}
	t.Logf("edges: %+v", edges.Edges)

	germanWords := []string{
		"DIENTE", "UNTER", "VERLIEBTE", "KÄMPFTE", "LEBTE", "WAR", "IST",
		"HAT", "HATTE", "WURDE", "BEFAHL", "LIEBTE",
	}
	for _, e := range edges.Edges {
		rt := strings.ToUpper(e.RelationType)
		for _, de := range germanWords {
			if strings.Contains(rt, de) {
				t.Errorf("German word in relation_type: %q", e.RelationType)
			}
		}
		// Must be SCREAMING_SNAKE_CASE: only uppercase letters and underscores
		for _, r := range e.RelationType {
			if (r < 'A' || r > 'Z') && r != '_' {
				t.Errorf("relation_type %q contains non-uppercase/non-underscore char %q", e.RelationType, string(r))
				break
			}
		}
	}
}

// TestExtractionQuality_CausalEdge verifies that causal language ("führte zu",
// "verursachte", "weil") produces CAUSES or LEADS_TO edges.
func TestExtractionQuality_CausalEdge(t *testing.T) {
	c := newTestClient()
	skipIfNoOllama(t, c)
	ctx, cancel := ctx2m()
	defer cancel()

	text := "Die Invasion Napoleons verursachte den Brand von Moskau und führte zum Rückzug der Grande Armée."
	entities, err := c.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract entities: %v", err)
	}
	edges, err := c.ExtractEdges(ctx, entities.Entities, text)
	if err != nil {
		t.Fatalf("extract edges: %v", err)
	}
	t.Logf("entities: %v", entityNames(entities.Entities))
	t.Logf("edges: %+v", edges.Edges)

	causalRelations := map[string]bool{
		"CAUSES": true, "CAUSED": true, "LEADS_TO": true, "LED_TO": true,
		"RESULTS_IN": true, "RESULTED_IN": true, "TRIGGERED": true, "TRIGGERED_BY": true,
	}
	found := false
	for _, e := range edges.Edges {
		if causalRelations[e.RelationType] {
			found = true
			t.Logf("causal edge found: %s — %s", e.RelationType, e.Fact)
		}
	}
	if !found {
		t.Errorf("no causal relation found for explicitly causal text, got types: %v", relationTypes(edges.Edges))
	}
}

// TestExtractionQuality_FamilyRelations verifies that family relations use
// specific types (MARRIED_TO, PARENT_OF, CHILD_OF) and NOT WORKS_AT.
func TestExtractionQuality_FamilyRelations(t *testing.T) {
	c := newTestClient()
	skipIfNoOllama(t, c)
	ctx, cancel := ctx2m()
	defer cancel()

	text := "Graf Ilja Rostow ist der Vater von Natascha, Sonja und Nikolaj. Er ist mit Gräfin Rostowa verheiratet."
	entities, err := c.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract entities: %v", err)
	}
	edges, err := c.ExtractEdges(ctx, entities.Entities, text)
	if err != nil {
		t.Fatalf("extract edges: %v", err)
	}
	t.Logf("entities: %v", entityNames(entities.Entities))
	t.Logf("edges: %+v", edges.Edges)

	familyRelations := map[string]bool{
		"MARRIED_TO": true, "PARENT_OF": true, "CHILD_OF": true,
		"FATHER_OF": true, "MOTHER_OF": true, "SON_OF": true, "DAUGHTER_OF": true,
		"SPOUSE_OF": true, "RELATED_TO": true,
	}

	for _, e := range edges.Edges {
		if e.RelationType == "WORKS_AT" {
			t.Errorf("WORKS_AT used for family relation — fact: %q", e.Fact)
		}
	}

	found := false
	for _, e := range edges.Edges {
		if familyRelations[e.RelationType] {
			found = true
		}
	}
	if !found {
		t.Errorf("no family relation found, got types: %v", relationTypes(edges.Edges))
	}
}

// TestExtractionQuality_Deduplication verifies that the same entity mentioned
// multiple times in different forms appears only once.
func TestExtractionQuality_Deduplication(t *testing.T) {
	c := newTestClient()
	skipIfNoOllama(t, c)
	ctx, cancel := ctx2m()
	defer cancel()

	// Napoleon mentioned 3× in different forms
	text := "Napoleon befahl den Angriff. Napoleons Armee war erschöpft. Der Kaiser Napoleon zog sich zurück."
	got, err := c.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	t.Logf("entities: %+v", got.Entities)

	napoleonCount := 0
	for _, e := range got.Entities {
		if strings.Contains(strings.ToLower(e.Name), "napoleon") {
			napoleonCount++
		}
	}
	if napoleonCount > 1 {
		t.Errorf("Napoleon deduplicated to %d entries — expected 1, got: %v", napoleonCount, entityNames(got.Entities))
	}
}

// TestExtractionQuality_AllCapsNormalization verifies that ALL-CAPS entity names
// from chapter headers get normalized to Title Case.
func TestExtractionQuality_AllCapsNormalization(t *testing.T) {
	c := newTestClient()
	skipIfNoOllama(t, c)
	ctx, cancel := ctx2m()
	defer cancel()

	text := `CHAPTER I

JONATHAN HARKER'S JOURNAL

3 May. Bistritz.—Left Munich at 8:35 P. M., on 1st May, arriving at Vienna early next morning. JONATHAN HARKER met COUNT DRACULA at the castle.`
	got, err := c.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	t.Logf("entities: %+v", got.Entities)

	for _, e := range got.Entities {
		if e.Name == strings.ToUpper(e.Name) && len(e.Name) > 3 {
			t.Errorf("ALL-CAPS entity found — should be Title Case: %q", e.Name)
		}
	}

	// Must find Jonathan Harker and Count Dracula as Persons
	foundHarker, foundDracula := false, false
	for _, e := range got.Entities {
		if strings.Contains(strings.ToLower(e.Name), "harker") {
			foundHarker = true
			if e.EntityType != "Person" {
				t.Errorf("Harker should be Person, got %q", e.EntityType)
			}
		}
		if strings.Contains(strings.ToLower(e.Name), "dracula") {
			foundDracula = true
			if e.EntityType != "Person" {
				t.Errorf("Dracula should be Person, got %q (fictional character = Person)", e.EntityType)
			}
		}
	}
	if !foundHarker {
		t.Errorf("Jonathan Harker not found, got: %v", entityNames(got.Entities))
	}
	if !foundDracula {
		t.Errorf("Count Dracula not found, got: %v", entityNames(got.Entities))
	}
}

// TestExtractionQuality_NameOrder verifies that person names are extracted in
// natural "First Last" order, not inverted "Last, First" or "Last First" forms.
func TestExtractionQuality_NameOrder(t *testing.T) {
	c := newTestClient()
	skipIfNoOllama(t, c)
	ctx, cancel := ctx2m()
	defer cancel()

	// Names appear in normal prose order — model must not invert them.
	text := `Dr. Anna Bergmann presented her findings at the conference.
	Karl von Moltke commanded the Prussian forces.
	Marie Curie discovered radioactivity together with her husband Pierre Curie.`

	got, err := c.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract: %v", err)
	}
	t.Logf("entities: %+v", got.Entities)

	// Inverted patterns: "Bergmann Anna", "Moltke Karl", "Curie Marie"
	invertedPatterns := []struct{ inverted, canonical string }{
		{"Bergmann Anna", "Anna Bergmann"},
		{"Moltke Karl", "Karl von Moltke"},
		{"Curie Marie", "Marie Curie"},
		{"Curie Pierre", "Pierre Curie"},
	}
	for _, p := range invertedPatterns {
		for _, e := range got.Entities {
			if strings.EqualFold(e.Name, p.inverted) {
				t.Errorf("inverted name %q found — expected %q", e.Name, p.canonical)
			}
		}
	}

	// All four persons must be found in correct order.
	mustFind := []string{"Anna Bergmann", "Karl von Moltke", "Marie Curie", "Pierre Curie"}
	for _, want := range mustFind {
		found := false
		for _, e := range got.Entities {
			if strings.EqualFold(e.Name, want) {
				found = true
			}
		}
		if !found {
			t.Errorf("person %q not found (or was inverted), got: %v", want, entityNames(got.Entities))
		}
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

func entityNames(entities []ExtractedEntity) []string {
	out := make([]string, len(entities))
	for i, e := range entities {
		out[i] = e.Name
	}
	return out
}

func relationTypes(edges []ExtractedEdge) []string {
	seen := map[string]bool{}
	var out []string
	for _, e := range edges {
		if !seen[e.RelationType] {
			seen[e.RelationType] = true
			out = append(out, e.RelationType)
		}
	}
	return out
}
