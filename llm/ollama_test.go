package llm

import "testing"

func TestCleanStructuredContent_StripsThinkBlock(t *testing.T) {
	raw := "<think>I should reason first</think>\n{\"extracted_entities\":[{\"name\":\"Melanie\"}]}"

	got := cleanStructuredContent(raw)
	want := "{\"extracted_entities\":[{\"name\":\"Melanie\"}]}"

	if got != want {
		t.Fatalf("cleanStructuredContent() = %q, want %q", got, want)
	}
}

func TestCleanStructuredContent_StripsUnclosedThinkBlock(t *testing.T) {
	raw := "{\"extracted_entities\":[]}\n<think>I forgot to close this"

	got := cleanStructuredContent(raw)
	want := "{\"extracted_entities\":[]}"

	if got != want {
		t.Fatalf("cleanStructuredContent() = %q, want %q", got, want)
	}
}

func TestCleanStructuredContent_ExtractsJSONObjectFromMixedOutput(t *testing.T) {
	raw := "Here is the JSON you asked for:\n```json\n{\"edges\":[{\"relation_type\":\"KNOWS\"}]}\n```\nExtra commentary"

	got := cleanStructuredContent(raw)
	want := "{\"edges\":[{\"relation_type\":\"KNOWS\"}]}"

	if got != want {
		t.Fatalf("cleanStructuredContent() = %q, want %q", got, want)
	}
}

func TestCleanStructuredContent_ExtractsJSONArrayFromMixedOutput(t *testing.T) {
	raw := "prefix\n[{\"relation_type\":\"KNOWS\"}]\nsuffix"

	got := cleanStructuredContent(raw)
	want := "[{\"relation_type\":\"KNOWS\"}]"

	if got != want {
		t.Fatalf("cleanStructuredContent() = %q, want %q", got, want)
	}
}

func TestCleanStructuredContent_FallsBackToSanitizedContent(t *testing.T) {
	raw := "not json at all"

	got := cleanStructuredContent(raw)

	if got != raw {
		t.Fatalf("cleanStructuredContent() = %q, want %q", got, raw)
	}
}
