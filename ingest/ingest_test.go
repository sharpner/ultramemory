package ingest

import "testing"

func TestChunk_SingleChunk(t *testing.T) {
	text := "short"
	got := chunk(text, 1500, 150)
	if len(got) != 1 || got[0] != "short" {
		t.Errorf("short text should produce 1 chunk, got %v", got)
	}
}

func TestChunk_Overlap(t *testing.T) {
	// 10 runes, size=6, overlap=2, step=4 → windows: [0:6], [4:10]
	text := "abcdefghij"
	got := chunk(text, 6, 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %v", len(got), got)
	}
	if got[0] != "abcdef" {
		t.Errorf("first chunk: want %q, got %q", "abcdef", got[0])
	}
	if got[1] != "efghij" {
		t.Errorf("second chunk: want %q, got %q", "efghij", got[1])
	}
}

func TestChunk_Unicode(t *testing.T) {
	// Emojis are multi-byte but single rune — chunking must work on runes, not bytes.
	text := "🐉🐉🐉🐉🐉🐉"
	got := chunk(text, 4, 1)
	for _, c := range got {
		if len([]rune(c)) > 4 {
			t.Errorf("chunk exceeds rune size 4: %q (%d runes)", c, len([]rune(c)))
		}
	}
}
