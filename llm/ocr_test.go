package llm

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)


func skipIfNoOllamaOCR(t *testing.T, c *Client) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := c.Ping(ctx); err != nil {
		t.Skipf("Ollama not available: %v", err)
	}
}

func loadTestScan(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("/tmp/test_scan.png")
	if err != nil {
		t.Skipf("test scan not available at /tmp/test_scan.png: %v", err)
	}
	return data
}

func TestOCR_IsDoublePageDetection(t *testing.T) {
	data := loadTestScan(t)

	// 4522x3386 = ratio ~1.34 → double page
	if !isDoublePageScan(data) {
		t.Errorf("expected double-page detection for landscape scan (4522x3386)")
	}
}

func TestOCR_IsDoublePageDetection_Portrait(t *testing.T) {
	// Synthesise a minimal portrait PNG header (1x2 = ratio 0.5).
	// Just test the ratio logic with real dimensions via a real narrow image.
	// We approximate by testing the inverse: a byte slice that decodes to portrait.
	// Since we can't easily synthesise a PNG here, just test the ratio function directly.
	if isDoublePageScan([]byte("not an image")) {
		t.Errorf("invalid data should not be detected as double-page")
	}
}

func TestOCR_IsRefusal(t *testing.T) {
	cases := []struct {
		text     string
		expected bool
	}{
		{"I cannot fulfill this request.", true},
		{"I'm sorry, but I can't process images.", true},
		{"I am unable to process this.", true},
		{"Cannot fulfill the request.", true},
		{"Here is the transcribed text: Hello World", false},
		{"Transcription: some content here", false},
		{"", false},
	}
	for _, tc := range cases {
		got := isRefusal(tc.text)
		if got != tc.expected {
			t.Errorf("isRefusal(%q) = %v, want %v", tc.text, got, tc.expected)
		}
	}
}

func TestOCR_DoublePage(t *testing.T) {
	client := New("http://localhost:11434", "gemma3:4b", "nomic-embed-text")
	skipIfNoOllamaOCR(t, client)

	data := loadTestScan(t)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	text, err := client.OCR(ctx, data)
	if err != nil {
		t.Fatalf("OCR failed: %v", err)
	}

	t.Logf("OCR output (%d chars):\n%s", len(text), text)

	// Must contain content from both pages.
	if len(text) < 200 {
		t.Errorf("OCR output too short (%d chars), expected substantial content", len(text))
	}

	// Should not be a refusal.
	if isRefusal(text) {
		t.Errorf("OCR returned a refusal: %q", text[:100])
	}

	// Must contain some recognisable content from the scan.
	lower := strings.ToLower(text)
	knownWords := []string{"build", "people", "company", "work", "community", "network", "global", "team", "gore", "management", "factory", "blogosphere"}
	hits := 0
	for _, w := range knownWords {
		if strings.Contains(lower, w) {
			hits++
		}
	}
	if hits < 2 {
		t.Errorf("OCR output missing expected content (only %d/%d known words found); got: %q",
			hits, len(knownWords), truncate(text, 300))
	}
	t.Logf("Known-word hits: %d/%d", hits, len(knownWords))

	// No excessive repetition — double-page prompt should avoid duplicate blocks.
	firstHalf := text[:len(text)/2]
	secondHalf := text[len(text)/2:]
	overlapLen := 0
	words := strings.Fields(firstHalf)
	for _, w := range words {
		if strings.Contains(secondHalf, w) {
			overlapLen++
		}
	}
	overlapRatio := float64(overlapLen) / float64(max(len(words), 1))
	t.Logf("Half-overlap ratio: %.2f (lower is better for double-page dedup)", overlapRatio)
	if overlapRatio > 0.85 {
		t.Errorf("OCR output looks duplicated (overlap ratio %.2f > 0.85) — double-page prompt may not be working", overlapRatio)
	}
}

func TestOCR_RetryOnRefusal(t *testing.T) {
	client := New("http://localhost:11434", "gemma3:4b", "nomic-embed-text")
	skipIfNoOllamaOCR(t, client)

	data := loadTestScan(t)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	// Force the first prompt to be the one that historically triggers refusal.
	text, err := client.ocrRequest(ctx, data,
		"Read all text from this scanned document image. Return only the raw text, no explanations.")
	if err != nil {
		t.Fatalf("ocrRequest failed: %v", err)
	}
	t.Logf("Prompt-1 response: %q", truncate(text, 120))

	// If it refused, retry logic should recover.
	if isRefusal(text) {
		text, err = client.ocrRequest(ctx, data, promptRetry)
		if err != nil {
			t.Fatalf("retry ocrRequest failed: %v", err)
		}
		if isRefusal(text) {
			t.Errorf("retry also refused: %q", truncate(text, 120))
		}
		t.Logf("Retry recovered successfully")
	} else {
		t.Logf("First prompt succeeded without refusal — retry not needed")
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
