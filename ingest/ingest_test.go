package ingest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sharpner/ultramemory/store"
)

func openIngestTestDB(t *testing.T) *store.DB {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "ingest-test-*.db")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	_ = f.Close()
	db, err := store.Open(f.Name())
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

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

func TestWalk_TextFile(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 5) // 225 chars
	if err := os.WriteFile(filepath.Join(dir, "doc.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Errorf("expected >= 1 chunk queued, got %d", n)
	}
}

func TestWalk_SkipsHiddenDirs(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	hidden := filepath.Join(dir, ".hidden")
	if err := os.MkdirAll(hidden, 0o755); err != nil {
		t.Fatal(err)
	}
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 5)
	if err := os.WriteFile(filepath.Join(hidden, "secret.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("hidden dir must be skipped, got %d chunks", n)
	}
}

func TestWalk_ShortChunksIgnored(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tiny.txt"), []byte("too short"), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("content < 50 chars must produce 0 chunks, got %d", n)
	}
}

func TestWalk_BinaryFileSkipped(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	// Invalid UTF-8 sequence in a .txt file.
	if err := os.WriteFile(filepath.Join(dir, "binary.txt"), []byte{0xff, 0xfe, 0x00, 0x01}, 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("invalid UTF-8 file must be skipped, got %d chunks", n)
	}
}

func TestWalk_UnknownExtensionSkipped(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 5)
	if err := os.WriteFile(filepath.Join(dir, "data.xyz"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("unknown extension .xyz must be skipped, got %d chunks", n)
	}
}
