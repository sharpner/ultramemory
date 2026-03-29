package store

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildDSN_PlainPath(t *testing.T) {
	dsn := buildDSN("/tmp/test.db")
	if !strings.HasPrefix(dsn, "file:") {
		t.Errorf("expected file: prefix, got %q", dsn)
	}
	if !strings.Contains(dsn, "busy_timeout") {
		t.Errorf("busy_timeout pragma missing from DSN: %q", dsn)
	}
}

func TestBuildDSN_PathWithSpaces(t *testing.T) {
	dsn := buildDSN("/tmp/my research/data.db")
	// The space must be escaped so it doesn't break the URI.
	if strings.Contains(dsn, " ") {
		t.Errorf("unescaped space in DSN: %q", dsn)
	}
	if !strings.Contains(dsn, "busy_timeout") {
		t.Errorf("PRAGMAs missing after escaped path: %q", dsn)
	}
}

func TestBuildDSN_PathWithQuestionMark(t *testing.T) {
	dsn := buildDSN("/tmp/weird?dir/test.db")
	// '?' in path must be escaped, not parsed as query separator.
	// The PRAGMAs must still appear after the real '?' separator.
	parts := strings.SplitN(dsn, "?", 2)
	if len(parts) != 2 {
		t.Fatalf("expected exactly one '?' separator, got %q", dsn)
	}
	if !strings.Contains(parts[1], "busy_timeout") {
		t.Errorf("PRAGMAs missing or broken by unescaped '?': %q", dsn)
	}
}

func TestOpen_PathWithSpaces(t *testing.T) {
	dir := t.TempDir()
	subdir := filepath.Join(dir, "my research")
	if err := os.MkdirAll(subdir, 0o755); err != nil {
		t.Fatal(err)
	}
	dbPath := filepath.Join(subdir, "test.db")

	db, err := Open(dbPath)
	if err != nil {
		t.Fatalf("Open with spaces in path: %v", err)
	}
	defer db.Close()

	// Verify PRAGMAs are active by checking WAL mode.
	var journalMode string
	err = db.sql.QueryRow("PRAGMA journal_mode").Scan(&journalMode)
	if err != nil {
		t.Fatal(err)
	}
	if journalMode != "wal" {
		t.Errorf("journal_mode = %q, want 'wal' (PRAGMAs not applied?)", journalMode)
	}
}

func TestRunMigrations_Idempotent(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "migrate-test.db")

	// Open once — runs migrations.
	db1, err := Open(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	db1.Close()

	// Open again — migrations should succeed (duplicate column = no-op).
	db2, err := Open(dbPath)
	if err != nil {
		t.Fatalf("second Open failed: %v", err)
	}
	db2.Close()
}
