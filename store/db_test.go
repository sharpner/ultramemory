package store

import (
	"database/sql"
	"errors"
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
	defer closeTestDB(t, db)

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
	closeTestDB(t, db1)

	// Open again — migrations should succeed (duplicate column = no-op).
	db2, err := Open(dbPath)
	if err != nil {
		t.Fatalf("second Open failed: %v", err)
	}
	closeTestDB(t, db2)
}

func TestOpen_StampsCurrentDBFormat(t *testing.T) {
	db := openDBForTest(t)

	var format string
	err := db.sql.QueryRow(`SELECT value FROM db_meta WHERE key = ?`, DBFormatKey).Scan(&format)
	if err != nil {
		t.Fatalf("read db format: %v", err)
	}
	if format != CurrentDBFormat() {
		t.Fatalf("db format = %q, want %q", format, CurrentDBFormat())
	}
}

func TestOpen_LegacyDBWithoutMetaGetsStamped(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "legacy.db")
	conn := openRawSQLite(t, dbPath)
	_, err := conn.Exec(`CREATE TABLE episodes (uuid TEXT PRIMARY KEY, content TEXT NOT NULL)`)
	if err != nil {
		t.Fatalf("create legacy schema: %v", err)
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("close legacy db: %v", err)
	}

	db, err := Open(dbPath)
	if CurrentDBFormat() == DBFormatOllama {
		if err != nil {
			t.Fatalf("Open legacy db: %v", err)
		}
		defer closeTestDB(t, db)
	}
	if CurrentDBFormat() != DBFormatOllama {
		if err == nil {
			t.Fatal("expected legacy db to be rejected by non-ollama build")
		}
		var mismatch *DBFormatMismatchError
		if !errors.As(err, &mismatch) {
			t.Fatalf("expected DBFormatMismatchError, got %T (%v)", err, err)
		}
		if mismatch.Actual != DBFormatOllama {
			t.Fatalf("legacy mismatch actual = %q, want %q", mismatch.Actual, DBFormatOllama)
		}
	}

	format := readDBFormatAtPath(t, dbPath)
	if format != DBFormatOllama {
		t.Fatalf("legacy db format = %q, want %q", format, DBFormatOllama)
	}
}

func TestOpen_RejectsWrongDBFormat(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "wrong-format.db")
	conn := openRawSQLite(t, dbPath)
	_, err := conn.Exec(`CREATE TABLE db_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)`)
	if err != nil {
		t.Fatalf("create db_meta: %v", err)
	}
	_, err = conn.Exec(`INSERT INTO db_meta (key, value) VALUES (?, ?)`, DBFormatKey, otherDBFormat())
	if err != nil {
		t.Fatalf("insert db format: %v", err)
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("close mismatch db: %v", err)
	}

	_, err = Open(dbPath)
	if err == nil {
		t.Fatal("expected wrong db format error, got nil")
	}

	var mismatch *DBFormatMismatchError
	if !errors.As(err, &mismatch) {
		t.Fatalf("expected DBFormatMismatchError, got %T (%v)", err, err)
	}
	if mismatch.Expected != CurrentDBFormat() {
		t.Fatalf("expected format = %q, want %q", mismatch.Expected, CurrentDBFormat())
	}
	if mismatch.Actual != otherDBFormat() {
		t.Fatalf("actual format = %q, want %q", mismatch.Actual, otherDBFormat())
	}
}

func openDBForTest(t *testing.T) *DB {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	db, err := Open(dbPath)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() {
		closeTestDB(t, db)
	})
	return db
}

func openRawSQLite(t *testing.T, path string) *sql.DB {
	t.Helper()
	conn, err := sql.Open("sqlite", buildDSN(path))
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	return conn
}

func readDBFormatAtPath(t *testing.T, path string) string {
	t.Helper()
	conn := openRawSQLite(t, path)
	defer closeTestSQLite(t, conn)

	var format string
	err := conn.QueryRow(`SELECT value FROM db_meta WHERE key = ?`, DBFormatKey).Scan(&format)
	if err != nil {
		t.Fatalf("read db format at path: %v", err)
	}
	return format
}

func closeTestDB(t *testing.T, db *DB) {
	t.Helper()
	if err := db.Close(); err != nil {
		t.Fatalf("close db: %v", err)
	}
}

func closeTestSQLite(t *testing.T, conn *sql.DB) {
	t.Helper()
	if err := conn.Close(); err != nil {
		t.Fatalf("close sqlite connection: %v", err)
	}
}

func otherDBFormat() string {
	if CurrentDBFormat() == DBFormatOllama {
		return DBFormatMistral
	}
	return DBFormatOllama
}
