// Package store handles all SQLite persistence for memory-local.
// Uses modernc.org/sqlite (pure Go, no CGO) with WAL mode.
package store

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	_ "modernc.org/sqlite"
)

const schema = `
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS episodes (
	uuid        TEXT PRIMARY KEY,
	content     TEXT NOT NULL,
	group_id    TEXT NOT NULL DEFAULT 'default',
	source      TEXT NOT NULL DEFAULT '',
	created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
	embedding   BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
	content,
	uuid UNINDEXED,
	tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS entities (
	uuid        TEXT PRIMARY KEY,
	name        TEXT NOT NULL,
	entity_type TEXT NOT NULL,
	group_id    TEXT NOT NULL DEFAULT 'default',
	created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
	embedding   BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
	name,
	uuid UNINDEXED,
	tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS edges (
	uuid        TEXT PRIMARY KEY,
	source_uuid TEXT NOT NULL,
	target_uuid TEXT NOT NULL,
	name        TEXT NOT NULL,
	fact        TEXT NOT NULL,
	group_id    TEXT NOT NULL DEFAULT 'default',
	valid_at    TEXT,
	invalid_at  TEXT,
	episodes    TEXT NOT NULL DEFAULT '[]',
	created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
	embedding   BLOB,
	FOREIGN KEY (source_uuid) REFERENCES entities(uuid),
	FOREIGN KEY (target_uuid) REFERENCES entities(uuid)
);

CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
	fact,
	uuid UNINDEXED,
	tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS entity_episodes (
	entity_uuid  TEXT NOT NULL,
	episode_uuid TEXT NOT NULL,
	PRIMARY KEY (entity_uuid, episode_uuid),
	FOREIGN KEY (entity_uuid)  REFERENCES entities(uuid),
	FOREIGN KEY (episode_uuid) REFERENCES episodes(uuid)
);

CREATE TABLE IF NOT EXISTS jobs (
	id         INTEGER PRIMARY KEY AUTOINCREMENT,
	type       TEXT NOT NULL,
	payload    TEXT NOT NULL,
	status     TEXT NOT NULL DEFAULT 'pending',
	attempts   INTEGER NOT NULL DEFAULT 0,
	max_attempts INTEGER NOT NULL DEFAULT 3,
	error      TEXT,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_pending
	ON jobs(created_at ASC)
	WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_entities_group
	ON entities(group_id, name);

CREATE INDEX IF NOT EXISTS idx_edges_src
	ON edges(source_uuid);

CREATE INDEX IF NOT EXISTS idx_edges_tgt
	ON edges(target_uuid);

CREATE INDEX IF NOT EXISTS idx_edges_group
	ON edges(group_id);
`

// DB wraps a SQLite database connection.
type DB struct {
	sql *sql.DB
}

// Open opens (or creates) the SQLite database at path.
func Open(path string) (*DB, error) {
	conn, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	// Single writer — prevents "database is locked" under concurrent workers.
	conn.SetMaxOpenConns(1)
	conn.SetMaxIdleConns(1)

	if _, err := conn.ExecContext(context.Background(), schema); err != nil {
		return nil, fmt.Errorf("apply schema: %w", err)
	}
	return &DB{sql: conn}, nil
}

// Close closes the database.
func (d *DB) Close() error {
	return d.sql.Close()
}

// SQL returns the raw *sql.DB for advanced usage.
func (d *DB) SQL() *sql.DB {
	return d.sql
}

// EncodeEmbedding serialises []float32 → []byte (little-endian).
func EncodeEmbedding(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// DecodeEmbedding deserialises []byte → []float32.
func DecodeEmbedding(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

// fts5Query converts a user query into an FTS5 match expression.
// Each word becomes a prefix term (word*) joined by AND.
// Special FTS5 characters are stripped.
func fts5Query(q string) string {
	words := strings.Fields(q)
	if len(words) == 0 {
		return ""
	}
	terms := make([]string, 0, len(words))
	for _, w := range words {
		// Strip FTS5 special chars to avoid syntax errors.
		w = strings.Map(func(r rune) rune {
			switch r {
			case '"', '\'', '(', ')', '*', '^', '-', '+', ':', '.':
				return -1
			}
			return r
		}, w)
		if w != "" {
			terms = append(terms, w+"*")
		}
	}
	// OR semantics: any matching term ranks higher; all terms present = highest score.
	return strings.Join(terms, " OR ")
}

// CosineSimilarity computes cosine similarity between two vectors.
// Returns 0 if either vector is nil or lengths differ.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
