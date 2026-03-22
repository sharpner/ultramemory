// Package ingest walks a directory and enqueues text chunks for graph extraction.
package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/store"
)

const (
	// chunkSize is the max rune count per text chunk pushed to the queue.
	chunkSize = 1500
	// chunkOverlap is the overlap between consecutive chunks (context continuity).
	chunkOverlap = 150
)

// Walker ingests files into the job queue.
type Walker struct {
	db      *store.DB
	groupID string
}

// New creates a Walker.
func New(db *store.DB, groupID string) *Walker {
	return &Walker{db: db, groupID: groupID}
}

// Walk recursively ingests all readable text files under root.
func (w *Walker) Walk(ctx context.Context, root string) (int, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return 0, fmt.Errorf("abs root: %w", err)
	}

	total := 0
	err = filepath.WalkDir(absRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable
		}
		if d.IsDir() {
			if path == absRoot {
				return nil // never skip the root itself
			}
			return skipDir(d.Name())
		}
		if !isText(path) {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			slog.Warn("skip unreadable file", "path", path, "err", err)
			return nil
		}
		if !utf8.Valid(data) {
			return nil // binary file
		}

		chunks := chunk(string(data), chunkSize, chunkOverlap)
		for _, c := range chunks {
			c = strings.TrimSpace(c)
			if len(c) < 50 {
				continue // too short to be meaningful
			}

			payload, _ := json.Marshal(graph.IngestPayload{
				Content: c,
				Source:  path,
				GroupID: w.groupID,
			})
			if err := w.db.PushJob(ctx, store.JobTypeIngest, string(payload)); err != nil {
				return fmt.Errorf("push job: %w", err)
			}
			total++
		}
		return nil
	})
	return total, err
}

// skipDir returns filepath.SkipDir for well-known noise directories.
func skipDir(name string) error {
	skip := map[string]bool{
		".git": true, "node_modules": true, "vendor": true,
		".next": true, "dist": true, "build": true, "__pycache__": true,
		".cache": true, "coverage": true, "testdata": true,
	}
	if skip[name] || strings.HasPrefix(name, ".") {
		return filepath.SkipDir
	}
	return nil
}

// isText returns true for file extensions we can meaningfully extract from.
func isText(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	ok := map[string]bool{
		".go": true, ".md": true, ".txt": true, ".yaml": true,
		".yml": true, ".json": true, ".toml": true, ".sh": true,
		".ts": true, ".tsx": true, ".js": true, ".jsx": true,
		".py": true, ".rs": true, ".sql": true, ".env": true,
		".proto": true, ".graphql": true, ".html": true, ".css": true,
	}
	return ok[ext]
}

// chunk splits text into overlapping windows of at most size runes.
func chunk(text string, size, overlap int) []string {
	runes := []rune(text)
	if len(runes) <= size {
		return []string{string(runes)}
	}

	var chunks []string
	step := size - overlap
	if step <= 0 {
		step = size
	}
	for i := 0; i < len(runes); i += step {
		end := i + size
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return chunks
}
