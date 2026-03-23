package store

import (
	"context"
	"fmt"
)

// Edge is a typed relationship between two entities.
type Edge struct {
	UUID       string
	SourceUUID string
	TargetUUID string
	Name       string // relation type e.g. WORKS_AT
	Fact       string // natural language fact
	GroupID    string
	ValidAt    *string
	InvalidAt  *string
	Episodes   string // JSON array of episode UUIDs
	Embedding  []float32
}

// UpsertEdge inserts or updates an edge.
// Deduplicates by (source_uuid, target_uuid, name, group_id).
func (d *DB) UpsertEdge(ctx context.Context, e Edge) error {
	var existing string
	_ = d.sql.QueryRowContext(ctx,
		`SELECT uuid FROM edges
		 WHERE source_uuid = ? AND target_uuid = ? AND name = ? AND group_id = ?
		 LIMIT 1`,
		e.SourceUUID, e.TargetUUID, e.Name, e.GroupID,
	).Scan(&existing)

	var err error
	var embBlob []byte
	if len(e.Embedding) > 0 {
		embBlob = EncodeEmbedding(e.Embedding)
	}

	if existing != "" {
		if _, err = d.sql.ExecContext(ctx,
			`UPDATE edges SET fact = ?, embedding = ? WHERE uuid = ?`,
			e.Fact, embBlob, existing,
		); err != nil {
			return err
		}
		// Keep FTS in sync with updated fact.
		if _, err := d.sql.ExecContext(ctx,
			`DELETE FROM edges_fts WHERE uuid = ?`, existing,
		); err != nil {
			return err
		}
		_, err = d.sql.ExecContext(ctx,
			`INSERT INTO edges_fts (uuid, fact) VALUES (?, ?)`, existing, e.Fact,
		)
		return err
	}

	_, err = d.sql.ExecContext(ctx, `
		INSERT INTO edges (uuid, source_uuid, target_uuid, name, fact, group_id, valid_at, invalid_at, episodes, embedding)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		e.UUID, e.SourceUUID, e.TargetUUID, e.Name, e.Fact,
		e.GroupID, e.ValidAt, e.InvalidAt, e.Episodes, embBlob,
	)
	if err != nil {
		return fmt.Errorf("insert edge: %w", err)
	}

	if _, err := d.sql.ExecContext(ctx,
		`DELETE FROM edges_fts WHERE uuid = ?`, e.UUID,
	); err != nil {
		return err
	}
	_, err = d.sql.ExecContext(ctx,
		`INSERT INTO edges_fts (uuid, fact) VALUES (?, ?)`, e.UUID, e.Fact,
	)
	return err
}

// CountEdges returns the total edge count for a group.
func (d *DB) CountEdges(ctx context.Context, groupID string) (int, error) {
	var n int
	err := d.sql.QueryRowContext(ctx,
		`SELECT count(*) FROM edges WHERE group_id = ?`, groupID,
	).Scan(&n)
	return n, err
}

// SearchEdgesFTS performs fulltext search over edge facts.
func (d *DB) SearchEdgesFTS(ctx context.Context, query, groupID string, limit int) ([]Edge, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT e.uuid, e.source_uuid, e.target_uuid, e.name, e.fact, e.embedding
		FROM edges_fts f
		JOIN edges e ON e.uuid = f.uuid
		WHERE edges_fts MATCH ? AND e.group_id = ?
		ORDER BY rank
		LIMIT ?`,
		fts5Query(query), groupID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Edge
	for rows.Next() {
		var e Edge
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.SourceUUID, &e.TargetUUID, &e.Name, &e.Fact, &blob); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}

// AllEdgesWithEmbeddings loads all edges with embeddings for vector search.
func (d *DB) AllEdgesWithEmbeddings(ctx context.Context, groupID string) ([]Edge, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid, source_uuid, target_uuid, name, fact, embedding
		 FROM edges
		 WHERE group_id = ? AND embedding IS NOT NULL`,
		groupID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Edge
	for rows.Next() {
		var e Edge
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.SourceUUID, &e.TargetUUID, &e.Name, &e.Fact, &blob); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}
