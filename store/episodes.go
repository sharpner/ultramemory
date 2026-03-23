package store

import (
	"context"
	"fmt"
	"time"
)

// Episode is a raw text chunk stored in the graph.
type Episode struct {
	UUID      string
	Content   string
	GroupID   string
	Source    string
	CreatedAt time.Time
	Embedding []float32
}

// UpsertEpisode inserts or replaces an episode (idempotent on uuid).
func (d *DB) UpsertEpisode(ctx context.Context, ep Episode) error {
	var embBlob []byte
	if len(ep.Embedding) > 0 {
		embBlob = EncodeEmbedding(ep.Embedding)
	}

	_, err := d.sql.ExecContext(ctx, `
		INSERT INTO episodes (uuid, content, group_id, source, embedding)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(uuid) DO UPDATE SET
			content   = excluded.content,
			embedding = excluded.embedding`,
		ep.UUID, ep.Content, ep.GroupID, ep.Source, embBlob,
	)
	if err != nil {
		return fmt.Errorf("upsert episode: %w", err)
	}

	// Keep FTS in sync (delete + reinsert is simplest).
	if _, err := d.sql.ExecContext(ctx,
		`DELETE FROM episodes_fts WHERE uuid = ?`, ep.UUID,
	); err != nil {
		return fmt.Errorf("fts delete: %w", err)
	}
	_, err = d.sql.ExecContext(ctx,
		`INSERT INTO episodes_fts (uuid, content) VALUES (?, ?)`,
		ep.UUID, ep.Content,
	)
	return err
}

// FirstEntitySource returns the source file of the first episode linked to an entity.
func (d *DB) FirstEntitySource(ctx context.Context, entityUUID, groupID string) string {
	var src string
	d.sql.QueryRowContext(ctx, `
		SELECT ep.source FROM episodes ep
		JOIN entity_episodes ee ON ee.episode_uuid = ep.uuid
		WHERE ee.entity_uuid = ? AND ep.group_id = ?
		LIMIT 1`,
		entityUUID, groupID,
	).Scan(&src)
	return src
}

// FirstEdgeSource returns the source file of the first episode linked to an edge.
func (d *DB) FirstEdgeSource(ctx context.Context, edgeUUID string) string {
	var src string
	d.sql.QueryRowContext(ctx, `
		SELECT ep.source FROM episodes ep
		WHERE ep.uuid IN (
			SELECT value FROM json_each((SELECT episodes FROM edges WHERE uuid = ?))
		)
		LIMIT 1`,
		edgeUUID,
	).Scan(&src)
	return src
}

// CountEpisodes returns the total episode count for a group.
func (d *DB) CountEpisodes(ctx context.Context, groupID string) (int, error) {
	var n int
	err := d.sql.QueryRowContext(ctx,
		`SELECT count(*) FROM episodes WHERE group_id = ?`, groupID,
	).Scan(&n)
	return n, err
}
