package store

import (
	"context"
	"fmt"
	"strings"
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
	).Scan(&src) //nolint:errcheck
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
	).Scan(&src) //nolint:errcheck
	return src
}

// SearchEpisodesFTS performs fulltext search over episode content.
func (d *DB) SearchEpisodesFTS(ctx context.Context, query, groupID string, limit int) ([]Episode, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT e.uuid, e.content, e.source, e.embedding
		FROM episodes_fts f
		JOIN episodes e ON e.uuid = f.uuid
		WHERE episodes_fts MATCH ? AND e.group_id = ?
		ORDER BY rank
		LIMIT ?`,
		fts5Query(query), groupID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Episode
	for rows.Next() {
		var ep Episode
		var blob []byte
		if err := rows.Scan(&ep.UUID, &ep.Content, &ep.Source, &blob); err != nil {
			return nil, err
		}
		ep.GroupID = groupID
		ep.Embedding = DecodeEmbedding(blob)
		out = append(out, ep)
	}
	return out, rows.Err()
}

// AllEpisodesWithEmbeddings loads all episodes with embeddings for vector search.
func (d *DB) AllEpisodesWithEmbeddings(ctx context.Context, groupID string) ([]Episode, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid, content, source, embedding
		 FROM episodes
		 WHERE group_id = ? AND embedding IS NOT NULL`,
		groupID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Episode
	for rows.Next() {
		var ep Episode
		var blob []byte
		if err := rows.Scan(&ep.UUID, &ep.Content, &ep.Source, &blob); err != nil {
			return nil, err
		}
		ep.GroupID = groupID
		ep.Embedding = DecodeEmbedding(blob)
		out = append(out, ep)
	}
	return out, rows.Err()
}

// EpisodesForEntities returns episodes linked to any of the given entity UUIDs,
// ordered by recency (latest source first). Used for MAGMA episode backfill.
func (d *DB) EpisodesForEntities(ctx context.Context, entityUUIDs []string, groupID string, limit int) ([]Episode, error) {
	if len(entityUUIDs) == 0 {
		return nil, nil
	}
	ph := strings.Repeat("?,", len(entityUUIDs)-1) + "?"
	args := make([]any, 0, len(entityUUIDs)+2)
	for _, u := range entityUUIDs {
		args = append(args, u)
	}
	args = append(args, groupID, limit)
	rows, err := d.sql.QueryContext(ctx, `
		SELECT DISTINCT e.uuid, e.content, e.source, e.embedding
		FROM episodes e
		JOIN entity_episodes ee ON ee.episode_uuid = e.uuid
		WHERE ee.entity_uuid IN (`+ph+`) AND e.group_id = ?
		ORDER BY e.created_at DESC
		LIMIT ?`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Episode
	for rows.Next() {
		var ep Episode
		var blob []byte
		if err := rows.Scan(&ep.UUID, &ep.Content, &ep.Source, &blob); err != nil {
			return nil, err
		}
		ep.GroupID = groupID
		ep.Embedding = DecodeEmbedding(blob)
		out = append(out, ep)
	}
	return out, rows.Err()
}

// CountEpisodes returns the total episode count for a group.
func (d *DB) CountEpisodes(ctx context.Context, groupID string) (int, error) {
	var n int
	err := d.sql.QueryRowContext(ctx,
		`SELECT count(*) FROM episodes WHERE group_id = ?`, groupID,
	).Scan(&n)
	return n, err
}
