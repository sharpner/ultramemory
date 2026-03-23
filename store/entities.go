package store

import (
	"context"
	"database/sql"
	"fmt"
)

// Entity is a named node in the knowledge graph.
type Entity struct {
	UUID        string
	Name        string
	EntityType  string
	GroupID     string
	Embedding   []float32
	Description string
}

// UpsertEntity inserts or merges an entity by name+group (case-insensitive).
// Returns the canonical UUID that should be used.
func (d *DB) UpsertEntity(ctx context.Context, e Entity) (string, error) {
	var existing string
	err := d.sql.QueryRowContext(ctx,
		`SELECT uuid FROM entities
		 WHERE group_id = ? AND lower(name) = lower(?)
		 LIMIT 1`,
		e.GroupID, e.Name,
	).Scan(&existing)
	if err != nil && err != sql.ErrNoRows {
		return "", fmt.Errorf("lookup entity: %w", err)
	}
	if existing != "" {
		if len(e.Embedding) > 0 {
			_, err = d.sql.ExecContext(ctx,
				`UPDATE entities SET embedding = ?, description = ? WHERE uuid = ?`,
				EncodeEmbedding(e.Embedding), e.Description, existing,
			)
		}
		return existing, err
	}

	var embBlob []byte
	if len(e.Embedding) > 0 {
		embBlob = EncodeEmbedding(e.Embedding)
	}
	_, err = d.sql.ExecContext(ctx, `
		INSERT INTO entities (uuid, name, entity_type, group_id, embedding, description)
		VALUES (?, ?, ?, ?, ?, ?)`,
		e.UUID, e.Name, e.EntityType, e.GroupID, embBlob, e.Description,
	)
	if err != nil {
		return "", fmt.Errorf("insert entity: %w", err)
	}
	if _, err := d.sql.ExecContext(ctx,
		`DELETE FROM entities_fts WHERE uuid = ?`, e.UUID,
	); err != nil {
		return "", err
	}
	_, err = d.sql.ExecContext(ctx,
		`INSERT INTO entities_fts (uuid, name) VALUES (?, ?)`, e.UUID, e.Name,
	)
	return e.UUID, err
}

// LinkEntityEpisode creates the many-to-many association.
func (d *DB) LinkEntityEpisode(ctx context.Context, entityUUID, episodeUUID string) error {
	_, err := d.sql.ExecContext(ctx,
		`INSERT OR IGNORE INTO entity_episodes (entity_uuid, episode_uuid) VALUES (?, ?)`,
		entityUUID, episodeUUID,
	)
	return err
}

// CountEntities returns the total entity count for a group.
func (d *DB) CountEntities(ctx context.Context, groupID string) (int, error) {
	var n int
	err := d.sql.QueryRowContext(ctx,
		`SELECT count(*) FROM entities WHERE group_id = ?`, groupID,
	).Scan(&n)
	return n, err
}

// AllEntitiesWithEmbeddings loads all entities that have embeddings for vector search.
func (d *DB) AllEntitiesWithEmbeddings(ctx context.Context, groupID string) ([]Entity, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid, name, entity_type, embedding, description
		 FROM entities
		 WHERE group_id = ? AND embedding IS NOT NULL`,
		groupID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Entity
	for rows.Next() {
		var e Entity
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.Name, &e.EntityType, &blob, &e.Description); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}

// EntitiesForEpisodes returns entities linked to any of the given episode UUIDs
// via the entity_episodes join table. Used for episode→entity MAGMA seed expansion:
// FTS episode hits often link to entities not directly matched by entity FTS.
func (d *DB) EntitiesForEpisodes(ctx context.Context, episodeUUIDs []string, groupID string) ([]Entity, error) {
	if len(episodeUUIDs) == 0 {
		return nil, nil
	}
	ph := placeholders(len(episodeUUIDs))
	args := make([]any, 0, len(episodeUUIDs)+1)
	for _, u := range episodeUUIDs {
		args = append(args, u)
	}
	args = append(args, groupID)
	rows, err := d.sql.QueryContext(ctx, `
		SELECT DISTINCT e.uuid, e.name, e.entity_type, e.embedding, e.description
		FROM entities e
		JOIN entity_episodes ee ON ee.entity_uuid = e.uuid
		WHERE ee.episode_uuid IN (`+ph+`) AND e.group_id = ?`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Entity
	for rows.Next() {
		var e Entity
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.Name, &e.EntityType, &blob, &e.Description); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}

// SearchEntitiesFTS performs fulltext search on entity names.
func (d *DB) SearchEntitiesFTS(ctx context.Context, query, groupID string, limit int) ([]Entity, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT e.uuid, e.name, e.entity_type, e.embedding, e.description
		FROM entities_fts f
		JOIN entities e ON e.uuid = f.uuid
		WHERE entities_fts MATCH ? AND e.group_id = ?
		ORDER BY rank
		LIMIT ?`,
		fts5Query(query), groupID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Entity
	for rows.Next() {
		var e Entity
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.Name, &e.EntityType, &blob, &e.Description); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}
