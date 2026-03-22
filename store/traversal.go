package store

import (
	"context"
	"fmt"
)

// NeighborEntity is an entity connected via an edge (bidirectional).
type NeighborEntity struct {
	UUID       string
	Name       string
	EntityType string
	GroupID    string
	EdgeFact   string
}

// GetNeighbors returns all entities bidirectionally connected to uuid via an edge.
func (d *DB) GetNeighbors(ctx context.Context, uuid, groupID string) ([]NeighborEntity, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT e.uuid, e.name, e.entity_type, e.group_id, ed.fact
		FROM edges ed
		JOIN entities e ON e.uuid = ed.target_uuid
		WHERE ed.source_uuid = ? AND ed.group_id = ?
		UNION
		SELECT e.uuid, e.name, e.entity_type, e.group_id, ed.fact
		FROM edges ed
		JOIN entities e ON e.uuid = ed.source_uuid
		WHERE ed.target_uuid = ? AND ed.group_id = ?`,
		uuid, groupID, uuid, groupID,
	)
	if err != nil {
		return nil, fmt.Errorf("get neighbors: %w", err)
	}
	defer rows.Close()

	var out []NeighborEntity
	for rows.Next() {
		var n NeighborEntity
		if err := rows.Scan(&n.UUID, &n.Name, &n.EntityType, &n.GroupID, &n.EdgeFact); err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	return out, rows.Err()
}
