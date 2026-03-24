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
	EdgeName   string    // relation name (e.g. "CAUSES", "KNOWS") for intent routing
	Embedding  []float32 // entity embedding for semantic affinity scoring
}

// GetNeighbors returns all entities connected to uuid, including:
//   - Direct edge neighbors (bidirectional)
//   - Episodic co-occurrence: entities extracted from the same episode (Synapse §3.1)
//
// Episodic bridging enables temporal/narrative chain traversal that direct edges miss.
func (d *DB) GetNeighbors(ctx context.Context, uuid, groupID string) ([]NeighborEntity, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT e.uuid, e.name, e.entity_type, e.group_id, ed.fact, ed.name, e.embedding
		FROM edges ed
		JOIN entities e ON e.uuid = ed.target_uuid
		WHERE ed.source_uuid = ? AND ed.group_id = ?
		UNION
		SELECT e.uuid, e.name, e.entity_type, e.group_id, ed.fact, ed.name, e.embedding
		FROM edges ed
		JOIN entities e ON e.uuid = ed.source_uuid
		WHERE ed.target_uuid = ? AND ed.group_id = ?
`,
		uuid, groupID, uuid, groupID,
	)
	if err != nil {
		return nil, fmt.Errorf("get neighbors: %w", err)
	}
	defer rows.Close() //nolint:errcheck

	var out []NeighborEntity
	for rows.Next() {
		var n NeighborEntity
		var blob []byte
		if err := rows.Scan(&n.UUID, &n.Name, &n.EntityType, &n.GroupID, &n.EdgeFact, &n.EdgeName, &blob); err != nil {
			return nil, err
		}
		n.Embedding = DecodeEmbedding(blob)
		out = append(out, n)
	}
	return out, rows.Err()
}
