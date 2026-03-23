package store

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"gonum.org/v1/gonum/graph/community"
	"gonum.org/v1/gonum/graph/simple"
)

// CommunityResult holds statistics from community detection.
type CommunityResult struct {
	Communities int
	Entities    int
}

// DetectCommunities runs Louvain community detection on the entity graph
// for the given group and writes community_id back to the entities table.
// Resolution controls granularity: higher = more, smaller communities.
func (d *DB) DetectCommunities(ctx context.Context, groupID string, resolution float64) (CommunityResult, error) {
	if resolution <= 0 {
		resolution = 1.0
	}

	// ── 1. Load all entities for this group ──────────────────────────────────
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid FROM entities WHERE group_id = ?`, groupID)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("load entities: %w", err)
	}
	defer rows.Close()

	var uuids []string
	uuidToID := map[string]int64{}
	idToUUID := map[int64]string{}
	var nextID int64

	for rows.Next() {
		var uuid string
		if err := rows.Scan(&uuid); err != nil {
			return CommunityResult{}, err
		}
		uuids = append(uuids, uuid)
		uuidToID[uuid] = nextID
		idToUUID[nextID] = uuid
		nextID++
	}
	if err := rows.Err(); err != nil {
		return CommunityResult{}, err
	}

	if len(uuids) < 2 {
		return CommunityResult{Entities: len(uuids)}, nil
	}

	// ── 2. Load all edges and build gonum graph ──────────────────────────────
	g := simple.NewWeightedUndirectedGraph(0, 0)

	// Add all nodes.
	for id := int64(0); id < nextID; id++ {
		g.AddNode(simple.Node(id))
	}

	// Add edges (weight = count of edges between two entities).
	edgeWeights := map[[2]int64]float64{}
	edgeRows, err := d.sql.QueryContext(ctx,
		`SELECT source_uuid, target_uuid FROM edges WHERE group_id = ?`, groupID)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("load edges: %w", err)
	}
	defer edgeRows.Close()

	for edgeRows.Next() {
		var src, tgt string
		if err := edgeRows.Scan(&src, &tgt); err != nil {
			return CommunityResult{}, err
		}
		srcID, ok1 := uuidToID[src]
		tgtID, ok2 := uuidToID[tgt]
		if !ok1 || !ok2 || srcID == tgtID {
			continue
		}
		// Canonical edge key (lower ID first).
		key := [2]int64{srcID, tgtID}
		if srcID > tgtID {
			key = [2]int64{tgtID, srcID}
		}
		edgeWeights[key]++
	}
	if err := edgeRows.Err(); err != nil {
		return CommunityResult{}, err
	}

	for key, w := range edgeWeights {
		g.SetWeightedEdge(g.NewWeightedEdge(simple.Node(key[0]), simple.Node(key[1]), w))
	}

	// ── 3. Run Louvain ───────────────────────────────────────────────────────
	reduced := community.Modularize(g, resolution, nil)
	communities := reduced.Communities()

	// ── 4. Write community_id back to database ───────────────────────────────
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	stmt, err := tx.PrepareContext(ctx,
		`UPDATE entities SET community_id = ? WHERE uuid = ? AND group_id = ?`)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("prepare: %w", err)
	}
	defer stmt.Close()

	for communityID, members := range communities {
		for _, node := range members {
			uuid := idToUUID[node.ID()]
			if _, err := stmt.ExecContext(ctx, communityID, uuid, groupID); err != nil {
				return CommunityResult{}, fmt.Errorf("update community: %w", err)
			}
		}
	}

	if err := tx.Commit(); err != nil {
		return CommunityResult{}, fmt.Errorf("commit: %w", err)
	}

	slog.Info("community detection complete",
		"group", groupID,
		"entities", len(uuids),
		"communities", len(communities),
	)

	return CommunityResult{
		Communities: len(communities),
		Entities:    len(uuids),
	}, nil
}

// EntityCommunityID returns the community_id for a given entity UUID.
func (d *DB) EntityCommunityID(ctx context.Context, uuid string) int {
	var id int
	err := d.sql.QueryRowContext(ctx,
		`SELECT community_id FROM entities WHERE uuid = ?`, uuid).Scan(&id)
	if err != nil {
		return -1
	}
	return id
}

// CommunityMap returns a map of entity UUID → community_id for all entities
// in the group that have been assigned a community (community_id >= 0).
func (d *DB) CommunityMap(ctx context.Context, groupID string) (map[string]int, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid, community_id FROM entities WHERE group_id = ? AND community_id >= 0`,
		groupID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	m := map[string]int{}
	for rows.Next() {
		var uuid string
		var cid int
		if err := rows.Scan(&uuid, &cid); err != nil {
			return nil, err
		}
		m[uuid] = cid
	}
	return m, rows.Err()
}

// EntitiesInCommunity returns all entity UUIDs in the given community.
func (d *DB) EntitiesInCommunity(ctx context.Context, groupID string, communityID int) ([]string, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid FROM entities WHERE group_id = ? AND community_id = ?`,
		groupID, communityID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var uuids []string
	for rows.Next() {
		var uuid string
		if err := rows.Scan(&uuid); err != nil {
			return nil, err
		}
		uuids = append(uuids, uuid)
	}
	return uuids, rows.Err()
}

// CommunityInput holds the data needed to generate a community report.
type CommunityInput struct {
	CommunityID int
	EntityNames []string // names of entities in the community
	KeyFacts    []string // top edge facts connecting community members
}

// CommunityInputsForGroup returns CommunityInput for all communities in a group
// that have at least minMembers entities. Used by the report generation step.
func (d *DB) CommunityInputsForGroup(ctx context.Context, groupID string, minMembers int) ([]CommunityInput, error) {
	// Load community → entity names
	rows, err := d.sql.QueryContext(ctx,
		`SELECT community_id, name FROM entities WHERE group_id = ? AND community_id >= 0 ORDER BY community_id`,
		groupID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	communityNames := map[int][]string{}
	for rows.Next() {
		var cid int
		var name string
		if err := rows.Scan(&cid, &name); err != nil {
			return nil, err
		}
		communityNames[cid] = append(communityNames[cid], name)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	var inputs []CommunityInput
	for cid, names := range communityNames {
		if len(names) < minMembers {
			continue
		}
		// Load top 5 edge facts involving community members.
		// Use a subquery to get entity UUIDs for this community.
		factRows, err := d.sql.QueryContext(ctx, `
			SELECT fact FROM edges
			WHERE group_id = ? AND (
				source_uuid IN (SELECT uuid FROM entities WHERE group_id = ? AND community_id = ?)
				OR target_uuid IN (SELECT uuid FROM entities WHERE group_id = ? AND community_id = ?)
			)
			LIMIT 5`,
			groupID, groupID, cid, groupID, cid)
		if err != nil {
			return nil, err
		}
		var facts []string
		for factRows.Next() {
			var f string
			if err := factRows.Scan(&f); err != nil {
				factRows.Close()
				return nil, err
			}
			facts = append(facts, f)
		}
		factRows.Close()
		if err := factRows.Err(); err != nil {
			return nil, err
		}
		inputs = append(inputs, CommunityInput{
			CommunityID: cid,
			EntityNames: names,
			KeyFacts:    facts,
		})
	}
	return inputs, nil
}

// StoreCommunityReport persists a generated report for a community.
func (d *DB) StoreCommunityReport(ctx context.Context, groupID string, communityID int, report string) error {
	_, err := d.sql.ExecContext(ctx,
		`INSERT OR REPLACE INTO community_reports (community_id, group_id, report) VALUES (?, ?, ?)`,
		communityID, groupID, report)
	return err
}

// CommunityReportsForIDs returns stored reports for the given community IDs.
func (d *DB) CommunityReportsForIDs(ctx context.Context, groupID string, communityIDs []int) ([]string, error) {
	if len(communityIDs) == 0 {
		return nil, nil
	}
	args := make([]any, 0, len(communityIDs)+1)
	args = append(args, groupID)
	placeholders := make([]string, len(communityIDs))
	for i, cid := range communityIDs {
		placeholders[i] = "?"
		args = append(args, cid)
	}
	query := `SELECT report FROM community_reports WHERE group_id = ? AND community_id IN (` +
		strings.Join(placeholders, ",") + `) AND report != ''`
	rows, err := d.sql.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var reports []string
	for rows.Next() {
		var r string
		if err := rows.Scan(&r); err != nil {
			return nil, err
		}
		reports = append(reports, r)
	}
	return reports, rows.Err()
}
