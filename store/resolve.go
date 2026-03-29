package store

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"strings"
)

// ResolveConfig controls entity resolution behaviour.
type ResolveConfig struct {
	Threshold float64 // cosine similarity threshold, e.g. 0.85
	DryRun    bool    // when true, compute clusters but write nothing
}

// ResolveResult summarises what was (or would be) done.
type ResolveResult struct {
	ClustersFound    int
	EntitiesMerged   int
	EdgesRetargeted  int
	EpisodesRelinked int
}

// ResolveEntities merges near-duplicate entities within groupID using
// cosine similarity on stored embeddings. Entities without embeddings
// are skipped. When cfg.DryRun is true, planned merges are logged but
// no rows are modified.
func (d *DB) ResolveEntities(ctx context.Context, groupID string, cfg ResolveConfig) (ResolveResult, error) {
	entities, err := d.AllEntitiesWithEmbeddings(ctx, groupID)
	if err != nil {
		return ResolveResult{}, fmt.Errorf("load entities: %w", err)
	}

	// Group by entity_type so we only compare within the same type.
	byType := make(map[string][]Entity)
	for _, e := range entities {
		byType[e.EntityType] = append(byType[e.EntityType], e)
	}

	// Union-Find over positions in the global entities slice.
	uuidToIdx := make(map[string]int, len(entities))
	for i, e := range entities {
		uuidToIdx[e.UUID] = i
	}
	uf := newUnionFind(len(entities))

	for _, group := range byType {
		for i := 0; i < len(group); i++ {
			for j := i + 1; j < len(group); j++ {
				// Token Jaccard on names is the primary signal.
				// Embedding cosine confirms a name-based match, never triggers alone.
				nameSim := tokenJaccard(group[i].Name, group[j].Name)
				if nameSim < 0.5 {
					continue
				}
				embSim := CosineSimilarity(group[i].Embedding, group[j].Embedding)
				if embSim < cfg.Threshold && nameSim < cfg.Threshold {
					continue
				}
				uf.union(uuidToIdx[group[i].UUID], uuidToIdx[group[j].UUID])
			}
		}
	}

	// Collect clusters with more than one member.
	rootToMembers := make(map[int][]int)
	for i := range entities {
		root := uf.find(i)
		rootToMembers[root] = append(rootToMembers[root], i)
	}

	var mergeClusters [][]Entity
	for _, members := range rootToMembers {
		if len(members) < 2 {
			continue
		}
		cluster := make([]Entity, len(members))
		for i, idx := range members {
			cluster[i] = entities[idx]
		}
		mergeClusters = append(mergeClusters, cluster)
	}

	result := ResolveResult{ClustersFound: len(mergeClusters)}

	if len(mergeClusters) == 0 {
		return result, nil
	}

	if cfg.DryRun {
		for _, cluster := range mergeClusters {
			canonical := pickCanonical(ctx, d, cluster, groupID)
			names := make([]string, len(cluster))
			for i, e := range cluster {
				names[i] = e.Name
			}
			slog.Info("dry-run: would merge",
				"canonical", canonical.Name,
				"duplicates", strings.Join(names, ", "),
			)
		}
		return result, nil
	}

	for _, cluster := range mergeClusters {
		canonical := pickCanonical(ctx, d, cluster, groupID)
		dupes := make([]Entity, 0, len(cluster)-1)
		for _, e := range cluster {
			if e.UUID != canonical.UUID {
				dupes = append(dupes, e)
			}
		}

		edgesRetargeted, episodesRelinked, err := d.mergeCluster(ctx, groupID, canonical, dupes)
		if err != nil {
			return result, fmt.Errorf("merge cluster (canonical=%s): %w", canonical.Name, err)
		}
		result.EntitiesMerged += len(dupes)
		result.EdgesRetargeted += edgesRetargeted
		result.EpisodesRelinked += episodesRelinked
	}

	return result, nil
}

// pickCanonical selects the entity with the most edges in groupID.
// Tiebreaker: longest name.
func pickCanonical(ctx context.Context, d *DB, cluster []Entity, groupID string) Entity {
	best := cluster[0]
	bestEdges := edgeCount(ctx, d, best.UUID, groupID)

	for _, e := range cluster[1:] {
		count := edgeCount(ctx, d, e.UUID, groupID)
		if count > bestEdges {
			best = e
			bestEdges = count
			continue
		}
		if count == bestEdges && len(e.Name) > len(best.Name) {
			best = e
		}
	}
	return best
}

// edgeCount returns the number of edges where uuid appears as source or target.
func edgeCount(ctx context.Context, d *DB, uuid, groupID string) int {
	var n int
	err := d.sql.QueryRowContext(ctx,
		`SELECT count(*) FROM edges
		 WHERE (source_uuid = ? OR target_uuid = ?) AND group_id = ?`,
		uuid, uuid, groupID,
	).Scan(&n)
	if err != nil {
		slog.Warn("edgeCount query failed, defaulting to 0", "uuid", uuid, "err", err)
	}
	return n
}

// mergeCluster retargets edges and episode links from dupes to canonical,
// then deletes the duplicate entity rows. Runs inside a single transaction.
func (d *DB) mergeCluster(ctx context.Context, groupID string, canonical Entity, dupes []Entity) (edgesRetargeted, episodesRelinked int, err error) {
	dupeUUIDs := make([]string, len(dupes))
	for i, e := range dupes {
		dupeUUIDs[i] = e.UUID
	}

	ph := placeholders(len(dupeUUIDs))
	dupeArgs := stringsToAny(dupeUUIDs)

	tx, err := d.sql.BeginTx(ctx, &sql.TxOptions{})
	if err != nil {
		return 0, 0, fmt.Errorf("begin tx: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	// Retarget source side of edges.
	srcArgs := append([]any{canonical.UUID}, dupeArgs...)
	srcArgs = append(srcArgs, groupID)
	res, txErr := tx.ExecContext(ctx,
		`UPDATE edges SET source_uuid = ?
		 WHERE source_uuid IN (`+ph+`) AND group_id = ?`,
		srcArgs...,
	)
	if txErr != nil {
		return 0, 0, fmt.Errorf("retarget source: %w", txErr)
	}
	n, _ := res.RowsAffected()
	edgesRetargeted += int(n)

	// Retarget target side of edges.
	tgtArgs := append([]any{canonical.UUID}, dupeArgs...)
	tgtArgs = append(tgtArgs, groupID)
	res, txErr = tx.ExecContext(ctx,
		`UPDATE edges SET target_uuid = ?
		 WHERE target_uuid IN (`+ph+`) AND group_id = ?`,
		tgtArgs...,
	)
	if txErr != nil {
		return 0, 0, fmt.Errorf("retarget target: %w", txErr)
	}
	n, _ = res.RowsAffected()
	edgesRetargeted += int(n)

	// Delete self-loops created by the merge.
	if _, txErr = tx.ExecContext(ctx,
		`DELETE FROM edges
		 WHERE source_uuid = target_uuid AND source_uuid = ? AND group_id = ?`,
		canonical.UUID, groupID,
	); txErr != nil {
		return 0, 0, fmt.Errorf("delete self-loops: %w", txErr)
	}

	// Deduplicate edges that now share (source_uuid, target_uuid, name, group_id).
	if _, txErr = tx.ExecContext(ctx,
		`DELETE FROM edges
		 WHERE rowid NOT IN (
		   SELECT min(rowid) FROM edges
		   WHERE group_id = ?
		   GROUP BY source_uuid, target_uuid, name, group_id
		 ) AND group_id = ?`,
		groupID, groupID,
	); txErr != nil {
		return 0, 0, fmt.Errorf("dedup edges: %w", txErr)
	}

	// Relink entity_episodes: copy dupe links to canonical, then delete dupes.
	relinkArgs := append([]any{canonical.UUID}, dupeArgs...)
	res, txErr = tx.ExecContext(ctx,
		`INSERT OR IGNORE INTO entity_episodes (entity_uuid, episode_uuid)
		 SELECT ?, episode_uuid FROM entity_episodes
		 WHERE entity_uuid IN (`+ph+`)`,
		relinkArgs...,
	)
	if txErr != nil {
		return 0, 0, fmt.Errorf("relink episodes: %w", txErr)
	}
	n, _ = res.RowsAffected()
	episodesRelinked += int(n)

	if _, txErr = tx.ExecContext(ctx,
		`DELETE FROM entity_episodes WHERE entity_uuid IN (`+ph+`)`,
		dupeArgs...,
	); txErr != nil {
		return 0, 0, fmt.Errorf("delete old entity_episodes: %w", txErr)
	}

	// Clean up content-less FTS table.
	if _, txErr = tx.ExecContext(ctx,
		`DELETE FROM entities_fts WHERE uuid IN (`+ph+`)`,
		dupeArgs...,
	); txErr != nil {
		return 0, 0, fmt.Errorf("delete entities_fts: %w", txErr)
	}

	// Delete the duplicate entity rows.
	deleteArgs := append(dupeArgs, groupID)
	if _, txErr = tx.ExecContext(ctx,
		`DELETE FROM entities WHERE uuid IN (`+ph+`) AND group_id = ?`,
		deleteArgs...,
	); txErr != nil {
		return 0, 0, fmt.Errorf("delete entities: %w", txErr)
	}

	if txErr = tx.Commit(); txErr != nil {
		return 0, 0, fmt.Errorf("commit: %w", txErr)
	}
	return edgesRetargeted, episodesRelinked, nil
}

// placeholders returns n comma-separated `?` markers, e.g. "?,?,?" for n=3.
func placeholders(n int) string {
	if n <= 0 {
		return ""
	}
	return strings.Repeat("?,", n-1) + "?"
}

// stringsToAny converts []string to []any for use in variadic SQL args.
func stringsToAny(ss []string) []any {
	out := make([]any, len(ss))
	for i, s := range ss {
		out[i] = s
	}
	return out
}

// tokenJaccard computes the Jaccard similarity between lowercased token sets.
// "Jonathan Harker" vs "Harker Jonathan" → 1.0 (same tokens).
// "Mr. Harker" vs "Jonathan Harker" → 0.5 (one shared token out of three unique).
func tokenJaccard(a, b string) float64 {
	tokensA := strings.Fields(strings.ToLower(a))
	tokensB := strings.Fields(strings.ToLower(b))
	if len(tokensA) == 0 || len(tokensB) == 0 {
		return 0
	}
	setA := make(map[string]bool, len(tokensA))
	for _, t := range tokensA {
		setA[t] = true
	}
	setB := make(map[string]bool, len(tokensB))
	for _, t := range tokensB {
		setB[t] = true
	}
	intersection := 0
	for t := range setA {
		if setB[t] {
			intersection++
		}
	}
	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

// unionFind implements Union-Find with path compression and union by rank.
type unionFind struct {
	parent []int
	rank   []int
}

func newUnionFind(n int) *unionFind {
	uf := &unionFind{
		parent: make([]int, n),
		rank:   make([]int, n),
	}
	for i := range uf.parent {
		uf.parent[i] = i
	}
	return uf
}

func (uf *unionFind) find(x int) int {
	if uf.parent[x] != x {
		uf.parent[x] = uf.find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *unionFind) union(x, y int) {
	rx, ry := uf.find(x), uf.find(y)
	if rx == ry {
		return
	}
	if uf.rank[rx] < uf.rank[ry] {
		rx, ry = ry, rx
	}
	uf.parent[ry] = rx
	if uf.rank[rx] == uf.rank[ry] {
		uf.rank[rx]++
	}
}
