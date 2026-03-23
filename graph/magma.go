package graph

import (
	"context"
	"math"
	"sort"

	"github.com/sharpner/ultramemory/store"
)

// MAGMAConfig configures the beam-search spreading activation.
//
// Full transition per MAGMA paper (arxiv.org/abs/2601.03236), Eq. 5:
//
//	S(n_j | n_i, q) = exp(λ₁·φ(type(e_ij), T_q) + λ₂·sim(n⃗_j, q⃗))
//
// Score propagation, Algorithm 1 line 10 (additive with decay):
//
//	score_v = score_u · γ + S(n_j | n_i, q)
//
// Traversal terminates via MaxHops (MaxDepth=5) and MaxNodes (Budget=200).
// There is no threshold in the paper; Threshold=0 disables output filtering.
type MAGMAConfig struct {
	BeamWidth int     // candidates to keep per hop — paper: BeamWidth (default: 10)
	MaxHops   int     // maximum traversal depth — paper: MaxDepth=5 (default: 5)
	Threshold float64 // minimum score for output inclusion; 0 = include all (default: 0)
	MaxNodes  int     // visited node budget — paper: Budget=200 (default: 200)
	Decay     float64 // γ: parent-score decay per hop (default: 0.5)
	Lambda1   float64 // λ₁: edge-intent structural alignment coefficient — paper: 1.0 (default: 1.0)
	Lambda2   float64 // λ₂: semantic affinity coefficient — paper: 0.3–0.7 (default: 0.5)
}

// DefaultMAGMAConfig returns paper-aligned defaults (arxiv.org/abs/2601.03236).
func DefaultMAGMAConfig() MAGMAConfig {
	return MAGMAConfig{
		BeamWidth: 10,
		MaxHops:   5,
		Threshold: 0,
		MaxNodes:  200,
		Decay:     0.5,
		Lambda1:   1.0,
		Lambda2:   0.5,
	}
}

// ActivatedNode is an entity with an accumulated activation score from graph traversal.
type ActivatedNode struct {
	UUID       string
	Name       string
	EntityType string
	Activation float64
}

// GraphTraverser retrieves neighbors of a node — satisfied by *store.DB.
type GraphTraverser interface {
	GetNeighbors(ctx context.Context, uuid, groupID string) ([]store.NeighborEntity, error)
}

type beamEntry struct {
	uuid  string
	name  string
	etype string
	score float64
	depth int
}

// SpreadMAGMA performs beam-search spreading activation from seed nodes.
//
// Implements Algorithm 1 from the MAGMA paper (arxiv.org/abs/2601.03236):
//
//  1. Seeds initialised at score=1.0, added to visited.
//  2. Per hop: expand each frontier node, compute transition score S for every
//     unvisited neighbor, accumulate bestScores additively, add to Candidates.
//  3. Candidates trimmed to BeamWidth; survivors marked visited.
//  4. Terminates at MaxHops or when visited exceeds MaxNodes (Budget).
//
// EdgeUnknown (empty/unrecognized edge name) yields φ=0 — unknown edges are
// traversal-neutral, only embedding similarity contributes via λ₂.
// Scores accumulate across paths so hub nodes rank higher than leaf nodes.
func SpreadMAGMA(ctx context.Context, g GraphTraverser, seeds []ActivatedNode, query string, queryEmb []float32, groupID string, cfg MAGMAConfig) ([]ActivatedNode, error) {
	if len(seeds) == 0 {
		return nil, nil
	}
	applyMAGMADefaults(&cfg)
	intent := ClassifyIntent(query)

	bestScores := make(map[string]float64, len(seeds))
	nameMap := make(map[string]string, len(seeds))
	etypeMap := make(map[string]string, len(seeds))
	visited := make(map[string]bool, len(seeds))

	beam := make([]beamEntry, 0, len(seeds))
	for _, s := range seeds {
		bestScores[s.UUID] = 1.0
		nameMap[s.UUID] = s.Name
		etypeMap[s.UUID] = s.EntityType
		visited[s.UUID] = true
		beam = append(beam, beamEntry{uuid: s.UUID, name: s.Name, etype: s.EntityType, score: 1.0})
	}

	for hop := 0; hop < cfg.MaxHops && len(beam) > 0; hop++ {
		// Budget check: paper terminates when visited.Size() >= Budget.
		if len(visited) >= cfg.MaxNodes {
			break
		}

		var next []beamEntry
		for _, candidate := range beam {
			neighbors, err := g.GetNeighbors(ctx, candidate.uuid, groupID)
			if err != nil {
				continue
			}
			// Synapse fan effect: dilute energy across outgoing edges.
			// High-degree hubs spread less energy per edge than leaf nodes.
			fanOut := float64(len(neighbors))
			if fanOut < 1 {
				fanOut = 1
			}
			for _, nb := range neighbors {
				if visited[nb.UUID] {
					continue
				}
				// Eq. 5: S(n_j|n_i,q) = exp(λ₁·φ + λ₂·sim)
				t := computeTransition(nb.EdgeName, nb.Embedding, queryEmb, intent, cfg.Lambda1, cfg.Lambda2)
				// Algorithm 1 line 10: score_v = score_u · γ + S  (additive)
				// Fan effect applied: divide propagated score by out-degree.
				newScore := (candidate.score*cfg.Decay + t) / fanOut
				bestScores[nb.UUID] += newScore
				if _, ok := nameMap[nb.UUID]; !ok {
					nameMap[nb.UUID] = nb.Name
					etypeMap[nb.UUID] = nb.EntityType
				}
				next = append(next, beamEntry{
					uuid:  nb.UUID,
					name:  nb.Name,
					etype: nb.EntityType,
					score: newScore,
					depth: candidate.depth + 1,
				})
			}
		}
		// Algorithm 1 line 17: CurrentFrontier ← Candidates.TopK(BeamWidth)
		sort.Slice(next, func(i, j int) bool { return next[i].score > next[j].score })
		if len(next) > cfg.BeamWidth {
			next = next[:cfg.BeamWidth]
		}
		// Algorithm 1 line 16: Visited.AddAll(NextFrontier)
		for _, e := range next {
			visited[e.uuid] = true
		}
		beam = next
	}

	out := make([]ActivatedNode, 0, len(bestScores))
	for uuid, score := range bestScores {
		if cfg.Threshold > 0 && score < cfg.Threshold {
			continue
		}
		out = append(out, ActivatedNode{
			UUID:       uuid,
			Name:       nameMap[uuid],
			EntityType: etypeMap[uuid],
			Activation: score,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Activation > out[j].Activation })
	if len(out) > cfg.MaxNodes {
		out = out[:cfg.MaxNodes]
	}

	// Synapse lateral inhibition: winner-take-all competition.
	// Top-M nodes suppress weaker ones. Nodes that survive are
	// structurally distinct winners, reducing context noise.
	out = lateralInhibition(out, 7, 0.15)

	return out, nil
}

// computeTransition computes the MAGMA transition score (Eq. 5):
//
//	S = exp(λ₁·φ(edge_type, intent) + λ₂·cos_sim(nodeEmb, queryEmb))
//
// EdgeUnknown yields φ=0; missing embeddings yield cos_sim=0.
// With both absent: S = exp(0) = 1.0 (traversal-neutral).
func computeTransition(edgeName string, nodeEmb, queryEmb []float32, intent QueryIntent, lambda1, lambda2 float64) float64 {
	phi := edgePhi(classifyEdge(edgeName), intent)
	embSim := 0.0
	if len(nodeEmb) > 0 && len(queryEmb) > 0 {
		embSim = float64(store.CosineSimilarity(nodeEmb, queryEmb))
	}
	return math.Exp(lambda1*phi + lambda2*embSim)
}

func applyMAGMADefaults(cfg *MAGMAConfig) {
	if cfg.BeamWidth == 0 {
		cfg.BeamWidth = 10
	}
	if cfg.MaxHops == 0 {
		cfg.MaxHops = 5
	}
	if cfg.MaxNodes == 0 {
		cfg.MaxNodes = 200
	}
	if cfg.Decay == 0 {
		cfg.Decay = 0.5
	}
	if cfg.Lambda1 == 0 {
		cfg.Lambda1 = 1.0
	}
	if cfg.Lambda2 == 0 {
		cfg.Lambda2 = 0.5
	}
	// Threshold: 0 means "no output filtering" — do not fill with a default.
}

// lateralInhibition implements Synapse-style winner-take-all competition.
// For each node, subtract beta × sum of (stronger_score - my_score) for the
// top-M strongest competitors. Nodes whose adjusted score drops to ≤0 are removed.
// The input must be sorted descending by Activation.
func lateralInhibition(nodes []ActivatedNode, topM int, beta float64) []ActivatedNode {
	if len(nodes) <= 1 {
		return nodes
	}
	if topM > len(nodes) {
		topM = len(nodes)
	}

	adjusted := make([]ActivatedNode, len(nodes))
	copy(adjusted, nodes)

	for i := range adjusted {
		var inhibition float64
		for k := 0; k < topM && k < i; k++ {
			inhibition += nodes[k].Activation - nodes[i].Activation
		}
		adjusted[i].Activation = nodes[i].Activation - beta*inhibition
	}

	// Remove nodes suppressed to ≤ 0.
	out := adjusted[:0]
	for _, n := range adjusted {
		if n.Activation <= 0 {
			continue
		}
		out = append(out, n)
	}
	return out
}
