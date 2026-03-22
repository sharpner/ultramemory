package graph

import (
	"context"
	"math"
	"sort"

	"github.com/sharpner/ultramemory/store"
)

// MAGMAConfig configures the beam-search spreading activation.
//
// Full transition per MAGMA paper (arxiv.org/abs/2601.03236):
//
//	S(n_j | n_i, q) = exp(λ₁·φ(edge_type, intent) + λ₂·cos_sim(n_j.embedding, query))
//
// Score propagation (multiplicative decay):
//
//	score_v = score_u · γ · S(n_j | n_i, q)
type MAGMAConfig struct {
	BeamWidth int     // candidates to keep per hop (default: 10)
	MaxHops   int     // maximum traversal depth (default: 3)
	Threshold float64 // minimum activation score to keep (default: 0.1)
	MaxNodes  int     // maximum nodes to return (default: 50)
	Decay     float64 // γ: score decay per hop (default: 0.5)
	Lambda1   float64 // edge-intent coefficient for φ term (default: 0.5)
	Lambda2   float64 // semantic coefficient for exp(λ₂·sim) (default: 0.5)
}

// DefaultMAGMAConfig returns sensible defaults.
func DefaultMAGMAConfig() MAGMAConfig {
	return MAGMAConfig{
		BeamWidth: 10,
		MaxHops:   3,
		Threshold: 0.1,
		MaxNodes:  50,
		Decay:     0.5,
		Lambda1:   0.5,
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
// The full MAGMA transition formula is applied:
//
//	S(n_j|n_i,q) = exp(λ₁·φ(edge_type, intent(query)) + λ₂·cos_sim(n_j.embedding, queryEmb))
//
// EdgeUnknown (empty or unrecognized edge name) yields φ=0 so unknown edges
// are traversal-neutral — only embedding similarity contributes.
// Scores accumulate across paths so hub nodes rank higher.
// A visited set prevents cycles from inflating scores.
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
		var next []beamEntry
		for _, candidate := range beam {
			neighbors, err := g.GetNeighbors(ctx, candidate.uuid, groupID)
			if err != nil {
				continue
			}
			for _, nb := range neighbors {
				if visited[nb.UUID] {
					continue
				}
				t := computeTransition(nb.EdgeName, nb.Embedding, queryEmb, intent, cfg.Lambda1, cfg.Lambda2)
				newScore := candidate.score * cfg.Decay * t
				bestScores[nb.UUID] += newScore
				if _, ok := nameMap[nb.UUID]; !ok {
					nameMap[nb.UUID] = nb.Name
					etypeMap[nb.UUID] = nb.EntityType
				}
				if newScore >= cfg.Threshold {
					next = append(next, beamEntry{
						uuid:  nb.UUID,
						name:  nb.Name,
						etype: nb.EntityType,
						score: newScore,
						depth: candidate.depth + 1,
					})
				}
			}
		}
		sort.Slice(next, func(i, j int) bool { return next[i].score > next[j].score })
		if len(next) > cfg.BeamWidth {
			next = next[:cfg.BeamWidth]
		}
		// Mark the surviving beam as visited before the next hop.
		for _, e := range next {
			visited[e.uuid] = true
		}
		beam = next
	}

	out := make([]ActivatedNode, 0, len(bestScores))
	for uuid, score := range bestScores {
		if score < cfg.Threshold {
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
	return out, nil
}

// computeTransition computes the full MAGMA transition score:
//
//	S = exp(λ₁·φ(edge_type, intent) + λ₂·cos_sim(nodeEmb, queryEmb))
//
// EdgeUnknown (empty/unrecognized edgeName) yields φ=0 so λ₁ has no effect.
// Missing embeddings yield cos_sim=0 so λ₂ has no effect.
// With both absent: S = exp(0) = 1.0 (neutral, no bias).
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
		cfg.MaxHops = 3
	}
	if cfg.Threshold == 0 {
		cfg.Threshold = 0.1
	}
	if cfg.MaxNodes == 0 {
		cfg.MaxNodes = 50
	}
	if cfg.Decay == 0 {
		cfg.Decay = 0.5
	}
	if cfg.Lambda1 == 0 {
		cfg.Lambda1 = 0.5
	}
	if cfg.Lambda2 == 0 {
		cfg.Lambda2 = 0.5
	}
}
