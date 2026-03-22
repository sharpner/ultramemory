package graph

import (
	"context"
	"sort"
	"strings"

	"github.com/sharpner/ultramemory/store"
)

// MAGMAConfig configures the beam-search spreading activation.
type MAGMAConfig struct {
	BeamWidth   int     // candidates to keep per hop (default: 10)
	MaxHops     int     // maximum traversal depth (default: 3)
	Threshold   float64 // minimum activation score to keep (default: 0.1)
	MaxNodes    int     // maximum nodes to return (default: 50)
	DecayFactor float64 // base decay per hop (default: 0.5)
}

// DefaultMAGMAConfig returns sensible defaults.
func DefaultMAGMAConfig() MAGMAConfig {
	return MAGMAConfig{
		BeamWidth:   10,
		MaxHops:     3,
		Threshold:   0.1,
		MaxNodes:    50,
		DecayFactor: 0.5,
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
// Each hop retrieves neighbors, computes keyword-based semantic affinity, and
// accumulates scores so nodes reachable via multiple paths rank higher.
func SpreadMAGMA(ctx context.Context, g GraphTraverser, seeds []ActivatedNode, query, groupID string, cfg MAGMAConfig) ([]ActivatedNode, error) {
	if len(seeds) == 0 {
		return nil, nil
	}
	applyMAGMADefaults(&cfg)

	keywords := magmaKeywords(query)
	bestScores := make(map[string]float64, len(seeds))
	nameMap := make(map[string]string, len(seeds))
	etypeMap := make(map[string]string, len(seeds))

	beam := make([]beamEntry, 0, len(seeds))
	for _, s := range seeds {
		bestScores[s.UUID] = 1.0
		nameMap[s.UUID] = s.Name
		etypeMap[s.UUID] = s.EntityType
		beam = append(beam, beamEntry{uuid: s.UUID, name: s.Name, etype: s.EntityType, score: 1.0})
	}

	for hop := 0; hop < cfg.MaxHops && len(beam) > 0; hop++ {
		var next []beamEntry
		for _, candidate := range beam {
			if candidate.depth >= cfg.MaxHops {
				continue
			}
			neighbors, err := g.GetNeighbors(ctx, candidate.uuid, groupID)
			if err != nil {
				continue
			}
			for _, nb := range neighbors {
				affinity := magmaAffinity(nb.Name+" "+nb.EdgeFact, keywords)
				newScore := candidate.score * cfg.DecayFactor * affinity
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
	if cfg.DecayFactor == 0 {
		cfg.DecayFactor = 0.5
	}
}

// magmaAffinity returns 0.3–1.0 based on keyword hits in text; 0.5 if no keywords.
func magmaAffinity(text string, keywords []string) float64 {
	if len(keywords) == 0 {
		return 0.5
	}
	lower := strings.ToLower(text)
	matches := 0
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			matches++
		}
	}
	return 0.3 + 0.7*float64(matches)/float64(len(keywords))
}

var magmaStopWords = map[string]bool{
	"der": true, "die": true, "das": true, "ein": true, "eine": true,
	"und": true, "oder": true, "aber": true, "wenn": true, "weil": true,
	"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
	"is": true, "are": true, "was": true, "were": true, "be": true,
	"in": true, "on": true, "at": true, "to": true, "for": true,
	"of": true, "with": true, "by": true, "from": true, "as": true,
	"ist": true, "sind": true, "war": true, "mit": true, "von": true,
	"zu": true, "für": true, "aus": true, "bei": true, "nach": true,
}

func magmaKeywords(query string) []string {
	words := strings.Fields(strings.ToLower(query))
	out := make([]string, 0, len(words))
	for _, w := range words {
		if len(w) >= 3 && !magmaStopWords[w] {
			out = append(out, w)
		}
	}
	return out
}
