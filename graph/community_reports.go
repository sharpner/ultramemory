package graph

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/store"
)

// GenerateCommunityReports stores fact-based community summaries for communities
// with ≥3 Person members (Leiden §4 community context).
//
// Reports are built directly from edge facts — no LLM generation.
// LLM-generated prose summaries caused hallucinations in testing (e.g., adding
// "LGBTQ+ support group" membership not present in the actual conversation),
// which degraded open-domain retrieval by -2.8% and overall by -1.7%.
// Fact-only reports are grounded, verifiable, and prevent context pollution.
func GenerateCommunityReports(ctx context.Context, db *store.DB, _ *llm.Client, groupID string) error {
	inputs, err := db.CommunityInputsForGroup(ctx, groupID, 3)
	if err != nil {
		return fmt.Errorf("load community inputs: %w", err)
	}
	if len(inputs) == 0 {
		return nil
	}

	generated := 0
	for _, inp := range inputs {
		if len(inp.KeyFacts) == 0 {
			continue
		}
		// Build a fact-only report: entity roster + key facts.
		// No LLM call — prevents hallucination of training-data knowledge.
		var sb strings.Builder
		sb.WriteString("People: ")
		sb.WriteString(strings.Join(inp.EntityNames, ", "))
		sb.WriteString(". Key facts: ")
		sb.WriteString(strings.Join(inp.KeyFacts, " "))

		report := sb.String()
		if err := db.StoreCommunityReport(ctx, groupID, inp.CommunityID, report); err != nil {
			slog.Warn("store community report failed", "community", inp.CommunityID, "err", err)
			continue
		}
		generated++
	}
	slog.Info("community reports generated (fact-only)", "group", groupID, "count", generated, "total", len(inputs))
	return nil
}
