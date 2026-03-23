package graph

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/store"
)

const communityReportPrompt = `You summarize a group of people and their relationships in ONE concise sentence.

Given the people and key facts below, write exactly one sentence (max 30 words) that captures who these people are and how they are connected.

People: %s
Key facts:
%s

One sentence summary:`

// GenerateCommunityReports generates LLM summaries for communities with ≥3 members
// and stores them in the community_reports table (Leiden §4 community context).
// Skips communities that already have a report stored.
func GenerateCommunityReports(ctx context.Context, db *store.DB, client *llm.Client, groupID string) error {
	inputs, err := db.CommunityInputsForGroup(ctx, groupID, 3)
	if err != nil {
		return fmt.Errorf("load community inputs: %w", err)
	}
	if len(inputs) == 0 {
		return nil
	}

	generated := 0
	for _, inp := range inputs {
		facts := strings.Join(inp.KeyFacts, "\n- ")
		if facts == "" {
			facts = "(no direct connections found)"
		}
		prompt := fmt.Sprintf(communityReportPrompt,
			strings.Join(inp.EntityNames, ", "),
			"- "+facts,
		)

		report, err := client.Answer(ctx, "", prompt, 64)
		if err != nil {
			slog.Warn("community report generation failed",
				"community", inp.CommunityID, "err", err)
			continue
		}
		// Clean up: take first sentence only, strip trailing whitespace.
		report = strings.TrimSpace(report)
		if dot := strings.Index(report, "."); dot >= 0 {
			report = report[:dot+1]
		}
		if err := db.StoreCommunityReport(ctx, groupID, inp.CommunityID, report); err != nil {
			slog.Warn("store community report failed", "community", inp.CommunityID, "err", err)
			continue
		}
		generated++
	}
	slog.Info("community reports generated", "group", groupID, "count", generated, "total", len(inputs))
	return nil
}
