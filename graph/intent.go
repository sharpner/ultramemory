package graph

import "strings"

// QueryIntent classifies what kind of answer the user is looking for.
type QueryIntent int

const (
	IntentWhat     QueryIntent = iota // default: general information
	IntentWhy                         // causal: reason, cause, because, why
	IntentWhen                        // temporal: when, date, time, year
	IntentEntity                      // entity-focused: who, which person, company
)

// EdgeClass classifies an edge by its structural role in the graph.
type EdgeClass int

const (
	EdgeUnknown    EdgeClass = iota // empty / unrecognized name → φ=0 (neutral)
	EdgeRelational                  // KNOWS, RELATED_TO, MEMBER_OF
	EdgeCausal                      // CAUSES, RESULTS_IN, LEADS_TO
	EdgeTemporal                    // BORN_ON, OCCURRED_AT, HAPPENED_ON
	EdgeAttributive                 // HAS, IS_A, WORKS_AT, LOCATED_AT
)

// intentPhi[intent][edgeClass] = φ weight ∈ [0,1].
// Rows = QueryIntent (0..3), Cols = EdgeClass (0..4).
// EdgeUnknown always yields 0.0 so unknown edges are traversal-neutral.
var intentPhi = [4][5]float64{
	// Unknown  Relational  Causal  Temporal  Attributive
	0: {0.0, 0.5, 0.5, 0.5, 0.5}, // IntentWhat
	1: {0.0, 0.5, 1.0, 0.3, 0.4}, // IntentWhy
	2: {0.0, 0.3, 0.3, 1.0, 0.3}, // IntentWhen
	3: {0.0, 1.0, 0.4, 0.3, 0.6}, // IntentEntity
}

// ClassifyIntent returns the dominant intent of a search query.
func ClassifyIntent(query string) QueryIntent {
	for _, w := range strings.Fields(strings.ToLower(query)) {
		switch w {
		case "why", "reason", "cause", "causes", "because":
			return IntentWhy
		case "when", "date", "time", "year", "before", "after", "since":
			return IntentWhen
		case "who", "which", "person", "people", "company", "organization":
			return IntentEntity
		}
	}
	return IntentWhat
}

// classifyEdge infers EdgeClass from a relation name like "WORKS_AT" or "CAUSES".
func classifyEdge(edgeName string) EdgeClass {
	if edgeName == "" {
		return EdgeUnknown
	}
	n := strings.ToUpper(strings.TrimSpace(edgeName))
	// Use short substrings so TRIGGERED_BY matches "TRIGGER", RESULTED_IN matches "RESULT".
	for _, s := range []string{"CAUSE", "RESULT", "LEADS_TO", "TRIGGER", "PRODUCES"} {
		if strings.Contains(n, s) {
			return EdgeCausal
		}
	}
	for _, s := range []string{"BORN_ON", "OCCURRED_AT", "HAPPENED_ON", "FOUNDED_ON", "DIED_ON", "AT_DATE"} {
		if strings.Contains(n, s) {
			return EdgeTemporal
		}
	}
	// Social/structural relations between entities.
	// MARRIED_TO, ALLIED_WITH, OPPOSES, LOVES, PARENT_OF, CHILD_OF, COMMANDS, SERVES_UNDER
	// all fall through from the edgeSystem prompt but were previously EdgeAttributive.
	for _, s := range []string{"KNOWS", "RELATED_TO", "MEMBER_OF", "PART_OF", "CONNECTED_TO",
		"MARRIED", "ALLIED", "OPPOS", "LOVES", "PARENT", "CHILD_OF", "COMMAND", "SERVES"} {
		if strings.Contains(n, s) {
			return EdgeRelational
		}
	}
	// Unknown name but non-empty → treat as attributive (HAS, IS_A, WORKS_AT etc.)
	return EdgeAttributive
}

// edgePhi returns the φ weight for an (EdgeClass, QueryIntent) pair.
func edgePhi(ec EdgeClass, intent QueryIntent) float64 {
	return intentPhi[intent][ec]
}
