package graph

import "testing"

func TestClassifyIntent(t *testing.T) {
	tests := []struct {
		query  string
		intent QueryIntent
	}{
		{"why did alice leave", IntentWhy},
		{"the reason for departure", IntentWhy},
		{"because of what happened", IntentWhy},
		{"what causes the error", IntentWhy},
		{"when was bob born", IntentWhen},
		{"what date did it happen", IntentWhen},
		{"before 2020", IntentWhen},
		{"who is alice", IntentEntity},
		{"which company does bob work for", IntentEntity},
		{"person named bob", IntentEntity},
		{"tell me about alice", IntentWhat},
		{"alice schmidt engineer", IntentWhat},
		{"", IntentWhat},
		{"knowledge graph retrieval", IntentWhat},
	}
	for _, tc := range tests {
		got := ClassifyIntent(tc.query)
		if got != tc.intent {
			t.Errorf("ClassifyIntent(%q) = %v, want %v", tc.query, got, tc.intent)
		}
	}
}

func TestClassifyEdge(t *testing.T) {
	tests := []struct {
		name  string
		class EdgeClass
	}{
		{"CAUSES", EdgeCausal},
		{"RESULTS_IN", EdgeCausal},
		{"LEADS_TO", EdgeCausal},
		{"CAUSED_BY", EdgeCausal},
		{"BORN_ON", EdgeTemporal},
		{"OCCURRED_AT", EdgeTemporal},
		{"HAPPENED_ON", EdgeTemporal},
		{"FOUNDED_ON", EdgeTemporal},
		{"KNOWS", EdgeRelational},
		{"MEMBER_OF", EdgeRelational},
		{"RELATED_TO", EdgeRelational},
		{"PART_OF", EdgeRelational},
		{"WORKS_AT", EdgeAttributive},
		{"IS_A", EdgeAttributive},
		{"HAS", EdgeAttributive},
		{"LOCATED_AT", EdgeAttributive},
		{"RANDOM_EDGE", EdgeAttributive}, // non-empty but unrecognized → Attributive
		{"", EdgeUnknown},               // empty → Unknown
	}
	for _, tc := range tests {
		got := classifyEdge(tc.name)
		if got != tc.class {
			t.Errorf("classifyEdge(%q) = %v, want %v", tc.name, got, tc.class)
		}
	}
}

func TestEdgePhi(t *testing.T) {
	// EdgeUnknown always yields 0.0 regardless of intent.
	for _, intent := range []QueryIntent{IntentWhat, IntentWhy, IntentWhen, IntentEntity} {
		if phi := edgePhi(EdgeUnknown, intent); phi != 0.0 {
			t.Errorf("edgePhi(EdgeUnknown, %v) = %f, want 0.0", intent, phi)
		}
	}
	// Domain-matched pairs yield maximum weight.
	if phi := edgePhi(EdgeCausal, IntentWhy); phi != 1.0 {
		t.Errorf("edgePhi(EdgeCausal, IntentWhy) = %f, want 1.0", phi)
	}
	if phi := edgePhi(EdgeTemporal, IntentWhen); phi != 1.0 {
		t.Errorf("edgePhi(EdgeTemporal, IntentWhen) = %f, want 1.0", phi)
	}
	if phi := edgePhi(EdgeRelational, IntentEntity); phi != 1.0 {
		t.Errorf("edgePhi(EdgeRelational, IntentEntity) = %f, want 1.0", phi)
	}
	// IntentWhat uses uniform weights for all known edge classes.
	for _, ec := range []EdgeClass{EdgeRelational, EdgeCausal, EdgeTemporal, EdgeAttributive} {
		if phi := edgePhi(ec, IntentWhat); phi != 0.5 {
			t.Errorf("edgePhi(%v, IntentWhat) = %f, want 0.5", ec, phi)
		}
	}
}
