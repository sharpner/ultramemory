package llm

import (
	"encoding/json"
	"testing"
)

func TestExtractedEdge_UnmarshalJSON_NormalTypes(t *testing.T) {
	raw := `{"relation_type":"WORKS_AT","source_entity_id":0,"target_entity_id":1,"fact":"Alice works at Google","valid_at":"2020-01-01","invalid_at":null}`
	var edge ExtractedEdge
	if err := json.Unmarshal([]byte(raw), &edge); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if edge.SourceEntityID != 0 {
		t.Errorf("SourceEntityID = %d, want 0", edge.SourceEntityID)
	}
	if edge.TargetEntityID != 1 {
		t.Errorf("TargetEntityID = %d, want 1", edge.TargetEntityID)
	}
	if edge.ValidAt == nil || *edge.ValidAt != "2020-01-01" {
		t.Errorf("ValidAt = %v, want 2020-01-01", edge.ValidAt)
	}
	if edge.InvalidAt != nil {
		t.Errorf("InvalidAt = %v, want nil", edge.InvalidAt)
	}
}

func TestExtractedEdge_UnmarshalJSON_StringIDs(t *testing.T) {
	raw := `{"relation_type":"KNOWS","source_entity_id":"2","target_entity_id":"3","fact":"they know each other"}`
	var edge ExtractedEdge
	if err := json.Unmarshal([]byte(raw), &edge); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if edge.SourceEntityID != 2 {
		t.Errorf("SourceEntityID = %d, want 2", edge.SourceEntityID)
	}
	if edge.TargetEntityID != 3 {
		t.Errorf("TargetEntityID = %d, want 3", edge.TargetEntityID)
	}
}

func TestExtractedEdge_UnmarshalJSON_FloatIDs(t *testing.T) {
	raw := `{"relation_type":"KNOWS","source_entity_id":1.0,"target_entity_id":2.0,"fact":"test"}`
	var edge ExtractedEdge
	if err := json.Unmarshal([]byte(raw), &edge); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if edge.SourceEntityID != 1 {
		t.Errorf("SourceEntityID = %d, want 1", edge.SourceEntityID)
	}
	if edge.TargetEntityID != 2 {
		t.Errorf("TargetEntityID = %d, want 2", edge.TargetEntityID)
	}
}

func TestExtractedEdge_UnmarshalJSON_NumberValidAt(t *testing.T) {
	raw := `{"relation_type":"BORN_IN","source_entity_id":0,"target_entity_id":1,"fact":"born there","valid_at":2024,"invalid_at":2025}`
	var edge ExtractedEdge
	if err := json.Unmarshal([]byte(raw), &edge); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if edge.ValidAt == nil || *edge.ValidAt != "2024" {
		t.Errorf("ValidAt = %v, want '2024'", edge.ValidAt)
	}
	if edge.InvalidAt == nil || *edge.InvalidAt != "2025" {
		t.Errorf("InvalidAt = %v, want '2025'", edge.InvalidAt)
	}
}

func TestExtractedEdge_UnmarshalJSON_MixedTypes(t *testing.T) {
	// Worst case: string IDs + number dates (as seen in production logs)
	raw := `{"relation_type":"LIVES_IN","source_entity_id":"0","target_entity_id":1,"fact":"lives there","valid_at":2024,"invalid_at":null}`
	var edge ExtractedEdge
	if err := json.Unmarshal([]byte(raw), &edge); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if edge.SourceEntityID != 0 {
		t.Errorf("SourceEntityID = %d, want 0", edge.SourceEntityID)
	}
	if edge.TargetEntityID != 1 {
		t.Errorf("TargetEntityID = %d, want 1", edge.TargetEntityID)
	}
	if edge.ValidAt == nil || *edge.ValidAt != "2024" {
		t.Errorf("ValidAt = %v, want '2024'", edge.ValidAt)
	}
	if edge.InvalidAt != nil {
		t.Errorf("InvalidAt = %v, want nil", edge.InvalidAt)
	}
}

func TestExtractedEdge_UnmarshalJSON_BareArray(t *testing.T) {
	raw := `[{"relation_type":"KNOWS","source_entity_id":"1","target_entity_id":"2","fact":"test","valid_at":2020}]`
	var edges []ExtractedEdge
	if err := json.Unmarshal([]byte(raw), &edges); err != nil {
		t.Fatalf("unmarshal array: %v", err)
	}
	if len(edges) != 1 {
		t.Fatalf("got %d edges, want 1", len(edges))
	}
	if edges[0].SourceEntityID != 1 {
		t.Errorf("SourceEntityID = %d, want 1", edges[0].SourceEntityID)
	}
	if edges[0].ValidAt == nil || *edges[0].ValidAt != "2020" {
		t.Errorf("ValidAt = %v, want '2020'", edges[0].ValidAt)
	}
}
