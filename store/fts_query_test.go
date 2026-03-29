package store

import (
	"strings"
	"testing"
)

func TestFts5Query_PossessiveSplit(t *testing.T) {
	// "Caroline's" must produce "Caroline*" not "Carolines*"
	// FTS5 unicode61 tokenizes "Caroline's" → ["caroline", "s"].
	// Old code stripped ' → "Carolines" → "Carolines*" which doesn't match "caroline".
	q := fts5Query("Caroline's job")
	if strings.Contains(q, "Carolines*") {
		t.Errorf("possessive bug: got %q, must not contain 'Carolines*'", q)
	}
	if !strings.Contains(q, "Caroline*") {
		t.Errorf("possessive fix: got %q, must contain 'Caroline*'", q)
	}
}

func TestFts5Query_ShortTokensDropped(t *testing.T) {
	// Possessive "s" (len=1) must be dropped.
	q := fts5Query("Caroline's")
	if strings.Contains(q, " OR s*") || strings.HasSuffix(q, " OR s*") || q == "s*" {
		t.Errorf("short token 's' not dropped: got %q", q)
	}
	if !strings.Contains(q, "Caroline*") {
		t.Errorf("entity name missing: got %q, want Caroline*", q)
	}
}

func TestFts5Query_Empty(t *testing.T) {
	if q := fts5Query(""); q != "" {
		t.Errorf("empty query should return empty string, got %q", q)
	}
}

func TestFts5Query_NumericEntityName(t *testing.T) {
	// Single-digit entity names like "4" must not be dropped.
	q := fts5Query("4")
	if q == "" {
		t.Fatal("numeric entity name '4' was dropped, want '4*'")
	}
	if q != "4*" {
		t.Errorf("got %q, want %q", q, "4*")
	}
}

func TestFts5Query_MixedShortTokens(t *testing.T) {
	// "d 4" — "d" is alphabetic noise and may be dropped, but "4" must survive.
	q := fts5Query("d 4")
	if q == "" {
		t.Fatal("query 'd 4' produced empty FTS query")
	}
	if !strings.Contains(q, "4*") {
		t.Errorf("numeric token missing: got %q", q)
	}
}

func TestFts5Query_NormalQuery(t *testing.T) {
	// Regular query without possessives should still work.
	q := fts5Query("When did Caroline attend yoga")
	if !strings.Contains(q, "Caroline*") {
		t.Errorf("Caroline missing from query: %q", q)
	}
	if !strings.Contains(q, "yoga*") {
		t.Errorf("yoga missing from query: %q", q)
	}
}
