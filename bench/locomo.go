// Package bench implements evaluation benchmarks for ultramemory.
package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/store"
)

// ── Category mapping ─────────────────────────────────────────────────────────

var categoryName = map[int]string{
	1: "single-hop",
	2: "multi-hop",
	3: "temporal",
	4: "open-domain",
	5: "adversarial",
}

const qaSystem = `You answer questions about a conversation between two people.
Use ONLY the provided context. Be extremely concise — a few words at most.
For dates, give the format used in the context (e.g. "7 May 2023").
For names, give the full name as it appears in the context.
If the context does not contain the answer, respond with exactly: unknown`

const chunkSize = 1500

// ── Types ────────────────────────────────────────────────────────────────────

type rawConversation struct {
	SampleID string                     `json:"sample_id"`
	Conv     map[string]json.RawMessage `json:"conversation"`
	QA       []rawQA                    `json:"qa"`
}

type rawQA struct {
	Question string          `json:"question"`
	Answer   json.RawMessage `json:"answer"`
	Category int             `json:"category"`
	Evidence []string        `json:"evidence"`
}

type rawTurn struct {
	Speaker     string   `json:"speaker"`
	DiaID       string   `json:"dia_id"`
	Text        string   `json:"text"`
	ImgURL      []string `json:"img_url"`
	BlipCaption string   `json:"blip_caption"`
}

// Conversation is one parsed LoCoMo sample.
type Conversation struct {
	SampleID string
	SpeakerA string
	SpeakerB string
	Sessions []Session
	QA       []QA
}

// Session is one dialogue session with a timestamp.
type Session struct {
	Number   int
	DateTime string
	Turns    []Turn
}

// Turn is one dialogue utterance.
type Turn struct {
	Speaker string
	DiaID   string
	Text    string
}

// QA is one annotated question-answer pair.
type QA struct {
	Question string
	Answer   string
	Category int
	Evidence []string
}

// CategoryScore holds aggregated metrics for one QA category.
type CategoryScore struct {
	Category string
	Count    int
	AvgF1    float64
	AvgEM    float64
}

// Result holds the full benchmark output.
type Result struct {
	Overall    CategoryScore
	ByCategory []CategoryScore
	Duration   time.Duration
}

type qaScore struct {
	category int
	f1       float64
	em       float64
}

// ── Public API ───────────────────────────────────────────────────────────────

// RunLoCoMo evaluates ultramemory against the LoCoMo benchmark.
// Set limit > 0 to evaluate only the first N conversations.
func RunLoCoMo(ctx context.Context, dataPath string, db *store.DB, client *llm.Client, resolveThreshold float64, limit int) (*Result, error) {
	conversations, err := parseLoCoMo(dataPath)
	if err != nil {
		return nil, err
	}
	if limit > 0 && limit < len(conversations) {
		conversations = conversations[:limit]
	}

	start := time.Now()
	var scores []qaScore

	for ci, conv := range conversations {
		groupID := "locomo-" + conv.SampleID
		ext := graph.New(db, client, resolveThreshold)

		slog.Info("ingesting conversation",
			"conv", ci+1, "of", len(conversations),
			"sample", conv.SampleID,
			"sessions", len(conv.Sessions),
			"qa", len(conv.QA),
		)

		// Ingest all sessions.
		for _, sess := range conv.Sessions {
			text := formatSession(sess)
			chunks := chunkText(text, chunkSize)
			for _, chunk := range chunks {
				if len(strings.TrimSpace(chunk)) < 50 {
					continue
				}
				source := fmt.Sprintf("locomo/%s/session_%d", conv.SampleID, sess.Number)
				if err := ext.Process(ctx, chunk, source, groupID); err != nil {
					slog.Warn("extraction failed", "conv", conv.SampleID, "session", sess.Number, "err", err)
				}
			}
		}

		slog.Info("evaluating QA", "conv", conv.SampleID, "questions", len(conv.QA))

		// Evaluate each QA question.
		for qi, qa := range conv.QA {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}

			results, err := graph.Search(ctx, db, client, qa.Question, groupID, 10)
			if err != nil {
				slog.Warn("search failed", "question", qa.Question, "err", err)
				scores = append(scores, qaScore{qa.Category, 0, 0})
				continue
			}

			context := formatContext(results)
			prompt := fmt.Sprintf("Context:\n%s\n\nQuestion: %s", context, qa.Question)
			answer, err := client.Answer(ctx, qaSystem, prompt)
			if err != nil {
				slog.Warn("answer failed", "question", qa.Question, "err", err)
				scores = append(scores, qaScore{qa.Category, 0, 0})
				continue
			}

			f1 := tokenF1(answer, qa.Answer)
			em := exactMatch(answer, qa.Answer)
			scores = append(scores, qaScore{qa.Category, f1, em})

			if (qi+1)%25 == 0 {
				slog.Info("qa progress",
					"conv", conv.SampleID,
					"done", qi+1,
					"of", len(conv.QA),
				)
			}
		}
	}

	return aggregate(scores, time.Since(start)), nil
}

// PrintResult outputs the benchmark results as a table.
func PrintResult(r *Result) {
	fmt.Printf("\n── LoCoMo Benchmark Results ──────────────────────────────\n\n")
	fmt.Printf("%-14s  %5s  %7s  %7s\n", "Category", "Count", "F1", "EM")
	fmt.Printf("%-14s  %5s  %7s  %7s\n", "──────────────", "─────", "───────", "───────")
	for _, c := range r.ByCategory {
		fmt.Printf("%-14s  %5d  %6.1f%%  %6.1f%%\n", c.Category, c.Count, c.AvgF1*100, c.AvgEM*100)
	}
	fmt.Printf("%-14s  %5s  %7s  %7s\n", "──────────────", "─────", "───────", "───────")
	fmt.Printf("%-14s  %5d  %6.1f%%  %6.1f%%\n", "OVERALL", r.Overall.Count, r.Overall.AvgF1*100, r.Overall.AvgEM*100)
	fmt.Printf("\nDuration: %s\n", r.Duration.Round(time.Second))
}

// ── Parsing ──────────────────────────────────────────────────────────────────

func parseLoCoMo(path string) ([]Conversation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read locomo: %w", err)
	}

	var raw []rawConversation
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse locomo: %w", err)
	}

	conversations := make([]Conversation, 0, len(raw))
	for _, rc := range raw {
		conv := Conversation{
			SampleID: rc.SampleID,
			QA:       make([]QA, len(rc.QA)),
		}

		// Parse speaker names.
		if v, ok := rc.Conv["speaker_a"]; ok {
			_ = json.Unmarshal(v, &conv.SpeakerA)
		}
		if v, ok := rc.Conv["speaker_b"]; ok {
			_ = json.Unmarshal(v, &conv.SpeakerB)
		}

		// Parse sessions by iterating numbered keys.
		for i := 1; i <= 100; i++ {
			key := "session_" + strconv.Itoa(i)
			turnsRaw, ok := rc.Conv[key]
			if !ok {
				break
			}

			sess := Session{Number: i}
			if dtRaw, ok := rc.Conv[key+"_date_time"]; ok {
				_ = json.Unmarshal(dtRaw, &sess.DateTime)
			}

			var turns []rawTurn
			if err := json.Unmarshal(turnsRaw, &turns); err != nil {
				continue // skip malformed sessions
			}

			for _, t := range turns {
				text := t.Text
				if text == "" && t.BlipCaption != "" {
					text = "[Image: " + t.BlipCaption + "]"
				}
				if text == "" {
					continue
				}
				sess.Turns = append(sess.Turns, Turn{
					Speaker: t.Speaker,
					DiaID:   t.DiaID,
					Text:    text,
				})
			}

			if len(sess.Turns) > 0 {
				conv.Sessions = append(conv.Sessions, sess)
			}
		}

		for i, q := range rc.QA {
			conv.QA[i] = QA{
				Question: q.Question,
				Answer:   rawAnswerToString(q.Answer),
				Category: q.Category,
				Evidence: q.Evidence,
			}
		}

		conversations = append(conversations, conv)
	}

	return conversations, nil
}

func rawAnswerToString(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	// Numeric answers (e.g. 2022, 3).
	return strings.Trim(string(raw), " \t\n")
}

// ── Evaluation helpers ───────────────────────────────────────────────────────

func formatSession(s Session) string {
	var b strings.Builder
	if s.DateTime != "" {
		b.WriteString("[" + s.DateTime + "]\n")
	}
	for _, t := range s.Turns {
		b.WriteString(t.Speaker + ": " + t.Text + "\n")
	}
	return b.String()
}

func formatContext(results []graph.SearchResult) string {
	if len(results) == 0 {
		return "(no relevant facts found)"
	}
	var b strings.Builder
	for i, r := range results {
		switch r.Type {
		case "entity":
			fmt.Fprintf(&b, "%d. %s (%s)\n", i+1, r.Title, r.Body)
		case "edge":
			fmt.Fprintf(&b, "%d. %s\n", i+1, r.Body)
		case "episode":
			// Truncate long episode content to keep prompt manageable.
			body := r.Body
			if len(body) > 500 {
				body = body[:500] + "..."
			}
			fmt.Fprintf(&b, "%d. [dialogue] %s\n", i+1, body)
		}
	}
	return b.String()
}

func chunkText(text string, size int) []string {
	runes := []rune(text)
	if len(runes) <= size {
		return []string{string(runes)}
	}
	var chunks []string
	for i := 0; i < len(runes); i += size {
		end := i + size
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return chunks
}

// ── Scoring (SQuAD-style F1 and Exact Match) ────────────────────────────────

func normalizeAnswer(s string) string {
	s = strings.ToLower(s)
	// Remove punctuation.
	s = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, s)
	// Remove articles.
	for _, article := range []string{" a ", " an ", " the "} {
		s = strings.ReplaceAll(s, article, " ")
	}
	return strings.Join(strings.Fields(s), " ")
}

func tokenF1(prediction, gold string) float64 {
	predTokens := strings.Fields(normalizeAnswer(prediction))
	goldTokens := strings.Fields(normalizeAnswer(gold))
	if len(predTokens) == 0 || len(goldTokens) == 0 {
		return 0
	}

	goldCounts := map[string]int{}
	for _, t := range goldTokens {
		goldCounts[t]++
	}

	common := 0
	for _, t := range predTokens {
		if goldCounts[t] > 0 {
			common++
			goldCounts[t]--
		}
	}

	if common == 0 {
		return 0
	}
	precision := float64(common) / float64(len(predTokens))
	recall := float64(common) / float64(len(goldTokens))
	return 2 * precision * recall / (precision + recall)
}

func exactMatch(prediction, gold string) float64 {
	if normalizeAnswer(prediction) == normalizeAnswer(gold) {
		return 1
	}
	return 0
}

// ── Aggregation ──────────────────────────────────────────────────────────────

func aggregate(scores []qaScore, duration time.Duration) *Result {
	type acc struct {
		f1Sum float64
		emSum float64
		count int
	}
	byCategory := map[int]*acc{}
	overall := &acc{}

	for _, s := range scores {
		if _, ok := byCategory[s.category]; !ok {
			byCategory[s.category] = &acc{}
		}
		byCategory[s.category].f1Sum += s.f1
		byCategory[s.category].emSum += s.em
		byCategory[s.category].count++
		overall.f1Sum += s.f1
		overall.emSum += s.em
		overall.count++
	}

	result := &Result{Duration: duration}
	if overall.count > 0 {
		result.Overall = CategoryScore{
			Category: "overall",
			Count:    overall.count,
			AvgF1:    overall.f1Sum / float64(overall.count),
			AvgEM:    overall.emSum / float64(overall.count),
		}
	}

	cats := make([]int, 0, len(byCategory))
	for c := range byCategory {
		cats = append(cats, c)
	}
	sort.Ints(cats)

	for _, c := range cats {
		a := byCategory[c]
		name := categoryName[c]
		if name == "" {
			name = "cat-" + strconv.Itoa(c)
		}
		result.ByCategory = append(result.ByCategory, CategoryScore{
			Category: name,
			Count:    a.count,
			AvgF1:    a.f1Sum / float64(a.count),
			AvgEM:    a.emSum / float64(a.count),
		})
	}

	return result
}
