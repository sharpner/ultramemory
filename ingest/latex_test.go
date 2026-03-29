package ingest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseMacros(t *testing.T) {
	dir := t.TempDir()
	sty := `\newcommand{\Method}{\texttt{kvtc}}
\newcommand{\llama}{\textbf{Llama 3.1 8B}}
\newcommand{\mistral}{$\mathtt{Ministral}$}
\renewcommand{\foo}[1]{bar}
`
	if err := os.WriteFile(filepath.Join(dir, "custom.sty"), []byte(sty), 0o644); err != nil {
		t.Fatal(err)
	}

	macros := parseMacros(dir)

	cases := []struct {
		name string
		want string
	}{
		{"Method", "kvtc"},
		{"llama", "Llama 3.1 8B"},
	}
	for _, tc := range cases {
		got := macros[tc.name]
		if got != tc.want {
			t.Errorf("macro %q = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestApplyMacros(t *testing.T) {
	macros := map[string]string{
		"Method": "KVTC",
		"llama":  "Llama 3.1 8B",
	}
	text := `We present \Method{}, a lightweight coder. Testing on \llama{} and \Method.`
	got := applyMacros(text, macros)

	for _, want := range []string{"We present KVTC", "Testing on Llama 3.1 8B", "and KVTC"} {
		if !strings.Contains(got, want) {
			t.Errorf("applyMacros: missing %q in %q", want, got)
		}
	}
	if strings.Contains(got, `\Method`) {
		t.Errorf("unreplaced macro in output: %q", got)
	}
}

func TestCleanDetexOutput(t *testing.T) {
	input := "Results in smith2024deep show improvements. We present , a coder.\n\n\n\nNext section."
	got := cleanDetexOutput(input)

	if strings.Contains(got, "smith2024deep") {
		t.Errorf("cite-key leak not removed: %q", got)
	}
	if strings.Contains(got, " , a") {
		t.Errorf("orphan comma not cleaned: %q", got)
	}
	if strings.Contains(got, "\n\n\n") {
		t.Errorf("blank lines not collapsed: %q", got)
	}
}

func TestFallbackStrip(t *testing.T) {
	tex := `\documentclass{article}
\usepackage{amsmath}
\begin{document}

\section{Introduction}
We present KVTC, a lightweight coder achieving 20x compression.
The formula $E = mc^2$ is inline math.

\begin{equation}
\frac{1}{n}\sum_{i=1}^{n} x_i
\end{equation}

Results in \cite{smith2024} confirm this. We use \textbf{bold text} and \textit{italic}.

\begin{table}
\begin{tabular}{lcc}
Model & CR & MMLU \\
Llama 3.1 8B & 20x & 65.2 \\
\end{tabular}
\end{table}

Tested on Llama 3 and Mistral NeMo at NVIDIA.
\end{document}
After document should be stripped.`

	got := fallbackStrip(tex)

	// Prose must survive.
	for _, want := range []string{
		"We present KVTC",
		"20x compression",
		"bold text",
		"italic",
		"Tested on Llama 3",
		"Mistral NeMo",
		"NVIDIA",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("fallbackStrip: missing %q", want)
		}
	}

	// LaTeX noise must be gone.
	for _, noise := range []string{
		`\usepackage`,
		`\documentclass`,
		`\begin{equation}`,
		`\frac{`,
		`\cite{`,
		`\textbf{`,
		"After document",
	} {
		if strings.Contains(got, noise) {
			t.Errorf("fallbackStrip: noise %q survived", noise)
		}
	}
}

func TestSanitizeTeX_RealPaper(t *testing.T) {
	// Use KVTC paper if available.
	paperDir := filepath.Join(os.Getenv("HOME"), ".cache/nanochat/knowledge/2511.01815")
	mainTeX := filepath.Join(paperDir, "main.tex")
	if _, err := os.Stat(mainTeX); err != nil {
		t.Skip("KVTC paper not available at", mainTeX)
	}

	db := openIngestTestDB(t)
	w := New(db, "test")
	ctx := context.Background()

	text, err := w.sanitizeTeX(ctx, mainTeX)
	if err != nil {
		t.Fatalf("sanitizeTeX: %v", err)
	}

	if len(text) < 1000 {
		t.Fatalf("suspiciously short output: %d chars", len(text))
	}

	// Key entities must survive macro resolution + detex.
	for _, want := range []string{
		"Staniszewski",
		"NVIDIA",
		"Llama",
		"compression",
		"PCA",
		"quantization",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("entity %q not found in sanitized output", want)
		}
	}

	// KVTC must be resolved from \Method macro.
	if !strings.Contains(text, "kvtc") && !strings.Contains(text, "KVTC") {
		t.Error("Method macro not resolved — 'kvtc'/'KVTC' not found")
	}

	// LaTeX noise must be gone.
	for _, noise := range []string{
		`\usepackage`,
		`\documentclass`,
		`\begin{equation}`,
	} {
		if strings.Contains(text, noise) {
			t.Errorf("LaTeX noise %q survived sanitization", noise)
		}
	}

	// No braces should remain (LaTeX fully stripped).
	if strings.Count(text, "{") > 5 {
		t.Errorf("too many remaining braces (%d) — LaTeX not fully stripped", strings.Count(text, "{"))
	}

	t.Logf("Sanitized output: %d chars (from %d raw)", len(text),
		func() int { b, _ := os.ReadFile(mainTeX); return len(b) }())
}

func TestSanitizeTeX_AllPapers(t *testing.T) {
	knowledgeDir := filepath.Join(os.Getenv("HOME"), ".cache/nanochat/knowledge")
	if _, err := os.Stat(knowledgeDir); err != nil {
		t.Skip("knowledge dir not available")
	}

	db := openIngestTestDB(t)
	w := New(db, "test")
	ctx := context.Background()

	papers, _ := filepath.Glob(filepath.Join(knowledgeDir, "*/main.tex"))
	if len(papers) == 0 {
		t.Skip("no papers found")
	}

	for _, mainTeX := range papers {
		paper := filepath.Base(filepath.Dir(mainTeX))
		t.Run(paper, func(t *testing.T) {
			text, err := w.sanitizeTeX(ctx, mainTeX)
			if err != nil {
				t.Fatalf("sanitizeTeX: %v", err)
			}
			raw, _ := os.ReadFile(mainTeX)
			// Wrapper main.tex files (only \input{} calls) produce very little output — that's expected.
			if len(text) < 100 && len(raw) > 1000 {
				t.Errorf("suspiciously short: %d chars from %d raw", len(text), len(raw))
			}
			// No LaTeX command leaks.
			if strings.Count(text, `\begin{`) > 0 {
				t.Errorf("LaTeX \\begin leaked (%d)", strings.Count(text, `\begin{`))
			}
			t.Logf("%d chars output", len(text))
		})
	}
}
