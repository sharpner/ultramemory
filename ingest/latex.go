package ingest

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// LaTeX macro patterns compiled once.
var (
	// Matches \newcommand{\Name}{...body...} — name is captured, body extracted by balancedBrace.
	reMacroDef = regexp.MustCompile(`\\(?:re)?newcommand\{\\(\w+)\}(?:\[\d+\])?`)

	// \texttt{X}, \textbf{X}, \textit{X}, \emph{X}, \textsc{X}, \text{X} → X
	reFormatCmd = regexp.MustCompile(`\\(?:texttt|textbf|textit|textsc|emph|text)\{([^}]*)\}`)

	// Cite-key leaks: barewords like "smith2024deep" or "yuan2024kvcache..."
	reCiteKey = regexp.MustCompile(`\b[a-z]{2,20}\d{4}[a-z]{2,40}\b`)

	// Multiple blank lines → single blank line
	reBlankLines = regexp.MustCompile(`\n{3,}`)

	// Orphan punctuation from removed macros: "We present , a" → "We present a"
	reOrphanComma = regexp.MustCompile(` , `)
	reOrphanColon = regexp.MustCompile(` : `)
)

// sanitizeTeX converts a .tex file to clean prose via:
// 1. Resolve custom macros from companion .sty files
// 2. Run detex to strip LaTeX commands
// 3. Clean up residual artifacts
func (w *Walker) sanitizeTeX(ctx context.Context, path string) (string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read tex: %w", err)
	}
	text := string(raw)

	// Step 1: resolve custom macros from .sty files in the same directory.
	dir := filepath.Dir(path)
	macros := parseMacros(dir)
	text = applyMacros(text, macros)

	// Step 2: run detex if available.
	if w.detexBin != "" {
		cleaned, err := w.runDetex(ctx, text)
		if err != nil {
			return "", fmt.Errorf("detex: %w", err)
		}
		text = cleaned
	} else {
		// Fallback: lightweight Go-only stripping (no detex installed).
		text = fallbackStrip(text)
	}

	// Step 3: clean up residual noise.
	text = cleanDetexOutput(text)
	return text, nil
}

// parseMacros scans all .sty and .tex files in dir for \newcommand definitions.
// Returns a map of macro name → expansion (with inner formatting unwrapped).
func parseMacros(dir string) map[string]string {
	macros := map[string]string{}
	files, _ := filepath.Glob(filepath.Join(dir, "*.sty"))
	texFiles, _ := filepath.Glob(filepath.Join(dir, "*.tex"))
	files = append(files, texFiles...)

	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		content := string(data)
		for _, loc := range reMacroDef.FindAllStringSubmatchIndex(content, -1) {
			name := content[loc[2]:loc[3]]
			// Body starts at the '{' after the match.
			bodyStart := loc[1]
			body := extractBraced(content, bodyStart)
			if body == "" {
				continue
			}
			// Unwrap formatting: \texttt{kvtc} → kvtc
			body = reFormatCmd.ReplaceAllString(body, "$1")
			macros[name] = body
		}
	}
	return macros
}

// extractBraced extracts the content inside the next {...} starting at pos.
// Handles nested braces. Returns empty string if no braced group found.
func extractBraced(s string, pos int) string {
	// Find opening brace.
	for pos < len(s) && s[pos] != '{' {
		pos++
	}
	if pos >= len(s) {
		return ""
	}
	depth := 0
	start := pos + 1
	for i := pos; i < len(s); i++ {
		switch s[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[start:i]
			}
		}
	}
	return ""
}

// applyMacros replaces \MacroName{} and \MacroName with their expansion.
func applyMacros(text string, macros map[string]string) string {
	for name, body := range macros {
		// \Method{} → body
		text = strings.ReplaceAll(text, `\`+name+`{}`, body)
		// \Method → body (only at word boundary, not \MethodFull)
		re := regexp.MustCompile(`\\` + regexp.QuoteMeta(name) + `\b`)
		text = re.ReplaceAllString(text, body)
	}
	return text
}

// runDetex writes text to a temp file and runs detex on it.
func (w *Walker) runDetex(ctx context.Context, text string) (string, error) {
	tmp, err := os.CreateTemp("", "ultramemory-detex-*.tex")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.WriteString(text); err != nil {
		tmp.Close()
		return "", err
	}
	tmp.Close()

	out, err := exec.CommandContext(ctx, w.detexBin, "-l", tmp.Name()).Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// fallbackStrip does lightweight LaTeX stripping when detex is not installed.
// Less thorough than detex but catches the worst noise sources.
func fallbackStrip(text string) string {
	// Strip preamble.
	if idx := strings.Index(text, `\begin{document}`); idx >= 0 {
		text = text[idx+len(`\begin{document}`):]
	}
	if idx := strings.Index(text, `\end{document}`); idx >= 0 {
		text = text[:idx]
	}

	// Strip display math environments.
	for _, env := range []string{"equation", "equation*", "align", "align*", "gather", "multline"} {
		re := regexp.MustCompile(`(?s)\\begin\{` + regexp.QuoteMeta(env) + `\}.*?\\end\{` + regexp.QuoteMeta(env) + `\}`)
		text = re.ReplaceAllString(text, "")
	}
	// \[...\] and $$...$$
	text = regexp.MustCompile(`(?s)\\\[.*?\\]`).ReplaceAllString(text, "")
	text = regexp.MustCompile(`(?s)\$\$.*?\$\$`).ReplaceAllString(text, "")

	// Strip inline math $...$
	text = regexp.MustCompile(`\$[^$]+?\$`).ReplaceAllString(text, "")

	// Strip citations/refs.
	text = regexp.MustCompile(`\\(?:citep?|citet|ref|cref|label|eqref)\{[^}]*\}`).ReplaceAllString(text, "")

	// Unwrap formatting commands.
	text = reFormatCmd.ReplaceAllString(text, "$1")

	// Strip remaining commands with no arguments.
	text = regexp.MustCompile(`\\[a-zA-Z]+`).ReplaceAllString(text, "")

	// Strip table/figure environments.
	for _, env := range []string{"table", "table*", "figure", "figure*", "tabular"} {
		re := regexp.MustCompile(`(?s)\\begin\{` + regexp.QuoteMeta(env) + `\}.*?\\end\{` + regexp.QuoteMeta(env) + `\}`)
		text = re.ReplaceAllString(text, "")
	}

	// Remove remaining braces.
	text = strings.ReplaceAll(text, "{", "")
	text = strings.ReplaceAll(text, "}", "")

	return text
}

// cleanDetexOutput removes residual noise from detex output.
func cleanDetexOutput(text string) string {
	// Remove cite-key leaks (e.g. "yuan2024kvcachecompressionreturn").
	text = reCiteKey.ReplaceAllString(text, "")

	// Fix orphan punctuation from removed macros.
	text = reOrphanComma.ReplaceAllString(text, " ")
	text = reOrphanColon.ReplaceAllString(text, " ")

	// Collapse blank lines.
	text = reBlankLines.ReplaceAllString(text, "\n\n")

	return strings.TrimSpace(text)
}
