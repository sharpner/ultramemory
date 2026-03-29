package ingest

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// LaTeX patterns compiled once at package init.
var (
	// \newcommand{\Name}{...body...} — name captured, body extracted by extractBraced.
	reMacroDef = regexp.MustCompile(`\\(?:re)?newcommand\{\\(\w+)\}(?:\[\d+\])?`)

	// \texttt{X}, \textbf{X}, \textit{X}, \emph{X}, \textsc{X}, \text{X} → X
	reFormatCmd = regexp.MustCompile(`\\(?:texttt|textbf|textit|textsc|emph|text)\{([^}]*)\}`)

	// Cite-key leaks: barewords like "smith2024deeplearning" (min 3+4+4 chars to avoid false positives).
	reCiteKey = regexp.MustCompile(`\b[a-z]{3,20}\d{4}[a-z]{4,40}\b`)

	reBlankLines  = regexp.MustCompile(`\n{3,}`)
	reOrphanComma = regexp.MustCompile(` , `)
	reOrphanColon = regexp.MustCompile(` : `)

	// fallbackStrip static patterns — compiled once.
	reDisplayMathBracket = regexp.MustCompile(`(?s)\\\[.*?\\]`)
	reDisplayMathDollar  = regexp.MustCompile(`(?s)\$\$.*?\$\$`)
	reInlineMath         = regexp.MustCompile(`\$[^$]+?\$`)
	reCiteRef            = regexp.MustCompile(`\\(?:citep?|citet|ref|cref|label|eqref)\{[^}]*\}`)
	reGenericCommand     = regexp.MustCompile(`\\[a-zA-Z]+`)
)

// sanitizeTeX converts a .tex file to clean prose via:
// 1. Resolve custom macros from companion .sty files
// 2. Run detex (with Go fallback) to strip LaTeX commands
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

	// Step 2: strip LaTeX. Try detex first, fall back to Go stripper on failure.
	text = w.stripLaTeX(ctx, text)

	// Step 3: clean up residual noise.
	text = cleanDetexOutput(text)
	return text, nil
}

// stripLaTeX runs detex if available, falling back to Go stripper if detex
// is missing or fails at runtime.
func (w *Walker) stripLaTeX(ctx context.Context, text string) string {
	if w.detexBin == "" {
		return fallbackStrip(text)
	}
	cleaned, err := w.runDetex(ctx, text)
	if err != nil {
		slog.Warn("detex failed, using fallback stripper", "err", err)
		return fallbackStrip(text)
	}
	return cleaned
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
			slog.Warn("skip macro file", "path", f, "err", err)
			continue
		}
		content := string(data)
		for _, loc := range reMacroDef.FindAllStringSubmatchIndex(content, -1) {
			name := content[loc[2]:loc[3]]
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
// Sorted iteration ensures deterministic output.
func applyMacros(text string, macros map[string]string) string {
	names := make([]string, 0, len(macros))
	for name := range macros {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		body := macros[name]
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
		// Include detex's stderr in the error for diagnostics.
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && len(exitErr.Stderr) > 0 {
			return "", fmt.Errorf("exit %d: %s", exitErr.ExitCode(), strings.TrimSpace(string(exitErr.Stderr)))
		}
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

	// Strip table/figure environments FIRST (before generic command strip destroys \begin/\end).
	for _, env := range []string{"table", "table*", "figure", "figure*", "tabular"} {
		re := regexp.MustCompile(`(?s)\\begin\{` + regexp.QuoteMeta(env) + `\}.*?\\end\{` + regexp.QuoteMeta(env) + `\}`)
		text = re.ReplaceAllString(text, "")
	}

	// Strip display math environments.
	for _, env := range []string{"equation", "equation*", "align", "align*", "gather", "multline"} {
		re := regexp.MustCompile(`(?s)\\begin\{` + regexp.QuoteMeta(env) + `\}.*?\\end\{` + regexp.QuoteMeta(env) + `\}`)
		text = re.ReplaceAllString(text, "")
	}
	text = reDisplayMathBracket.ReplaceAllString(text, "")
	text = reDisplayMathDollar.ReplaceAllString(text, "")

	// Strip inline math $...$
	text = reInlineMath.ReplaceAllString(text, "")

	// Strip citations/refs.
	text = reCiteRef.ReplaceAllString(text, "")

	// Unwrap formatting commands.
	text = reFormatCmd.ReplaceAllString(text, "$1")

	// Strip remaining commands with no arguments.
	text = reGenericCommand.ReplaceAllString(text, "")

	// Remove remaining braces.
	text = strings.ReplaceAll(text, "{", "")
	text = strings.ReplaceAll(text, "}", "")

	return text
}

// cleanDetexOutput removes residual noise from detex output.
func cleanDetexOutput(text string) string {
	text = reCiteKey.ReplaceAllString(text, "")
	text = reOrphanComma.ReplaceAllString(text, " ")
	text = reOrphanColon.ReplaceAllString(text, " ")
	text = reBlankLines.ReplaceAllString(text, "\n\n")
	return strings.TrimSpace(text)
}
