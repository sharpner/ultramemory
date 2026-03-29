package ingest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"encoding/json"

	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/store"
)

func openIngestTestDB(t *testing.T) *store.DB {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "ingest-test-*.db")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	_ = f.Close()
	db, err := store.Open(f.Name())
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestChunk_SingleChunk(t *testing.T) {
	text := "short"
	got := chunk(text, 1500, 150)
	if len(got) != 1 || got[0] != "short" {
		t.Errorf("short text should produce 1 chunk, got %v", got)
	}
}

func TestChunk_Overlap(t *testing.T) {
	// 10 runes, size=6, overlap=2, step=4 → windows: [0:6], [4:10]
	text := "abcdefghij"
	got := chunk(text, 6, 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %v", len(got), got)
	}
	if got[0] != "abcdef" {
		t.Errorf("first chunk: want %q, got %q", "abcdef", got[0])
	}
	if got[1] != "efghij" {
		t.Errorf("second chunk: want %q, got %q", "efghij", got[1])
	}
}

func TestChunk_Unicode(t *testing.T) {
	// Emojis are multi-byte but single rune — chunking must work on runes, not bytes.
	text := "🐉🐉🐉🐉🐉🐉"
	got := chunk(text, 4, 1)
	for _, c := range got {
		if len([]rune(c)) > 4 {
			t.Errorf("chunk exceeds rune size 4: %q (%d runes)", c, len([]rune(c)))
		}
	}
}

func TestWalk_TextFile(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 5) // 225 chars
	if err := os.WriteFile(filepath.Join(dir, "doc.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Errorf("expected >= 1 chunk queued, got %d", n)
	}
}

func TestWalk_TexFile(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	tex := `\documentclass{article}
\author{John Smith}
\title{On Upper Bounds of Innerproduct Estimation}
\begin{document}
\maketitle
We present a novel approach to bounding the innerproduct estimator proposed by Thompson and Liu at MIT.
\end{document}`
	if err := os.WriteFile(filepath.Join(dir, "paper.tex"), []byte(tex), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Errorf("expected >= 1 chunk queued for .tex file, got %d", n)
	}
}

func TestWalk_BibFile(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	bib := `@article{smith2024bounds,
  author  = {John Smith and Jane Doe},
  title   = {Upper and Lower Bounds for Innerproduct Estimation},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  pages   = {1--42}
}`
	if err := os.WriteFile(filepath.Join(dir, "refs.bib"), []byte(bib), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Errorf("expected >= 1 chunk queued for .bib file, got %d", n)
	}
}

func TestWalk_TexMathContent(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	// Realistic LaTeX with heavy math, inline formulas, and environments.
	tex := `\documentclass{article}
\usepackage{amsmath}
\author{Thompson and Liu}
\title{Tight Bounds on the Unbiased Innerproduct Estimator}
\begin{document}
\maketitle

\section{Introduction}
We derive tight upper and lower bounds for the innerproduct estimator $\hat{\theta} = \frac{1}{n}\sum_{i=1}^{n} x_i y_i$
originally proposed by Richardson et al.\ at Stanford University.

\section{Main Result}
\begin{theorem}[Main Bound]
For all $\epsilon > 0$ and $n \geq 1$, the mean squared error satisfies
\begin{equation}
\text{MSE}(\hat{\theta}) \leq \frac{\|x\|_2^2 \|y\|_2^2}{n} + \mathcal{O}\!\left(\frac{1}{n^2}\right).
\label{eq:main}
\end{equation}
\end{theorem}

\begin{proof}
By the Cauchy--Schwarz inequality and Jensen's inequality applied to $f(z) = z^2$, we have
\[
\mathbb{E}\!\left[(\hat{\theta} - \theta)^2\right] = \text{Var}(\hat{\theta}) = \frac{1}{n}\text{Var}(x_1 y_1).
\]
The claim follows from bounding $\text{Var}(x_1 y_1) \leq \|x\|_2^2\|y\|_2^2$ via H\"older's inequality.
\end{proof}

\section{Experiments}
We evaluated our bounds on datasets from the UCI Machine Learning Repository and compared against
the bootstrap estimator of Efron and Tibshirani. Experiments were conducted on an NVIDIA A100 GPU
cluster at the Massachusetts Institute of Technology.

\bibliographystyle{plain}
\bibliography{refs}
\end{document}`
	if err := os.WriteFile(filepath.Join(dir, "bounds.tex"), []byte(tex), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Errorf("expected >= 1 chunk for math-heavy .tex, got %d", n)
	}

	// Drain queue and verify chunked content preserves formulas.
	var combined string
	for {
		j, err := db.NextJob(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if j == nil {
			break
		}
		var p graph.IngestPayload
		if err := json.Unmarshal([]byte(j.Payload), &p); err != nil {
			t.Fatal(err)
		}
		combined += p.Content
	}
	// Key entities and math fragments must survive chunking.
	for _, want := range []string{
		"Thompson", "Liu", "Richardson", "Stanford",
		"Cauchy--Schwarz", "Jensen", "Efron", "Tibshirani",
		`\hat{\theta}`, `\frac{1}{n}`, `\text{MSE}`,
		"Massachusetts Institute of Technology",
	} {
		if !strings.Contains(combined, want) {
			t.Errorf("expected %q in chunked content, not found", want)
		}
	}
}

func TestWalk_ArxivTexPaper(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()

	// Realistic excerpt from arXiv:2511.01815 (KVTC paper) — includes preamble,
	// author metadata, abstract, math, citations, and table input refs.
	tex := `\documentclass{article}
\usepackage{iclr2026_conference,times}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{amsmath}

\title{KV Cache Transform Coding\\for Compact Storage in LLM Inference}

\author{Konrad Staniszewski$^{1,2}$ \& Adrian {\L}a\'ncucki$^{1}$\\
NVIDIA$^{1}$, University of Warsaw$^{2}$\\
\texttt{kstaniszewsk@nvidia.com}
}

\begin{document}
\maketitle

\begin{abstract}
Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management.
KV caches can be reused across conversation turns via shared-prefix prompts that are common in
iterative code editing and chat. We present KVTC, a lightweight transform coder that compresses
KV caches for compact on-GPU and off-GPU storage. Drawing on classical media compression, KVTC
combines PCA-based feature decorrelation, adaptive quantization, and entropy coding. It requires only
a brief initial calibration and leaves model parameters unchanged. By exploiting redundancies in KV
caches, KVTC achieves up to 20$\times$ compression while maintaining reasoning and long-context accuracy,
and 40$\times$ or higher for specific use cases. We test KVTC with Llama 3, Mistral NeMo, and R1-Qwen 2.5
models across benchmarks including AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper
and RULER. It consistently outperforms inference-time baselines such as token eviction, quantization,
and SVD-based methods, while achieving higher compression ratios.
\end{abstract}

\section{Introduction}

Chat-based interfaces, commonly used for interacting with large language models (LLMs), enable users
to iteratively refine answers across open-domain dialogues and specialized tasks, such as code
generation \citep{chiang2024chatbotarenaopenplatform,kopf2023openassistantconversationsdemocratizing}.
Each conversational turn extends the key--value (KV) cache associated with a conversation, storing
hidden activations for every previous token. For modern Transformer models, this cache can easily
occupy multiple gigabytes.

CacheGen \citep{liu2024cachegenkvcachecompression} compresses caches for transmission, offering at
most 8.6$\times$ KV cache reduction in comparison to a 16-bit baseline. SVDq
\citep{yankun2025svdq125bit410xkey} and xKV \citep{chang2025xkvcrosslayersvdkvcache} pursue low-rank
compression during prefill, but both require calculation of per-prompt SVD.

\section{Preliminaries}\label{sec:preliminaries}

\paragraph{KV Cache Structure}
During decoding in autoregressive Transformers with multi-head self-attention, the keys and values
produced for each processed token are cached to avoid recomputation. For $l$ layers, $h$ heads, head
dimension $d_{\mathrm{head}}$ and sequence length $t$, a 16-bit KV cache occupies
\begin{equation}
M_{\mathrm{KV}} = 2 \cdot l \cdot h \cdot d_{\mathrm{head}} \cdot t \cdot 2 \;\text{bytes}.
\label{eq:kv-size}
\end{equation}

\paragraph{PCA-based Decorrelation}
Let $X \in \mathbb{R}^{n \times d}$ be the matrix of $n$ cached key (or value) vectors. We compute
the SVD of the centered data matrix: $\bar{X} = U \Sigma V^\top$. The orthonormal basis $V$ serves as
the PCA transform, and the projected coefficients $C = \bar{X} V$ are decorrelated.

\section{Results}

\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Model} & \textbf{CR} & \textbf{MMLU} \\
\hline
Llama 3.1 8B   & 20$\times$ & 65.2 \\
Mistral NeMo   & 18$\times$ & 63.8 \\
R1-Qwen 2.5    & 22$\times$ & 67.1 \\
\hline
\end{tabular}
\caption{Compression ratios and MMLU accuracy for KVTC across models.}
\end{table}

\bibliographystyle{plainnat}
\bibliography{refs}
\end{document}`

	if err := os.WriteFile(filepath.Join(dir, "main.tex"), []byte(tex), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Fatalf("expected >= 1 chunk for arXiv .tex paper, got %d", n)
	}

	// Drain queue and verify content integrity.
	var combined string
	for {
		j, err := db.NextJob(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if j == nil {
			break
		}
		var p graph.IngestPayload
		if err := json.Unmarshal([]byte(j.Payload), &p); err != nil {
			t.Fatal(err)
		}
		combined += p.Content + "\n"
	}

	// Authors and affiliations must survive.
	for _, want := range []string{
		"Konrad Staniszewski", "Adrian",
		"NVIDIA", "University of Warsaw",
	} {
		if !strings.Contains(combined, want) {
			t.Errorf("author/affiliation %q not found in chunks", want)
		}
	}

	// Key technical entities and model names.
	for _, want := range []string{
		"KVTC", "Llama 3", "Mistral NeMo", "R1-Qwen 2.5",
		"AIME25", "GSM8K", "MMLU", "Qasper", "RULER",
		"CacheGen", "SVDq", "xKV",
	} {
		if !strings.Contains(combined, want) {
			t.Errorf("entity %q not found in chunks", want)
		}
	}

	// Math formulas must be preserved verbatim.
	for _, want := range []string{
		`d_{\mathrm{head}}`,
		`M_{\mathrm{KV}}`,
		`U \Sigma V^\top`,
		`\bar{X} V`,
		`20$\times$`,
	} {
		if !strings.Contains(combined, want) {
			t.Errorf("formula %q not found in chunks", want)
		}
	}

	// Citations must survive chunking.
	for _, want := range []string{
		`\citep{liu2024cachegenkvcachecompression}`,
		`\citep{yankun2025svdq125bit410xkey}`,
	} {
		if !strings.Contains(combined, want) {
			t.Errorf("citation %q not found in chunks", want)
		}
	}
}

func TestWalk_WithSourceOverride(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 5)
	if err := os.WriteFile(filepath.Join(dir, "paper.tex"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	arxivURL := "https://arxiv.org/abs/2511.01815"
	n, err := New(db, "g").WithSource(arxivURL).Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 1 {
		t.Fatal("expected >= 1 chunk")
	}

	// Verify the source in the payload is the arXiv URL, not the file path.
	job, err := db.NextJob(context.Background())
	if err != nil || job == nil {
		t.Fatal("expected a job")
	}
	var p graph.IngestPayload
	if err := json.Unmarshal([]byte(job.Payload), &p); err != nil {
		t.Fatal(err)
	}
	if p.Source != arxivURL {
		t.Errorf("source = %q, want %q", p.Source, arxivURL)
	}
}

func TestWalk_SkipsHiddenDirs(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	hidden := filepath.Join(dir, ".hidden")
	if err := os.MkdirAll(hidden, 0o755); err != nil {
		t.Fatal(err)
	}
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 5)
	if err := os.WriteFile(filepath.Join(hidden, "secret.txt"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("hidden dir must be skipped, got %d chunks", n)
	}
}

func TestWalk_ShortChunksIgnored(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tiny.txt"), []byte("too short"), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("content < 50 chars must produce 0 chunks, got %d", n)
	}
}

func TestWalk_BinaryFileSkipped(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	// Invalid UTF-8 sequence in a .txt file.
	if err := os.WriteFile(filepath.Join(dir, "binary.txt"), []byte{0xff, 0xfe, 0x00, 0x01}, 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("invalid UTF-8 file must be skipped, got %d chunks", n)
	}
}

func TestWalk_UnknownExtensionSkipped(t *testing.T) {
	db := openIngestTestDB(t)
	dir := t.TempDir()
	content := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 5)
	if err := os.WriteFile(filepath.Join(dir, "data.xyz"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	n, err := New(db, "g").Walk(context.Background(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("unknown extension .xyz must be skipped, got %d chunks", n)
	}
}
