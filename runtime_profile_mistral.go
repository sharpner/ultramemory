//go:build mistral

package main

import (
	"fmt"
	"os"

	"github.com/sharpner/ultramemory/llm"
)

const (
	buildProviderName     = "mistral"
	defaultExtractModel   = "mistral-3b-latest"
	defaultEmbeddingModel = "mistral-embed"
	defaultOCRModel       = "mistral-ocr-latest"
	defaultLLMParallel    = 4
)

const buildEnvironmentHelp = `  MEMORY_MODEL               extraction model             (default: mistral-3b-latest)
  MEMORY_EMBED_MODEL         embedding model              (default: mistral-embed)
  MISTRAL_API_KEY            Mistral API key              (required)`

func newDefaultRuntime(extractModel, embedModel string) (*appRuntime, error) {
	apiKey := os.Getenv("MISTRAL_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("MISTRAL_API_KEY not set — required for mistral build")
	}

	extractor := llm.NewMistral(apiKey, extractModel)
	embedder := llm.NewMistral(apiKey, embedModel)
	ocr := llm.NewMistral(apiKey, defaultOCRModel)

	return &appRuntime{
		extractor: extractor,
		embedder:  embedder,
		answerer:  extractor,
		ocr:       ocr,
		checker:   extractor,
		warmer:    extractor,
	}, nil
}
