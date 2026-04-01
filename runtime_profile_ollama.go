//go:build !mistral

package main

import "github.com/sharpner/ultramemory/llm"

const (
	buildProviderName     = "ollama"
	defaultOllama         = "http://localhost:11434"
	defaultExtractModel   = "gemma3:4b"
	defaultEmbeddingModel = "mxbai-embed-large"
	defaultLLMParallel    = 1
)

const buildEnvironmentHelp = `  MEMORY_OLLAMA              Ollama base URL              (default: http://localhost:11434)
  MEMORY_MODEL               extraction model             (default: gemma3:4b)
  MEMORY_EMBED_MODEL         embedding model              (default: mxbai-embed-large)
  MISTRAL_API_KEY            Mistral API key              (optional; bench -qa-model/-judge only)`

func newDefaultRuntime(extractModel, embedModel string) (*appRuntime, error) {
	client := llm.New(envOr("MEMORY_OLLAMA", defaultOllama), extractModel, embedModel)
	return &appRuntime{
		extractor: client,
		embedder:  client,
		answerer:  client,
		ocr:       client,
		checker:   client,
		warmer:    client,
	}, nil
}
