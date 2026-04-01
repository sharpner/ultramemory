package main

import (
	"context"

	"github.com/sharpner/ultramemory/llm"
)

type appRuntime struct {
	extractor llm.EntityExtractor
	embedder  llm.Embedder
	answerer  llm.Answerer
	ocr       llm.OCR
	checker   llm.HealthChecker
	warmer    llm.Warmer
}

func (r *appRuntime) ping(ctx context.Context) error {
	if r == nil {
		return nil
	}
	if r.checker == nil {
		return nil
	}
	return r.checker.Ping(ctx)
}

func (r *appRuntime) warmup(ctx context.Context) error {
	if r == nil {
		return nil
	}
	if r.warmer == nil {
		return nil
	}
	return r.warmer.Warmup(ctx)
}
