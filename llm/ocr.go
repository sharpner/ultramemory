//go:build !mistral

package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"strings"
)

const (
	// doublPageRatio is the min width/height ratio to treat a scan as a double-page spread.
	doublePageRatio = 1.3
)

var (
	promptSingle = "Transcribe all text from this document scan exactly as it appears. Output only the raw text, preserving headings and paragraphs. No explanations."

	promptDouble = "This is a double-page book scan showing two pages side by side. " +
		"Transcribe the LEFT page top-to-bottom first, then the RIGHT page top-to-bottom. " +
		"Output only the raw text, preserving headings and paragraphs. No explanations."

	promptRetry = "Look at this image carefully. Write out word for word every piece of text you can see. Start immediately with the first word."

	refusalPrefixes = []string{
		"i cannot", "i'm sorry", "i am sorry", "i'm unable", "i am unable",
		"cannot fulfill", "cannot process", "unable to process",
	}
)

// OCR extracts text from a scanned image. Detects double-page spreads and retries once on refusal.
func (c *Client) OCR(ctx context.Context, imageBytes []byte) (string, error) {
	isDouble := isDoublePageScan(imageBytes)
	prompt := promptSingle
	if isDouble {
		prompt = promptDouble
	}

	text, err := c.ocrRequest(ctx, imageBytes, prompt)
	if err != nil {
		return "", err
	}
	if isRefusal(text) {
		text, err = c.ocrRequest(ctx, imageBytes, promptRetry)
		if err != nil {
			return "", err
		}
		if isRefusal(text) {
			return "", fmt.Errorf("model refused to transcribe image after retry")
		}
	}
	return strings.TrimSpace(text), nil
}

func (c *Client) ocrRequest(ctx context.Context, imageBytes []byte, prompt string) (string, error) {
	b64 := base64.StdEncoding.EncodeToString(imageBytes)

	body, err := json.Marshal(map[string]any{
		"model":  c.extractModel,
		"prompt": prompt,
		"images": []string{b64},
		"stream": false,
	})
	if err != nil {
		return "", fmt.Errorf("marshal ocr request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("ocr request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	data, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", err
	}

	var result struct {
		Response string `json:"response"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("decode ocr response: %w", err)
	}
	return result.Response, nil
}

func isDoublePageScan(data []byte) bool {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return false
	}
	if cfg.Height == 0 {
		return false
	}
	return float64(cfg.Width)/float64(cfg.Height) > doublePageRatio
}

func isRefusal(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	for _, prefix := range refusalPrefixes {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}
