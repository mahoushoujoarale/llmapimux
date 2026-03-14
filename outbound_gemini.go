package llmapimux

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// GeminiClient sends IR Requests to the Gemini GenerateContent API.
type GeminiClient struct {
	HTTPClient *http.Client // optional, uses http.DefaultClient if nil
}

// Send encodes an IR Request to Gemini JSON, sends it, and returns a decoded IR Response.
func (c *GeminiClient) Send(ctx context.Context, req *Request, cfg OutboundConfig) (*Response, error) {
	model, body, err := EncodeGeminiRequest(req)
	if err != nil {
		return nil, fmt.Errorf("gemini outbound encode request: %w", err)
	}
	if len(req.OutboundExtra) > 0 {
		body, err = mergeOutboundExtra(body, req.OutboundExtra)
		if err != nil {
			return nil, fmt.Errorf("gemini outbound merge extra: %w", err)
		}
	}

	url := trimBaseURL(cfg.BaseURL) + "/v1beta/models/" + model + ":generateContent"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini outbound new request: %w", err)
	}

	applyExtraHeaders(httpReq, cfg)
	httpReq.Header.Set("x-goog-api-key", cfg.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := httpClientForProxy(c.HTTPClient, cfg.ProxyURL).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini outbound send: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("gemini outbound read response: %w", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, newUpstreamHTTPError("gemini outbound status", httpResp.StatusCode, httpResp.Header, respBody)
	}

	return DecodeGeminiResponse(respBody)
}

// SendStream encodes an IR Request, sends it to the streaming endpoint, and returns a channel of StreamResults.
func (c *GeminiClient) SendStream(ctx context.Context, req *Request, cfg OutboundConfig) (<-chan StreamResult, error) {
	model, body, err := EncodeGeminiRequest(req)
	if err != nil {
		return nil, fmt.Errorf("gemini outbound encode stream request: %w", err)
	}
	if len(req.OutboundExtra) > 0 {
		body, err = mergeOutboundExtra(body, req.OutboundExtra)
		if err != nil {
			return nil, fmt.Errorf("gemini outbound stream merge extra: %w", err)
		}
	}

	url := trimBaseURL(cfg.BaseURL) + "/v1beta/models/" + model + ":streamGenerateContent?alt=sse"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini outbound new stream request: %w", err)
	}

	applyExtraHeaders(httpReq, cfg)
	httpReq.Header.Set("x-goog-api-key", cfg.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := httpClientForProxy(c.HTTPClient, cfg.ProxyURL).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini outbound send stream: %w", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		body, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, newUpstreamHTTPError("gemini outbound stream status", httpResp.StatusCode, httpResp.Header, body)
	}

	ct := httpResp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		httpResp.Body.Close()
		return nil, fmt.Errorf("gemini outbound stream: unexpected Content-Type %q", ct)
	}

	ch := make(chan StreamResult)
	go func() {
		defer close(ch)
		defer httpResp.Body.Close()

		reader := NewSSEReader(httpResp.Body)

		// drainTrailingUsage reads any remaining SSE chunks after a stop event,
		// looking for usage-only chunks. Gemini sometimes sends usageMetadata
		// in a separate trailing chunk after the finishReason chunk.
		drainTrailingUsage := func(stop *StreamEvent) {
			for {
				data, err := reader.Read()
				if err != nil {
					break // EOF or error — done draining
				}
				trailingEvents, err := DecodeGeminiStreamChunk(data)
				if err != nil || len(trailingEvents) == 0 {
					continue
				}
				// Merge usage from trailing chunks into the stop event
				for _, trailing := range trailingEvents {
					if trailing.Usage != nil {
						if stop.Usage == nil {
							stop.Usage = trailing.Usage
						} else {
							mergeStreamUsage(stop.Usage, trailing.Usage)
						}
					}
				}
			}
			select {
			case ch <- StreamResult{Event: stop}:
			case <-ctx.Done():
			}
		}

		for {
			data, err := reader.Read()
			if err == io.EOF {
				return
			}
			if err != nil {
				select {
				case ch <- StreamResult{Err: fmt.Errorf("gemini outbound stream read: %w", err)}:
				case <-ctx.Done():
				}
				return
			}

			events, err := DecodeGeminiStreamChunk(data)
			if err != nil {
				select {
				case ch <- StreamResult{Err: fmt.Errorf("gemini outbound stream decode: %w", err)}:
				case <-ctx.Done():
				}
				return
			}

			// Skip empty event slices
			if len(events) == 0 {
				continue
			}

			for _, event := range events {
				// Gemini sometimes sends text content and finishReason in the same chunk.
				// DecodeGeminiStreamChunk emits this as StreamEventDelta with StopReason set.
				// Split into two events: content-only delta first, then a stop event.
				// Do NOT send the original mixed event to avoid duplicating the delta content.
				// This ensures downstream encoders (e.g. Anthropic) emit content_block_delta
				// before the stop reason, rather than discarding the text content.
				//
				// After splitting, drain any trailing chunks for usage before emitting stop.
				// Gemini may send usageMetadata in a separate chunk after finishReason.
				if event.Type == StreamEventDelta && event.StopReason != nil {
					if event.Delta != nil {
						contentOnly := &StreamEvent{
							Type:  StreamEventDelta,
							Index: event.Index,
							Delta: event.Delta,
						}
						select {
						case ch <- StreamResult{Event: contentOnly}:
						case <-ctx.Done():
							return
						}
					}
					stop := &StreamEvent{
						Type:       StreamEventStop,
						StopReason: event.StopReason,
						Usage:      event.Usage,
					}
					drainTrailingUsage(stop)
					return
				}

				// For a plain stop event (no delta), also drain trailing usage.
				if event.Type == StreamEventStop {
					drainTrailingUsage(event)
					return
				}

				select {
				case ch <- StreamResult{Event: event}:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return ch, nil
}
