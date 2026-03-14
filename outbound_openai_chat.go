package llmapimux

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAIChatClient sends IR Requests to the OpenAI Chat Completions API.
type OpenAIChatClient struct {
	HTTPClient *http.Client // optional, uses http.DefaultClient if nil
}

// Send encodes an IR Request to OpenAI Chat JSON, sends it, and returns a decoded IR Response.
func (c *OpenAIChatClient) Send(ctx context.Context, req *Request, cfg OutboundConfig) (*Response, error) {
	body, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		return nil, fmt.Errorf("openai chat outbound encode request: %w", err)
	}
	if len(req.OutboundExtra) > 0 {
		body, err = mergeOutboundExtra(body, req.OutboundExtra)
		if err != nil {
			return nil, fmt.Errorf("openai chat outbound merge extra: %w", err)
		}
	}

	url := trimBaseURL(cfg.BaseURL) + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai chat outbound new request: %w", err)
	}

	applyExtraHeaders(httpReq, cfg)
	httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := httpClientForProxy(c.HTTPClient, cfg.ProxyURL).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai chat outbound send: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai chat outbound read response: %w", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, newUpstreamHTTPError("openai chat outbound status", httpResp.StatusCode, httpResp.Header, respBody)
	}

	return DecodeOpenAIChatResponse(respBody)
}

// SendStream encodes an IR Request, sends it with stream=true, and returns a channel of StreamResults.
func (c *OpenAIChatClient) SendStream(ctx context.Context, req *Request, cfg OutboundConfig) (<-chan StreamResult, error) {
	outboundReq := *req
	outboundReq.Stream = true

	body, err := EncodeOpenAIChatRequest(&outboundReq)
	if err != nil {
		return nil, fmt.Errorf("openai chat outbound encode stream request: %w", err)
	}
	if len(req.OutboundExtra) > 0 {
		body, err = mergeOutboundExtra(body, req.OutboundExtra)
		if err != nil {
			return nil, fmt.Errorf("openai chat outbound stream merge extra: %w", err)
		}
	}

	url := trimBaseURL(cfg.BaseURL) + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai chat outbound new stream request: %w", err)
	}

	applyExtraHeaders(httpReq, cfg)
	httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := httpClientForProxy(c.HTTPClient, cfg.ProxyURL).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai chat outbound send stream: %w", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		body, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, newUpstreamHTTPError("openai chat outbound stream status", httpResp.StatusCode, httpResp.Header, body)
	}

	ct := httpResp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		httpResp.Body.Close()
		return nil, fmt.Errorf("openai chat outbound stream: unexpected Content-Type %q", ct)
	}

	ch := make(chan StreamResult)
	go func() {
		defer close(ch)
		defer httpResp.Body.Close()

		reader := NewSSEReader(httpResp.Body)
		for {
			data, err := reader.Read()
			if err == io.EOF {
				return
			}
			if err != nil {
				select {
				case ch <- StreamResult{Err: fmt.Errorf("openai chat outbound stream read: %w", err)}:
				case <-ctx.Done():
				}
				return
			}

			// [DONE] sentinel signals end of stream
			if string(data) == "[DONE]" {
				return
			}

			event, err := DecodeOpenAIChatStreamChunk(data)
			if err != nil {
				select {
				case ch <- StreamResult{Err: fmt.Errorf("openai chat outbound stream decode: %w", err)}:
				case <-ctx.Done():
				}
				return
			}

			// Skip nil events
			if event == nil {
				continue
			}

			select {
			case ch <- StreamResult{Event: event}:
			case <-ctx.Done():
				return
			}

			// Stop after stop event
			if event.Type == StreamEventStop {
				return
			}
		}
	}()

	return ch, nil
}
