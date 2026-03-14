package llmapimux

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// AnthropicClient sends IR Requests to the Anthropic Messages API.
type AnthropicClient struct {
	HTTPClient *http.Client // optional, uses http.DefaultClient if nil
}

// Send encodes an IR Request to Anthropic JSON, sends it, and returns a decoded IR Response.
func (c *AnthropicClient) Send(ctx context.Context, req *Request, cfg OutboundConfig) (*Response, error) {
	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic outbound encode request: %w", err)
	}
	if len(req.OutboundExtra) > 0 {
		body, err = mergeOutboundExtra(body, req.OutboundExtra)
		if err != nil {
			return nil, fmt.Errorf("anthropic outbound merge extra: %w", err)
		}
	}

	url := trimBaseURL(cfg.BaseURL) + "/v1/messages"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic outbound new request: %w", err)
	}

	applyExtraHeaders(httpReq, cfg)
	httpReq.Header.Set("x-api-key", cfg.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := httpClientForProxy(c.HTTPClient, cfg.ProxyURL).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic outbound send: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("anthropic outbound read response: %w", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, newUpstreamHTTPError("anthropic outbound status", httpResp.StatusCode, httpResp.Header, respBody)
	}

	return DecodeAnthropicResponse(respBody)
}

// SendStream encodes an IR Request, sends it with stream=true, and returns a channel of StreamResults.
func (c *AnthropicClient) SendStream(ctx context.Context, req *Request, cfg OutboundConfig) (<-chan StreamResult, error) {
	outboundReq := *req
	outboundReq.Stream = true

	body, err := EncodeAnthropicRequest(&outboundReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic outbound encode stream request: %w", err)
	}
	if len(req.OutboundExtra) > 0 {
		body, err = mergeOutboundExtra(body, req.OutboundExtra)
		if err != nil {
			return nil, fmt.Errorf("anthropic outbound stream merge extra: %w", err)
		}
	}

	url := trimBaseURL(cfg.BaseURL) + "/v1/messages"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic outbound new stream request: %w", err)
	}

	applyExtraHeaders(httpReq, cfg)
	httpReq.Header.Set("x-api-key", cfg.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := httpClientForProxy(c.HTTPClient, cfg.ProxyURL).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic outbound send stream: %w", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		body, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, newUpstreamHTTPError("anthropic outbound stream status", httpResp.StatusCode, httpResp.Header, body)
	}

	ct := httpResp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		httpResp.Body.Close()
		return nil, fmt.Errorf("anthropic outbound stream: unexpected Content-Type %q", ct)
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
				case ch <- StreamResult{Err: fmt.Errorf("anthropic outbound stream read: %w", err)}:
				case <-ctx.Done():
				}
				return
			}

			eventType := reader.LastEventType()
			event, err := DecodeAnthropicStreamEvent(eventType, data)
			if err != nil {
				select {
				case ch <- StreamResult{Err: fmt.Errorf("anthropic outbound stream decode: %w", err)}:
				case <-ctx.Done():
				}
				return
			}

			// Skip nil events (e.g. ping)
			if event == nil {
				continue
			}

			select {
			case ch <- StreamResult{Event: event}:
			case <-ctx.Done():
				return
			}

			// Stop after message_stop
			if event.Type == StreamEventStop {
				return
			}
		}
	}()

	return ch, nil
}
