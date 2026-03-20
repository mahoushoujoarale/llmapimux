package llmapimux

import (
	"context"
	"fmt"
	"io"
	"net/http"
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

	respBody, err := doSend(ctx, c.HTTPClient, cfg, body,
		trimBaseURL(cfg.BaseURL)+"/v1/messages",
		[][2]string{{"x-api-key", cfg.APIKey}, {"anthropic-version", "2023-06-01"}},
		"anthropic outbound")
	if err != nil {
		return nil, err
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

	httpResp, err := doStreamSetup(ctx, c.HTTPClient, cfg, body,
		trimBaseURL(cfg.BaseURL)+"/v1/messages",
		[][2]string{{"x-api-key", cfg.APIKey}, {"anthropic-version", "2023-06-01"}},
		"anthropic outbound")
	if err != nil {
		return nil, err
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
