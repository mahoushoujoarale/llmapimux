package llmapimux

import (
	"context"
	"fmt"
	"io"
	"net/http"
)

// OpenAIResponsesClient sends IR Requests to the OpenAI Responses API.
type OpenAIResponsesClient struct {
	HTTPClient *http.Client // optional, uses http.DefaultClient if nil
}

// Send encodes an IR Request to OpenAI Responses JSON, sends it, and returns a decoded IR Response.
func (c *OpenAIResponsesClient) Send(ctx context.Context, req *Request, cfg OutboundConfig) (*Response, error) {
	body, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		return nil, fmt.Errorf("openai responses outbound encode request: %w", err)
	}
	if len(req.OutboundExtra) > 0 {
		body, err = mergeOutboundExtra(body, req.OutboundExtra)
		if err != nil {
			return nil, fmt.Errorf("openai responses outbound merge extra: %w", err)
		}
	}

	respBody, err := doSend(ctx, c.HTTPClient, cfg, body,
		trimBaseURL(cfg.BaseURL)+"/v1/responses",
		[][2]string{{"Authorization", "Bearer " + cfg.APIKey}},
		"openai responses outbound")
	if err != nil {
		return nil, err
	}
	return DecodeOpenAIResponsesResponse(respBody)
}

// SendStream encodes an IR Request, sends it with stream=true, and returns a channel of StreamResults.
func (c *OpenAIResponsesClient) SendStream(ctx context.Context, req *Request, cfg OutboundConfig) (<-chan StreamResult, error) {
	outboundReq := *req
	outboundReq.Stream = true

	body, err := EncodeOpenAIResponsesRequest(&outboundReq)
	if err != nil {
		return nil, fmt.Errorf("openai responses outbound encode stream request: %w", err)
	}
	if len(req.OutboundExtra) > 0 {
		body, err = mergeOutboundExtra(body, req.OutboundExtra)
		if err != nil {
			return nil, fmt.Errorf("openai responses outbound stream merge extra: %w", err)
		}
	}

	httpResp, err := doStreamSetup(ctx, c.HTTPClient, cfg, body,
		trimBaseURL(cfg.BaseURL)+"/v1/responses",
		[][2]string{{"Authorization", "Bearer " + cfg.APIKey}},
		"openai responses outbound")
	if err != nil {
		return nil, err
	}

	ch := make(chan StreamResult)
	go func() {
		defer close(ch)
		defer httpResp.Body.Close()

		// reasoningIndices tracks output_index values for reasoning/unknown items that
		// should be suppressed (e.g. reasoning items produced by o-series models).
		// indexRemap remaps the remaining output indices to a compact 0-based sequence
		// so that the first non-reasoning block always starts at index 0.
		reasoningIndices := map[int]bool{}
		indexRemap := map[int]int{}
		nextSeqIndex := 0

		reader := NewSSEReader(httpResp.Body)
		for {
			data, err := reader.Read()
			if err == io.EOF {
				return
			}
			if err != nil {
				select {
				case ch <- StreamResult{Err: fmt.Errorf("openai responses outbound stream read: %w", err)}:
				case <-ctx.Done():
				}
				return
			}

			// [DONE] sentinel signals end of stream
			if string(data) == "[DONE]" {
				return
			}

			eventType := reader.LastEventType()
			event, err := DecodeOpenAIResponsesStreamEvent(eventType, data)
			if err != nil {
				select {
				case ch <- StreamResult{Err: fmt.Errorf("openai responses outbound stream decode: %w", err)}:
				case <-ctx.Done():
				}
				return
			}

			// Skip nil events (some events may return nil)
			if event == nil {
				continue
			}

			// For indexed events, handle reasoning suppression and index remapping.
			switch event.Type {
			case StreamEventContentBlockStart, StreamEventContentBlockStop, StreamEventDelta:
				if event.Type == StreamEventContentBlockStart && event.Delta == nil {
					// No delta type means this is a reasoning or unknown item — suppress it.
					reasoningIndices[event.Index] = true
					continue
				}
				if reasoningIndices[event.Index] {
					if event.Type == StreamEventContentBlockStop {
						delete(reasoningIndices, event.Index)
					}
					continue
				}
				// Remap to compact sequential index starting at 0.
				if _, ok := indexRemap[event.Index]; !ok {
					indexRemap[event.Index] = nextSeqIndex
					nextSeqIndex++
				}
				event.Index = indexRemap[event.Index]
			}

			select {
			case ch <- StreamResult{Event: event}:
			case <-ctx.Done():
				return
			}

			// Stop after response.completed
			if event.Type == StreamEventStop {
				return
			}
		}
	}()

	return ch, nil
}
