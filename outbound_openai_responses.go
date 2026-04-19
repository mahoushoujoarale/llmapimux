package llmapimux

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const openAIResponsesWebSearchSourcesInclude = "web_search_call.action.sources"

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
	body, err = ensureOpenAIResponsesWebSearchInclude(body, req)
	if err != nil {
		return nil, fmt.Errorf("openai responses outbound include merge: %w", err)
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
	body, err = ensureOpenAIResponsesWebSearchInclude(body, &outboundReq)
	if err != nil {
		return nil, fmt.Errorf("openai responses outbound stream include merge: %w", err)
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

func ensureOpenAIResponsesWebSearchInclude(body []byte, req *Request) ([]byte, error) {
	if !requestHasToolType(req, "web_search") {
		return body, nil
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("unmarshal request body: %w", err)
	}

	var include []string
	if includeRaw, ok := raw["include"]; ok && len(includeRaw) > 0 {
		if err := json.Unmarshal(includeRaw, &include); err != nil {
			return nil, fmt.Errorf("unmarshal include: %w", err)
		}
	}
	for _, item := range include {
		if item == openAIResponsesWebSearchSourcesInclude {
			return body, nil
		}
	}
	include = append(include, openAIResponsesWebSearchSourcesInclude)
	includeRaw, err := json.Marshal(include)
	if err != nil {
		return nil, fmt.Errorf("marshal include: %w", err)
	}
	raw["include"] = includeRaw
	return json.Marshal(raw)
}

func requestHasToolType(req *Request, toolType string) bool {
	if req == nil {
		return false
	}
	for _, tool := range req.Tools {
		if tool.Type == toolType {
			return true
		}
	}
	return false
}
