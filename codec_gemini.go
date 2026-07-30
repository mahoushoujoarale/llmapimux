package llmapimux

import (
	"encoding/json"
	"net/http"
	"strings"
)

// geminiCodec implements inboundCodec for the Gemini GenerateContent protocol.
type geminiCodec struct{}

func (c *geminiCodec) Protocol() Protocol {
	return ProtocolGemini
}

func (c *geminiCodec) KnownFields() map[string]bool {
	return geminiKnownFields
}

func (c *geminiCodec) ExtractAPIKey(r *http.Request) string {
	if key := r.Header.Get("x-goog-api-key"); key != "" {
		return key
	}
	return r.URL.Query().Get("key")
}

func (c *geminiCodec) DecodeRequest(r *http.Request, body []byte) (*Request, error) {
	req, err := decodeRequestWithInboundProtocol(body, ProtocolGemini, func(body []byte) (*Request, error) {
		return DecodeGeminiRequest(r.URL.Path, body)
	})
	if err != nil {
		return nil, err
	}
	// Gemini streaming is determined by the URL path, not the body.
	req.Stream = strings.Contains(r.URL.Path, ":streamGenerateContent")
	return req, nil
}

func (c *geminiCodec) WriteError(w http.ResponseWriter, statusCode int, msg string) {
	writeGeminiError(w, statusCode, msg)
}

func (c *geminiCodec) EncodeResponse(resp *Response) ([]byte, error) {
	return EncodeGeminiResponse(resp)
}

func (c *geminiCodec) WriteStreamingResponse(sseWriter *SSEWriter, ch <-chan StreamResult) {
	var accumulatedUsage Usage
	var lastStopReason StopReason

	// Gemini has no incremental function-call mechanism: a functionCall part must
	// carry complete `args` JSON. Every other protocol streams tool calls
	// incrementally — Anthropic puts the name/id on content_block_start and the
	// arguments in partial-JSON fragments, OpenAI does the same across tool_call
	// deltas — so fragments must be buffered per output index and flushed as one
	// part. Forwarding a fragment directly yields unparsable args (and previously
	// aborted the whole stream, because marshalling invalid raw JSON fails).
	type pendingToolCall struct {
		tool ToolUseContent
		args strings.Builder
	}
	pending := map[int]*pendingToolCall{}
	var pendingOrder []int

	flushPending := func() bool {
		for _, idx := range pendingOrder {
			p := pending[idx]
			if p == nil {
				continue
			}
			delete(pending, idx)
			tool := p.tool
			args := strings.TrimSpace(p.args.String())
			if args == "" || !json.Valid([]byte(args)) {
				// Never emit invalid args; an empty object is the safest fallback.
				args = "{}"
			}
			tool.Arguments = json.RawMessage(args)
			data, err := EncodeGeminiStreamChunk(&StreamEvent{
				Type:  StreamEventDelta,
				Index: idx,
				Delta: &ContentPart{
					Type:    ContentTypeToolUse,
					ToolUse: &tool,
				},
			})
			if err != nil {
				return false
			}
			if data == nil {
				continue
			}
			if sseWriter.WriteData(data) != nil {
				return false
			}
		}
		pendingOrder = pendingOrder[:0]
		return true
	}

	// bufferTool records a tool-call fragment, returning false if the caller should
	// stop processing.
	bufferTool := func(index int, tu *ToolUseContent) {
		p, ok := pending[index]
		if !ok {
			p = &pendingToolCall{}
			pending[index] = p
			pendingOrder = append(pendingOrder, index)
		}
		if tu.ID != "" {
			p.tool.ID = tu.ID
		}
		if tu.Name != "" {
			p.tool.Name = tu.Name
		}
		if frag := string(tu.Arguments); frag != "" && frag != "{}" {
			p.args.WriteString(frag)
		}
	}

	for result := range ch {
		if result.Err != nil {
			// The status code is already committed, so report the failure in-band
			// using Gemini's error envelope rather than closing the connection
			// silently, which a client cannot distinguish from a clean end.
			writeGeminiStreamError(sseWriter, result.Err.Error())
			return
		}

		// An in-band IR error event (e.g. an Anthropic "error" SSE event) has no
		// Gemini chunk representation — surface it as an error envelope.
		if result.Event != nil && result.Event.Type == StreamEventError {
			msg := "upstream stream error"
			if result.Event.Error != nil && result.Event.Error.Message != "" {
				msg = result.Event.Error.Message
			}
			writeGeminiStreamError(sseWriter, msg)
			return
		}

		// Accumulate usage from early events (e.g. Anthropic message_start
		// carries PromptTokens in StreamEventStart.Response.Usage).
		// Gemini only emits usage in the final chunk, so we must defer it.
		if result.Event != nil {
			if result.Event.Usage != nil {
				mergeStreamUsage(&accumulatedUsage, result.Event.Usage)
			}
			if result.Event.Response != nil && result.Event.Response.Usage.PromptTokens != 0 {
				mergeStreamUsage(&accumulatedUsage, &result.Event.Response.Usage)
			}
			// Capture stop reasons from delta events (e.g. Anthropic message_delta).
			if result.Event.StopReason != nil {
				lastStopReason = *result.Event.StopReason
			}
		}

		// Tool-call buffering.
		if ev := result.Event; ev != nil {
			isToolPart := ev.Delta != nil && ev.Delta.Type == ContentTypeToolUse && ev.Delta.ToolUse != nil
			switch {
			case (ev.Type == StreamEventContentBlockStart || ev.Type == StreamEventDelta) && isToolPart:
				bufferTool(ev.Index, ev.Delta.ToolUse)
				continue
			case ev.Type == StreamEventContentBlockStop, ev.Type == StreamEventStop:
				if !flushPending() {
					return
				}
			}
		}

		// On stop event, inject accumulated usage and stop reason.
		if result.Event != nil && result.Event.Type == StreamEventStop {
			if result.Event.Usage == nil && accumulatedUsage.PromptTokens != 0 {
				result.Event.Usage = &Usage{}
				*result.Event.Usage = accumulatedUsage
			} else if result.Event.Usage != nil && result.Event.Usage.PromptTokens == 0 && accumulatedUsage.PromptTokens != 0 {
				mergeStreamUsage(result.Event.Usage, &accumulatedUsage)
			}
			if result.Event.StopReason == nil && lastStopReason != "" {
				result.Event.StopReason = &lastStopReason
			}
		}

		data, err := EncodeGeminiStreamChunk(result.Event)
		if err != nil {
			break
		}
		if data == nil {
			continue
		}
		// Gemini uses JSON array SSE data (not event: lines), so use WriteData.
		if err := sseWriter.WriteData(data); err != nil {
			break
		}
	}

	// Guard against an upstream that ends without a stop event.
	flushPending()
}

// writeGeminiStreamError emits an in-band SSE chunk carrying Gemini's error
// envelope. Used when a stream fails after the HTTP 200 has been committed.
func writeGeminiStreamError(sseWriter *SSEWriter, message string) {
	data, err := json.Marshal(map[string]any{
		"error": map[string]any{
			"code":    http.StatusBadGateway,
			"message": message,
			"status":  httpStatusToGeminiStatus(http.StatusBadGateway),
		},
	})
	if err != nil {
		return
	}
	sseWriter.WriteData(data) //nolint:errcheck
}

// writeGeminiError writes a Gemini-formatted error response.
func writeGeminiError(w http.ResponseWriter, statusCode int, message string) {
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"code":    statusCode,
			"message": message,
			"status":  httpStatusToGeminiStatus(statusCode),
		},
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(body) //nolint:errcheck
}

// httpStatusToGeminiStatus maps an HTTP status code to a Gemini status string.
func httpStatusToGeminiStatus(code int) string {
	switch code {
	case http.StatusBadRequest:
		return "INVALID_ARGUMENT"
	case http.StatusUnauthorized:
		return "UNAUTHENTICATED"
	case http.StatusForbidden:
		return "PERMISSION_DENIED"
	case http.StatusNotFound:
		return "NOT_FOUND"
	case http.StatusTooManyRequests:
		return "RESOURCE_EXHAUSTED"
	case http.StatusBadGateway:
		return "UNAVAILABLE"
	default:
		return "INTERNAL"
	}
}
