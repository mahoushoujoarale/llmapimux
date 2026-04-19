package llmapimux

import (
	"encoding/json"
	"net/http"
	"sort"
)

// anthropicCodec implements inboundCodec for the Anthropic Messages protocol.
type anthropicCodec struct{}

func (c *anthropicCodec) Protocol() Protocol {
	return ProtocolAnthropic
}

func (c *anthropicCodec) KnownFields() map[string]bool {
	return anthropicKnownFields
}

func (c *anthropicCodec) ExtractAPIKey(r *http.Request) string {
	if key := r.Header.Get("x-api-key"); key != "" {
		return key
	}
	return parseBearerAPIKey(r)
}

func (c *anthropicCodec) DecodeRequest(r *http.Request, body []byte) (*Request, error) {
	return decodeRequestWithInboundProtocol(body, ProtocolAnthropic, DecodeAnthropicRequest)
}

func (c *anthropicCodec) WriteError(w http.ResponseWriter, statusCode int, msg string) {
	writeAnthropicError(w, statusCode, "api_error", msg)
}

func (c *anthropicCodec) EncodeResponse(resp *Response) ([]byte, error) {
	return EncodeAnthropicResponse(resp)
}

func (c *anthropicCodec) WriteStreamingResponse(sseWriter *SSEWriter, ch <-chan StreamResult) {
	// Track open content blocks: Anthropic index → content type.
	// This normalises IR streams from protocols that don't produce content_block_start /
	// content_block_stop lifecycle events (e.g. OpenAI Chat, Gemini) so the Anthropic SSE
	// output is always well-formed for SDK accumulators.
	//
	// Gemini sends all deltas on IR index 0 regardless of content type (thinking, text,
	// tool_use). The Anthropic protocol requires each content block to have a unique index
	// and consistent type. We remap IR indices to sequential Anthropic indices and detect
	// content-type changes to close/reopen blocks automatically.
	openBlockType := map[int]ContentType{} // Anthropic index → content type
	nextIndex := 0                         // next Anthropic block index to assign
	messageStartSent := false

	// sourceIndexMap tracks the current Anthropic index assigned to each IR source index.
	// When the content type changes on a source index, the old Anthropic block is closed
	// and a new one is opened with the next available index.
	sourceIndexMap := map[int]int{} // IR source index → current Anthropic index

	writeSSE := func(event *StreamEvent) bool {
		eventType, data, err := EncodeAnthropicStreamEvent(event)
		if err != nil {
			return false
		}
		return sseWriter.WriteEvent(eventType, data) == nil
	}

	injectBlockStart := func(index int, deltaType ContentType, delta *ContentPart) bool {
		blockPart := ContentPart{Type: deltaType}
		if deltaType == ContentTypeText {
			blockPart.Text = &TextContent{}
		} else if deltaType == ContentTypeToolUse && delta != nil && delta.ToolUse != nil {
			blockPart.ToolUse = &ToolUseContent{
				ID:   delta.ToolUse.ID,
				Name: delta.ToolUse.Name,
			}
		} else if deltaType == ContentTypeServerToolUse && delta != nil && delta.ServerToolUse != nil {
			blockPart.ServerToolUse = &ServerToolUseContent{
				ID:   delta.ServerToolUse.ID,
				Name: delta.ServerToolUse.Name,
			}
		} else if deltaType == ContentTypeThinking {
			blockPart.Thinking = &ThinkingContent{}
		}
		synthetic := &StreamEvent{
			Type:  StreamEventContentBlockStart,
			Index: index,
			Delta: &blockPart,
		}
		return writeSSE(synthetic)
	}

loop:
	for result := range ch {
		if result.Err != nil {
			// Cannot change status code at this point — just stop.
			break
		}

		event := result.Event

		// Inject message_start if the upstream skipped it (e.g. Gemini starts with delta).
		if !messageStartSent {
			messageStartSent = true
			if event.Type != StreamEventStart {
				synthetic := &StreamEvent{
					Type:     StreamEventStart,
					Response: &Response{},
				}
				if !writeSSE(synthetic) {
					break
				}
			}
		}

		// Track upstream-provided content_block_start to prevent double-injection on subsequent deltas.
		if event.Type == StreamEventContentBlockStart {
			openBlockType[event.Index] = event.Delta.Type
			sourceIndexMap[event.Index] = event.Index
		}
		// Remove from tracking when upstream provides a real content_block_stop,
		// so StreamEventStop doesn't emit a duplicate synthetic stop for the same index.
		if event.Type == StreamEventContentBlockStop {
			delete(openBlockType, event.Index)
		}

		// For deltas: detect content-type changes and remap IR index → Anthropic index.
		if event.Type == StreamEventDelta && event.Delta != nil &&
			(event.Delta.Type == ContentTypeText || event.Delta.Type == ContentTypeToolUse || event.Delta.Type == ContentTypeServerToolUse || event.Delta.Type == ContentTypeThinking || event.Delta.Type == ContentTypeRefusal) {

			srcIdx := event.Index
			anthIdx, mapped := sourceIndexMap[srcIdx]

			if mapped {
				if openBlockType[anthIdx] != event.Delta.Type {
					// Content type changed on same source index — close old block.
					if !writeSSE(&StreamEvent{Type: StreamEventContentBlockStop, Index: anthIdx}) {
						break
					}
					delete(openBlockType, anthIdx)
					mapped = false
				}
			}

			if !mapped {
				// Assign new Anthropic index and inject content_block_start.
				anthIdx = nextIndex
				nextIndex++
				sourceIndexMap[srcIdx] = anthIdx
				openBlockType[anthIdx] = event.Delta.Type
				if !injectBlockStart(anthIdx, event.Delta.Type, event.Delta) {
					break
				}
			}

			event.Index = anthIdx
		}

		// When a StreamEventStop arrives, close any open content blocks first.
		// Anthropic SDK accumulator updates AsAny() only on ContentBlockStopEvent,
		// so omitting it leaves the block's JSON.raw stale (empty text).
		// Also, if the upstream encoded stop_reason directly in StreamEventStop
		// (e.g. OpenAI Chat finish_reason), emit a separate message_delta first.
		if event.Type == StreamEventStop {
			indices := make([]int, 0, len(openBlockType))
			for idx := range openBlockType {
				indices = append(indices, idx)
			}
			sort.Ints(indices)
			for _, idx := range indices {
				stopBlock := &StreamEvent{
					Type:  StreamEventContentBlockStop,
					Index: idx,
				}
				if !writeSSE(stopBlock) {
					break loop
				}
			}
			if event.StopReason != nil {
				stopReason := *event.StopReason
				messageDelta := &StreamEvent{
					Type:       StreamEventDelta,
					StopReason: &stopReason,
					Usage:      event.Usage,
				}
				if !writeSSE(messageDelta) {
					break
				}
			}
		}

		if !writeSSE(event) {
			break
		}
	}
}

// writeAnthropicError writes an Anthropic-formatted error response.
func writeAnthropicError(w http.ResponseWriter, statusCode int, errType, message string) {
	body, _ := json.Marshal(map[string]any{
		"type": "error",
		"error": map[string]string{
			"type":    errType,
			"message": message,
		},
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(body) //nolint:errcheck
}
