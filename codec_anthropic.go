package llmapimux

import (
	"encoding/json"
	"net/http"
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
	// Normalise arbitrary IR streams into a well-formed Anthropic SSE stream.
	//
	// The Anthropic protocol requires content blocks to be strictly sequential:
	// block N must be closed with content_block_stop before block N+1 is opened,
	// and every block needs a unique index and a stable type. IR streams coming
	// from other protocols do not honour that:
	//   - OpenAI Chat / Gemini emit no content_block_start / content_block_stop
	//     lifecycle events at all.
	//   - Gemini puts every delta on IR index 0 regardless of content type.
	//   - OpenAI Chat may interleave a text delta and several tool_call deltas on
	//     different IR indices without ever closing the previous one.
	//
	// We therefore keep exactly one Anthropic block open at a time: opening a new
	// block implicitly closes the current one.
	type openBlock struct {
		index       int
		contentType ContentType
	}
	var current *openBlock // the single currently open Anthropic block, if any
	nextIndex := 0         // next Anthropic block index to assign
	messageStartSent := false
	// accumulatedUsage collects usage arriving on events that have no Anthropic
	// representation (e.g. OpenAI Chat usage-only deltas) so it can be folded into
	// the terminating message_delta instead of being lost.
	var accumulatedUsage Usage
	var lastStopReason StopReason
	var lastStopSequence string
	// messageDeltaSent guards against emitting a second message_delta on
	// StreamEventStop when the upstream already sent one (native Anthropic sends
	// message_delta then message_stop).
	messageDeltaSent := false
	// sourceIndexMap remembers which Anthropic index an IR source index currently
	// maps to, so repeated deltas on the same source index reuse the same block.
	sourceIndexMap := map[int]int{}

	writeSSE := func(event *StreamEvent) bool {
		eventType, data, err := EncodeAnthropicStreamEvent(event)
		if err != nil {
			return false
		}
		if data == nil {
			// Event has no Anthropic representation (e.g. a usage-only delta);
			// skip it without tearing down the stream.
			return true
		}
		return sseWriter.WriteEvent(eventType, data) == nil
	}

	// closeCurrent emits content_block_stop for the open block, if any.
	closeCurrent := func() bool {
		if current == nil {
			return true
		}
		idx := current.index
		current = nil
		return writeSSE(&StreamEvent{Type: StreamEventContentBlockStop, Index: idx})
	}

	// blockStartPart builds the content_block payload for a synthetic
	// content_block_start. Anthropic requires the block skeleton (empty text,
	// tool id/name) to be present up front.
	blockStartPart := func(deltaType ContentType, delta *ContentPart) ContentPart {
		part := ContentPart{Type: deltaType}
		switch deltaType {
		case ContentTypeText, ContentTypeRefusal:
			// Refusal has no Anthropic block type and degrades to text.
			part.Type = ContentTypeText
			part.Text = &TextContent{}
		case ContentTypeThinking:
			part.Thinking = &ThinkingContent{}
		case ContentTypeToolUse:
			if delta != nil && delta.ToolUse != nil {
				part.ToolUse = &ToolUseContent{ID: delta.ToolUse.ID, Name: delta.ToolUse.Name}
			} else {
				part.ToolUse = &ToolUseContent{}
			}
		case ContentTypeServerToolUse:
			if delta != nil && delta.ServerToolUse != nil {
				part.ServerToolUse = &ServerToolUseContent{ID: delta.ServerToolUse.ID, Name: delta.ServerToolUse.Name}
			} else {
				part.ServerToolUse = &ServerToolUseContent{}
			}
		}
		return part
	}

	// openBlockAt closes any open block and opens a new one at the next index.
	openBlockAt := func(deltaType ContentType, delta *ContentPart) (int, bool) {
		if !closeCurrent() {
			return 0, false
		}
		idx := nextIndex
		nextIndex++
		part := blockStartPart(deltaType, delta)
		if !writeSSE(&StreamEvent{Type: StreamEventContentBlockStart, Index: idx, Delta: &part}) {
			return 0, false
		}
		current = &openBlock{index: idx, contentType: deltaType}
		return idx, true
	}

	// isBlockDelta reports whether a delta type maps onto an Anthropic content block.
	isBlockDelta := func(t ContentType) bool {
		switch t {
		case ContentTypeText, ContentTypeToolUse, ContentTypeServerToolUse,
			ContentTypeThinking, ContentTypeRefusal:
			return true
		}
		return false
	}

	for result := range ch {
		if result.Err != nil {
			// Status code is already committed — surface the failure as an SSE
			// error event so the client can distinguish it from a clean end of
			// stream, then stop.
			writeSSE(&StreamEvent{
				Type:  StreamEventError,
				Error: &StreamError{Type: "api_error", Message: result.Err.Error()},
			})
			return
		}

		event := result.Event
		if event == nil {
			continue
		}

		// Collect usage and stop metadata from every event so nothing is dropped
		// when an event itself has no Anthropic representation.
		if event.Usage != nil {
			mergeStreamUsage(&accumulatedUsage, event.Usage)
		}
		if event.Response != nil {
			mergeStreamUsage(&accumulatedUsage, &event.Response.Usage)
		}
		if event.StopReason != nil {
			lastStopReason = *event.StopReason
		}
		if event.StopSequence != "" {
			lastStopSequence = event.StopSequence
		}

		// Inject message_start if the upstream skipped it (e.g. Gemini starts with a delta).
		if !messageStartSent {
			messageStartSent = true
			if event.Type != StreamEventStart {
				if !writeSSE(&StreamEvent{Type: StreamEventStart, Response: &Response{}}) {
					return
				}
			}
		}

		switch event.Type {
		case StreamEventContentBlockStart:
			// Upstream provided a real block start. Close whatever is open and
			// re-index onto our own sequential numbering so upstream indices that
			// are sparse or reused cannot collide.
			deltaType := ContentTypeText
			if event.Delta != nil {
				deltaType = event.Delta.Type
			}
			// A nil Delta is legitimate: DecodeOpenAIResponsesStreamEvent produces
			// one for output_item.added carrying an item type we do not map. Treat
			// it as an empty text block rather than dereferencing nil.
			idx, ok := openBlockAt(deltaType, event.Delta)
			if !ok {
				return
			}
			sourceIndexMap[event.Index] = idx
			continue

		case StreamEventContentBlockStop:
			// Only honour a stop for the block we currently have open; stops for
			// already-closed or unknown indices are redundant.
			if current != nil {
				if mapped, ok := sourceIndexMap[event.Index]; !ok || mapped == current.index {
					if !closeCurrent() {
						return
					}
				}
			}
			delete(sourceIndexMap, event.Index)
			continue

		case StreamEventDelta:
			// A delta carrying a stop reason and no content is a native Anthropic
			// message_delta passing through; record that so StreamEventStop does
			// not synthesise a duplicate.
			if event.Delta == nil && event.StopReason != nil {
				messageDeltaSent = true
			}
			if event.Delta != nil && isBlockDelta(event.Delta.Type) {
				srcIdx := event.Index
				anthIdx, mapped := sourceIndexMap[srcIdx]
				// Reopen when this source index has no block, when its block was
				// superseded by another source index, or when the content type
				// changed (Gemini switches type on the same index).
				needNew := !mapped || current == nil || current.index != anthIdx ||
					current.contentType != event.Delta.Type
				if needNew {
					newIdx, ok := openBlockAt(event.Delta.Type, event.Delta)
					if !ok {
						return
					}
					sourceIndexMap[srcIdx] = newIdx
					anthIdx = newIdx
				}
				event.Index = anthIdx
			}

		case StreamEventStop:
			// Close the open block before finishing: the Anthropic SDK accumulator
			// only refreshes AsAny() on content_block_stop, so omitting it leaves
			// the block's raw JSON stale.
			if !closeCurrent() {
				return
			}
			// Anthropic carries stop_reason and final usage in message_delta, which
			// must precede message_stop. Emit it whenever we have either — the
			// reason may have arrived on an earlier delta (native Anthropic) or be
			// folded into the stop event (OpenAI Chat finish_reason).
			stopReason := lastStopReason
			if event.StopReason != nil {
				stopReason = *event.StopReason
			}
			if !messageDeltaSent && (stopReason != "" || accumulatedUsage != (Usage{})) {
				if stopReason == "" {
					stopReason = StopReasonEndTurn
				}
				usage := accumulatedUsage
				md := &StreamEvent{
					Type:         StreamEventDelta,
					StopReason:   &stopReason,
					StopSequence: lastStopSequence,
					Usage:        &usage,
				}
				if !writeSSE(md) {
					return
				}
				messageDeltaSent = true
			}
		}

		if !writeSSE(event) {
			return
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
