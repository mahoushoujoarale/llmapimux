package llmapimux

import (
	"encoding/json"
	"net/http"
	"strings"
)

// pendingToolBlock buffers a tool_use / server_tool_use stream that cannot be
// emitted live. See WriteStreamingResponse for why buffering is required.
type pendingToolBlock struct {
	isServer bool
	id       string
	name     string
	args     strings.Builder
}

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
	//
	// Tool calls are the exception. OpenAI Chat streams parallel tool_call
	// deltas keyed by their own array index, and nothing stops a provider from
	// interleaving them (tc0 args, tc1 args, more tc0 args, ...). The
	// one-open-block model cannot express that: resuming tc0 after tc1 opened
	// would require appending to a closed block, so opening a fresh block
	// produces an id-less tool_use fragment carrying the remainder of the
	// arguments — a corrupt assistant message for clients such as Claude Code,
	// which then fail the turn and rewrite history (duplicated user turns,
	// broken prompt-cache prefixes).
	//
	// Synthetic tool deltas (no real content_block_start from the upstream) are
	// therefore accumulated per source index and flushed as complete blocks at
	// the end of the stream, in first-seen order. Text/thinking still stream
	// live; a text block arriving between tool deltas is emitted before them,
	// which only reorders blocks in the rare text-after-tool-call case and
	// never corrupts content. Deltas that DID come with a real
	// content_block_start (native Anthropic upstream) keep streaming live
	// untouched.
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
	// liveToolStarted records source indices whose tool block was opened by a
	// real upstream content_block_start (native Anthropic pass-through). Those
	// keep streaming live; only synthetic tool deltas are buffered.
	liveToolStarted := map[int]bool{}
	// pendingTools buffers synthetic tool deltas keyed by IR source index;
	// toolOrder preserves first-seen order for the final flush.
	pendingTools := map[int]*pendingToolBlock{}
	var toolOrder []int

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

	// bufferToolDelta accumulates a synthetic tool_use / server_tool_use delta
	// for the end-of-stream flush.
	bufferToolDelta := func(srcIdx int, delta *ContentPart) {
		tb := pendingTools[srcIdx]
		if tb == nil {
			tb = &pendingToolBlock{}
			pendingTools[srcIdx] = tb
			toolOrder = append(toolOrder, srcIdx)
		}
		if delta.ToolUse != nil {
			if delta.ToolUse.ID != "" {
				tb.id = delta.ToolUse.ID
			}
			if delta.ToolUse.Name != "" {
				tb.name = delta.ToolUse.Name
			}
			tb.args.WriteString(string(delta.ToolUse.Arguments))
		}
		if delta.ServerToolUse != nil {
			tb.isServer = true
			if delta.ServerToolUse.ID != "" {
				tb.id = delta.ServerToolUse.ID
			}
			if delta.ServerToolUse.Name != "" {
				tb.name = delta.ServerToolUse.Name
			}
			tb.args.WriteString(string(delta.ServerToolUse.Arguments))
		}
	}

	// flushPendingTools emits every buffered tool call as one complete content
	// block — content_block_start with the id/name skeleton, a single
	// input_json_delta carrying the full accumulated arguments, and
	// content_block_stop — in first-seen order. Idempotent: buffers are cleared
	// after the flush.
	flushPendingTools := func() bool {
		for _, srcIdx := range toolOrder {
			tb := pendingTools[srcIdx]
			blockType := ContentTypeToolUse
			startPart := ContentPart{ToolUse: &ToolUseContent{ID: tb.id, Name: tb.name}}
			if tb.isServer {
				blockType = ContentTypeServerToolUse
				startPart = ContentPart{ServerToolUse: &ServerToolUseContent{ID: tb.id, Name: tb.name}}
			}
			startPart.Type = blockType
			idx, ok := openBlockAt(blockType, &startPart)
			if !ok {
				return false
			}
			if tb.args.Len() > 0 {
				args := json.RawMessage(tb.args.String())
				deltaPart := ContentPart{Type: blockType}
				if tb.isServer {
					deltaPart.ServerToolUse = &ServerToolUseContent{Arguments: args}
				} else {
					deltaPart.ToolUse = &ToolUseContent{Arguments: args}
				}
				if !writeSSE(&StreamEvent{Type: StreamEventDelta, Index: idx, Delta: &deltaPart}) {
					return false
				}
			}
			if !closeCurrent() {
				return false
			}
		}
		toolOrder = nil
		pendingTools = map[int]*pendingToolBlock{}
		return true
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
			if deltaType == ContentTypeToolUse || deltaType == ContentTypeServerToolUse {
				// A real block start: this tool streams live from here on. If a
				// synthetic buffer had already accumulated for the same index
				// (mixed-lifecycle upstream), drop it so the call is not emitted twice.
				liveToolStarted[event.Index] = true
				if _, buffered := pendingTools[event.Index]; buffered {
					delete(pendingTools, event.Index)
					filtered := toolOrder[:0]
					for _, k := range toolOrder {
						if k != event.Index {
							filtered = append(filtered, k)
						}
					}
					toolOrder = filtered
				}
			}
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
				// Synthetic tool deltas (OpenAI Chat tool_call deltas carry no
				// block lifecycle) are buffered so interleaved parallel calls
				// cannot be split into id-less fragments; they flush as complete
				// blocks at end of stream. Tool deltas with a real upstream
				// content_block_start keep streaming live below.
				switch event.Delta.Type {
				case ContentTypeToolUse, ContentTypeServerToolUse:
					if !liveToolStarted[event.Index] {
						bufferToolDelta(event.Index, event.Delta)
						continue
					}
				}
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
			// Emit buffered tool calls as complete blocks before the terminating
			// message_delta / message_stop, in first-seen order.
			if !flushPendingTools() {
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
