package llmapimux

import (
	"encoding/json"
	"net/http"
)

// openaiResponsesCodec implements inboundCodec for the OpenAI Responses protocol.
type openaiResponsesCodec struct{}

func (c *openaiResponsesCodec) Protocol() Protocol {
	return ProtocolOpenAIResponses
}

func (c *openaiResponsesCodec) KnownFields() map[string]bool {
	return openaiResponsesKnownFields
}

func (c *openaiResponsesCodec) ExtractAPIKey(r *http.Request) string {
	return parseBearerAPIKey(r)
}

func (c *openaiResponsesCodec) DecodeRequest(r *http.Request, body []byte) (*Request, error) {
	return decodeRequestWithInboundProtocol(body, ProtocolOpenAIResponses, DecodeOpenAIResponsesRequest)
}

func (c *openaiResponsesCodec) WriteError(w http.ResponseWriter, statusCode int, msg string) {
	writeOpenAIError(w, statusCode, msg)
}

func (c *openaiResponsesCodec) EncodeResponse(resp *Response) ([]byte, error) {
	return EncodeOpenAIResponsesResponse(resp)
}

func (c *openaiResponsesCodec) WriteStreamingResponse(sseWriter *SSEWriter, ch <-chan StreamResult) {
	var accumulatedUsage Usage
	var lastStopReason StopReason

	// The Responses API grammar requires response.created before any output, and
	// requires a response.output_item.added carrying the function name before any
	// function_call_arguments.delta. Cross-protocol IR streams do not always
	// produce a StreamEventStart (Gemini has no equivalent) and carry the tool
	// name on content_block_start, so both are synthesised here. Without them SDKs
	// see a stream with no beginning and unnamed tool calls.
	createdSent := false
	// announcedTools tracks output indices whose response.output_item.added has
	// already been emitted, so the synthetic one is sent at most once per tool.
	announcedTools := map[int]bool{}
	ensureCreated := func(ev *StreamEvent) bool {
		if createdSent {
			return true
		}
		createdSent = true
		if ev != nil && ev.Type == StreamEventStart {
			return true // the real start event is about to be written
		}
		_, data, err := EncodeOpenAIResponsesStreamEvent(&StreamEvent{
			Type:     StreamEventStart,
			Response: &Response{},
		})
		if err != nil || data == nil {
			return true
		}
		return sseWriter.WriteEvent("response.created", data) == nil
	}

	for result := range ch {
		if result.Err != nil {
			// The status code is already committed, so report the failure in-band
			// as a Responses `error` event instead of closing the connection with
			// no signal.
			writeOpenAIResponsesStreamError(sseWriter, "server_error", "", result.Err.Error())
			return
		}
		if !ensureCreated(result.Event) {
			return
		}
		if result.Event != nil && result.Event.Type == StreamEventStart {
			createdSent = true
		}

		// Accumulate usage from early events (e.g. Anthropic message_start
		// carries PromptTokens in StreamEventStart.Response.Usage).
		// OpenAI Responses only emits usage in response.completed, so we
		// must defer it.
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

		// A tool name arriving on a StreamEventDelta (which is how OpenAI Chat and
		// Gemini sources carry it) has no place in
		// response.function_call_arguments.delta, so synthesise the
		// response.output_item.added that the Responses grammar requires first.
		if ev := result.Event; ev != nil && ev.Type == StreamEventDelta &&
			ev.Delta != nil && ev.Delta.Type == ContentTypeToolUse && ev.Delta.ToolUse != nil &&
			ev.Delta.ToolUse.Name != "" && !announcedTools[ev.Index] {
			announcedTools[ev.Index] = true
			_, data, err := EncodeOpenAIResponsesStreamEvent(&StreamEvent{
				Type:  StreamEventContentBlockStart,
				Index: ev.Index,
				Delta: &ContentPart{
					Type: ContentTypeToolUse,
					ToolUse: &ToolUseContent{
						ID:   ev.Delta.ToolUse.ID,
						Name: ev.Delta.ToolUse.Name,
					},
				},
			})
			if err == nil && data != nil {
				if err := sseWriter.WriteEvent("response.output_item.added", data); err != nil {
					return
				}
			}
		}
		if ev := result.Event; ev != nil && ev.Type == StreamEventContentBlockStart &&
			ev.Delta != nil && ev.Delta.Type == ContentTypeToolUse {
			announcedTools[ev.Index] = true
		}

		eventType, data, err := EncodeOpenAIResponsesStreamEvent(result.Event)
		if err != nil || data == nil {
			// Some IR events (e.g. usage-only deltas) have no OpenAI Responses
			// streaming representation, signalled either by an error or by nil
			// data. Skip them — their usage has already been accumulated above.
			continue
		}
		if err := sseWriter.WriteEvent(eventType, data); err != nil {
			break
		}
	}
	// OpenAI Responses API does NOT use a [DONE] sentinel
}

// writeOpenAIResponsesStreamError emits an in-band Responses `error` SSE event.
// Used when a stream fails after the HTTP 200 has already been committed.
func writeOpenAIResponsesStreamError(sseWriter *SSEWriter, errType, code, message string) {
	payload := map[string]any{
		"type":    "error",
		"message": message,
	}
	if errType != "" {
		payload["type"] = errType
	}
	if code != "" {
		payload["code"] = code
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	sseWriter.WriteEvent("error", data) //nolint:errcheck
}
