package llmapimux

import (
	"encoding/json"
	"net/http"
)

// openaiChatCodec implements inboundCodec for the OpenAI Chat Completions protocol.
type openaiChatCodec struct{}

func (c *openaiChatCodec) Protocol() Protocol {
	return ProtocolOpenAIChat
}

func (c *openaiChatCodec) KnownFields() map[string]bool {
	return openaiChatKnownFields
}

func (c *openaiChatCodec) ExtractAPIKey(r *http.Request) string {
	return parseBearerAPIKey(r)
}

func (c *openaiChatCodec) DecodeRequest(r *http.Request, body []byte) (*Request, error) {
	return decodeRequestWithInboundProtocol(body, ProtocolOpenAIChat, DecodeOpenAIChatRequest)
}

func (c *openaiChatCodec) WriteError(w http.ResponseWriter, statusCode int, msg string) {
	writeOpenAIError(w, statusCode, msg)
}

func (c *openaiChatCodec) EncodeResponse(resp *Response) ([]byte, error) {
	return EncodeOpenAIChatResponse(resp)
}

func (c *openaiChatCodec) WriteStreamingResponse(sseWriter *SSEWriter, ch <-chan StreamResult) {
	completed := false
	var accumulatedUsage Usage
	var lastStopReason StopReason
	var streamCreated int64
	var streamID string
	var streamModel string

	for result := range ch {
		if result.Err != nil {
			// The status code is already committed, so the failure has to be
			// reported in-band. Emit an OpenAI-shaped error chunk followed by
			// [DONE] so the client can tell a mid-stream failure from a clean
			// end of stream instead of just seeing the connection close.
			writeOpenAIChatStreamError(sseWriter, result.Err)
			return
		}

		// Accumulate usage from early events (e.g. Anthropic message_start
		// carries PromptTokens in StreamEventStart.Response.Usage).
		// OpenAI Chat only emits usage in the final stop chunk, so we
		// must defer it.
		if result.Event != nil {
			if result.Event.Usage != nil {
				mergeStreamUsage(&accumulatedUsage, result.Event.Usage)
			}
			if result.Event.Response != nil && result.Event.Response.Usage.PromptTokens != 0 {
				mergeStreamUsage(&accumulatedUsage, &result.Event.Response.Usage)
			}
			// Capture stop reasons that arrive on non-stop events.
			// Anthropic sends stop_reason in message_delta (StreamEventDelta),
			// but message_stop decodes to StreamEventStop with nil StopReason.
			if result.Event.StopReason != nil {
				lastStopReason = *result.Event.StopReason
			}
			// Capture ID/Model/Created from the start event for propagation.
			if result.Event.Type == StreamEventStart && result.Event.Response != nil {
				streamID = result.Event.Response.ID
				streamModel = result.Event.Response.Model
				streamCreated = result.Event.Response.Created
			}
		}

		// On stop event, inject accumulated usage and stop reason.
		if result.Event != nil && result.Event.Type == StreamEventStop {
			// Inject accumulated usage into the stop event if it has none
			// or only partial usage (e.g. only CompletionTokens from message_delta).
			if result.Event.Usage == nil && accumulatedUsage.PromptTokens != 0 {
				result.Event.Usage = &Usage{}
				*result.Event.Usage = accumulatedUsage
			} else if result.Event.Usage != nil && result.Event.Usage.PromptTokens == 0 && accumulatedUsage.PromptTokens != 0 {
				mergeStreamUsage(result.Event.Usage, &accumulatedUsage)
			}
			// Inject captured stop reason if the stop event has none.
			// Anthropic message_stop → StreamEventStop{StopReason: nil}
			// but the actual reason was in message_delta → StreamEventDelta.StopReason.
			if result.Event.StopReason == nil && lastStopReason != "" {
				result.Event.StopReason = &lastStopReason
			}
		}

		// An in-band IR error event (e.g. an Anthropic "error" SSE event) has no
		// native Chat Completions chunk shape. Surface it as an error payload plus
		// dropping it silently.
		if result.Event != nil && result.Event.Type == StreamEventError {
			msg := "upstream stream error"
			if result.Event.Error != nil && result.Event.Error.Message != "" {
				msg = result.Event.Error.Message
			}
			errType := "api_error"
			code := ""
			if result.Event.Error != nil {
				if result.Event.Error.Type != "" {
					errType = result.Event.Error.Type
				}
				code = result.Event.Error.Code
			}
			writeOpenAIChatStreamErrorPayload(sseWriter, errType, code, msg)
			return
		}

		// Propagate ID/Model/Created to delta and stop events so every chunk
		// carries these fields, matching OpenAI's streaming behavior.
		if result.Event != nil && result.Event.Response == nil &&
			(streamID != "" || streamModel != "" || streamCreated != 0) {
			result.Event.Response = &Response{
				ID:      streamID,
				Model:   streamModel,
				Created: streamCreated,
			}
		}

		data, err := EncodeOpenAIChatStreamChunk(result.Event)
		if err != nil {
			break
		}
		if data == nil {
			continue
		}
		if err := sseWriter.WriteData(data); err != nil {
			break
		}
		if result.Event != nil && result.Event.Type == StreamEventStop {
			completed = true
		}
	}

	if completed {
		// Write the sentinel only after a complete upstream stop event.
		sseWriter.WriteDone() //nolint:errcheck
	}
}

// writeOpenAIChatStreamError emits a transport-level failure as an in-band SSE
// error chunk followed by [DONE].
func writeOpenAIChatStreamError(sseWriter *SSEWriter, err error) {
	writeOpenAIChatStreamErrorPayload(sseWriter, "api_error", "", err.Error())
}

// writeOpenAIChatStreamErrorPayload emits an OpenAI-shaped error object on the
// SSE stream, then the [DONE] sentinel so clients terminate cleanly.
func writeOpenAIChatStreamErrorPayload(sseWriter *SSEWriter, errType, code, message string) {
	payload := map[string]any{
		"message": message,
		"type":    errType,
		"param":   nil,
	}
	if code != "" {
		payload["code"] = code
	} else {
		payload["code"] = nil
	}
	data, err := json.Marshal(map[string]any{"error": payload})
	if err != nil {
		return
	}
	if err := sseWriter.WriteData(data); err != nil {
		return
	}
	sseWriter.WriteDone() //nolint:errcheck
}

// writeOpenAIError writes an OpenAI-formatted error response (shared by Chat and Responses handlers).
func writeOpenAIError(w http.ResponseWriter, statusCode int, message string) {
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "invalid_request_error",
			"code":    nil,
			"param":   nil,
		},
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(body) //nolint:errcheck
}
