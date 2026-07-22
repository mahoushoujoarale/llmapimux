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

	for result := range ch {
		if result.Err != nil {
			break
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

// writeOpenAIError writes an OpenAI-formatted error response (shared by Chat and Responses handlers).
func writeOpenAIError(w http.ResponseWriter, statusCode int, message string) {
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "invalid_request_error",
			"code":    nil,
		},
	})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(body) //nolint:errcheck
}
