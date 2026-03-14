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

	for result := range ch {
		if result.Err != nil {
			break
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
		// Write the [DONE] sentinel only after a complete upstream stop event.
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
