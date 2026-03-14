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
	for result := range ch {
		if result.Err != nil {
			// Cannot change status code at this point — just stop.
			break
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
