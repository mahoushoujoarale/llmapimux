package llmapimux

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// openaiAnnotationWire is the wire format for OpenAI annotation JSON objects.
type openaiAnnotationWire struct {
	Type       string `json:"type,omitempty"`
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	StartIndex *int   `json:"start_index,omitempty"`
	EndIndex   *int   `json:"end_index,omitempty"`
}

// defaultJSONSchemaName is used when a request carries a JSON Schema but no
// schema name. OpenAI's Chat (response_format.json_schema.name) and Responses
// (text.format.name) APIs both require the field, while Anthropic and Gemini have
// no equivalent, so a cross-protocol request would otherwise be rejected.
const defaultJSONSchemaName = "response"

// decodeOpenAIStop decodes the "stop" field which can be a string or array of strings.
// Shared by OpenAI Chat and OpenAI Responses decoders.
func decodeOpenAIStop(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		return []string{s}, nil
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, err
	}
	return arr, nil
}

// parseOpenAIAnnotations parses OpenAI annotation raw JSON objects into IR Citation structs.
// Shared by OpenAI Chat and OpenAI Responses decoders.
func parseOpenAIAnnotations(raw []json.RawMessage) ([]Citation, error) {
	citations := make([]Citation, 0, len(raw))
	for _, r := range raw {
		var wire openaiAnnotationWire
		if err := json.Unmarshal(r, &wire); err != nil {
			return nil, fmt.Errorf("unmarshal annotation: %w", err)
		}
		citations = append(citations, Citation{
			Kind:  wire.Type,
			URL:   wire.URL,
			Title: wire.Title,
			Start: wire.StartIndex,
			End:   wire.EndIndex,
		})
	}
	return citations, nil
}

// encodeOpenAIAnnotations converts IR Citation structs to OpenAI annotation wire format.
// Shared by OpenAI Chat and OpenAI Responses encoders.
func encodeOpenAIAnnotations(citations []Citation) []json.RawMessage {
	result := make([]json.RawMessage, 0, len(citations))
	for _, c := range citations {
		wire := openaiAnnotationWire{
			Type:       c.Kind,
			URL:        c.URL,
			Title:      c.Title,
			StartIndex: c.Start,
			EndIndex:   c.End,
		}
		data, _ := json.Marshal(wire)
		result = append(result, data)
	}
	return result
}

// generateRequestID returns a UUID v4 string.
func generateRequestID() string {
	var uuid [16]byte
	rand.Read(uuid[:])
	uuid[6] = (uuid[6] & 0x0f) | 0x40
	uuid[8] = (uuid[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		uuid[0:4], uuid[4:6], uuid[6:8], uuid[8:10], uuid[10:16])
}

// toolResultText extracts and concatenates all text parts from a ToolResultContent.
// Shared by OpenAI Chat, OpenAI Responses, and Gemini encoders.
//
// Non-text parts (images, documents) have no representation in the flat string
// tool-result formats used by OpenAI and Gemini, so a short placeholder is
// substituted to keep the model aware that content was present rather than
// silently emitting nothing.
func toolResultText(result *ToolResultContent) string {
	var texts []string
	for _, c := range result.Content {
		switch c.Type {
		case ContentTypeText:
			if c.Text != nil {
				texts = append(texts, c.Text.Text)
			}
		case ContentTypeImage:
			texts = append(texts, "[image omitted: not representable in this tool result format]")
		case ContentTypeDocument:
			texts = append(texts, "[document omitted: not representable in this tool result format]")
		}
	}
	return strings.Join(texts, "")
}

// toolErrorPrefix is prepended to tool-result text when ToolResultContent.IsError
// is set and the target protocol has no structured error flag. Anthropic carries
// is_error natively; OpenAI Chat/Responses and Gemini do not, so without this the
// model cannot tell a successful result from a failed one.
const toolErrorPrefix = "Error: "

// toolResultTextWithError renders tool-result content for protocols that lack a
// structured is_error flag, marking failures inline so the signal survives.
func toolResultTextWithError(result *ToolResultContent) string {
	text := toolResultText(result)
	if result.IsError {
		return toolErrorPrefix + text
	}
	return text
}

// hasMediaContent returns true if any message or system prompt contains image or document parts.
func hasMediaContent(req *Request) bool {
	for _, part := range req.SystemPrompt {
		if part.Type == ContentTypeImage || part.Type == ContentTypeDocument {
			return true
		}
	}
	for _, msg := range req.Messages {
		for _, part := range msg.Content {
			if part.Type == ContentTypeImage || part.Type == ContentTypeDocument {
				return true
			}
		}
	}
	return false
}

// parseBearerAPIKey extracts a Bearer token from Authorization header.
func parseBearerAPIKey(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if strings.HasPrefix(auth, "Bearer ") {
		return auth[len("Bearer "):]
	}
	return ""
}

// decodeRequestWithInboundProtocol decodes with fn and stamps inbound protocol.
func decodeRequestWithInboundProtocol(body []byte, inbound Protocol, fn func([]byte) (*Request, error)) (*Request, error) {
	req, err := fn(body)
	if err != nil {
		return nil, err
	}
	req.InboundProtocol = inbound
	return req, nil
}
