package llmapimux

import (
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGenerateRequestID(t *testing.T) {
	id := generateRequestID()
	if id == "" {
		t.Fatal("empty request ID")
	}
	// Should look like a UUID: 8-4-4-4-12
	parts := strings.Split(id, "-")
	if len(parts) != 5 {
		t.Errorf("unexpected format: %s", id)
	}
	// Two calls should produce different IDs
	id2 := generateRequestID()
	if id == id2 {
		t.Error("two calls returned the same ID")
	}
}

func TestHasMediaContent(t *testing.T) {
	textOnly := &Request{
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}},
		},
	}
	if hasMediaContent(textOnly) {
		t.Error("text-only request should not have media")
	}

	withImage := &Request{
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "look"}},
				{Type: ContentTypeImage, Image: &ImageContent{URL: "https://example.com/img.png"}},
			}},
		},
	}
	if !hasMediaContent(withImage) {
		t.Error("request with image should have media")
	}

	withDoc := &Request{
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeDocument, Document: &DocumentContent{URL: "https://example.com/doc.pdf"}},
			}},
		},
	}
	if !hasMediaContent(withDoc) {
		t.Error("request with document should have media")
	}

	withSystemImage := &Request{
		SystemPrompt: []ContentPart{
			{Type: ContentTypeImage, Image: &ImageContent{URL: "https://example.com/img.png"}},
		},
	}
	if !hasMediaContent(withSystemImage) {
		t.Error("system prompt with image should have media")
	}
}

func TestParseBearerAPIKey(t *testing.T) {
	t.Run("valid bearer", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		r.Header.Set("Authorization", "Bearer sk-test-123")
		if got := parseBearerAPIKey(r); got != "sk-test-123" {
			t.Fatalf("parseBearerAPIKey() = %q, want %q", got, "sk-test-123")
		}
	})

	t.Run("non bearer", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		r.Header.Set("Authorization", "Basic abc")
		if got := parseBearerAPIKey(r); got != "" {
			t.Fatalf("parseBearerAPIKey() = %q, want empty", got)
		}
	})

	t.Run("missing header", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		if got := parseBearerAPIKey(r); got != "" {
			t.Fatalf("parseBearerAPIKey() = %q, want empty", got)
		}
	})
}

func TestDecodeRequestWithInboundProtocol(t *testing.T) {
	decoded, err := decodeRequestWithInboundProtocol([]byte("ignored"), ProtocolOpenAIChat, func(_ []byte) (*Request, error) {
		return &Request{Model: "gpt-4o"}, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if decoded.InboundProtocol != ProtocolOpenAIChat {
		t.Fatalf("InboundProtocol = %s, want %s", decoded.InboundProtocol, ProtocolOpenAIChat)
	}

	wantErr := errors.New("decode failed")
	_, err = decodeRequestWithInboundProtocol([]byte("ignored"), ProtocolGemini, func(_ []byte) (*Request, error) {
		return nil, wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}
