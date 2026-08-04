package llmapimux

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCodecProtocols(t *testing.T) {
	tests := []struct {
		name     string
		codec    inboundCodec
		expected Protocol
	}{
		{"openai_chat", &openaiChatCodec{}, ProtocolOpenAIChat},
		{"openai_responses", &openaiResponsesCodec{}, ProtocolOpenAIResponses},
		{"anthropic", &anthropicCodec{}, ProtocolAnthropic},
		{"gemini", &geminiCodec{}, ProtocolGemini},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.codec.Protocol(); got != tt.expected {
				t.Errorf("Protocol() = %s, want %s", got, tt.expected)
			}
		})
	}
}

func TestCodecDecodeRequest_SetsInboundProtocol(t *testing.T) {
	tests := []struct {
		name     string
		codec    inboundCodec
		path     string
		body     []byte
		expected Protocol
	}{
		{
			name:     "openai_chat",
			codec:    &openaiChatCodec{},
			path:     "/v1/chat/completions",
			body:     []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`),
			expected: ProtocolOpenAIChat,
		},
		{
			name:     "openai_responses",
			codec:    &openaiResponsesCodec{},
			path:     "/v1/responses",
			body:     []byte(`{"model":"gpt-4o","input":"hi"}`),
			expected: ProtocolOpenAIResponses,
		},
		{
			name:     "anthropic",
			codec:    &anthropicCodec{},
			path:     "/v1/messages",
			body:     []byte(`{"model":"claude-sonnet-4-20250514","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`),
			expected: ProtocolAnthropic,
		},
		{
			name:     "gemini",
			codec:    &geminiCodec{},
			path:     "/v1/models/gemini-2.5-pro:generateContent",
			body:     []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`),
			expected: ProtocolGemini,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequest("POST", tt.path, nil)
			req, err := tt.codec.DecodeRequest(r, tt.body)
			if err != nil {
				t.Fatalf("DecodeRequest error: %v", err)
			}
			if req.InboundProtocol != tt.expected {
				t.Fatalf("InboundProtocol = %s, want %s", req.InboundProtocol, tt.expected)
			}
		})
	}
}

func TestCodecDecodeRequest_PostDecodeBehaviorContract(t *testing.T) {
	// Guardrail: codecs are protocol decoders only.
	// They must set InboundProtocol and should not eagerly populate RawExtra.
	tests := []struct {
		name  string
		codec inboundCodec
		path  string
		body  []byte
	}{
		{
			name:  "openai_chat",
			codec: &openaiChatCodec{},
			path:  "/v1/chat/completions",
			body:  []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"x_guardrail":{"k":"v"}}`),
		},
		{
			name:  "openai_responses",
			codec: &openaiResponsesCodec{},
			path:  "/v1/responses",
			body:  []byte(`{"model":"gpt-4o","input":"hi","x_guardrail":{"k":"v"}}`),
		},
		{
			name:  "anthropic",
			codec: &anthropicCodec{},
			path:  "/v1/messages",
			body:  []byte(`{"model":"claude-sonnet-4-20250514","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"x_guardrail":{"k":"v"}}`),
		},
		{
			name:  "gemini",
			codec: &geminiCodec{},
			path:  "/v1/models/gemini-2.5-pro:generateContent",
			body:  []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"x_guardrail":{"k":"v"}}`),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := httptest.NewRequest("POST", tt.path, nil)
			req, err := tt.codec.DecodeRequest(r, tt.body)
			if err != nil {
				t.Fatalf("DecodeRequest error: %v", err)
			}
			if req.InboundProtocol == "" {
				t.Fatal("InboundProtocol should be set during decode")
			}
			if req.RawExtra != nil {
				t.Fatal("RawExtra should not be eagerly populated by codec")
			}
		})
	}
}

func TestCodecKnownFields(t *testing.T) {
	tests := []struct {
		name  string
		codec inboundCodec
		key   string
	}{
		{"openai_chat", &openaiChatCodec{}, "model"},
		{"openai_responses", &openaiResponsesCodec{}, "model"},
		{"anthropic", &anthropicCodec{}, "model"},
		{"gemini", &geminiCodec{}, "contents"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kf := tt.codec.KnownFields()
			if !kf[tt.key] {
				t.Errorf("KnownFields missing %q", tt.key)
			}
		})
	}
}

func TestCodecExtractAPIKey(t *testing.T) {
	t.Run("openai_bearer", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		r.Header.Set("Authorization", "Bearer sk-test-123")
		key := (&openaiChatCodec{}).ExtractAPIKey(r)
		if key != "sk-test-123" {
			t.Errorf("got %q", key)
		}
	})
	t.Run("anthropic_x_api_key", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		r.Header.Set("x-api-key", "sk-ant-456")
		key := (&anthropicCodec{}).ExtractAPIKey(r)
		if key != "sk-ant-456" {
			t.Errorf("got %q", key)
		}
	})
	t.Run("anthropic_bearer", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		r.Header.Set("Authorization", "Bearer sk-ant-bearer-789")
		key := (&anthropicCodec{}).ExtractAPIKey(r)
		if key != "sk-ant-bearer-789" {
			t.Errorf("got %q", key)
		}
	})
	t.Run("anthropic_x_api_key_takes_precedence", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		r.Header.Set("x-api-key", "from-header")
		r.Header.Set("Authorization", "Bearer from-bearer")
		key := (&anthropicCodec{}).ExtractAPIKey(r)
		if key != "from-header" {
			t.Errorf("got %q, want x-api-key to take precedence", key)
		}
	})
	t.Run("anthropic_no_auth", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		key := (&anthropicCodec{}).ExtractAPIKey(r)
		if key != "" {
			t.Errorf("got %q, want empty", key)
		}
	})
	t.Run("gemini_header", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/", nil)
		r.Header.Set("x-goog-api-key", "gem-789")
		key := (&geminiCodec{}).ExtractAPIKey(r)
		if key != "gem-789" {
			t.Errorf("got %q", key)
		}
	})
	t.Run("gemini_query", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/?key=gem-query", nil)
		key := (&geminiCodec{}).ExtractAPIKey(r)
		if key != "gem-query" {
			t.Errorf("got %q", key)
		}
	})
}

func TestCodecDecodeRequest_DoesNotSetRawExtra(t *testing.T) {
	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false,"service_tier":"priority","seed":42}`)
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	codec := &openaiChatCodec{}

	req, err := codec.DecodeRequest(r, body)
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "gpt-4o" {
		t.Errorf("Model = %s", req.Model)
	}
	if req.InboundProtocol != ProtocolOpenAIChat {
		t.Errorf("InboundProtocol = %s", req.InboundProtocol)
	}
	if req.RawExtra != nil {
		t.Fatal("RawExtra should be nil before on-demand extraction")
	}
}

func TestCodecDecodeRequest_GeminiSetsStreamFromURL(t *testing.T) {
	body := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
	codec := &geminiCodec{}

	t.Run("streaming", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/v1/models/gemini-2.5-pro:streamGenerateContent", nil)
		req, err := codec.DecodeRequest(r, body)
		if err != nil {
			t.Fatal(err)
		}
		if !req.Stream {
			t.Error("expected Stream=true for :streamGenerateContent URL")
		}
	})

	t.Run("non-streaming", func(t *testing.T) {
		r := httptest.NewRequest("POST", "/v1/models/gemini-2.5-pro:generateContent", nil)
		req, err := codec.DecodeRequest(r, body)
		if err != nil {
			t.Fatal(err)
		}
		if req.Stream {
			t.Error("expected Stream=false for :generateContent URL")
		}
	})
}

func TestCodecWriteError(t *testing.T) {
	tests := []struct {
		name   string
		codec  inboundCodec
		substr string
	}{
		{"openai_chat", &openaiChatCodec{}, `"error"`},
		{"openai_responses", &openaiResponsesCodec{}, `"error"`},
		{"anthropic", &anthropicCodec{}, `"error"`},
		{"gemini", &geminiCodec{}, `"error"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			tt.codec.WriteError(w, http.StatusBadRequest, "test error")
			if w.Code != http.StatusBadRequest {
				t.Errorf("status = %d", w.Code)
			}
			if w.Header().Get("Content-Type") != "application/json" {
				t.Errorf("Content-Type = %s", w.Header().Get("Content-Type"))
			}
		})
	}
}

func TestCodecWriteError_IncludesParamField(t *testing.T) {
	tests := []struct {
		name      string
		codec     inboundCodec
		wantParam bool
	}{
		{"openai_chat", &openaiChatCodec{}, true},
		{"openai_responses", &openaiResponsesCodec{}, true},
		{"anthropic", &anthropicCodec{}, false},
		{"gemini", &geminiCodec{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			tt.codec.WriteError(w, http.StatusBadRequest, "test error")
			body := w.Body.String()
			if tt.wantParam {
				if !strings.Contains(body, `"param":null`) {
					t.Errorf("body missing param:null: %s", body)
				}
				if !strings.Contains(body, `"type":"invalid_request_error"`) {
					t.Errorf("body missing type field: %s", body)
				}
			} else {
				if strings.Contains(body, `"param"`) {
					t.Errorf("body should not contain param field: %s", body)
				}
			}
		})
	}
}

func TestOpenAIChatCodec_EncodeResponse_EmptyContentWithToolCalls(t *testing.T) {
	resp := &Response{
		ID:   "chatcmpl-empty",
		Model: "gpt-4o",
		Content: []ContentPart{
			{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`)}},
		},
		StopReason: StopReasonToolUse,
		Usage:      Usage{PromptTokens: 10, CompletionTokens: 5},
	}

	data, err := (&openaiChatCodec{}).EncodeResponse(resp)
	if err != nil {
		t.Fatalf("EncodeResponse error: %v", err)
	}
	if !strings.Contains(string(data), `"content":""`) {
		t.Errorf("content should be empty string when only tool_calls present, got:\n%s", string(data))
	}
}

func TestOpenAIChatCodec_EncodeResponse_NoContentNoToolCalls_OmitsContent(t *testing.T) {
	resp := &Response{
		ID:         "chatcmpl-none",
		Model:      "gpt-4o",
		StopReason: StopReasonEndTurn,
		Usage:      Usage{PromptTokens: 10, CompletionTokens: 0},
	}

	data, err := (&openaiChatCodec{}).EncodeResponse(resp)
	if err != nil {
		t.Fatalf("EncodeResponse error: %v", err)
	}
	if strings.Contains(string(data), `"content"`) {
		t.Errorf("content should be omitted when no text/tool_calls/reasoning, got:\n%s", string(data))
	}
}

func TestOpenAIChatCodec_DecodeResponse_PreservesCreated(t *testing.T) {
	body := `{"id":"chatcmpl-1","model":"gpt-4o","created":1700000000,"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`
	resp, err := DecodeOpenAIChatResponse([]byte(body))
	if err != nil {
		t.Fatalf("DecodeOpenAIChatResponse error: %v", err)
	}
	if resp.Created != 1700000000 {
		t.Errorf("Created = %d, want 1700000000", resp.Created)
	}
}

func TestOpenAIChatCodec_EncodeResponse_PreservesCreated(t *testing.T) {
	resp := &Response{
		ID:      "chatcmpl-1",
		Model:   "gpt-4o",
		Created: 1700000000,
		Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "ok"}}},
		StopReason: StopReasonEndTurn,
		Usage:      Usage{PromptTokens: 5, CompletionTokens: 1},
	}

	data, err := (&openaiChatCodec{}).EncodeResponse(resp)
	if err != nil {
		t.Fatalf("EncodeResponse error: %v", err)
	}
	if !strings.Contains(string(data), `"created":1700000000`) {
		t.Errorf("response missing created field, got:\n%s", string(data))
	}
}
