package llmapimux

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// This file covers negative and malformed input. Every decoder here is reachable
// directly from untrusted HTTP input, so none of them may panic — a panic in a
// decoder takes down the whole gateway process, not just one request. Before these
// tests the entire malformed-input surface was untested.

// malformedBodies is the shared corpus of hostile and degenerate request bodies.
type malformedBody struct {
	name       string
	body       string
	mustReject bool
}

func malformedBodies() []malformedBody {
	return []malformedBody{
		// codeflicker-fix: EDGE-Issue-004/tb3m3jp0fdxv42afyew5
		// These are structurally invalid for every request protocol, not merely
		// unusual inputs that a provider may choose to accept for compatibility.
		{"empty", ``, true},
		{"whitespace", `   `, true},
		{"not_json", `hello world`, true},
		{"truncated_object", `{"model":`, true},
		{"truncated_array", `{"messages":[`, true},
		{"json_null", `null`, true},
		{"json_array", `[1,2,3]`, true},
		{"json_string", `"just a string"`, true},
		{"json_number", `42`, true},
		{"json_bool", `true`, true},
		{"empty_object", `{}`, false},
		{"nested_nulls", `{"model":null,"messages":null,"tools":null,"tool_choice":null}`, false},
		{"wrong_types", `{"model":123,"messages":"not-an-array","max_tokens":"lots"}`, false},
		{"messages_wrong_element", `{"model":"m","messages":[42,"x",null]}`, false},
		{"content_wrong_type", `{"model":"m","messages":[{"role":"user","content":42}]}`, false},
		{"content_array_of_numbers", `{"model":"m","messages":[{"role":"user","content":[1,2,3]}]}`, false},
		{"unknown_content_type", `{"model":"m","messages":[{"role":"user","content":[{"type":"quantum_state","q":1}]}]}`, false},
		{"missing_role", `{"model":"m","messages":[{"content":"hi"}]}`, false},
		{"unknown_role", `{"model":"m","messages":[{"role":"wizard","content":"hi"}]}`, false},
		{"empty_messages", `{"model":"m","messages":[]}`, false},
		{"empty_content_array", `{"model":"m","messages":[{"role":"user","content":[]}]}`, false},
		{"tool_choice_no_tools", `{"model":"m","messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"tool","name":"ghost"}}`, false},
		{"tool_no_schema", `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"t"}]}`, false},
		{"tool_bad_schema", `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"t","input_schema":"not-an-object","parameters":"not-an-object"}]}`, false},
		{"negative_max_tokens", `{"model":"m","max_tokens":-5,"messages":[{"role":"user","content":"hi"}]}`, false},
		{"huge_max_tokens", `{"model":"m","max_tokens":99999999999,"messages":[{"role":"user","content":"hi"}]}`, false},
		{"bad_temperature", `{"model":"m","temperature":"hot","messages":[{"role":"user","content":"hi"}]}`, false},
		{"tool_result_no_id", `{"model":"m","messages":[{"role":"user","content":[{"type":"tool_result","content":"x"}]}]}`, false},
		{"tool_use_bad_input", `{"model":"m","messages":[{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":"not-an-object"}]}]}`, false},
		{"image_no_source", `{"model":"m","messages":[{"role":"user","content":[{"type":"image"}]}]}`, false},
		{"image_bad_base64", `{"model":"m","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"!!!not-base64!!!"}}]}]}`, false},
		{"deeply_nested_schema", `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"t","input_schema":` + deepSchema(60) + `}]}`, false},
		{"duplicate_keys", `{"model":"a","model":"b","messages":[{"role":"user","content":"hi"}]}`, false},
		{"gemini_shape", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, false},
		{"chat_shape", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, false},
		{"responses_shape", `{"model":"m","input":"hi"}`, false},
		{"anthropic_shape", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`, false},
	}
}

// deepSchema builds a nested JSON Schema `depth` levels deep, to check the
// recursive schema converters do not blow the stack on adversarial input.
func deepSchema(depth int) string {
	s := `{"type":"string"}`
	for i := 0; i < depth; i++ {
		s = `{"type":"object","properties":{"a":` + s + `}}`
	}
	return s
}

// TestMalformedInput_DecodersNeverPanic feeds the whole malformed corpus to every
// request decoder. A decoder may return an error or a degenerate-but-valid IR; it
// may never panic.
func TestMalformedInput_DecodersNeverPanic(t *testing.T) {
	decoders := []struct {
		name   string
		decode func(string) (*Request, error)
	}{
		{"anthropic", func(b string) (*Request, error) { return DecodeAnthropicRequest([]byte(b)) }},
		{"openai_chat", func(b string) (*Request, error) { return DecodeOpenAIChatRequest([]byte(b)) }},
		{"openai_responses", func(b string) (*Request, error) { return DecodeOpenAIResponsesRequest([]byte(b)) }},
		{"gemini", func(b string) (*Request, error) {
			return DecodeGeminiRequest("/v1beta/models/gemini-2.0:generateContent", []byte(b))
		}},
	}

	for _, d := range decoders {
		for _, tc := range malformedBodies() {
			t.Run(d.name+"/"+tc.name, func(t *testing.T) {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("decoder panicked on %s: %v", tc.name, r)
					}
				}()
				req, err := d.decode(tc.body)
				if tc.mustReject {
					if err == nil {
						t.Fatalf("decoder accepted input that must be rejected: %s", tc.body)
					}
					return
				}
				if err != nil {
					return // rejecting other malformed input is also permitted
				}
				if req == nil {
					t.Fatal("decoder returned nil request and nil error")
				}
				// A successfully decoded request must survive re-encoding into every
				// protocol without panicking, since that is what the gateway does next.
				assertRequestReencodesWithoutPanic(t, req)
			})
		}
	}
}

// assertRequestReencodesWithoutPanic re-encodes an IR request into all four
// protocols. Encoders are allowed to fail, but must not panic.
func assertRequestReencodesWithoutPanic(t *testing.T, req *Request) {
	t.Helper()
	if req.Model == "" {
		req.Model = "m"
	}
	encoders := []struct {
		name   string
		encode func(*Request) ([]byte, error)
	}{
		{"anthropic", EncodeAnthropicRequest},
		{"openai_chat", EncodeOpenAIChatRequest},
		{"openai_responses", EncodeOpenAIResponsesRequest},
		{"gemini", func(r *Request) ([]byte, error) {
			_, b, err := EncodeGeminiRequest(r)
			return b, err
		}},
	}
	for _, e := range encoders {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("%s encoder panicked: %v", e.name, r)
				}
			}()
			_, _ = e.encode(req)
		}()
	}
}

// TestMalformedInput_ResponseDecodersNeverPanic covers the upstream-response
// decode path, which is equally untrusted: a proxy or a misbehaving provider can
// return anything.
func TestMalformedInput_ResponseDecodersNeverPanic(t *testing.T) {
	bodies := []string{
		``, `   `, `not json`, `null`, `[]`, `{}`, `42`, `"str"`,
		`{"choices":null}`,
		`{"choices":[]}`,
		`{"choices":[{}]}`,
		`{"choices":[{"message":null}]}`,
		`{"choices":[{"message":{"content":42}}]}`,
		`{"choices":[{"message":{"tool_calls":[{"function":null}]}}]}`,
		`{"content":null}`,
		`{"content":[{"type":"unknown_block"}]}`,
		`{"content":[{"type":"tool_use","input":"not-an-object"}]}`,
		`{"candidates":null}`,
		`{"candidates":[]}`,
		`{"candidates":[{"content":null}]}`,
		`{"candidates":[{"content":{"parts":null}}]}`,
		`{"candidates":[{"content":{"parts":[{"inlineData":{"data":"!!!"}}]}}]}`,
		`{"output":null}`,
		`{"output":[{"type":"unknown_item"}]}`,
		`{"output":[{"type":"message","content":42}]}`,
		`{"usage":{"input_tokens":"many"}}`,
	}

	decoders := []struct {
		name   string
		decode func([]byte) (*Response, error)
	}{
		{"anthropic", DecodeAnthropicResponse},
		{"openai_chat", DecodeOpenAIChatResponse},
		{"openai_responses", DecodeOpenAIResponsesResponse},
		{"gemini", DecodeGeminiResponse},
	}

	for _, d := range decoders {
		for i, body := range bodies {
			t.Run(d.name+"/"+strings.NewReplacer("/", "_", " ", "_").Replace(truncateForName(body, i)), func(t *testing.T) {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("response decoder panicked on %q: %v", body, r)
					}
				}()
				resp, err := d.decode([]byte(body))
				if err != nil || resp == nil {
					return
				}
				assertResponseReencodesWithoutPanic(t, resp)
			})
		}
	}
}

func assertResponseReencodesWithoutPanic(t *testing.T, resp *Response) {
	t.Helper()
	encoders := []struct {
		name   string
		encode func(*Response) ([]byte, error)
	}{
		{"anthropic", EncodeAnthropicResponse},
		{"openai_chat", EncodeOpenAIChatResponse},
		{"openai_responses", EncodeOpenAIResponsesResponse},
		{"gemini", EncodeGeminiResponse},
	}
	for _, e := range encoders {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("%s response encoder panicked: %v", e.name, r)
				}
			}()
			_, _ = e.encode(resp)
		}()
	}
}

// TestMalformedInput_StreamDecodersNeverPanic covers the SSE chunk decoders, which
// see one untrusted fragment at a time.
func TestMalformedInput_StreamDecodersNeverPanic(t *testing.T) {
	chunks := []string{
		``, `   `, `not json`, `null`, `[]`, `{}`, `42`,
		`{"type":"content_block_delta"}`,
		`{"type":"content_block_delta","index":0,"delta":null}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"unknown_delta"}}`,
		`{"type":"content_block_start","index":0}`,
		`{"type":"message_delta"}`,
		`{"type":"message_delta","delta":null}`,
		`{"type":"unknown_event"}`,
		`{"choices":null}`,
		`{"choices":[{"delta":null}]}`,
		`{"choices":[{"delta":{"tool_calls":null}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"function":null}]}}]}`,
		`{"choices":[{"delta":{"content":42}}]}`,
		`{"candidates":null}`,
		`{"candidates":[{"content":{"parts":null}}]}`,
		`{"candidates":[{"finishReason":"WHAT_IS_THIS"}]}`,
		`{"type":"response.output_text.delta"}`,
		`{"type":"response.output_text.delta","delta":42}`,
		`{"type":"response.completed","response":null}`,
		`{"type":"response.unknown_event"}`,
	}

	for i, chunk := range chunks {
		name := truncateForName(chunk, i)
		t.Run("anthropic/"+name, func(t *testing.T) {
			mustNotPanic(t, func() { _, _ = DecodeAnthropicStreamEvent("", []byte(chunk)) })
		})
		t.Run("openai_chat/"+name, func(t *testing.T) {
			mustNotPanic(t, func() { _, _ = DecodeOpenAIChatStreamChunks([]byte(chunk)) })
		})
		t.Run("openai_responses/"+name, func(t *testing.T) {
			mustNotPanic(t, func() { _, _ = DecodeOpenAIResponsesStreamEvent("", []byte(chunk)) })
		})
		t.Run("gemini/"+name, func(t *testing.T) {
			mustNotPanic(t, func() { _, _ = DecodeGeminiStreamChunk([]byte(chunk)) })
		})
	}
}

// TestMalformedInput_StreamEncodersNeverPanic covers the encode direction with
// degenerate IR events, which arise from cross-protocol conversion.
func TestMalformedInput_StreamEncodersNeverPanic(t *testing.T) {
	events := []*StreamEvent{
		nil,
		{},
		{Type: StreamEventStart},
		{Type: StreamEventDelta},
		{Type: StreamEventDelta, Delta: &ContentPart{}},
		{Type: StreamEventDelta, Delta: &ContentPart{Type: ContentTypeText}},
		{Type: StreamEventDelta, Delta: &ContentPart{Type: ContentTypeToolUse}},
		{Type: StreamEventDelta, Delta: &ContentPart{Type: ContentTypeThinking}},
		{Type: StreamEventDelta, Delta: &ContentPart{Type: "made_up_type"}},
		{Type: StreamEventContentBlockStart},
		{Type: StreamEventContentBlockStart, Delta: &ContentPart{Type: ContentTypeToolUse}},
		{Type: StreamEventContentBlockStop},
		{Type: StreamEventStop},
		{Type: StreamEventError},
		{Type: StreamEventError, Error: &StreamError{}},
		{Type: "totally_unknown_event"},
		{Type: StreamEventDelta, Index: -1, Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: "x"}}},
	}

	for i, ev := range events {
		name := "event_" + itoa(i)
		t.Run("anthropic/"+name, func(t *testing.T) {
			mustNotPanic(t, func() { _, _, _ = EncodeAnthropicStreamEvent(ev) })
		})
		t.Run("openai_chat/"+name, func(t *testing.T) {
			mustNotPanic(t, func() { _, _ = EncodeOpenAIChatStreamChunk(ev) })
		})
		t.Run("openai_responses/"+name, func(t *testing.T) {
			mustNotPanic(t, func() { _, _, _ = EncodeOpenAIResponsesStreamEvent(ev) })
		})
		t.Run("gemini/"+name, func(t *testing.T) {
			mustNotPanic(t, func() { _, _ = EncodeGeminiStreamChunk(ev) })
		})
	}
}

// TestMalformedInput_HandlersReturnErrorNotPanic drives malformed bodies through
// the real HTTP handlers. Bad input must produce a 4xx/5xx in the inbound
// protocol's error shape, never a panic or a 200 with garbage.
func TestMalformedInput_HandlersReturnErrorNotPanic(t *testing.T) {
	// The upstream should never be reached for a request that fails to decode.
	var upstreamHits int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	paths := map[Protocol]string{
		ProtocolAnthropic:       "/v1/messages",
		ProtocolOpenAIChat:      "/v1/chat/completions",
		ProtocolOpenAIResponses: "/v1/responses",
		ProtocolGemini:          "/v1beta/models/gemini-2.0:generateContent",
	}

	for _, inbound := range allProtocols {
		for _, tc := range malformedBodies() {
			t.Run(protocolName(inbound)+"/"+tc.name, func(t *testing.T) {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("handler panicked on %s: %v", tc.name, r)
					}
				}()

				beforeHits := upstreamHits
				mux := NewMux(&staticRouter{result: RouteResult{
					Protocol: ProtocolOpenAIChat,
					BaseURL:  upstream.URL,
					APIKey:   "k",
					Model:    "gpt-4o",
				}})
				req := httptest.NewRequest("POST", paths[inbound], strings.NewReader(tc.body))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				inboundHandler(mux, inbound).ServeHTTP(w, req)

				if tc.mustReject {
					if w.Code < 400 {
						t.Errorf("status = %d, want >=400 for input that must be rejected", w.Code)
					}
					if upstreamHits != beforeHits {
						t.Errorf("must-reject input reached upstream: hits before=%d after=%d", beforeHits, upstreamHits)
					}
					assertErrorShape(t, inbound, w.Body.Bytes())
					return
				}

				if w.Code == http.StatusOK {
					// Accepting the body is fine for shapes that happen to be valid,
					// but the response must still be well-formed for the client.
					if _, err := extractResponseText(inbound, w.Body.Bytes()); err != nil {
						t.Errorf("200 with a malformed %s response: %v\n%s",
							protocolName(inbound), err, w.Body.String())
					}
					return
				}
				if w.Code < 400 {
					t.Errorf("status = %d, want 200 or >=400", w.Code)
				}
				// The error must be shaped for the inbound protocol so SDKs can parse it.
				assertErrorShape(t, inbound, w.Body.Bytes())
			})
		}
	}
}

// assertErrorShape checks an error body matches the inbound protocol's envelope.
func assertErrorShape(t *testing.T, p Protocol, body []byte) {
	t.Helper()
	s := string(body)
	if strings.TrimSpace(s) == "" {
		t.Error("error response body is empty")
		return
	}
	if !strings.Contains(s, `"error"`) {
		t.Errorf("%s error response has no \"error\" object: %s", protocolName(p), s)
	}
	if p == ProtocolAnthropic && !strings.Contains(s, `"type":"error"`) {
		t.Errorf("anthropic error response must set type=error: %s", s)
	}
}

// TestMalformedInput_UnsupportedMethodAndPath covers the non-POST and wrong-path
// cases, which must not reach the decoder at all.
func TestMalformedInput_UnsupportedMethod(t *testing.T) {
	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat, BaseURL: "http://127.0.0.1:1", APIKey: "k", Model: "m",
	}})
	for _, inbound := range allProtocols {
		for _, method := range []string{"GET", "PUT", "DELETE", "PATCH"} {
			t.Run(protocolName(inbound)+"/"+method, func(t *testing.T) {
				req := httptest.NewRequest(method, "/v1/messages", nil)
				w := httptest.NewRecorder()
				func() {
					defer func() {
						if r := recover(); r != nil {
							t.Fatalf("handler panicked on %s: %v", method, r)
						}
					}()
					inboundHandler(mux, inbound).ServeHTTP(w, req)
				}()
				if w.Code == http.StatusOK {
					t.Errorf("%s %s returned 200", method, "/v1/messages")
				}
			})
		}
	}
}

// TestMalformedInput_HugeAndPathologicalBodies guards against pathological but
// syntactically valid input.
func TestMalformedInput_PathologicalBodies(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"many_messages", `{"model":"m","max_tokens":10,"messages":[` +
			strings.TrimSuffix(strings.Repeat(`{"role":"user","content":"x"},`, 500), ",") + `]}`},
		{"long_text", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"` +
			strings.Repeat("a", 100000) + `"}]}`},
		{"many_content_parts", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[` +
			strings.TrimSuffix(strings.Repeat(`{"type":"text","text":"x"},`, 500), ",") + `]}]}`},
		{"many_tools", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],"tools":[` +
			strings.TrimSuffix(strings.Repeat(`{"name":"t","input_schema":{"type":"object"}},`, 200), ",") + `]}`},
		{"deep_schema", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"x"}],` +
			`"tools":[{"name":"t","input_schema":` + deepSchema(200) + `}]}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mustNotPanic(t, func() {
				req, err := DecodeAnthropicRequest([]byte(tc.body))
				if err != nil || req == nil {
					return
				}
				assertRequestReencodesWithoutPanic(t, req)
			})
		})
	}
}

// --- helpers ---

func mustNotPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panicked: %v", r)
		}
	}()
	fn()
}

// truncateForName renders a body fragment as a short, stable subtest name.
func truncateForName(s string, i int) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "empty_" + itoa(i)
	}
	if len(s) > 24 {
		s = s[:24]
	}
	return itoa(i) + "_" + strings.NewReplacer(
		" ", "_", `"`, "", "{", "", "}", "", "[", "", "]", "",
		":", "-", ",", "-", "/", "_",
	).Replace(s)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	if neg {
		return "-" + string(b)
	}
	return string(b)
}
