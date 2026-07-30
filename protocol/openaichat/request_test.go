package openaichat

import (
	"encoding/json"
	"testing"
)

// --- ChatMessage ---

// ChatMessage.Content is json.RawMessage because the API accepts both a plain
// string and an array of content parts; both forms must survive.
func TestChatMessage_ContentDualForm(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"string", `{"role":"user","content":"hi"}`},
		{"array", `{"role":"user","content":[{"type":"text","text":"hi"}]}`},
		{"array_multimodal", `{"role":"user","content":[{"type":"text","text":"hi"},` +
			`{"type":"image_url","image_url":{"url":"https://x/y.png","detail":"low"}}]}`},
		{"tool_array", `{"role":"tool","tool_call_id":"t","content":[{"type":"text","text":"r"}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var m ChatMessage
			if err := json.Unmarshal([]byte(tc.body), &m); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if len(m.Content) == 0 {
				t.Fatal("Content is empty")
			}
			out, err := json.Marshal(m)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var want, got map[string]any
			if err := json.Unmarshal([]byte(tc.body), &want); err != nil {
				t.Fatalf("unmarshal want: %v", err)
			}
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatalf("unmarshal got: %v", err)
			}
			for k := range want {
				if _, ok := got[k]; !ok {
					t.Errorf("field %q lost on round-trip: %s", k, out)
				}
			}
		})
	}
}

// A message with no content (an assistant turn that only makes tool calls) must
// omit the field rather than emit null, which some providers reject.
func TestChatMessage_OmitsAbsentContent(t *testing.T) {
	m := ChatMessage{
		Role:      "assistant",
		ToolCalls: []ToolCall{{ID: "t", Type: "function", Function: ToolCallFunction{Name: "f", Arguments: "{}"}}},
	}
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := raw["content"]; ok {
		t.Errorf("content emitted for a tool-call-only assistant message: %s", out)
	}
	if _, ok := raw["tool_calls"]; !ok {
		t.Errorf("tool_calls missing: %s", out)
	}
}

// reasoning_signature and reasoning_redacted are gateway-added fields that let
// Anthropic thinking blocks survive a trip through the Chat message history.
// Providers ignore unknown message fields, so emitting them is safe.
func TestChatMessage_ReasoningFieldsRoundTrip(t *testing.T) {
	reasoning := "thought"
	sig := "sig-abc"
	redacted := "enc-xyz"
	m := ChatMessage{
		Role:               "assistant",
		ReasoningContent:   &reasoning,
		ReasoningSignature: &sig,
		ReasoningRedacted:  &redacted,
	}
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back ChatMessage
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.ReasoningContent == nil || *back.ReasoningContent != reasoning {
		t.Errorf("ReasoningContent = %v", back.ReasoningContent)
	}
	if back.ReasoningSignature == nil || *back.ReasoningSignature != sig {
		t.Errorf("ReasoningSignature = %v", back.ReasoningSignature)
	}
	if back.ReasoningRedacted == nil || *back.ReasoningRedacted != redacted {
		t.Errorf("ReasoningRedacted = %v", back.ReasoningRedacted)
	}

	// They must be absent on a message that carries no reasoning.
	out, err = json.Marshal(ChatMessage{Role: "user"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, absent := range []string{"reasoning_content", "reasoning_signature", "reasoning_redacted"} {
		if _, ok := raw[absent]; ok {
			t.Errorf("field %q emitted for a plain user message: %s", absent, out)
		}
	}
}

// --- ToolCall.Index ---

// Streaming clients key argument fragments by tool_call index, so index 0 must be
// emitted rather than omitted — otherwise every fragment of the first tool call
// looks like it has no index at all.
func TestToolCall_IndexZeroIsEmitted(t *testing.T) {
	out, err := json.Marshal(ToolCall{Index: 0, ID: "t", Type: "function"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(raw["index"]) != "0" {
		t.Errorf("index = %s, want 0; streaming clients key fragments by index", raw["index"])
	}
}

// --- ChatResponseJSONSchema.Strict ---

// OpenAI requires response_format.json_schema.name, and strict must round-trip so
// a same-protocol passthrough does not silently relax the schema.
func TestChatResponseJSONSchema_StrictRoundTrip(t *testing.T) {
	body := []byte(`{"name":"Answer","strict":true,"schema":{"type":"object"}}`)
	var s ChatResponseJSONSchema
	if err := json.Unmarshal(body, &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if s.Name != "Answer" {
		t.Errorf("Name = %q", s.Name)
	}
	if !s.Strict {
		t.Error("Strict = false, want true")
	}
	out, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(raw["strict"]) != "true" {
		t.Errorf("strict = %s, want true", raw["strict"])
	}
	if string(raw["name"]) != `"Answer"` {
		t.Errorf("name = %s, want \"Answer\"", raw["name"])
	}
}

// --- ChatRequest ---

// A minimal request must not emit nulls for optional fields.
func TestChatRequest_MarshalOmitsUnsetFields(t *testing.T) {
	req := ChatRequest{
		Model:    "gpt-4o",
		Messages: []ChatMessage{{Role: "user"}},
	}
	out, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, field := range []string{"tools", "tool_choice", "response_format",
		"temperature", "top_p", "max_tokens", "max_completion_tokens", "stop"} {
		if _, ok := raw[field]; ok {
			t.Errorf("optional field %q emitted for a minimal request: %s", field, out)
		}
	}
}

// The API accepts both max_tokens (legacy) and max_completion_tokens; both must
// decode so an inbound request using either is understood.
func TestChatRequest_BothMaxTokenFieldsDecode(t *testing.T) {
	var legacy ChatRequest
	if err := json.Unmarshal([]byte(`{"model":"m","max_tokens":100}`), &legacy); err != nil {
		t.Fatalf("unmarshal legacy: %v", err)
	}
	if legacy.MaxTokens == nil || *legacy.MaxTokens != 100 {
		t.Errorf("MaxTokens = %v, want 100", legacy.MaxTokens)
	}

	var modern ChatRequest
	if err := json.Unmarshal([]byte(`{"model":"m","max_completion_tokens":200}`), &modern); err != nil {
		t.Fatalf("unmarshal modern: %v", err)
	}
	if modern.MaxCompletionTokens == nil || *modern.MaxCompletionTokens != 200 {
		t.Errorf("MaxCompletionTokens = %v, want 200", modern.MaxCompletionTokens)
	}
}

// tool_choice is a union of a string and an object; keeping it as RawMessage means
// both forms round-trip untouched.
func TestChatRequest_ToolChoiceDualForm(t *testing.T) {
	for _, body := range []string{
		`{"model":"m","tool_choice":"auto"}`,
		`{"model":"m","tool_choice":"required"}`,
		`{"model":"m","tool_choice":{"type":"function","function":{"name":"f"}}}`,
	} {
		var req ChatRequest
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatalf("unmarshal %s: %v", body, err)
		}
		if len(req.ToolChoice) == 0 {
			t.Errorf("ToolChoice empty for %s", body)
		}
		out, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var want, got map[string]any
		if err := json.Unmarshal([]byte(body), &want); err != nil {
			t.Fatalf("unmarshal want: %v", err)
		}
		if err := json.Unmarshal(out, &got); err != nil {
			t.Fatalf("unmarshal got: %v", err)
		}
		if !jsonEqual(want["tool_choice"], got["tool_choice"]) {
			t.Errorf("tool_choice changed: got %v, want %v", got["tool_choice"], want["tool_choice"])
		}
	}
}

func jsonEqual(a, b any) bool {
	ab, err1 := json.Marshal(a)
	bb, err2 := json.Marshal(b)
	if err1 != nil || err2 != nil {
		return false
	}
	return string(ab) == string(bb)
}
