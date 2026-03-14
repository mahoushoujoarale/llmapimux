package llmapimux

import (
	"encoding/json"
	"testing"
)

func TestContentPartTextJSON(t *testing.T) {
	part := ContentPart{
		Type: ContentTypeText,
		Text: &TextContent{Text: "hello"},
	}
	data, err := json.Marshal(part)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ContentPart
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Type != ContentTypeText || decoded.Text.Text != "hello" {
		t.Fatalf("round-trip failed: got %+v", decoded)
	}
}

func TestContentPartToolUseJSON(t *testing.T) {
	part := ContentPart{
		Type: ContentTypeToolUse,
		ToolUse: &ToolUseContent{
			ID:        "call_123",
			Name:      "read_file",
			Arguments: json.RawMessage(`{"path":"/tmp/foo"}`),
		},
	}
	data, err := json.Marshal(part)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ContentPart
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.ToolUse.ID != "call_123" || decoded.ToolUse.Name != "read_file" {
		t.Fatalf("round-trip failed: got %+v", decoded)
	}
	if string(decoded.ToolUse.Arguments) != `{"path":"/tmp/foo"}` {
		t.Fatalf("arguments round-trip failed: got %s", decoded.ToolUse.Arguments)
	}
}

func TestRequestBasicFields(t *testing.T) {
	temp := 0.7
	req := Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}},
		},
		MaxTokens:   1024,
		Temperature: &temp,
		Stream:      true,
	}
	if req.Model != "gpt-4o" || req.MaxTokens != 1024 || *req.Temperature != 0.7 {
		t.Fatalf("unexpected: %+v", req)
	}
}

func TestToolChoiceAllowedToolNames(t *testing.T) {
	tc := ToolChoice{
		Type:             "tool",
		ToolName:         "search",
		AllowedToolNames: []string{"search", "read_file"},
	}
	data, err := json.Marshal(tc)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ToolChoice
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Type != "tool" {
		t.Errorf("Type = %q, want %q", decoded.Type, "tool")
	}
	if decoded.ToolName != "search" {
		t.Errorf("ToolName = %q, want %q", decoded.ToolName, "search")
	}
	if len(decoded.AllowedToolNames) != 2 {
		t.Fatalf("AllowedToolNames len = %d, want 2", len(decoded.AllowedToolNames))
	}
	if decoded.AllowedToolNames[0] != "search" || decoded.AllowedToolNames[1] != "read_file" {
		t.Errorf("AllowedToolNames = %v, want [search read_file]", decoded.AllowedToolNames)
	}
}

func TestToolChoiceAllowParallelCalls(t *testing.T) {
	trueVal := true
	falseVal := false

	// Test with true
	tc := ToolChoice{Type: "auto", AllowParallelCalls: &trueVal}
	data, err := json.Marshal(tc)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ToolChoice
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.AllowParallelCalls == nil {
		t.Fatal("AllowParallelCalls is nil, want non-nil")
	}
	if !*decoded.AllowParallelCalls {
		t.Errorf("AllowParallelCalls = false, want true")
	}

	// Test with false
	tc2 := ToolChoice{Type: "auto", AllowParallelCalls: &falseVal}
	data2, err := json.Marshal(tc2)
	if err != nil {
		t.Fatal(err)
	}
	var decoded2 ToolChoice
	if err := json.Unmarshal(data2, &decoded2); err != nil {
		t.Fatal(err)
	}
	if decoded2.AllowParallelCalls == nil {
		t.Fatal("AllowParallelCalls is nil, want non-nil")
	}
	if *decoded2.AllowParallelCalls {
		t.Errorf("AllowParallelCalls = true, want false")
	}
}

func TestToolChoiceAllowParallelCallsNilOmitted(t *testing.T) {
	// When AllowParallelCalls is nil it should be omitted from JSON.
	tc := ToolChoice{Type: "auto"}
	data, err := json.Marshal(tc)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if _, exists := m["allow_parallel_calls"]; exists {
		t.Error("allow_parallel_calls should be omitted from JSON when nil")
	}
}

func TestProviderExtensionsOnRequest(t *testing.T) {
	ext := ProviderExtensions{
		"openai/service_tier": json.RawMessage(`"auto"`),
		"anthropic/thinking":  json.RawMessage(`{"type":"enabled","budget_tokens":5000}`),
	}
	req := Request{
		Model:              "gpt-4o",
		ProviderExtensions: ext,
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Request
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.ProviderExtensions) != 2 {
		t.Fatalf("ProviderExtensions len = %d, want 2", len(decoded.ProviderExtensions))
	}
	if string(decoded.ProviderExtensions["openai/service_tier"]) != `"auto"` {
		t.Errorf("ProviderExtensions[openai/service_tier] = %s, want %q", decoded.ProviderExtensions["openai/service_tier"], `"auto"`)
	}
	if string(decoded.ProviderExtensions["anthropic/thinking"]) != `{"type":"enabled","budget_tokens":5000}` {
		t.Errorf("ProviderExtensions[anthropic/thinking] = %s", decoded.ProviderExtensions["anthropic/thinking"])
	}
}

func TestProviderExtensionsOnResponse(t *testing.T) {
	ext := ProviderExtensions{
		"gemini/safety_ratings": json.RawMessage(`[{"category":"HARM_CATEGORY_HATE","probability":"NEGLIGIBLE"}]`),
	}
	resp := Response{
		Model:              "gemini-2.5-pro",
		ProviderExtensions: ext,
	}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Response
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.ProviderExtensions) != 1 {
		t.Fatalf("ProviderExtensions len = %d, want 1", len(decoded.ProviderExtensions))
	}
	got := string(decoded.ProviderExtensions["gemini/safety_ratings"])
	want := `[{"category":"HARM_CATEGORY_HATE","probability":"NEGLIGIBLE"}]`
	if got != want {
		t.Errorf("ProviderExtensions[gemini/safety_ratings] = %s, want %s", got, want)
	}
}

func TestContentPartRefusalJSON(t *testing.T) {
	part := ContentPart{
		Type:    ContentTypeRefusal,
		Refusal: &RefusalContent{Refusal: "I cannot help with that"},
	}
	data, err := json.Marshal(part)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ContentPart
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Type != ContentTypeRefusal {
		t.Errorf("Type = %q, want %q", decoded.Type, ContentTypeRefusal)
	}
	if decoded.Refusal == nil || decoded.Refusal.Refusal != "I cannot help with that" {
		t.Fatalf("round-trip failed: got %+v", decoded)
	}
}

func TestStreamEventErrorJSON(t *testing.T) {
	se := StreamEvent{
		Type: StreamEventError,
		Error: &StreamError{
			Type:    "server_error",
			Code:    "rate_limit",
			Message: "Too many requests",
			Param:   "model",
		},
	}
	data, err := json.Marshal(se)
	if err != nil {
		t.Fatal(err)
	}
	var decoded StreamEvent
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Type != StreamEventError {
		t.Errorf("Type = %q, want %q", decoded.Type, StreamEventError)
	}
	if decoded.Error == nil {
		t.Fatal("Error is nil")
	}
	if decoded.Error.Type != "server_error" {
		t.Errorf("Error.Type = %q, want %q", decoded.Error.Type, "server_error")
	}
	if decoded.Error.Code != "rate_limit" {
		t.Errorf("Error.Code = %q, want %q", decoded.Error.Code, "rate_limit")
	}
	if decoded.Error.Message != "Too many requests" {
		t.Errorf("Error.Message = %q, want %q", decoded.Error.Message, "Too many requests")
	}
}

func TestStreamEventIncompleteDetailsJSON(t *testing.T) {
	sr := StopReasonMaxTokens
	se := StreamEvent{
		Type:              StreamEventStop,
		StopReason:        &sr,
		IncompleteDetails: &IncompleteDetails{Reason: "max_output_tokens"},
	}
	data, err := json.Marshal(se)
	if err != nil {
		t.Fatal(err)
	}
	var decoded StreamEvent
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.IncompleteDetails == nil {
		t.Fatal("IncompleteDetails is nil")
	}
	if decoded.IncompleteDetails.Reason != "max_output_tokens" {
		t.Errorf("IncompleteDetails.Reason = %q, want %q", decoded.IncompleteDetails.Reason, "max_output_tokens")
	}
}

func TestProviderExtensionsNilOmitted(t *testing.T) {
	// When ProviderExtensions is nil it should be omitted from JSON.
	req := Request{Model: "gpt-4o"}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if _, exists := m["provider_extensions"]; exists {
		t.Error("provider_extensions should be omitted from JSON when nil")
	}
}
