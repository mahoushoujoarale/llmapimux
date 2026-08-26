package llmapimux

import (
	"encoding/json"
	"strings"
	"testing"
)

// These tests cover conversion behaviour that affects prompt-cache hit rate and
// request validity. Each one pins down a specific regression:
//
//   - cache_control breakpoints must survive an Anthropic round-trip.
//   - Synthetic Gemini tool-call IDs must be identical across identical requests.
//   - An assistant turn must never encode to a Chat message with no content and
//     no tool_calls.
//   - Interleaved tool_result / text ordering inside one turn must be preserved.
//   - A mid-conversation system message must stay where it was, not be hoisted.

func TestAnthropicRoundTrip_PreservesCacheControl(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4",
		"max_tokens": 1024,
		"system": [
			{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
		],
		"tools": [
			{"name": "get_weather", "description": "d", "input_schema": {"type":"object"},
			 "cache_control": {"type": "ephemeral"}}
		],
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "TURN1", "cache_control": {"type": "ephemeral"}}
			]},
			{"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
			{"role": "user", "content": [
				{"type": "text", "text": "TURN2", "cache_control": {"type": "ephemeral"}}
			]}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("DecodeAnthropicRequest: %v", err)
	}
	out, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("EncodeAnthropicRequest: %v", err)
	}

	// Four breakpoints in (system, tool, two user turns), four expected out.
	if got, want := strings.Count(string(out), `"cache_control"`), 4; got != want {
		t.Errorf("cache_control count = %d, want %d\nbody: %s", got, want, out)
	}

	// Each breakpoint must still be attached to the block it marked, not floated
	// to some other position.
	var decoded struct {
		System []struct {
			Text         string          `json:"text"`
			CacheControl json.RawMessage `json:"cache_control"`
		} `json:"system"`
		Messages []struct {
			Content []struct {
				Text         string          `json:"text"`
				CacheControl json.RawMessage `json:"cache_control"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("unmarshal encoded body: %v", err)
	}
	if len(decoded.System) != 1 || len(decoded.System[0].CacheControl) == 0 {
		t.Errorf("system block lost its cache_control: %+v", decoded.System)
	}
	if len(decoded.Messages) != 3 {
		t.Fatalf("message count = %d, want 3", len(decoded.Messages))
	}
	for _, idx := range []int{0, 2} {
		blocks := decoded.Messages[idx].Content
		if len(blocks) != 1 || len(blocks[0].CacheControl) == 0 {
			t.Errorf("messages[%d] lost its cache_control: %+v", idx, blocks)
		}
	}
	// The un-marked assistant turn must not gain one.
	if blocks := decoded.Messages[1].Content; len(blocks) == 1 && len(blocks[0].CacheControl) != 0 {
		t.Errorf("assistant turn gained an unexpected cache_control: %s", blocks[0].CacheControl)
	}
}

func TestAnthropicRoundTrip_DoesNotInventCacheControl(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("DecodeAnthropicRequest: %v", err)
	}
	out, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("EncodeAnthropicRequest: %v", err)
	}
	if strings.Contains(string(out), "cache_control") {
		t.Errorf("cache_control appeared without being requested: %s", out)
	}
}

// TestAnthropicBlockExtra_DoesNotLeakCrossProtocol checks that a breakpoint
// picked up from Anthropic is not replayed onto a protocol that has no such
// concept, where it would be an unknown field.
func TestAnthropicBlockExtra_DoesNotLeakCrossProtocol(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4",
		"max_tokens": 16,
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
		]}]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("DecodeAnthropicRequest: %v", err)
	}

	chat, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("EncodeOpenAIChatRequest: %v", err)
	}
	if strings.Contains(string(chat), "cache_control") {
		t.Errorf("cache_control leaked into Chat Completions body: %s", chat)
	}

	_, gem, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("EncodeGeminiRequest: %v", err)
	}
	if strings.Contains(string(gem), "cache_control") {
		t.Errorf("cache_control leaked into Gemini body: %s", gem)
	}
}

func TestGeminiSyntheticToolIDs_AreDeterministic(t *testing.T) {
	const path = "/v1/models/gemini-2.5-pro:generateContent"
	body := []byte(`{
		"contents": [
			{"role": "user", "parts": [{"text": "weather?"}]},
			{"role": "model", "parts": [
				{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
				{"functionCall": {"name": "get_weather", "args": {"city": "NY"}}}
			]},
			{"role": "user", "parts": [
				{"functionResponse": {"name": "get_weather", "response": {"t": 1}}},
				{"functionResponse": {"name": "get_weather", "response": {"t": 2}}}
			]}
		]
	}`)

	encodeOnce := func() string {
		req, err := DecodeGeminiRequest(path, body)
		if err != nil {
			t.Fatalf("DecodeGeminiRequest: %v", err)
		}
		out, err := EncodeAnthropicRequest(req)
		if err != nil {
			t.Fatalf("EncodeAnthropicRequest: %v", err)
		}
		return string(out)
	}

	first := encodeOnce()
	for i := 0; i < 5; i++ {
		if got := encodeOnce(); got != first {
			t.Fatalf("run %d differs from run 0, cache prefix is unstable\nrun0: %s\nrun%d: %s",
				i+1, first, i+1, got)
		}
	}
}

// TestGeminiSyntheticToolIDs_DistinguishParallelCalls guards against the
// determinism fix collapsing two distinct calls onto one ID, which would make
// tool results ambiguous.
func TestGeminiSyntheticToolIDs_DistinguishParallelCalls(t *testing.T) {
	const path = "/v1/models/gemini-2.5-pro:generateContent"
	// Same name and identical args: only the ordinal separates them.
	body := []byte(`{
		"contents": [
			{"role": "model", "parts": [
				{"functionCall": {"name": "ping", "args": {}}},
				{"functionCall": {"name": "ping", "args": {}}}
			]}
		]
	}`)

	req, err := DecodeGeminiRequest(path, body)
	if err != nil {
		t.Fatalf("DecodeGeminiRequest: %v", err)
	}
	if len(req.Messages) != 1 || len(req.Messages[0].Content) != 2 {
		t.Fatalf("unexpected IR shape: %+v", req.Messages)
	}
	a := req.Messages[0].Content[0].ToolUse.ID
	b := req.Messages[0].Content[1].ToolUse.ID
	if a == "" || b == "" {
		t.Fatalf("empty synthetic IDs: %q %q", a, b)
	}
	if a == b {
		t.Errorf("parallel calls collapsed onto the same ID: %q", a)
	}
}

func TestOpenAIChat_AssistantMessageNeverContentless(t *testing.T) {
	// An assistant turn whose only part has no Chat representation. Without the
	// fix this encodes to a bare {"role":"assistant"}, which providers reject.
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "hi"}},
			}},
			{Role: RoleAssistant, Content: []ContentPart{
				{Type: ContentTypeRedactedThinking, RedactedThinking: &RedactedThinkingContent{Data: "opaque"}},
			}},
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "again"}},
			}},
		},
	}

	out, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("EncodeOpenAIChatRequest: %v", err)
	}

	var decoded struct {
		Messages []struct {
			Role      string          `json:"role"`
			Content   json.RawMessage `json:"content"`
			ToolCalls json.RawMessage `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for i, m := range decoded.Messages {
		if len(m.Content) == 0 && len(m.ToolCalls) == 0 {
			t.Errorf("messages[%d] (role %q) has neither content nor tool_calls: %s", i, m.Role, out)
		}
	}
}

// TestOpenAIChat_AssistantToolCallOnlyKeepsNilContent verifies the fix did not
// start emitting an empty string alongside tool_calls, which is a valid but
// different payload and would change the cached prefix.
func TestOpenAIChat_AssistantToolCallOnlyKeepsNilContent(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleAssistant, Content: []ContentPart{
				{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{
					ID: "call_1", Name: "f", Arguments: json.RawMessage(`{}`),
				}},
			}},
		},
	}
	out, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("EncodeOpenAIChatRequest: %v", err)
	}
	var decoded struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(decoded.Messages) != 1 {
		t.Fatalf("message count = %d, want 1", len(decoded.Messages))
	}
	if _, ok := decoded.Messages[0]["content"]; ok {
		t.Errorf("tool-call-only assistant message gained a content field: %s", out)
	}
	if _, ok := decoded.Messages[0]["tool_calls"]; !ok {
		t.Errorf("tool_calls missing: %s", out)
	}
}

func TestOpenAIChat_PreservesInterleavedToolResultOrdering(t *testing.T) {
	// Anthropic permits a user turn that interleaves tool results with text.
	// The text usually qualifies the results, so the order must be kept.
	body := []byte(`{
		"model": "claude-sonnet-4",
		"max_tokens": 64,
		"messages": [
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "call_1", "name": "f", "input": {}}
			]},
			{"role": "user", "content": [
				{"type": "text", "text": "BEFORE"},
				{"type": "tool_result", "tool_use_id": "call_1", "content": "RESULT"},
				{"type": "text", "text": "AFTER"}
			]}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("DecodeAnthropicRequest: %v", err)
	}
	out, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("EncodeOpenAIChatRequest: %v", err)
	}

	var decoded struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Expected: assistant(tool_call), user["BEFORE"], tool(RESULT), user["AFTER"].
	var roles []string
	for _, m := range decoded.Messages {
		roles = append(roles, m.Role)
	}
	want := []string{"assistant", "user", "tool", "user"}
	if len(roles) != len(want) {
		t.Fatalf("roles = %v, want %v\nbody: %s", roles, want, out)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("roles = %v, want %v\nbody: %s", roles, want, out)
		}
	}

	beforeIdx := strings.Index(string(out), "BEFORE")
	resultIdx := strings.Index(string(out), "RESULT")
	afterIdx := strings.Index(string(out), "AFTER")
	if !(beforeIdx < resultIdx && resultIdx < afterIdx) {
		t.Errorf("ordering not preserved: BEFORE@%d RESULT@%d AFTER@%d\nbody: %s",
			beforeIdx, resultIdx, afterIdx, out)
	}
}

func TestOpenAIChat_MidConversationSystemStaysInPlace(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": "LEADING"},
			{"role": "user", "content": "q1"},
			{"role": "assistant", "content": "a1"},
			{"role": "system", "content": "MIDWAY"},
			{"role": "user", "content": "q2"}
		]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("DecodeOpenAIChatRequest: %v", err)
	}

	// The leading system message is hoisted; the mid-conversation one is not.
	if len(req.SystemPrompt) != 1 || req.SystemPrompt[0].Text.Text != "LEADING" {
		t.Errorf("SystemPrompt = %+v, want only LEADING", req.SystemPrompt)
	}
	var roles []Role
	for _, m := range req.Messages {
		roles = append(roles, m.Role)
	}
	want := []Role{RoleUser, RoleAssistant, RoleSystem, RoleUser}
	if len(roles) != len(want) {
		t.Fatalf("IR roles = %v, want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("IR roles = %v, want %v", roles, want)
		}
	}

	out, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("EncodeOpenAIChatRequest: %v", err)
	}
	// MIDWAY must come after a1, not be merged into the leading system message.
	leadingIdx := strings.Index(string(out), "LEADING")
	a1Idx := strings.Index(string(out), "a1")
	midIdx := strings.Index(string(out), "MIDWAY")
	if leadingIdx == -1 || a1Idx == -1 || midIdx == -1 {
		t.Fatalf("missing content in encoded body: %s", out)
	}
	if !(leadingIdx < a1Idx && a1Idx < midIdx) {
		t.Errorf("MIDWAY was hoisted: LEADING@%d a1@%d MIDWAY@%d\nbody: %s",
			leadingIdx, a1Idx, midIdx, out)
	}
}

// TestMidConversationSystem_EncodesValidlyForAllProtocols checks that the new
// RoleSystem message does not produce a role that a target API would reject.
func TestMidConversationSystem_EncodesValidlyForAllProtocols(t *testing.T) {
	req := &Request{
		Model:     "m",
		MaxTokens: 64,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "q1"}},
			}},
			{Role: RoleSystem, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "MIDWAY"}},
			}},
		},
	}

	anth, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("EncodeAnthropicRequest: %v", err)
	}
	var anthDecoded struct {
		Messages []struct{ Role string } `json:"messages"`
	}
	if err := json.Unmarshal(anth, &anthDecoded); err != nil {
		t.Fatalf("unmarshal anthropic: %v", err)
	}
	for i, m := range anthDecoded.Messages {
		if m.Role != "user" && m.Role != "assistant" {
			t.Errorf("anthropic messages[%d].role = %q, not accepted by the API", i, m.Role)
		}
	}

	_, gem, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("EncodeGeminiRequest: %v", err)
	}
	var gemDecoded struct {
		Contents []struct{ Role string } `json:"contents"`
	}
	if err := json.Unmarshal(gem, &gemDecoded); err != nil {
		t.Fatalf("unmarshal gemini: %v", err)
	}
	for i, c := range gemDecoded.Contents {
		if c.Role != "user" && c.Role != "model" {
			t.Errorf("gemini contents[%d].role = %q, not accepted by the API", i, c.Role)
		}
	}
	if !strings.Contains(string(gem), "MIDWAY") {
		t.Errorf("gemini body dropped the mid-conversation instruction: %s", gem)
	}

	resp, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("EncodeOpenAIResponsesRequest: %v", err)
	}
	if !strings.Contains(string(resp), "MIDWAY") {
		t.Errorf("responses body dropped the mid-conversation instruction: %s", resp)
	}
}

func TestOpenAIResponses_MidConversationSystemStaysInPlace(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "developer", "content": "LEADING"},
			{"type": "message", "role": "user", "content": "q1"},
			{"type": "message", "role": "assistant", "content": "a1"},
			{"type": "message", "role": "developer", "content": "MIDWAY"},
			{"type": "message", "role": "user", "content": "q2"}
		]
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("DecodeOpenAIResponsesRequest: %v", err)
	}
	if len(req.SystemPrompt) != 1 || req.SystemPrompt[0].Text.Text != "LEADING" {
		t.Errorf("SystemPrompt = %+v, want only LEADING", req.SystemPrompt)
	}

	var roles []Role
	for _, m := range req.Messages {
		roles = append(roles, m.Role)
	}
	want := []Role{RoleUser, RoleAssistant, RoleSystem, RoleUser}
	if len(roles) != len(want) {
		t.Fatalf("IR roles = %v, want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("IR roles = %v, want %v", roles, want)
		}
	}

	out, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("EncodeOpenAIResponsesRequest: %v", err)
	}
	a1Idx := strings.Index(string(out), "a1")
	midIdx := strings.Index(string(out), "MIDWAY")
	if a1Idx == -1 || midIdx == -1 {
		t.Fatalf("missing content in encoded body: %s", out)
	}
	if a1Idx > midIdx {
		t.Errorf("MIDWAY was hoisted ahead of a1: %s", out)
	}
}

// TestOpenAIChat_LeadingSystemRunStillConsolidated confirms the documented
// _consolidate_system_messages behaviour is unchanged for the common case where
// every system message precedes the conversation.
func TestOpenAIChat_LeadingSystemRunStillConsolidated(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": "S1"},
			{"role": "developer", "content": "S2"},
			{"role": "user", "content": "q"}
		]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("DecodeOpenAIChatRequest: %v", err)
	}
	if len(req.SystemPrompt) != 2 {
		t.Fatalf("SystemPrompt len = %d, want 2: %+v", len(req.SystemPrompt), req.SystemPrompt)
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != RoleUser {
		t.Errorf("Messages = %+v, want a single user turn", req.Messages)
	}

	out, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("EncodeOpenAIChatRequest: %v", err)
	}
	var decoded struct {
		Messages []struct{ Role string } `json:"messages"`
	}
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(decoded.Messages) != 2 {
		t.Fatalf("message count = %d, want 2 (one system + one user): %s", len(decoded.Messages), out)
	}
	if decoded.Messages[0].Role != "developer" {
		t.Errorf("messages[0].role = %q, want developer", decoded.Messages[0].Role)
	}
}
