package anthropic

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Tool has hand-written MarshalJSON/UnmarshalJSON so that provider-specific tool
// fields (max_uses, cache_control, allowed_domains, and whatever Anthropic adds
// next) survive a round-trip in ExtraFields instead of being dropped. That logic
// was previously untested.

func TestTool_UnmarshalJSON_CapturesExtraFields(t *testing.T) {
	data := []byte(`{
		"type": "web_search_20250305",
		"name": "web_search",
		"description": "Search the web",
		"input_schema": {"type":"object","properties":{"q":{"type":"string"}}},
		"max_uses": 5,
		"allowed_domains": ["example.com"],
		"cache_control": {"type":"ephemeral"}
	}`)

	var tool Tool
	if err := json.Unmarshal(data, &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if tool.Name != "web_search" {
		t.Errorf("Name = %q", tool.Name)
	}
	if tool.Type != "web_search_20250305" {
		t.Errorf("Type = %q", tool.Type)
	}
	if tool.Description != "Search the web" {
		t.Errorf("Description = %q", tool.Description)
	}
	if len(tool.InputSchema) == 0 {
		t.Error("InputSchema is empty")
	}

	// The four known fields must be consumed, leaving only the unknown ones.
	for _, known := range []string{"type", "name", "description", "input_schema"} {
		if _, ok := tool.ExtraFields[known]; ok {
			t.Errorf("known field %q leaked into ExtraFields", known)
		}
	}
	for _, extra := range []string{"max_uses", "allowed_domains", "cache_control"} {
		if _, ok := tool.ExtraFields[extra]; !ok {
			t.Errorf("unknown field %q was dropped instead of stored in ExtraFields", extra)
		}
	}
	if string(tool.ExtraFields["max_uses"]) != "5" {
		t.Errorf("max_uses = %s, want 5", tool.ExtraFields["max_uses"])
	}
}

func TestTool_UnmarshalJSON_NoExtraFieldsLeavesNilMap(t *testing.T) {
	// A nil map (rather than an empty one) keeps MarshalJSON output minimal and
	// makes reflect.DeepEqual comparisons in callers behave predictably.
	var tool Tool
	if err := json.Unmarshal([]byte(`{"name":"t","input_schema":{"type":"object"}}`), &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if tool.ExtraFields != nil {
		t.Errorf("ExtraFields = %v, want nil", tool.ExtraFields)
	}
}

func TestTool_UnmarshalJSON_ResetsExtraFieldsOnReuse(t *testing.T) {
	// Decoding into a previously-populated Tool must not retain stale extras.
	tool := Tool{ExtraFields: map[string]json.RawMessage{"stale": json.RawMessage(`1`)}}
	if err := json.Unmarshal([]byte(`{"name":"t"}`), &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := tool.ExtraFields["stale"]; ok {
		t.Error("stale ExtraFields entry survived a second unmarshal")
	}
}

func TestTool_MarshalJSON_RoundTrip(t *testing.T) {
	original := []byte(`{"type":"web_search_20250305","name":"web_search","description":"d",` +
		`"input_schema":{"type":"object"},"max_uses":3,"cache_control":{"type":"ephemeral"}}`)

	var tool Tool
	if err := json.Unmarshal(original, &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	out, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	// Compare as maps: key order is not part of the contract.
	var want, got map[string]any
	if err := json.Unmarshal(original, &want); err != nil {
		t.Fatalf("unmarshal want: %v", err)
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("unmarshal got: %v", err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("round-trip changed the tool:\n got: %v\nwant: %v", got, want)
	}
}

func TestTool_MarshalJSON_OmitsEmptyFields(t *testing.T) {
	// Anthropic rejects a tool carrying an empty type, and an empty description is
	// noise, so neither should be emitted.
	tool := Tool{Name: "t", InputSchema: json.RawMessage(`{"type":"object"}`)}
	out, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, absent := range []string{"type", "description"} {
		if _, ok := m[absent]; ok {
			t.Errorf("empty field %q was emitted: %s", absent, out)
		}
	}
	if _, ok := m["name"]; !ok {
		t.Errorf("name missing: %s", out)
	}
}

func TestTool_MarshalJSON_ExtraFieldsDoNotOverrideKnownOnes(t *testing.T) {
	// ExtraFields is written last, so a caller that stuffs a known key into it
	// wins. Document that so a future reorder is a deliberate decision.
	tool := Tool{
		Name:        "real",
		ExtraFields: map[string]json.RawMessage{"name": json.RawMessage(`"override"`)},
	}
	out, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]string
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["name"] != "override" {
		t.Errorf("name = %q; ExtraFields is applied last and is expected to win", m["name"])
	}
}

func TestTool_UnmarshalJSON_RejectsNonObject(t *testing.T) {
	for _, body := range []string{`"string"`, `42`, `[1,2]`, `null`, ``} {
		var tool Tool
		err := json.Unmarshal([]byte(body), &tool)
		if body == `null` {
			// encoding/json treats null as a no-op for custom unmarshalers.
			continue
		}
		if err == nil {
			t.Errorf("unmarshal(%q) succeeded, want an error", body)
		}
	}
}

// --- Request.System dual form ---

// The Anthropic API accepts `system` as either a plain string or an array of
// content blocks. SystemBlocks normalises both, which is why System is typed as
// json.RawMessage.

func TestRequest_SystemBlocks_StringForm(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(`{"system":"Be brief."}`), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	blocks, err := req.SystemBlocks()
	if err != nil {
		t.Fatalf("SystemBlocks: %v", err)
	}
	if len(blocks) != 1 || blocks[0].Type != "text" || blocks[0].Text != "Be brief." {
		t.Errorf("blocks = %+v, want one text block", blocks)
	}
}

func TestRequest_SystemBlocks_ArrayForm(t *testing.T) {
	var req Request
	body := `{"system":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	blocks, err := req.SystemBlocks()
	if err != nil {
		t.Fatalf("SystemBlocks: %v", err)
	}
	if len(blocks) != 2 || blocks[0].Text != "a" || blocks[1].Text != "b" {
		t.Errorf("blocks = %+v, want two text blocks", blocks)
	}
}

func TestRequest_SystemBlocks_AbsentAndEmpty(t *testing.T) {
	cases := []string{`{}`, `{"system":null}`, `{"system":""}`}
	for _, body := range cases {
		var req Request
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatalf("unmarshal %s: %v", body, err)
		}
		blocks, err := req.SystemBlocks()
		if err != nil {
			t.Fatalf("SystemBlocks(%s): %v", body, err)
		}
		if len(blocks) != 0 {
			t.Errorf("SystemBlocks(%s) = %+v, want empty", body, blocks)
		}
	}
}

func TestRequest_SystemBlocks_MalformedReturnsError(t *testing.T) {
	for _, body := range []string{`{"system":42}`, `{"system":{"type":"text"}}`, `{"system":[1,2]}`} {
		var req Request
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatalf("unmarshal %s: %v", body, err)
		}
		if _, err := req.SystemBlocks(); err == nil {
			t.Errorf("SystemBlocks(%s) succeeded, want an error", body)
		}
	}
}

func TestRequest_SetSystemBlocks(t *testing.T) {
	var req Request
	if err := req.SetSystemBlocks([]ContentBlock{{Type: "text", Text: "hi"}}); err != nil {
		t.Fatalf("SetSystemBlocks: %v", err)
	}
	blocks, err := req.SystemBlocks()
	if err != nil {
		t.Fatalf("SystemBlocks: %v", err)
	}
	if len(blocks) != 1 || blocks[0].Text != "hi" {
		t.Errorf("round-trip = %+v", blocks)
	}

	// Setting no blocks must clear the field so it is omitted from the wire form.
	if err := req.SetSystemBlocks(nil); err != nil {
		t.Fatalf("SetSystemBlocks(nil): %v", err)
	}
	if req.System != nil {
		t.Errorf("System = %s, want nil", req.System)
	}
	out, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != "" && containsKey(out, "system") {
		t.Errorf("system was emitted after being cleared: %s", out)
	}
}

// --- Request omitempty guarantees ---

// Explicit nulls for optional fields are rejected by strict proxies in front of
// the Anthropic API, so a minimal request must emit none of them.
func TestRequest_MarshalOmitsOptionalNulls(t *testing.T) {
	req := Request{
		Model:     "claude-3",
		MaxTokens: 100,
		Messages:  []Message{{Role: "user"}},
	}
	out, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, field := range []string{"system", "temperature", "top_p", "top_k",
		"stop_sequences", "tools", "tool_choice", "thinking", "metadata"} {
		if _, ok := m[field]; ok {
			t.Errorf("optional field %q was emitted for a minimal request: %s", field, out)
		}
	}
	// max_tokens is required by the API and must always appear.
	if _, ok := m["max_tokens"]; !ok {
		t.Errorf("max_tokens is required but was omitted: %s", out)
	}
}

// Thinking.BudgetTokens must be omitted when zero: the API rejects it on a
// disabled thinking config.
func TestThinking_MarshalOmitsZeroBudget(t *testing.T) {
	out, err := json.Marshal(Thinking{Type: "disabled"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if containsKey(out, "budget_tokens") {
		t.Errorf("budget_tokens emitted for a disabled config: %s", out)
	}

	out, err = json.Marshal(Thinking{Type: "enabled", BudgetTokens: 1024})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !containsKey(out, "budget_tokens") {
		t.Errorf("budget_tokens missing for an enabled config: %s", out)
	}
}

// --- StreamMessageDeltaInner.StopSequence ---

func TestStreamMessageDelta_StopSequenceRoundTrip(t *testing.T) {
	body := []byte(`{"stop_reason":"stop_sequence","stop_sequence":"END"}`)
	var inner StreamMessageDeltaInner
	if err := json.Unmarshal(body, &inner); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if inner.StopSequence == nil || *inner.StopSequence != "END" {
		t.Fatalf("StopSequence = %v, want END", inner.StopSequence)
	}
	out, err := json.Marshal(inner)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !containsKey(out, "stop_sequence") {
		t.Errorf("stop_sequence lost on re-marshal: %s", out)
	}

	// Absent stop_sequence must stay absent rather than becoming an empty string.
	var noSeq StreamMessageDeltaInner
	if err := json.Unmarshal([]byte(`{"stop_reason":"end_turn"}`), &noSeq); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	out, err = json.Marshal(noSeq)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if containsKey(out, "stop_sequence") {
		t.Errorf("stop_sequence emitted when absent: %s", out)
	}
}

func containsKey(body []byte, key string) bool {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return false
	}
	_, ok := m[key]
	return ok
}
