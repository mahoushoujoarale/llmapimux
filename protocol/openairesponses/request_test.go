package openairesponses

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Tool has hand-written MarshalJSON/UnmarshalJSON so that built-in tool
// configuration the struct does not model (web_search filters, file_search
// vector_store_ids, container settings, ...) survives a round-trip in ExtraFields
// instead of being dropped. That logic was previously untested.

func TestTool_UnmarshalJSON_CapturesExtraFields(t *testing.T) {
	data := []byte(`{
		"type": "file_search",
		"name": "search_docs",
		"description": "Search docs",
		"parameters": {"type":"object"},
		"strict": true,
		"vector_store_ids": ["vs_1"],
		"max_num_results": 10
	}`)

	var tool Tool
	if err := json.Unmarshal(data, &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if tool.Type != "file_search" || tool.Name != "search_docs" {
		t.Errorf("Type=%q Name=%q", tool.Type, tool.Name)
	}
	if tool.Description != "Search docs" {
		t.Errorf("Description = %q", tool.Description)
	}
	if !tool.Strict {
		t.Error("Strict = false, want true")
	}
	if len(tool.Parameters) == 0 {
		t.Error("Parameters is empty")
	}

	for _, known := range []string{"type", "name", "description", "parameters", "strict"} {
		if _, ok := tool.ExtraFields[known]; ok {
			t.Errorf("known field %q leaked into ExtraFields", known)
		}
	}
	for _, extra := range []string{"vector_store_ids", "max_num_results"} {
		if _, ok := tool.ExtraFields[extra]; !ok {
			t.Errorf("unknown field %q was dropped instead of stored in ExtraFields", extra)
		}
	}
}

func TestTool_UnmarshalJSON_NoExtraFieldsLeavesNilMap(t *testing.T) {
	var tool Tool
	if err := json.Unmarshal([]byte(`{"type":"function","name":"f"}`), &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if tool.ExtraFields != nil {
		t.Errorf("ExtraFields = %v, want nil", tool.ExtraFields)
	}
}

func TestTool_UnmarshalJSON_ResetsExtraFieldsOnReuse(t *testing.T) {
	tool := Tool{ExtraFields: map[string]json.RawMessage{"stale": json.RawMessage(`1`)}}
	if err := json.Unmarshal([]byte(`{"type":"function","name":"f"}`), &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := tool.ExtraFields["stale"]; ok {
		t.Error("stale ExtraFields entry survived a second unmarshal")
	}
}

func TestTool_MarshalJSON_RoundTrip(t *testing.T) {
	original := []byte(`{"type":"web_search","name":"web_search","description":"d",` +
		`"parameters":{"type":"object"},"strict":true,"search_context_size":"high"}`)

	var tool Tool
	if err := json.Unmarshal(original, &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	out, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

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

func TestTool_MarshalJSON_OmitsEmptyAndFalse(t *testing.T) {
	// strict=false is the default; emitting it explicitly is noise, and an empty
	// name is invalid for a function tool.
	tool := Tool{Type: "function", Name: "f"}
	out, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, absent := range []string{"description", "parameters", "strict"} {
		if _, ok := m[absent]; ok {
			t.Errorf("empty field %q was emitted: %s", absent, out)
		}
	}
}

func TestTool_UnmarshalJSON_RejectsNonObject(t *testing.T) {
	for _, body := range []string{`"string"`, `42`, `[1,2]`} {
		var tool Tool
		if err := json.Unmarshal([]byte(body), &tool); err == nil {
			t.Errorf("unmarshal(%q) succeeded, want an error", body)
		}
	}
}

func TestTool_UnmarshalJSON_RejectsWrongFieldTypes(t *testing.T) {
	// A wrong scalar type must be reported, not silently coerced.
	for _, body := range []string{
		`{"type":42}`,
		`{"name":[1]}`,
		`{"description":{"a":1}}`,
		`{"strict":"yes"}`,
	} {
		var tool Tool
		if err := json.Unmarshal([]byte(body), &tool); err == nil {
			t.Errorf("unmarshal(%s) succeeded, want an error", body)
		}
	}
}

// --- TextFormat / ContentPart additions ---

// TextFormat.Strict and ContentPart.Detail were added so that structured-output
// strictness and image detail survive a round-trip; verify the wire names.
func TestTextFormat_StrictRoundTrip(t *testing.T) {
	body := []byte(`{"type":"json_schema","name":"Answer","strict":true,"schema":{"type":"object"}}`)
	var f TextFormat
	if err := json.Unmarshal(body, &f); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !f.Strict {
		t.Error("Strict = false, want true")
	}
	if f.Name != "Answer" {
		t.Errorf("Name = %q, want Answer", f.Name)
	}
	out, err := json.Marshal(f)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(m["strict"]) != "true" {
		t.Errorf("strict = %s, want true", m["strict"])
	}
}

func TestContentPart_DetailRoundTrip(t *testing.T) {
	body := []byte(`{"type":"input_image","image_url":"https://x/y.png","detail":"high"}`)
	var p ContentPart
	if err := json.Unmarshal(body, &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if p.Detail != "high" {
		t.Errorf("Detail = %q, want high", p.Detail)
	}
	out, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(m["detail"]) != `"high"` {
		t.Errorf("detail = %s, want \"high\"", m["detail"])
	}

	// An absent detail must stay absent — OpenAI validates the enum.
	var noDetail ContentPart
	if err := json.Unmarshal([]byte(`{"type":"input_image","image_url":"u"}`), &noDetail); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	out, err = json.Marshal(noDetail)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// A fresh map: unmarshalling into a populated one merges rather than replaces.
	var m2 map[string]json.RawMessage
	if err := json.Unmarshal(out, &m2); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := m2["detail"]; ok {
		t.Errorf("detail emitted when absent: %s", out)
	}
}

// --- InputItem reasoning fields ---

// EncryptedContent and Signature let Anthropic redacted_thinking and thinking
// signatures survive a trip through the Responses input history.
func TestInputItem_ReasoningFieldsRoundTrip(t *testing.T) {
	item := InputItem{
		Type:             "reasoning",
		EncryptedContent: "enc-xyz",
		Signature:        "sig-abc",
		Summary:          []ReasoningSummary{{Type: "summary_text", Text: "thought"}},
	}
	out, err := json.Marshal(item)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back InputItem
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.EncryptedContent != "enc-xyz" {
		t.Errorf("EncryptedContent = %q", back.EncryptedContent)
	}
	if back.Signature != "sig-abc" {
		t.Errorf("Signature = %q", back.Signature)
	}
	if len(back.Summary) != 1 || back.Summary[0].Text != "thought" {
		t.Errorf("Summary = %+v", back.Summary)
	}

	// A plain message item must not emit the reasoning fields.
	out, err = json.Marshal(InputItem{Type: "message", Role: "user"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, absent := range []string{"encrypted_content", "signature", "summary"} {
		if _, ok := m[absent]; ok {
			t.Errorf("field %q emitted for a message item: %s", absent, out)
		}
	}
}

// --- InputItem.Output string-or-array tolerance ---

// function_call_output.output is a string on the Responses API, but clients
// (and the API's own output replayed back into input) may send an array of
// content parts instead. Unmarshalling must accept both.
func TestInputItem_OutputAcceptsStringAndArray(t *testing.T) {
	cases := []struct {
		name string
		json string
		want string
	}{
		{"string", `"42"`, "42"},
		{"null", `null`, ""},
		{"array one part", `[{"type":"output_text","text":"42"}]`, "42"},
		{"array multi parts", `[{"type":"output_text","text":"a"},{"type":"output_text","text":"b"}]`, "a\nb"},
		{"array non-text dropped", `[{"type":"input_image","image_url":"x"},{"type":"output_text","text":"ok"}]`, "ok"},
		{"array empty", `[]`, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var item InputItem
			if err := json.Unmarshal([]byte(`{"type":"function_call_output","call_id":"c1","output":`+c.json+`}`), &item); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if string(item.Output) != c.want {
				t.Errorf("Output = %q, want %q", item.Output, c.want)
			}
		})
	}
}

func TestInputItem_OutputStillMarshalsAsString(t *testing.T) {
	out, err := json.Marshal(InputItem{Type: "function_call_output", CallID: "c1", Output: "42"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(m["output"]) != `"42"` {
		t.Errorf("output = %s, want \"42\"", m["output"])
	}

	// An empty output must stay omitted (omitempty applies to the named string type).
	out, err = json.Marshal(InputItem{Type: "message", Role: "user"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m = nil
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := m["output"]; ok {
		t.Errorf("output emitted when empty: %s", out)
	}
}
