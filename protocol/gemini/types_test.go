package gemini

import (
	"encoding/json"
	"testing"
)

// --- Schema ---

// Gemini's Schema is the target of the JSON-Schema converter. Several keywords
// were added after they were found to be silently dropped, so the wire names and
// omitempty behaviour are pinned here.

func TestSchema_AllKeywordsRoundTrip(t *testing.T) {
	nullable := true
	minLen, maxLen := 2, 8
	minItems, maxItems := 1, 5
	min, max := 0.0, 100.0

	s := Schema{
		Type:        "OBJECT",
		Format:      "int32",
		Description: "d",
		Nullable:    &nullable,
		Properties: map[string]Schema{
			"a": {Type: "STRING", MinLength: &minLen, MaxLength: &maxLen, Pattern: "^a"},
			"n": {Type: "INTEGER", Enum: []string{"1", "2"}, Minimum: &min, Maximum: &max},
		},
		Required:         []string{"a"},
		Items:            &Schema{Type: "STRING"},
		MinItems:         &minItems,
		MaxItems:         &maxItems,
		AnyOf:            []Schema{{Type: "STRING"}, {Type: "NUMBER"}},
		PropertyOrdering: []string{"a", "n"},
	}

	out, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Schema
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if back.Type != "OBJECT" || back.Format != "int32" || back.Description != "d" {
		t.Errorf("scalars lost: %+v", back)
	}
	if back.Nullable == nil || !*back.Nullable {
		t.Errorf("Nullable = %v, want true", back.Nullable)
	}
	if len(back.Properties) != 2 {
		t.Fatalf("Properties = %v", back.Properties)
	}
	a := back.Properties["a"]
	if a.MinLength == nil || *a.MinLength != 2 || a.MaxLength == nil || *a.MaxLength != 8 {
		t.Errorf("string constraints lost: %+v", a)
	}
	if a.Pattern != "^a" {
		t.Errorf("Pattern = %q", a.Pattern)
	}
	n := back.Properties["n"]
	if len(n.Enum) != 2 || n.Enum[0] != "1" {
		t.Errorf("Enum = %v", n.Enum)
	}
	if n.Minimum == nil || *n.Minimum != 0 || n.Maximum == nil || *n.Maximum != 100 {
		t.Errorf("numeric bounds lost: %+v", n)
	}
	if back.MinItems == nil || *back.MinItems != 1 || back.MaxItems == nil || *back.MaxItems != 5 {
		t.Errorf("array bounds lost: %+v", back)
	}
	if len(back.AnyOf) != 2 {
		t.Errorf("AnyOf = %v", back.AnyOf)
	}
	if len(back.PropertyOrdering) != 2 || back.PropertyOrdering[0] != "a" {
		t.Errorf("PropertyOrdering = %v", back.PropertyOrdering)
	}
}

// Gemini rejects unknown and null schema fields, so a minimal schema must emit
// nothing but its type.
func TestSchema_MarshalOmitsUnsetKeywords(t *testing.T) {
	out, err := json.Marshal(Schema{Type: "STRING"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != `{"type":"STRING"}` {
		t.Errorf("minimal schema = %s, want {\"type\":\"STRING\"}", out)
	}
}

// A zero value must be distinguishable from "unset": minLength:0 and
// minimum:0 are meaningful constraints, which is why they are pointers.
func TestSchema_ZeroValuedConstraintsAreEmitted(t *testing.T) {
	zero := 0
	zeroF := 0.0
	out, err := json.Marshal(Schema{Type: "STRING", MinLength: &zero, Minimum: &zeroF})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(m["minLength"]) != "0" {
		t.Errorf("minLength = %s, want 0 (an explicit zero must be emitted)", m["minLength"])
	}
	if string(m["minimum"]) != "0" {
		t.Errorf("minimum = %s, want 0", m["minimum"])
	}
}

// Gemini serialises enum entries as strings even for INTEGER schemas, so the field
// must stay []string; a numeric enum on the wire is a decode error the caller needs
// to see rather than silently lose.
func TestSchema_EnumIsStringOnly(t *testing.T) {
	var s Schema
	if err := json.Unmarshal([]byte(`{"type":"INTEGER","enum":["1","2"]}`), &s); err != nil {
		t.Fatalf("unmarshal string enum: %v", err)
	}
	if len(s.Enum) != 2 {
		t.Errorf("Enum = %v", s.Enum)
	}
	if err := json.Unmarshal([]byte(`{"type":"INTEGER","enum":[1,2]}`), &s); err == nil {
		t.Error("numeric enum decoded silently; Gemini's wire format is string-only")
	}
}

// --- ThinkingConfig ---

// ThinkingBudget is a pointer because an explicit 0 is how the API is told to turn
// thinking off; with a plain int + omitempty that instruction vanished and thinking
// silently stayed on.
func TestThinkingConfig_ExplicitZeroBudgetIsEmitted(t *testing.T) {
	var tc ThinkingConfig
	tc.SetThinkingBudget(0)
	out, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(m["thinkingBudget"]) != "0" {
		t.Errorf("thinkingBudget = %s, want 0; an explicit disable must reach the API",
			m["thinkingBudget"])
	}
}

func TestThinkingConfig_UnsetBudgetIsOmitted(t *testing.T) {
	out, err := json.Marshal(ThinkingConfig{})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != "{}" {
		t.Errorf("empty config = %s, want {}", out)
	}
}

func TestThinkingConfig_Budget(t *testing.T) {
	var nilCfg *ThinkingConfig
	if got := nilCfg.Budget(); got != 0 {
		t.Errorf("nil.Budget() = %d, want 0", got)
	}
	if got := (&ThinkingConfig{}).Budget(); got != 0 {
		t.Errorf("unset Budget() = %d, want 0", got)
	}
	var tc ThinkingConfig
	tc.SetThinkingBudget(2048)
	if got := tc.Budget(); got != 2048 {
		t.Errorf("Budget() = %d, want 2048", got)
	}
}

func TestThinkingConfig_RoundTrip(t *testing.T) {
	include := true
	body := []byte(`{"thinkingBudget":1024,"includeThoughts":true,"thinkingLevel":"high"}`)
	var tc ThinkingConfig
	if err := json.Unmarshal(body, &tc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if tc.Budget() != 1024 {
		t.Errorf("Budget() = %d, want 1024", tc.Budget())
	}
	if tc.IncludeThoughts == nil || *tc.IncludeThoughts != include {
		t.Errorf("IncludeThoughts = %v", tc.IncludeThoughts)
	}
	if tc.ThinkingLevel != "high" {
		t.Errorf("ThinkingLevel = %q", tc.ThinkingLevel)
	}
}

// --- Part ---

// Part is a union: exactly one payload field should be set. An all-zero Part
// marshals to {} which Gemini rejects, which is why the converter guards against
// emitting one.
func TestPart_EmptyMarshalsToEmptyObject(t *testing.T) {
	out, err := json.Marshal(Part{})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != "{}" {
		t.Errorf("empty part = %s; the converter relies on this being detectable", out)
	}
}

func TestPart_UnionFieldsRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"text", `{"text":"hi"}`},
		{"thought", `{"text":"hmm","thought":true}`},
		{"inlineData", `{"inlineData":{"mimeType":"image/png","data":"aGk="}}`},
		{"functionCall", `{"functionCall":{"name":"f","args":{"a":1},"id":"c1"}}`},
		{"functionResponse", `{"functionResponse":{"name":"f","response":{"r":1},"id":"c1"}}`},
		{"fileData", `{"fileData":{"mimeType":"application/pdf","fileUri":"gs://x"}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var p Part
			if err := json.Unmarshal([]byte(tc.body), &p); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			out, err := json.Marshal(p)
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

// --- Request omitempty ---

// systemInstruction must be omitted entirely when absent; an explicit null is
// rejected.
func TestRequest_MarshalOmitsUnsetFields(t *testing.T) {
	req := Request{
		Contents: []Content{{Role: "user", Parts: []Part{{Text: "hi"}}}},
	}
	out, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, field := range []string{"systemInstruction", "tools", "toolConfig", "generationConfig"} {
		if _, ok := m[field]; ok {
			t.Errorf("optional field %q emitted for a minimal request: %s", field, out)
		}
	}
	if _, ok := m["contents"]; !ok {
		t.Errorf("contents is required but missing: %s", out)
	}
}
