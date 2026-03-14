package llmapimux

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestJsonFieldNames(t *testing.T) {
	type sample struct {
		Name  string `json:"name"`
		Value int    `json:"value,omitempty"`
		Skip  string `json:"-"`
		NoTag string
	}
	fields := jsonFieldNames(reflect.TypeOf(sample{}))
	if !fields["name"] {
		t.Error("missing 'name'")
	}
	if !fields["value"] {
		t.Error("missing 'value'")
	}
	if fields["Skip"] || fields["-"] {
		t.Error("should not include skipped field")
	}
	if fields["NoTag"] || fields[""] {
		t.Error("should not include untagged field")
	}
}

func TestCaptureRawExtra(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		knownFields map[string]bool
		wantNil     bool
		want        map[string]string
	}{
		{
			name:        "only known fields",
			body:        `{"model":"gpt-4o","temperature":0.7}`,
			knownFields: map[string]bool{"model": true, "temperature": true},
			wantNil:     true,
		},
		{
			name:        "unknown scalar",
			body:        `{"model":"gpt-4o","seed":42}`,
			knownFields: map[string]bool{"model": true},
			want:        map[string]string{"seed": `42`},
		},
		{
			name:        "unknown nested object",
			body:        `{"model":"gpt-4o","metadata":{"trace_id":"abc","nested":{"k":1}}}`,
			knownFields: map[string]bool{"model": true},
			want:        map[string]string{"metadata": `{"trace_id":"abc","nested":{"k":1}}`},
		},
		{
			name:        "unknown array",
			body:        `{"model":"gpt-4o","tags":["a",{"b":2}]}`,
			knownFields: map[string]bool{"model": true},
			want:        map[string]string{"tags": `["a",{"b":2}]`},
		},
		{
			name:        "invalid json",
			body:        `{"model":"gpt-4o",`,
			knownFields: map[string]bool{"model": true},
			wantNil:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extras := captureRawExtra([]byte(tt.body), tt.knownFields)
			if tt.wantNil {
				if extras != nil {
					t.Fatalf("extras = %v, want nil", extras)
				}
				return
			}
			if extras == nil {
				t.Fatal("extras is nil")
			}
			if len(extras) != len(tt.want) {
				t.Fatalf("len(extras) = %d, want %d", len(extras), len(tt.want))
			}
			for k, want := range tt.want {
				got, ok := extras[k]
				if !ok {
					t.Fatalf("missing key %q", k)
				}
				if string(got) != want {
					t.Fatalf("extras[%q] = %s, want %s", k, string(got), want)
				}
			}
		})
	}
}

func TestMergeRawExtra(t *testing.T) {
	tests := []struct {
		name        string
		encoded     string
		extras      map[string]json.RawMessage
		knownFields map[string]bool
		wantExact   string
		wantFields  map[string]string
	}{
		{
			name:      "empty extras returns original",
			encoded:   `{"model":"gpt-4o"}`,
			extras:    nil,
			wantExact: `{"model":"gpt-4o"}`,
		},
		{
			name:    "merge unknown fields",
			encoded: `{"model":"gpt-4o","temperature":0.7}`,
			extras: map[string]json.RawMessage{
				"seed": json.RawMessage(`42`),
			},
			knownFields: map[string]bool{"model": true, "temperature": true},
			wantFields: map[string]string{
				"model":       `"gpt-4o"`,
				"temperature": `0.7`,
				"seed":        `42`,
			},
		},
		{
			name:    "skip overwriting known fields",
			encoded: `{"model":"gpt-4o","temperature":0.7}`,
			extras: map[string]json.RawMessage{
				"temperature": json.RawMessage(`0.1`),
				"seed":        json.RawMessage(`42`),
			},
			knownFields: map[string]bool{"model": true, "temperature": true},
			wantFields: map[string]string{
				"model":       `"gpt-4o"`,
				"temperature": `0.7`,
				"seed":        `42`,
			},
		},
		{
			name:      "invalid encoded json returns original bytes",
			encoded:   `{"model":"gpt-4o",`,
			extras:    map[string]json.RawMessage{"seed": json.RawMessage(`42`)},
			wantExact: `{"model":"gpt-4o",`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := mergeRawExtra([]byte(tt.encoded), tt.extras, tt.knownFields)
			if err != nil {
				t.Fatalf("mergeRawExtra error: %v", err)
			}
			if tt.wantExact != "" {
				if string(result) != tt.wantExact {
					t.Fatalf("result = %s, want %s", string(result), tt.wantExact)
				}
				return
			}

			var got map[string]json.RawMessage
			if err := json.Unmarshal(result, &got); err != nil {
				t.Fatalf("unmarshal result: %v", err)
			}
			if len(got) != len(tt.wantFields) {
				t.Fatalf("len(fields) = %d, want %d", len(got), len(tt.wantFields))
			}
			for k, want := range tt.wantFields {
				if string(got[k]) != want {
					t.Fatalf("field %q = %s, want %s", k, string(got[k]), want)
				}
			}
		})
	}
}

func TestRawExtra_CaptureMergePreservationEquivalence(t *testing.T) {
	known := map[string]bool{"model": true, "messages": true, "stream": true}
	original := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false,"service_tier":"priority","seed":42,"metadata":{"trace":"abc"}}`)

	extras := captureRawExtra(original, known)
	if extras == nil {
		t.Fatal("extras is nil")
	}

	encodedKnownOnly := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":false}`)
	merged, err := mergeRawExtra(encodedKnownOnly, extras, known)
	if err != nil {
		t.Fatalf("mergeRawExtra error: %v", err)
	}

	var got map[string]json.RawMessage
	if err := json.Unmarshal(merged, &got); err != nil {
		t.Fatalf("unmarshal merged: %v", err)
	}
	var want map[string]json.RawMessage
	if err := json.Unmarshal(original, &want); err != nil {
		t.Fatalf("unmarshal original: %v", err)
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merged json differs from original\n got=%v\nwant=%v", got, want)
	}
}

func TestKnownFieldsVarsExist(t *testing.T) {
	// Verify that knownFields are populated from the protocol structs.
	if len(openaiChatKnownFields) == 0 {
		t.Error("openaiChatKnownFields is empty")
	}
	if !openaiChatKnownFields["model"] {
		t.Error("openaiChatKnownFields should contain 'model'")
	}
	if len(anthropicKnownFields) == 0 {
		t.Error("anthropicKnownFields is empty")
	}
	if len(geminiKnownFields) == 0 {
		t.Error("geminiKnownFields is empty")
	}
	if len(openaiResponsesKnownFields) == 0 {
		t.Error("openaiResponsesKnownFields is empty")
	}
}

func TestMergeOutboundExtra(t *testing.T) {
	tests := []struct {
		name       string
		encoded    string
		extras     map[string]json.RawMessage
		wantFields map[string]string
		wantErr    bool
	}{
		{
			name:       "nil extras returns original",
			encoded:    `{"model":"gpt-4o"}`,
			extras:     nil,
			wantFields: map[string]string{"model": `"gpt-4o"`},
		},
		{
			name:       "empty extras returns original",
			encoded:    `{"model":"gpt-4o"}`,
			extras:     map[string]json.RawMessage{},
			wantFields: map[string]string{"model": `"gpt-4o"`},
		},
		{
			name:    "single field injection",
			encoded: `{"model":"gpt-4o"}`,
			extras: map[string]json.RawMessage{
				"service_tier": json.RawMessage(`"priority"`),
			},
			wantFields: map[string]string{
				"model":        `"gpt-4o"`,
				"service_tier": `"priority"`,
			},
		},
		{
			name:    "multiple field injection",
			encoded: `{"model":"gpt-4o"}`,
			extras: map[string]json.RawMessage{
				"service_tier": json.RawMessage(`"priority"`),
				"seed":         json.RawMessage(`42`),
			},
			wantFields: map[string]string{
				"model":        `"gpt-4o"`,
				"service_tier": `"priority"`,
				"seed":         `42`,
			},
		},
		{
			name:    "override existing field",
			encoded: `{"model":"gpt-4o","temperature":0.7}`,
			extras: map[string]json.RawMessage{
				"temperature": json.RawMessage(`0.1`),
			},
			wantFields: map[string]string{
				"model":       `"gpt-4o"`,
				"temperature": `0.1`,
			},
		},
		{
			name:    "invalid base JSON returns error",
			encoded: `not-json`,
			extras: map[string]json.RawMessage{
				"foo": json.RawMessage(`"bar"`),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := mergeOutboundExtra([]byte(tt.encoded), tt.extras)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var got map[string]json.RawMessage
			if err := json.Unmarshal(result, &got); err != nil {
				t.Fatalf("unmarshal result: %v", err)
			}
			for k, want := range tt.wantFields {
				if string(got[k]) != want {
					t.Errorf("field %q = %s, want %s", k, string(got[k]), want)
				}
			}
		})
	}
}
