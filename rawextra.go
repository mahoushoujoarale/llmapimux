package llmapimux

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	anthropic "github.com/mahoushoujoarale/llmapimux/protocol/anthropic"
	gemini "github.com/mahoushoujoarale/llmapimux/protocol/gemini"
	"github.com/mahoushoujoarale/llmapimux/protocol/openaichat"
	"github.com/mahoushoujoarale/llmapimux/protocol/openairesponses"
)

// jsonFieldNames extracts JSON field names from a struct type's tags.
func jsonFieldNames(t reflect.Type) map[string]bool {
	fields := make(map[string]bool)
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("json")
		if tag == "" || tag == "-" {
			continue
		}
		name, _, _ := strings.Cut(tag, ",")
		if name != "" {
			fields[name] = true
		}
	}
	return fields
}

// Per-protocol known fields, scanned at init time.
var openaiChatKnownFields = jsonFieldNames(reflect.TypeOf(openaichat.ChatRequest{}))
var openaiResponsesKnownFields = jsonFieldNames(reflect.TypeOf(openairesponses.Request{}))
var anthropicKnownFields = jsonFieldNames(reflect.TypeOf(anthropic.Request{}))
var geminiKnownFields = jsonFieldNames(reflect.TypeOf(gemini.Request{}))

// extractRawExtra extracts all JSON fields from body that are NOT in knownFields.
func extractRawExtra(body []byte, knownFields map[string]bool) map[string]json.RawMessage {
	var allFields map[string]json.RawMessage
	if err := json.Unmarshal(body, &allFields); err != nil {
		return nil
	}
	for k := range allFields {
		if knownFields[k] {
			delete(allFields, k)
		}
	}
	if len(allFields) == 0 {
		return nil
	}
	return allFields
}

// populateRawExtraIfNeeded performs on-demand extraction and stores it on req.
func populateRawExtraIfNeeded(req *Request, body []byte, knownFields map[string]bool) {
	if req == nil || req.RawExtra != nil {
		return
	}
	req.RawExtra = extractRawExtra(body, knownFields)
}

// mergeRawExtra merges protocol-specific extras into encoded output.
// Fields in knownFields are skipped (they come from IR only).
func mergeRawExtra(encoded []byte, extras map[string]json.RawMessage, knownFields map[string]bool) ([]byte, error) {
	if len(extras) == 0 {
		return encoded, nil
	}
	var base map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &base); err != nil {
		return encoded, nil
	}
	for k, v := range extras {
		if !knownFields[k] {
			base[k] = v
		}
	}
	return json.Marshal(base)
}

// mergeOutboundExtra merges caller-injected extras into an encoded JSON body.
// Unlike mergeRawExtra, it does not filter by knownFields — callers may
// intentionally set any field. Returns error on invalid base JSON (caller-injected
// fields should not be silently lost).
func mergeOutboundExtra(encoded []byte, extras map[string]json.RawMessage) ([]byte, error) {
	if len(extras) == 0 {
		return encoded, nil
	}
	var base map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &base); err != nil {
		return nil, fmt.Errorf("mergeOutboundExtra unmarshal: %w", err)
	}
	for k, v := range extras {
		base[k] = v
	}
	return json.Marshal(base)
}
