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
//
// "Known" means "the IR carries this field", which is what makes it safe to
// exclude from RawExtra: the encoder will re-emit it from the IR. A field that is
// declared on the protocol struct but never mapped to the IR is *not* known in
// that sense — excluding it from RawExtra while also never encoding it makes the
// field disappear on a same-protocol passthrough. Such fields are listed in the
// unmapped sets below and deliberately left in RawExtra so they reach the upstream.
var openaiChatKnownFields = knownFieldsFor(reflect.TypeOf(openaichat.ChatRequest{}), openaiChatUnmappedFields)
var openaiResponsesKnownFields = knownFieldsFor(reflect.TypeOf(openairesponses.Request{}), openaiResponsesUnmappedFields)
var anthropicKnownFields = knownFieldsFor(reflect.TypeOf(anthropic.Request{}), anthropicUnmappedFields)
var geminiKnownFields = knownFieldsFor(reflect.TypeOf(gemini.Request{}), geminiUnmappedFields)

// openaiChatUnmappedFields are OpenAI Chat request fields with no IR
// representation. They round-trip via RawExtra on same-protocol passthrough and
// are dropped cross-protocol, like any other unrepresentable option.
var openaiChatUnmappedFields = []string{
	"service_tier", "store", "user", "seed", "logit_bias", "logprobs",
	"top_logprobs", "n", "frequency_penalty", "presence_penalty", "prediction",
	"modalities", "audio", "web_search_options", "prompt_cache_key", "safety_identifier",
}

// openaiResponsesUnmappedFields are OpenAI Responses request fields with no IR
// representation.
var openaiResponsesUnmappedFields = []string{
	"service_tier", "store", "user", "truncation", "include", "prompt",
	"prompt_cache_key", "safety_identifier", "background",
}

// anthropicUnmappedFields are Anthropic request fields with no IR representation.
var anthropicUnmappedFields = []string{
	"metadata", "container", "mcp_servers", "service_tier",
}

// geminiUnmappedFields are Gemini request fields with no IR representation.
var geminiUnmappedFields = []string{
	"safetySettings", "cachedContent", "labels",
}

// knownFieldsFor derives the set of struct-declared JSON field names, minus the
// names that are declared but never mapped to the IR.
func knownFieldsFor(t reflect.Type, unmapped []string) map[string]bool {
	fields := jsonFieldNames(t)
	for _, name := range unmapped {
		delete(fields, name)
	}
	return fields
}

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
