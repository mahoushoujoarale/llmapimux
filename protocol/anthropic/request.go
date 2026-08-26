package anthropic

import "encoding/json"

// Request is the JSON structure of an Anthropic Messages API request.
//
// Optional fields use omitempty: emitting explicit nulls for tool_choice,
// thinking, system, etc. causes strict gateways and proxies in front of the
// Anthropic API to reject the request. max_tokens has no omitempty because the
// API requires it, and stream is always meaningful.
//
// System is json.RawMessage because the API accepts either a plain string or an
// array of content blocks; see SystemBlocks.
type Request struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens"`
	System        json.RawMessage `json:"system,omitempty"`
	Messages      []Message       `json:"messages"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	TopK          *int            `json:"top_k,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	Tools         []Tool          `json:"tools,omitempty"`
	ToolChoice    *ToolChoice     `json:"tool_choice,omitempty"`
	Thinking      *Thinking       `json:"thinking,omitempty"`
	Metadata      json.RawMessage `json:"metadata,omitempty"`
}

// SystemBlocks decodes the dual-form system field into content blocks. A bare
// string is normalised to a single text block. Returns nil when system is absent.
func (r *Request) SystemBlocks() ([]ContentBlock, error) {
	raw := r.System
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, err
		}
		if text == "" {
			return nil, nil
		}
		return []ContentBlock{{Type: "text", Text: text}}, nil
	}
	var blocks []ContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, err
	}
	return blocks, nil
}

// SetSystemBlocks encodes content blocks into the system field.
func (r *Request) SetSystemBlocks(blocks []ContentBlock) error {
	if len(blocks) == 0 {
		r.System = nil
		return nil
	}
	data, err := json.Marshal(blocks)
	if err != nil {
		return err
	}
	r.System = data
	return nil
}

// Message represents a single message in the Anthropic API.
// Content can be a string or an array of content blocks.
type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// ContentBlock represents a single content block in the Anthropic API.
//
// ExtraFields captures block-level fields this struct does not model explicitly —
// most importantly cache_control, the prompt-caching breakpoint marker. Dropping
// it silently disables prompt caching for the whole request, so unknown fields are
// preserved through a round-trip via custom (Un)MarshalJSON rather than discarded.
type ContentBlock struct {
	Type string `json:"type"`

	// text
	Text string `json:"text,omitempty"`

	// image / document
	Source *Source `json:"source,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result
	ToolUseID  string          `json:"tool_use_id,omitempty"`
	ContentRaw json.RawMessage `json:"content,omitempty"`
	IsError    bool            `json:"is_error,omitempty"`
	ErrorCode  string          `json:"error_code,omitempty"`

	// thinking
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`

	// redacted_thinking
	Data string `json:"data,omitempty"`

	// citations (response path only)
	Citations []json.RawMessage `json:"citations,omitempty"`

	// ExtraFields holds block fields not modelled above (e.g. cache_control).
	ExtraFields map[string]json.RawMessage `json:"-"`
}

// contentBlockAlias mirrors ContentBlock without the custom marshaller, so the
// (Un)MarshalJSON methods below can delegate the known-field work to encoding/json
// without recursing.
type contentBlockAlias ContentBlock

// contentBlockKnownFields lists the JSON keys modelled by ContentBlock. Anything
// else found on the wire lands in ExtraFields.
var contentBlockKnownFields = map[string]bool{
	"type": true, "text": true, "source": true, "id": true, "name": true,
	"input": true, "tool_use_id": true, "content": true, "is_error": true,
	"error_code": true, "thinking": true, "signature": true, "data": true,
	"citations": true,
}

func (b ContentBlock) MarshalJSON() ([]byte, error) {
	known, err := json.Marshal(contentBlockAlias(b))
	if err != nil {
		return nil, err
	}
	if len(b.ExtraFields) == 0 {
		return known, nil
	}
	var merged map[string]json.RawMessage
	if err := json.Unmarshal(known, &merged); err != nil {
		return nil, err
	}
	// Known fields win: ExtraFields is only a carrier for unmodelled keys, and a
	// stale duplicate there must not override the IR-derived value.
	for k, v := range b.ExtraFields {
		if _, exists := merged[k]; !exists {
			merged[k] = v
		}
	}
	return json.Marshal(merged)
}

func (b *ContentBlock) UnmarshalJSON(data []byte) error {
	var alias contentBlockAlias
	if err := json.Unmarshal(data, &alias); err != nil {
		return err
	}
	*b = ContentBlock(alias)

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	for k := range raw {
		if contentBlockKnownFields[k] {
			delete(raw, k)
		}
	}
	// Reset rather than merge, so decoding into a reused value cannot retain
	// stale extras from a previous payload.
	if len(raw) > 0 {
		b.ExtraFields = raw
	} else {
		b.ExtraFields = nil
	}
	return nil
}

// Source represents the source of image or document content.
type Source struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // e.g. "image/png"
	Data      string `json:"data,omitempty"`       // base64-encoded data (when type=base64)
	URL       string `json:"url,omitempty"`        // URL (when type=url)
}

// Tool represents a tool in the Anthropic API.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
	// Type can be "custom", "web_search_20250305", etc. Empty string means custom.
	Type        string                     `json:"type"`
	ExtraFields map[string]json.RawMessage `json:"-"`
}

func (t Tool) MarshalJSON() ([]byte, error) {
	raw := make(map[string]json.RawMessage, 4+len(t.ExtraFields))
	if t.Name != "" {
		b, err := json.Marshal(t.Name)
		if err != nil {
			return nil, err
		}
		raw["name"] = b
	}
	if t.Description != "" {
		b, err := json.Marshal(t.Description)
		if err != nil {
			return nil, err
		}
		raw["description"] = b
	}
	if len(t.InputSchema) > 0 {
		raw["input_schema"] = t.InputSchema
	}
	if t.Type != "" {
		b, err := json.Marshal(t.Type)
		if err != nil {
			return nil, err
		}
		raw["type"] = b
	}
	for k, v := range t.ExtraFields {
		raw[k] = v
	}
	return json.Marshal(raw)
}

func (t *Tool) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	if v, ok := raw["name"]; ok {
		if err := json.Unmarshal(v, &t.Name); err != nil {
			return err
		}
		delete(raw, "name")
	}
	if v, ok := raw["description"]; ok {
		if err := json.Unmarshal(v, &t.Description); err != nil {
			return err
		}
		delete(raw, "description")
	}
	if v, ok := raw["input_schema"]; ok {
		t.InputSchema = v
		delete(raw, "input_schema")
	}
	if v, ok := raw["type"]; ok {
		if err := json.Unmarshal(v, &t.Type); err != nil {
			return err
		}
		delete(raw, "type")
	}
	if len(raw) > 0 {
		t.ExtraFields = raw
	} else {
		t.ExtraFields = nil
	}
	return nil
}

// ToolChoice represents the tool_choice field in the Anthropic API.
type ToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`                      // used when type = "tool"
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"` // inverse of AllowParallelCalls
}

// Thinking represents the thinking configuration in the Anthropic API.
// BudgetTokens is omitted when zero because the API rejects budget_tokens on a
// disabled thinking config.
type Thinking struct {
	Type         string `json:"type"`                   // "enabled" or "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"` // only when type = "enabled"
}
