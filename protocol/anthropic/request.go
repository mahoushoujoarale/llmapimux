package anthropic

import "encoding/json"

// Request is the JSON structure of an Anthropic Messages API request.
type Request struct {
	Model         string          `json:"model"`
	MaxTokens     int             `json:"max_tokens"`
	System        []ContentBlock  `json:"system"`
	Messages      []Message       `json:"messages"`
	Temperature   *float64        `json:"temperature"`
	TopP          *float64        `json:"top_p"`
	TopK          *int            `json:"top_k"`
	StopSequences []string        `json:"stop_sequences"`
	Stream        bool            `json:"stream"`
	Tools         []Tool          `json:"tools"`
	ToolChoice    *ToolChoice     `json:"tool_choice"`
	Thinking      *Thinking       `json:"thinking"`
	Metadata      json.RawMessage `json:"metadata,omitempty"`
}

// Message represents a single message in the Anthropic API.
// Content can be a string or an array of content blocks.
type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// ContentBlock represents a single content block in the Anthropic API.
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

	// thinking
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`

	// redacted_thinking
	Data string `json:"data,omitempty"`

	// citations (response path only)
	Citations []json.RawMessage `json:"citations,omitempty"`
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
type Thinking struct {
	Type         string `json:"type"`          // "enabled" or "disabled"
	BudgetTokens int    `json:"budget_tokens"` // only when type = "enabled"
}
