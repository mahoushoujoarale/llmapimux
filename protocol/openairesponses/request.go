package openairesponses

import "encoding/json"

// Request is the JSON structure of an OpenAI Responses API request.
type Request struct {
	Model             string          `json:"model"`
	Input             json.RawMessage `json:"input"`
	Instructions      string          `json:"instructions,omitempty"`
	MaxOutputTokens   *int            `json:"max_output_tokens,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	Tools             []Tool          `json:"tools,omitempty"`
	ToolChoice        json.RawMessage `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	Stop              json.RawMessage `json:"stop,omitempty"`
	Reasoning         *Reasoning      `json:"reasoning,omitempty"`
	Text              *Text           `json:"text,omitempty"`
	Truncation        json.RawMessage `json:"truncation,omitempty"`
	User              string          `json:"user,omitempty"`
	Metadata          json.RawMessage `json:"metadata,omitempty"`
	ServiceTier       string          `json:"service_tier,omitempty"`

	// Silently ignored fields
	PreviousResponseID string `json:"previous_response_id,omitempty"`
}

// Reasoning represents the reasoning configuration.
type Reasoning struct {
	Effort string `json:"effort,omitempty"`
}

// Text represents the text configuration.
type Text struct {
	Format *TextFormat `json:"format,omitempty"`
}

// TextFormat represents the text.format configuration.
type TextFormat struct {
	Type   string          `json:"type"`
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
}

// Tool represents a tool in the Responses API.
type Tool struct {
	Type        string                     `json:"type"`
	Name        string                     `json:"name,omitempty"`
	Description string                     `json:"description,omitempty"`
	Parameters  json.RawMessage            `json:"parameters,omitempty"`
	Strict      bool                       `json:"strict,omitempty"`
	ExtraFields map[string]json.RawMessage `json:"-"`
}

func (t Tool) MarshalJSON() ([]byte, error) {
	raw := make(map[string]json.RawMessage, 5+len(t.ExtraFields))
	if t.Type != "" {
		b, err := json.Marshal(t.Type)
		if err != nil {
			return nil, err
		}
		raw["type"] = b
	}
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
	if len(t.Parameters) > 0 {
		raw["parameters"] = t.Parameters
	}
	if t.Strict {
		b, err := json.Marshal(t.Strict)
		if err != nil {
			return nil, err
		}
		raw["strict"] = b
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

	if v, ok := raw["type"]; ok {
		if err := json.Unmarshal(v, &t.Type); err != nil {
			return err
		}
		delete(raw, "type")
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
	if v, ok := raw["parameters"]; ok {
		t.Parameters = v
		delete(raw, "parameters")
	}
	if v, ok := raw["strict"]; ok {
		if err := json.Unmarshal(v, &t.Strict); err != nil {
			return err
		}
		delete(raw, "strict")
	}
	if len(raw) > 0 {
		t.ExtraFields = raw
	} else {
		t.ExtraFields = nil
	}
	return nil
}

// ToolChoiceObj represents the object form of tool_choice.
type ToolChoiceObj struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

// InputItem is a single item in the input array.
type InputItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	ID        string          `json:"id,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Output    string          `json:"output,omitempty"`
}

// ContentPart represents a content part in an input message.
type ContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	FileData string `json:"file_data,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
	Filename string `json:"filename,omitempty"`
}
