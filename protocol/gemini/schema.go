package gemini

// Schema represents a schema in Gemini's format (uppercase type names).
type Schema struct {
	Type        string            `json:"type,omitempty"`
	Description string            `json:"description,omitempty"`
	Properties  map[string]Schema `json:"properties,omitempty"`
	Required    []string          `json:"required,omitempty"`
	Items       *Schema           `json:"items,omitempty"`
	Enum        []string          `json:"enum,omitempty"`
}
