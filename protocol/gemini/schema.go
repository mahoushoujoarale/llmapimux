package gemini

// Schema represents a schema in Gemini's format (uppercase type names).
//
// Gemini's OpenAPI subset supports more than the core structural keywords:
// format, nullable, enum and the numeric/string/array constraints are all
// accepted. They are modelled here so that JSON Schemas coming from other
// protocols keep their validation semantics instead of being flattened to bare
// types.
type Schema struct {
	Type        string            `json:"type,omitempty"`
	Format      string            `json:"format,omitempty"`
	Description string            `json:"description,omitempty"`
	Nullable    *bool             `json:"nullable,omitempty"`
	Properties  map[string]Schema `json:"properties,omitempty"`
	Required    []string          `json:"required,omitempty"`
	Items       *Schema           `json:"items,omitempty"`
	// Enum values are always strings on the Gemini wire, even for integer types.
	Enum []string `json:"enum,omitempty"`
	// String constraints.
	MinLength *int   `json:"minLength,omitempty"`
	MaxLength *int   `json:"maxLength,omitempty"`
	Pattern   string `json:"pattern,omitempty"`
	// Numeric constraints.
	Minimum *float64 `json:"minimum,omitempty"`
	Maximum *float64 `json:"maximum,omitempty"`
	// Array constraints.
	MinItems *int `json:"minItems,omitempty"`
	MaxItems *int `json:"maxItems,omitempty"`
	// Composition. Gemini supports anyOf; oneOf is mapped onto it since Gemini
	// has no exclusive-choice keyword.
	AnyOf []Schema `json:"anyOf,omitempty"`
	// PropertyOrdering is Gemini-specific and controls key order in generated JSON.
	PropertyOrdering []string `json:"propertyOrdering,omitempty"`
}
