package gemini

import "encoding/json"

// Request is the JSON structure of a Gemini GenerateContent API request.
// Field names use camelCase to match the official Gemini REST API (proto3 JSON encoding)
// and the google.golang.org/genai SDK serialization format.
type Request struct {
	Contents          []Content          `json:"contents,omitempty"`
	SystemInstruction *Content           `json:"systemInstruction,omitempty"`
	Tools             []ToolDeclaration  `json:"tools,omitempty"`
	ToolConfig        *ToolConfig        `json:"toolConfig,omitempty"`
	GenerationConfig  *GenerationConfig  `json:"generationConfig,omitempty"`
	SafetySettings    json.RawMessage    `json:"safetySettings,omitempty"`
	CachedContent     string             `json:"cachedContent,omitempty"`
}

// Content represents a content message with role and parts.
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

// Part is a union type for content parts in the Gemini API.
type Part struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *InlineData       `json:"inlineData,omitempty"`
	FileData         *FileData         `json:"fileData,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
	Thought          *bool             `json:"thought,omitempty"`
}

// InlineData holds inline binary data.
type InlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64-encoded
}

// FileData holds a reference to a file by URI.
type FileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

// FunctionCall represents a function call from the model.
type FunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
	ID   string          `json:"id,omitempty"`
}

// FunctionResponse represents the response to a function call.
type FunctionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response,omitempty"`
	ID       string          `json:"id,omitempty"`
}

// ToolDeclaration holds tool declarations.
type ToolDeclaration struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// FunctionDeclaration describes a function available to the model.
type FunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolConfig holds tool configuration.
type ToolConfig struct {
	FunctionCallingConfig *FunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// FunctionCallingConfig controls how the model uses functions.
type FunctionCallingConfig struct {
	Mode                 string   `json:"mode"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// GenerationConfig holds generation parameters.
type GenerationConfig struct {
	Temperature      *float64      `json:"temperature,omitempty"`
	TopP             *float64      `json:"topP,omitempty"`
	TopK             *int          `json:"topK,omitempty"`
	MaxOutputTokens  *int          `json:"maxOutputTokens,omitempty"`
	StopSequences    []string      `json:"stopSequences,omitempty"`
	ResponseMimeType string        `json:"responseMimeType,omitempty"`
	ResponseSchema   json.RawMessage `json:"responseSchema,omitempty"`
	ThinkingConfig   *ThinkingConfig `json:"thinkingConfig,omitempty"`
	CandidateCount   *int          `json:"candidateCount,omitempty"`
	ResponseLogprobs *bool         `json:"responseLogprobs,omitempty"`
	Logprobs         *int          `json:"logprobs,omitempty"`
	PresencePenalty  *float64      `json:"presencePenalty,omitempty"`
	FrequencyPenalty *float64      `json:"frequencyPenalty,omitempty"`
	Seed             *int          `json:"seed,omitempty"`
}

// ThinkingConfig controls extended thinking.
type ThinkingConfig struct {
	ThinkingBudget int    `json:"thinkingBudget,omitempty"`
	IncludeThoughts *bool `json:"includeThoughts,omitempty"`
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
}
