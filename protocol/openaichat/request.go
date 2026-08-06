package openaichat

import "encoding/json"

// ChatRequest is the JSON structure of an OpenAI Chat Completions API request.
type ChatRequest struct {
	Model               string               `json:"model"`
	Messages            []ChatMessage        `json:"messages"`
	Tools               []ChatTool           `json:"tools,omitempty"`
	ToolChoice          json.RawMessage      `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool                `json:"parallel_tool_calls,omitempty"`
	MaxTokens           *int                 `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                 `json:"max_completion_tokens,omitempty"`
	Temperature         *float64             `json:"temperature,omitempty"`
	TopP                *float64             `json:"top_p,omitempty"`
	Stop                json.RawMessage      `json:"stop,omitempty"`
	Stream              bool                 `json:"stream,omitempty"`
	ResponseFormat      *ChatResponseFormat  `json:"response_format,omitempty"`
	ReasoningEffort     string               `json:"reasoning_effort,omitempty"`
	FrequencyPenalty    *float64             `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64             `json:"presence_penalty,omitempty"`
	Seed                *int                 `json:"seed,omitempty"`
	N                   *int                 `json:"n,omitempty"`
	User                string               `json:"user,omitempty"`
	Logprobs            *bool                `json:"logprobs,omitempty"`
	TopLogprobs         *int                 `json:"top_logprobs,omitempty"`
	StreamOptions       json.RawMessage      `json:"stream_options,omitempty"`
	ServiceTier         string               `json:"service_tier,omitempty"`
	Store               *bool                `json:"store,omitempty"`
	Metadata            json.RawMessage      `json:"metadata,omitempty"`
	WebSearchOptions    json.RawMessage      `json:"web_search_options,omitempty"`
}

// ChatMessage represents a single message in the OpenAI Chat API.
// Content can be a string or an array of content parts.
//
// ReasoningSignature and ReasoningRedacted are not part of the OpenAI spec. They
// are emitted and consumed by this gateway so that Anthropic thinking blocks
// survive a round-trip through the Chat Completions message history — Anthropic
// rejects a replayed thinking block whose signature is missing, and requires
// redacted_thinking payloads to be returned verbatim. Providers ignore unknown
// message fields, so they are harmless when the target is a real OpenAI endpoint.
type ChatMessage struct {
	Role               string          `json:"role"`
	Content            json.RawMessage `json:"content,omitempty"`
	ReasoningContent   *string         `json:"reasoning_content,omitempty"`
	ReasoningSignature *string         `json:"reasoning_signature,omitempty"`
	ReasoningRedacted  *string         `json:"reasoning_redacted,omitempty"`
	ToolCalls          []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID         string          `json:"tool_call_id,omitempty"`
}

// ChatContentPart represents a single content part in an OpenAI Chat message.
type ChatContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *ChatImageURL `json:"image_url,omitempty"`
	VideoURL *ChatVideoURL `json:"video_url,omitempty"`
}

// ChatImageURL represents an image_url content part.
type ChatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ChatVideoURL represents a video_url content part.
type ChatVideoURL struct {
	URL string `json:"url"`
}

// ToolCall represents a tool call in the OpenAI Chat API.
// The Index field is used in streaming deltas only.
type ToolCall struct {
	Index    int              `json:"index"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction represents the function details within a tool call.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatTool represents a tool in the OpenAI Chat API.
type ChatTool struct {
	Type     string       `json:"type"`
	Function ChatFunction `json:"function"`
}

// ChatFunction represents the function details within a tool definition.
type ChatFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      bool            `json:"strict,omitempty"`
}

// ChatToolChoiceObj represents the object form of tool_choice.
type ChatToolChoiceObj struct {
	Type     string             `json:"type"`
	Function *ChatToolChoiceFunc `json:"function,omitempty"`
}

// ChatToolChoiceFunc represents the function within the object form of tool_choice.
type ChatToolChoiceFunc struct {
	Name string `json:"name"`
}

// ChatResponseFormat represents the response_format field.
type ChatResponseFormat struct {
	Type       string                 `json:"type"`
	JSONSchema *ChatResponseJSONSchema `json:"json_schema,omitempty"`
}

// ChatResponseJSONSchema represents the json_schema sub-field of response_format.
// Name is required by the OpenAI API when response_format.type is "json_schema".
type ChatResponseJSONSchema struct {
	Name   string          `json:"name,omitempty"`
	Schema json.RawMessage `json:"schema,omitempty"`
	Strict bool            `json:"strict,omitempty"`
}
