package llmapimux

import "encoding/json"

// Role represents the role of a message sender.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// StopReason represents the reason a model stopped generating.
type StopReason string

const (
	StopReasonEndTurn       StopReason = "end_turn"
	StopReasonMaxTokens     StopReason = "max_tokens"
	StopReasonToolUse       StopReason = "tool_use"
	StopReasonStopSequence  StopReason = "stop_sequence"
	StopReasonContentFilter StopReason = "content_filter"
	StopReasonPauseTurn     StopReason = "pause_turn"
)

// ContentType represents the type of a content part.
type ContentType string

const (
	ContentTypeText                ContentType = "text"
	ContentTypeImage               ContentType = "image"
	ContentTypeToolUse             ContentType = "tool_use"
	ContentTypeToolResult          ContentType = "tool_result"
	ContentTypeServerToolUse       ContentType = "server_tool_use"
	ContentTypeWebSearchToolResult ContentType = "web_search_tool_result"
	ContentTypeDocument            ContentType = "document"
	ContentTypeThinking            ContentType = "thinking"
	ContentTypeRedactedThinking    ContentType = "redacted_thinking"
	ContentTypeRefusal             ContentType = "refusal"
)

// CitationKind identifies the type of citation or annotation.
type CitationKind = string

const (
	CitationKindCharLocation    CitationKind = "char_location"
	CitationKindWebSearchResult CitationKind = "web_search_result"
	CitationKindURLCitation     CitationKind = "url_citation"
	CitationKindGemini          CitationKind = "gemini_citation"
)

// Citation represents a citation or annotation attached to a content part.
type Citation struct {
	Kind     CitationKind `json:"kind,omitempty"`
	Title    string       `json:"title,omitempty"`
	URL      string       `json:"url,omitempty"`
	Start    *int         `json:"start,omitempty"`
	End      *int         `json:"end,omitempty"`
	SourceID string       `json:"source_id,omitempty"`
}

// RefusalContent holds refusal text from the model.
type RefusalContent struct {
	Refusal string `json:"refusal"`
}

// ContentPart is a union type representing a single piece of content in a message.
type ContentPart struct {
	Type                ContentType                 `json:"type"`
	Text                *TextContent                `json:"text,omitempty"`
	Image               *ImageContent               `json:"image,omitempty"`
	ToolUse             *ToolUseContent             `json:"tool_use,omitempty"`
	ToolResult          *ToolResultContent          `json:"tool_result,omitempty"`
	ServerToolUse       *ServerToolUseContent       `json:"server_tool_use,omitempty"`
	WebSearchToolResult *WebSearchToolResultContent `json:"web_search_tool_result,omitempty"`
	Document            *DocumentContent            `json:"document,omitempty"`
	Thinking            *ThinkingContent            `json:"thinking,omitempty"`
	RedactedThinking    *RedactedThinkingContent    `json:"redacted_thinking,omitempty"`
	Refusal             *RefusalContent             `json:"refusal,omitempty"`
	Citations           []Citation                  `json:"citations,omitempty"`
}

// TextContent holds plain text.
type TextContent struct {
	Text string `json:"text"`
}

// ImageContent holds image data or a URL reference.
type ImageContent struct {
	Data      []byte `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

// DocumentContent holds document data or a URL reference.
type DocumentContent struct {
	Data      []byte `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Title     string `json:"title,omitempty"`
}

// ToolUseContent represents a tool call made by the model.
type ToolUseContent struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// ToolResultContent represents the result of a tool call.
type ToolResultContent struct {
	ToolUseID string        `json:"tool_use_id"`
	Name      string        `json:"name,omitempty"`
	Content   []ContentPart `json:"content,omitempty"`
	IsError   bool          `json:"is_error,omitempty"`
}

// ServerToolUseContent represents a built-in server tool call, such as Anthropic web_search.
type ServerToolUseContent struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// WebSearchResult is a single search hit returned by a built-in web search tool.
type WebSearchResult struct {
	Title string `json:"title,omitempty"`
	URL   string `json:"url,omitempty"`
}

// WebSearchToolResultContent represents the result block for a web search server tool call.
type WebSearchToolResultContent struct {
	ToolUseID string            `json:"tool_use_id"`
	Content   []WebSearchResult `json:"content,omitempty"`
	IsError   bool              `json:"is_error,omitempty"`
	ErrorCode string            `json:"error_code,omitempty"`
}

// ThinkingContent holds extended thinking output from the model.
type ThinkingContent struct {
	Thinking  string `json:"thinking"`
	Signature string `json:"signature,omitempty"`
}

// RedactedThinkingContent holds redacted thinking data that must round-trip exactly.
type RedactedThinkingContent struct {
	Data string `json:"data"`
}

// Message is a single turn in a conversation.
type Message struct {
	Role    Role          `json:"role"`
	Content []ContentPart `json:"content"`
}

// Tool describes a tool available to the model.
type Tool struct {
	Type        string                     `json:"type,omitempty"`
	Name        string                     `json:"name"`
	Description string                     `json:"description,omitempty"`
	Parameters  json.RawMessage            `json:"parameters,omitempty"`
	Strict      bool                       `json:"strict,omitempty"`
	ExtraFields map[string]json.RawMessage `json:"extra_fields,omitempty"`
}

// ToolChoice controls how the model selects tools.
type ToolChoice struct {
	Type               string   `json:"type"`
	ToolName           string   `json:"tool_name,omitempty"`
	AllowedToolNames   []string `json:"allowed_tool_names,omitempty"`
	AllowParallelCalls *bool    `json:"allow_parallel_calls,omitempty"`
}

// ProviderExtensions holds provider-specific extension fields as raw JSON values,
// keyed by a vendor-namespaced string (e.g. "anthropic/thinking").
// These are preserved during same-provider round-trips and silently dropped on cross-provider conversion.
type ProviderExtensions map[string]json.RawMessage

// ThinkingConfig controls extended thinking behavior.
type ThinkingConfig struct {
	Mode            string `json:"mode,omitempty"`
	BudgetTokens    int    `json:"budget_tokens,omitempty"`
	Effort          string `json:"effort,omitempty"`
	IncludeThoughts *bool  `json:"include_thoughts,omitempty"`
	Level           string `json:"level,omitempty"`
}

// ResponseFormat controls the output format of the model.
type ResponseFormat struct {
	Type       string          `json:"type"`
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

// Request is the unified intermediate representation of an LLM API request.
type Request struct {
	OriginalModel  string          `json:"-"`
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	SystemPrompt   []ContentPart   `json:"system_prompt,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     *ToolChoice     `json:"tool_choice,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	TopK           *int            `json:"top_k,omitempty"`
	StopSequences  []string        `json:"stop_sequences,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	Thinking       *ThinkingConfig `json:"thinking,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	// ProviderExtensions holds provider-specific extension fields.
	// Keys should be vendor-namespaced (e.g. "anthropic/thinking").
	// Silently dropped on cross-provider conversion.
	ProviderExtensions ProviderExtensions `json:"provider_extensions,omitempty"`
	// Protocol-specific fields preserved for same-protocol roundtrip.
	// Managed internally by the library — do not modify.
	RawExtra map[string]json.RawMessage `json:"-"`
	// Extra fields to merge into the outbound request body.
	// Set by RequestModifier before each send attempt.
	OutboundExtra   map[string]json.RawMessage `json:"-"`
	InboundProtocol Protocol                   `json:"-"`
}

// Response is the unified intermediate representation of an LLM API response.
type Response struct {
	ID                 string             `json:"id,omitempty"`
	Model              string             `json:"model,omitempty"`
	Content            []ContentPart      `json:"content,omitempty"`
	StopReason         StopReason         `json:"stop_reason,omitempty"`
	StopSequence       string             `json:"stop_sequence,omitempty"`
	Usage              Usage              `json:"usage"`
	ProviderExtensions ProviderExtensions `json:"provider_extensions,omitempty"`
}

// Usage tracks token consumption for a request/response pair.
type Usage struct {
	InputTokens         int `json:"input_tokens,omitempty"`
	OutputTokens        int `json:"output_tokens,omitempty"`
	TotalTokens         int `json:"total_tokens,omitempty"`
	CacheReadTokens     int `json:"cache_read_tokens,omitempty"`
	CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
	ThinkingTokens      int `json:"thinking_tokens,omitempty"`
}

// StreamEventType identifies the kind of streaming event.
type StreamEventType string

const (
	StreamEventStart             StreamEventType = "start"
	StreamEventDelta             StreamEventType = "delta"
	StreamEventContentBlockStart StreamEventType = "content_block_start"
	StreamEventContentBlockStop  StreamEventType = "content_block_stop"
	StreamEventStop              StreamEventType = "stop"
	StreamEventError             StreamEventType = "error"
)

// StreamError holds error information from a streaming error event.
type StreamError struct {
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Param   string `json:"param,omitempty"`
}

// IncompleteDetails holds information about why a response was incomplete.
type IncompleteDetails struct {
	Reason string `json:"reason,omitempty"`
}

// StreamEvent is a single event in a streaming response.
type StreamEvent struct {
	Type              StreamEventType    `json:"type"`
	Response          *Response          `json:"response,omitempty"`
	Index             int                `json:"index"`
	Delta             *ContentPart       `json:"delta,omitempty"`
	StopReason        *StopReason        `json:"stop_reason,omitempty"`
	Usage             *Usage             `json:"usage,omitempty"`
	Error             *StreamError       `json:"error,omitempty"`
	IncompleteDetails *IncompleteDetails `json:"incomplete_details,omitempty"`
}
