package openairesponses

import "encoding/json"

// Response is the JSON structure of an OpenAI Responses API response.
type Response struct {
	ID        string       `json:"id"`
	Object    string       `json:"object,omitempty"`
	Model     string       `json:"model,omitempty"`
	Status    string       `json:"status,omitempty"`
	Output    []OutputItem `json:"output,omitempty"`
	Usage     *Usage       `json:"usage,omitempty"`
	CreatedAt int64        `json:"created_at,omitempty"`
}

// OutputItem represents an item in the response output array.
type OutputItem struct {
	Type      string          `json:"type"`
	ID        string          `json:"id,omitempty"`
	Role      string          `json:"role,omitempty"`
	Content   []OutputContent `json:"content,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
}

// OutputContent represents a content part in a response output message.
type OutputContent struct {
	Type        string            `json:"type"`
	Text        string            `json:"text,omitempty"`
	Refusal     string            `json:"refusal,omitempty"`
	Annotations []json.RawMessage `json:"annotations,omitempty"`
}

// Usage represents usage information.
type Usage struct {
	InputTokens         int           `json:"input_tokens"`
	OutputTokens        int           `json:"output_tokens"`
	TotalTokens         int           `json:"total_tokens"`
	InputTokensDetails  *InputDetails `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *OutputDetails `json:"output_tokens_details,omitempty"`
}

// InputDetails represents detailed input token breakdown.
type InputDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// OutputDetails represents detailed output token breakdown.
type OutputDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}
