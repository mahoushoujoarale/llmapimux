package openaichat

import "encoding/json"

// ChatResponse is the JSON structure of an OpenAI Chat Completions API response.
type ChatResponse struct {
	ID               string           `json:"id"`
	Object           string           `json:"object"`
	Model            string           `json:"model"`
	Choices          []ChatChoice     `json:"choices"`
	Usage            *ChatUsage       `json:"usage,omitempty"`
	Created          int64            `json:"created,omitempty"`
	SystemFingerprint string          `json:"system_fingerprint,omitempty"`
	ServiceTier      string           `json:"service_tier,omitempty"`
}

// ChatChoice represents a single choice in the response.
type ChatChoice struct {
	Index        int              `json:"index"`
	Message      *ChatChoiceMessage `json:"message,omitempty"`
	Delta        *ChatChoiceMessage `json:"delta,omitempty"`
	FinishReason *string          `json:"finish_reason"`
}

// ChatChoiceMessage represents the message or delta within a choice.
type ChatChoiceMessage struct {
	Role        string            `json:"role,omitempty"`
	Content     *string           `json:"content,omitempty"`
	Refusal     *string           `json:"refusal,omitempty"`
	ToolCalls   []ToolCall        `json:"tool_calls,omitempty"`
	Annotations []json.RawMessage `json:"annotations,omitempty"`
}

// ChatUsage represents the usage information.
type ChatUsage struct {
	PromptTokens            int                    `json:"prompt_tokens"`
	CompletionTokens        int                    `json:"completion_tokens"`
	TotalTokens             int                    `json:"total_tokens"`
	PromptTokensDetails     *ChatPromptDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *ChatCompletionDetails `json:"completion_tokens_details,omitempty"`
}

// ChatPromptDetails represents detailed prompt token breakdown.
type ChatPromptDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
	AudioTokens  int `json:"audio_tokens,omitempty"`
}

// ChatCompletionDetails represents detailed completion token breakdown.
type ChatCompletionDetails struct {
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
	AudioTokens              int `json:"audio_tokens,omitempty"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
}
