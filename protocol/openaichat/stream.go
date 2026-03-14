package openaichat

// ChatStreamChunk is the JSON structure of an OpenAI Chat streaming chunk.
type ChatStreamChunk struct {
	ID               string       `json:"id,omitempty"`
	Object           string       `json:"object,omitempty"`
	Model            string       `json:"model,omitempty"`
	Choices          []ChatChoice `json:"choices,omitempty"`
	Usage            *ChatUsage   `json:"usage,omitempty"`
	Created          int64        `json:"created,omitempty"`
	SystemFingerprint string      `json:"system_fingerprint,omitempty"`
	ServiceTier      string       `json:"service_tier,omitempty"`
}
