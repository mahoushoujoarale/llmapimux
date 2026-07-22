package gemini

import "encoding/json"

// Response is the JSON structure of a Gemini GenerateContent API response.
type Response struct {
	Candidates    []Candidate    `json:"candidates,omitempty"`
	UsageMetadata *UsageMetadata `json:"usageMetadata,omitempty"`
	ModelVersion  string         `json:"modelVersion,omitempty"`
}

// Candidate represents a single candidate in the response.
type Candidate struct {
	Content           *Content         `json:"content,omitempty"`
	FinishReason      string           `json:"finishReason,omitempty"`
	CitationMetadata  *CitationMetadata `json:"citationMetadata,omitempty"`
	SafetyRatings     json.RawMessage  `json:"safetyRatings,omitempty"`
	TokenCount        *int             `json:"tokenCount,omitempty"`
	GroundingMetadata json.RawMessage  `json:"groundingMetadata,omitempty"`
}

// CitationMetadata holds citation information.
type CitationMetadata struct {
	CitationSources []CitationSource `json:"citationSources,omitempty"`
}

// CitationSource holds a single citation source.
type CitationSource struct {
	StartIndex int    `json:"startIndex,omitempty"`
	EndIndex   int    `json:"endIndex,omitempty"`
	URI        string `json:"uri,omitempty"`
	Title      string `json:"title,omitempty"`
	License    string `json:"license,omitempty"`
}

// UsageMetadata holds token usage information.
type UsageMetadata struct {
	PromptTokenCount           int `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount       int `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount            int `json:"totalTokenCount,omitempty"`
	ThoughtsTokenCount         int `json:"thoughtsTokenCount,omitempty"`
	CachedContentTokenCount    int `json:"cachedContentTokenCount,omitempty"`
	ToolUsePromptTokenCount    int `json:"toolUsePromptTokenCount,omitempty"`
}
