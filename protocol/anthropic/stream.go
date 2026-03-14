package anthropic

// StreamMessageStart is the JSON structure of the message_start SSE event.
type StreamMessageStart struct {
	Type    string        `json:"type"`
	Message StreamMessage `json:"message"`
}

// StreamMessage is the partial message in the message_start SSE event.
type StreamMessage struct {
	ID    string `json:"id"`
	Type  string `json:"type"`
	Model string `json:"model"`
	Role  string `json:"role"`
	Usage Usage  `json:"usage"`
}

// StreamContentBlockStart is the JSON structure of the content_block_start SSE event.
type StreamContentBlockStart struct {
	Type         string       `json:"type"`
	Index        int          `json:"index"`
	ContentBlock ContentBlock `json:"content_block"`
}

// StreamContentBlockDelta is the JSON structure of the content_block_delta SSE event.
type StreamContentBlockDelta struct {
	Type  string      `json:"type"`
	Index int         `json:"index"`
	Delta StreamDelta `json:"delta"`
}

// StreamDelta represents the delta payload within a content_block_delta event.
// Text and Thinking use *string so that empty strings are preserved in JSON
// (omitempty only omits nil pointers, not empty-string pointers).
// The Anthropic SDK requires these fields to be present even when empty.
type StreamDelta struct {
	Type        string  `json:"type"`
	Text        *string `json:"text,omitempty"`
	PartialJSON *string `json:"partial_json,omitempty"`
	Thinking    *string `json:"thinking,omitempty"`
	Signature   *string `json:"signature,omitempty"`
}

// StreamContentBlockStop is the JSON structure of the content_block_stop SSE event.
type StreamContentBlockStop struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

// StreamMessageDelta is the JSON structure of the message_delta SSE event.
type StreamMessageDelta struct {
	Type  string                  `json:"type"`
	Delta StreamMessageDeltaInner `json:"delta"`
	Usage Usage                   `json:"usage"`
}

// StreamMessageDeltaInner is the inner delta in a message_delta event.
type StreamMessageDeltaInner struct {
	StopReason string `json:"stop_reason"`
}
