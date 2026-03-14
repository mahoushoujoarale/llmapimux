package llmapimux

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"

	anthropic "github.com/llmapimux/llmapimux/protocol/anthropic"
)

// anthropicCitationWire is the wire format for Anthropic citation JSON objects.
type anthropicCitationWire struct {
	Type           string `json:"type,omitempty"`
	DocumentTitle  string `json:"document_title,omitempty"`
	Title          string `json:"title,omitempty"`
	URL            string `json:"url,omitempty"`
	StartCharIndex *int   `json:"start_char_index,omitempty"`
	EndCharIndex   *int   `json:"end_char_index,omitempty"`
	DocumentIndex  *int   `json:"document_index,omitempty"`
}


// Content can be a string or an array of content blocks.






// DecodeAnthropicRequest decodes an Anthropic Messages API JSON request body
// into the unified IR Request type.
func DecodeAnthropicRequest(body []byte) (*Request, error) {
	var raw anthropic.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode anthropic request: %w", err)
	}

	req := &Request{
		Model:         raw.Model,
		MaxTokens:     raw.MaxTokens,
		Temperature:   raw.Temperature,
		TopP:          raw.TopP,
		TopK:          raw.TopK,
		StopSequences: raw.StopSequences,
		Stream:        raw.Stream,
	}

	// System prompt
	if len(raw.System) > 0 {
		parts, err := convertAnthropicContentBlocks(raw.System)
		if err != nil {
			return nil, fmt.Errorf("decode anthropic request system: %w", err)
		}
		req.SystemPrompt = parts
	}

	// Messages
	if len(raw.Messages) > 0 {
		messages := make([]Message, 0, len(raw.Messages))
		for i, m := range raw.Messages {
			msg, err := convertAnthropicMessage(m)
			if err != nil {
				return nil, fmt.Errorf("decode anthropic request messages[%d]: %w", i, err)
			}
			messages = append(messages, msg)
		}
		req.Messages = messages
	}

	// Tools — skip server (non-custom) tools
	if len(raw.Tools) > 0 {
		tools := make([]Tool, 0, len(raw.Tools))
		for _, t := range raw.Tools {
			// Empty type or "custom" are custom tools; everything else is a server tool.
			if t.Type != "" && t.Type != "custom" {
				continue
			}
			tools = append(tools, Tool{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			})
		}
		if len(tools) > 0 {
			req.Tools = tools
		}
	}

	// Tool choice
	if raw.ToolChoice != nil {
		tc := &ToolChoice{}
		switch raw.ToolChoice.Type {
		case "auto":
			tc.Type = "auto"
		case "any":
			tc.Type = "required"
		case "tool":
			tc.Type = "tool"
			tc.ToolName = raw.ToolChoice.Name
		case "none":
			tc.Type = "none"
		default:
			tc.Type = raw.ToolChoice.Type
		}
		// disable_parallel_tool_use → AllowParallelCalls (inverse)
		if raw.ToolChoice.DisableParallelToolUse != nil {
			allow := !*raw.ToolChoice.DisableParallelToolUse
			tc.AllowParallelCalls = &allow
		}
		req.ToolChoice = tc
	}

	// Thinking config
	if raw.Thinking != nil {
		req.Thinking = &ThinkingConfig{
			Mode:         raw.Thinking.Type,
			BudgetTokens: raw.Thinking.BudgetTokens,
		}
	}

	return req, nil
}

// convertAnthropicMessage converts an anthropic.Message to an IR Message.
func convertAnthropicMessage(m anthropic.Message) (Message, error) {
	var role Role
	switch m.Role {
	case "user":
		role = RoleUser
	case "assistant":
		role = RoleAssistant
	default:
		role = Role(m.Role)
	}

	content, err := convertAnthropicMessageContent(m.Content)
	if err != nil {
		return Message{}, err
	}

	// Anthropic uses role "user" for tool result messages.
	// Promote to RoleTool for cross-protocol compatibility when all parts are tool results.
	if role == RoleUser && len(content) > 0 && allToolResults(content) {
		role = RoleTool
	}

	return Message{
		Role:    role,
		Content: content,
	}, nil
}

// allToolResults returns true if every content part is a tool result.
func allToolResults(parts []ContentPart) bool {
	for _, p := range parts {
		if p.Type != ContentTypeToolResult {
			return false
		}
	}
	return true
}

// convertAnthropicMessageContent handles the dual-form content: either a JSON
// string or a JSON array of content blocks.
func convertAnthropicMessageContent(raw json.RawMessage) ([]ContentPart, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	// Try string shorthand first
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, fmt.Errorf("unmarshal string content: %w", err)
		}
		return []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: text}},
		}, nil
	}

	// Otherwise expect an array of content blocks
	var blocks []anthropic.ContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, fmt.Errorf("unmarshal content blocks: %w", err)
	}
	return convertAnthropicContentBlocks(blocks)
}

// convertAnthropicContentBlocks converts a slice of anthropic.ContentBlock to IR ContentParts.
func convertAnthropicContentBlocks(blocks []anthropic.ContentBlock) ([]ContentPart, error) {
	parts := make([]ContentPart, 0, len(blocks))
	for i, b := range blocks {
		part, err := convertAnthropicContentBlock(b)
		if err != nil {
			return nil, fmt.Errorf("content block[%d]: %w", i, err)
		}
		parts = append(parts, part)
	}
	return parts, nil
}

// convertAnthropicContentBlock converts a single anthropic.ContentBlock to an IR ContentPart.
func convertAnthropicContentBlock(b anthropic.ContentBlock) (ContentPart, error) {
	switch b.Type {
	case "text":
		part := ContentPart{
			Type: ContentTypeText,
			Text: &TextContent{Text: b.Text},
		}
		if len(b.Citations) > 0 {
			citations, err := parseAnthropicCitations(b.Citations)
			if err != nil {
				return ContentPart{}, fmt.Errorf("parse anthropic citations: %w", err)
			}
			part.Citations = citations
		}
		return part, nil

	case "image":
		img, err := convertAnthropicImageSource(b.Source)
		if err != nil {
			return ContentPart{}, err
		}
		return ContentPart{
			Type:  ContentTypeImage,
			Image: img,
		}, nil

	case "document":
		doc, err := convertAnthropicDocumentSource(b.Source)
		if err != nil {
			return ContentPart{}, err
		}
		return ContentPart{
			Type:     ContentTypeDocument,
			Document: doc,
		}, nil

	case "tool_use":
		return ContentPart{
			Type: ContentTypeToolUse,
			ToolUse: &ToolUseContent{
				ID:        b.ID,
				Name:      b.Name,
				Arguments: b.Input,
			},
		}, nil

	case "tool_result":
		var content []ContentPart
		if len(b.ContentRaw) > 0 {
			var err error
			content, err = convertAnthropicMessageContent(b.ContentRaw)
			if err != nil {
				return ContentPart{}, fmt.Errorf("tool_result content: %w", err)
			}
		}
		return ContentPart{
			Type: ContentTypeToolResult,
			ToolResult: &ToolResultContent{
				ToolUseID: b.ToolUseID,
				Content:   content,
				IsError:   b.IsError,
			},
		}, nil

	case "thinking":
		return ContentPart{
			Type: ContentTypeThinking,
			Thinking: &ThinkingContent{
				Thinking:  b.Thinking,
				Signature: b.Signature,
			},
		}, nil

	case "redacted_thinking":
		return ContentPart{
			Type:             ContentTypeRedactedThinking,
			RedactedThinking: &RedactedThinkingContent{Data: b.Data},
		}, nil

	default:
		// Unknown content type — return a best-effort text part with empty text
		// rather than erroring, to be forward-compatible with new block types.
		return ContentPart{
			Type: ContentType(b.Type),
		}, nil
	}
}

// convertAnthropicImageSource converts an anthropic.Source to an ImageContent.
func convertAnthropicImageSource(src *anthropic.Source) (*ImageContent, error) {
	if src == nil {
		return &ImageContent{}, nil
	}
	switch src.Type {
	case "base64":
		data, err := base64.StdEncoding.DecodeString(src.Data)
		if err != nil {
			return nil, fmt.Errorf("decode image base64: %w", err)
		}
		return &ImageContent{
			Data:      data,
			MediaType: src.MediaType,
		}, nil
	case "url":
		return &ImageContent{
			URL: src.URL,
		}, nil
	default:
		return &ImageContent{}, nil
	}
}

// convertAnthropicDocumentSource converts an anthropic.Source to a DocumentContent.
func convertAnthropicDocumentSource(src *anthropic.Source) (*DocumentContent, error) {
	if src == nil {
		return &DocumentContent{}, nil
	}
	switch src.Type {
	case "base64":
		data, err := base64.StdEncoding.DecodeString(src.Data)
		if err != nil {
			return nil, fmt.Errorf("decode document base64: %w", err)
		}
		return &DocumentContent{
			Data:      data,
			MediaType: src.MediaType,
		}, nil
	case "url":
		return &DocumentContent{
			URL: src.URL,
		}, nil
	default:
		return &DocumentContent{}, nil
	}
}

// parseAnthropicCitations parses Anthropic citation raw JSON objects into IR Citation structs.
func parseAnthropicCitations(raw []json.RawMessage) ([]Citation, error) {
	citations := make([]Citation, 0, len(raw))
	for _, r := range raw {
		var wire anthropicCitationWire
		if err := json.Unmarshal(r, &wire); err != nil {
			return nil, fmt.Errorf("unmarshal citation: %w", err)
		}
		c := Citation{
			Kind:  wire.Type,
			URL:   wire.URL,
			Start: wire.StartCharIndex,
			End:   wire.EndCharIndex,
		}
		c.Title = wire.DocumentTitle
		if c.Title == "" {
			c.Title = wire.Title
		}
		if wire.DocumentIndex != nil {
			c.SourceID = strconv.Itoa(*wire.DocumentIndex)
		}
		citations = append(citations, c)
	}
	return citations, nil
}

// encodeAnthropicCitations converts IR Citation structs back to Anthropic citation wire format.
func encodeAnthropicCitations(citations []Citation) []json.RawMessage {
	result := make([]json.RawMessage, 0, len(citations))
	for _, c := range citations {
		var wire anthropicCitationWire
		switch {
		case c.Kind == CitationKindWebSearchResult || (c.URL != "" && c.Kind != CitationKindCharLocation):
			wire.Type = CitationKindWebSearchResult
			wire.Title = c.Title
			wire.URL = c.URL
		default:
			wire.Type = c.Kind
			wire.DocumentTitle = c.Title
			wire.StartCharIndex = c.Start
			wire.EndCharIndex = c.End
			if c.SourceID != "" {
				if docIdx, err := strconv.Atoi(c.SourceID); err == nil {
					wire.DocumentIndex = &docIdx
				}
			}
		}
		data, _ := json.Marshal(wire)
		result = append(result, data)
	}
	return result
}



// defaultAnthropicMaxTokens is the fallback max_tokens when the inbound request
// did not specify one. Anthropic requires a positive value; use a conservative
// default that works for all current Claude models.
const defaultAnthropicMaxTokens = 4096

// EncodeAnthropicRequest encodes a unified IR Request into an Anthropic Messages API JSON body.
func EncodeAnthropicRequest(req *Request) ([]byte, error) {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = defaultAnthropicMaxTokens
	}
	raw := anthropic.Request{
		Model:         req.Model,
		MaxTokens:     maxTokens,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		TopK:          req.TopK,
		StopSequences: req.StopSequences,
		Stream:        req.Stream,
	}

	// System prompt
	if len(req.SystemPrompt) > 0 {
		blocks, err := encodeAnthropicContentParts(req.SystemPrompt)
		if err != nil {
			return nil, fmt.Errorf("encode anthropic request system: %w", err)
		}
		raw.System = blocks
	}

	// Messages
	if len(req.Messages) > 0 {
		msgs := make([]anthropic.Message, 0, len(req.Messages))
		for i, m := range req.Messages {
			msg, err := encodeAnthropicMessage(m)
			if err != nil {
				return nil, fmt.Errorf("encode anthropic request messages[%d]: %w", i, err)
			}
			msgs = append(msgs, msg)
		}
		raw.Messages = msgs
	}

	// Tools
	if len(req.Tools) > 0 {
		tools := make([]anthropic.Tool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, anthropic.Tool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: t.Parameters,
				Type:        "custom",
			})
		}
		raw.Tools = tools
	}

	// Tool choice
	// Only emit tool_choice when Type is set. If Type is empty the IR ToolChoice
	// was synthesised only to carry AllowParallelCalls (e.g. from OpenAI
	// parallel_tool_calls with no tool_choice field). Anthropic requires
	// disable_parallel_tool_use to be nested inside a valid tool_choice object,
	// so when there is no Type we silently drop both fields.
	if req.ToolChoice != nil && req.ToolChoice.Type != "" {
		atc := &anthropic.ToolChoice{}
		switch req.ToolChoice.Type {
		case "auto":
			atc.Type = "auto"
		case "required":
			atc.Type = "any"
		case "tool":
			atc.Type = "tool"
			atc.Name = req.ToolChoice.ToolName
		case "none":
			atc.Type = "none"
		default:
			atc.Type = req.ToolChoice.Type
		}
		// AllowedToolNames degradation: single tool → named tool choice, multi-tool → silently drop
		if len(req.ToolChoice.AllowedToolNames) == 1 {
			atc.Type = "tool"
			atc.Name = req.ToolChoice.AllowedToolNames[0]
		}
		// len > 1: silently drop (cannot represent multi-allowlist in Anthropic)
		// AllowParallelCalls → disable_parallel_tool_use (inverse)
		if req.ToolChoice.AllowParallelCalls != nil {
			disable := !*req.ToolChoice.AllowParallelCalls
			atc.DisableParallelToolUse = &disable
		}
		raw.ToolChoice = atc
	}

	// Thinking config
	if req.Thinking != nil {
		at := &anthropic.Thinking{
			Type: req.Thinking.Mode,
		}
		if req.Thinking.Mode == "enabled" {
			at.BudgetTokens = req.Thinking.BudgetTokens
		}
		raw.Thinking = at
	}

	return json.Marshal(raw)
}

// encodeAnthropicMessage converts an IR Message to an anthropic.Message.
func encodeAnthropicMessage(m Message) (anthropic.Message, error) {
	var role string
	switch m.Role {
	case RoleUser:
		role = "user"
	case RoleAssistant:
		role = "assistant"
	case RoleTool:
		// Anthropic uses "user" role for tool result messages.
		role = "user"
	default:
		role = string(m.Role)
	}

	blocks, err := encodeAnthropicContentParts(m.Content)
	if err != nil {
		return anthropic.Message{}, err
	}

	content, err := json.Marshal(blocks)
	if err != nil {
		return anthropic.Message{}, fmt.Errorf("marshal message content: %w", err)
	}

	return anthropic.Message{
		Role:    role,
		Content: content,
	}, nil
}

// encodeAnthropicContentParts converts a slice of IR ContentParts to anthropic.ContentBlocks.
func encodeAnthropicContentParts(parts []ContentPart) ([]anthropic.ContentBlock, error) {
	blocks := make([]anthropic.ContentBlock, 0, len(parts))
	for i, p := range parts {
		block, err := encodeAnthropicContentPart(p)
		if err != nil {
			return nil, fmt.Errorf("content part[%d]: %w", i, err)
		}
		blocks = append(blocks, block)
	}
	return blocks, nil
}

// encodeAnthropicContentPart converts a single IR ContentPart to an anthropic.ContentBlock.
func encodeAnthropicContentPart(p ContentPart) (anthropic.ContentBlock, error) {
	switch p.Type {
	case ContentTypeText:
		b := anthropic.ContentBlock{Type: "text"}
		if p.Text != nil {
			b.Text = p.Text.Text
		}
		if len(p.Citations) > 0 {
			b.Citations = encodeAnthropicCitations(p.Citations)
		}
		return b, nil

	case ContentTypeImage:
		b := anthropic.ContentBlock{Type: "image"}
		if p.Image != nil {
			if len(p.Image.Data) > 0 {
				b.Source = &anthropic.Source{
					Type:      "base64",
					MediaType: p.Image.MediaType,
					Data:      base64.StdEncoding.EncodeToString(p.Image.Data),
				}
			} else if p.Image.URL != "" {
				b.Source = &anthropic.Source{
					Type: "url",
					URL:  p.Image.URL,
				}
			}
		}
		return b, nil

	case ContentTypeDocument:
		b := anthropic.ContentBlock{Type: "document"}
		if p.Document != nil {
			if len(p.Document.Data) > 0 {
				b.Source = &anthropic.Source{
					Type:      "base64",
					MediaType: p.Document.MediaType,
					Data:      base64.StdEncoding.EncodeToString(p.Document.Data),
				}
			} else if p.Document.URL != "" {
				b.Source = &anthropic.Source{
					Type: "url",
					URL:  p.Document.URL,
				}
			}
		}
		return b, nil

	case ContentTypeToolUse:
		b := anthropic.ContentBlock{Type: "tool_use"}
		if p.ToolUse != nil {
			b.ID = p.ToolUse.ID
			b.Name = p.ToolUse.Name
			b.Input = p.ToolUse.Arguments
		}
		return b, nil

	case ContentTypeToolResult:
		b := anthropic.ContentBlock{Type: "tool_result"}
		if p.ToolResult != nil {
			b.ToolUseID = p.ToolResult.ToolUseID
			b.IsError = p.ToolResult.IsError
			if len(p.ToolResult.Content) > 0 {
				nested, err := encodeAnthropicContentParts(p.ToolResult.Content)
				if err != nil {
					return anthropic.ContentBlock{}, fmt.Errorf("tool_result content: %w", err)
				}
				raw, err := json.Marshal(nested)
				if err != nil {
					return anthropic.ContentBlock{}, fmt.Errorf("tool_result content marshal: %w", err)
				}
				b.ContentRaw = raw
			}
		}
		return b, nil

	case ContentTypeThinking:
		b := anthropic.ContentBlock{Type: "thinking"}
		if p.Thinking != nil {
			b.Thinking = p.Thinking.Thinking
			b.Signature = p.Thinking.Signature
		}
		return b, nil

	case ContentTypeRedactedThinking:
		b := anthropic.ContentBlock{Type: "redacted_thinking"}
		if p.RedactedThinking != nil {
			b.Data = p.RedactedThinking.Data
		}
		return b, nil

	case ContentTypeRefusal:
		// Weak mapping: refusal → text (Anthropic has no native refusal type)
		b := anthropic.ContentBlock{Type: "text"}
		if p.Refusal != nil {
			b.Text = p.Refusal.Refusal
		}
		return b, nil

	default:
		return anthropic.ContentBlock{Type: string(p.Type)}, nil
	}
}

// DecodeAnthropicResponse decodes an Anthropic Messages API JSON response body
// into the unified IR Response type.
func DecodeAnthropicResponse(body []byte) (*Response, error) {
	var raw anthropic.Response
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode anthropic response: %w", err)
	}

	resp := &Response{
		ID:    raw.ID,
		Model: raw.Model,
		Usage: Usage{
			InputTokens:         raw.Usage.InputTokens,
			OutputTokens:        raw.Usage.OutputTokens,
			CacheCreationTokens: raw.Usage.CacheCreationInputTokens,
			CacheReadTokens:     raw.Usage.CacheReadInputTokens,
		},
	}

	// Content
	if len(raw.Content) > 0 {
		parts, err := convertAnthropicContentBlocks(raw.Content)
		if err != nil {
			return nil, fmt.Errorf("decode anthropic response content: %w", err)
		}
		resp.Content = parts
	}

	// Stop reason
	switch raw.StopReason {
	case "end_turn":
		resp.StopReason = StopReasonEndTurn
	case "max_tokens":
		resp.StopReason = StopReasonMaxTokens
	case "stop_sequence":
		resp.StopReason = StopReasonStopSequence
	case "tool_use":
		resp.StopReason = StopReasonToolUse
	case "pause_turn":
		resp.StopReason = StopReasonPauseTurn
	default:
		resp.StopReason = StopReason(raw.StopReason)
	}

	if raw.StopSequence != nil {
		resp.StopSequence = *raw.StopSequence
	}

	return resp, nil
}

// DecodeAnthropicStreamEvent decodes an Anthropic SSE event into the unified IR StreamEvent.
// eventType is from the SSE "event:" line, data is from the SSE "data:" line.
// Returns nil, nil for ping events (should be skipped by the caller).
func DecodeAnthropicStreamEvent(eventType string, data []byte) (*StreamEvent, error) {
	switch eventType {
	case "ping":
		return nil, nil

	case "message_start":
		var raw anthropic.StreamMessageStart
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode anthropic stream message_start: %w", err)
		}
		return &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    raw.Message.ID,
				Model: raw.Message.Model,
				Usage: Usage{
					InputTokens:         raw.Message.Usage.InputTokens,
					OutputTokens:        raw.Message.Usage.OutputTokens,
					CacheCreationTokens: raw.Message.Usage.CacheCreationInputTokens,
					CacheReadTokens:     raw.Message.Usage.CacheReadInputTokens,
				},
			},
		}, nil

	case "content_block_start":
		var raw anthropic.StreamContentBlockStart
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode anthropic stream content_block_start: %w", err)
		}
		part, err := convertAnthropicContentBlock(raw.ContentBlock)
		if err != nil {
			return nil, fmt.Errorf("decode anthropic stream content_block_start block: %w", err)
		}
		return &StreamEvent{
			Type:  StreamEventContentBlockStart,
			Index: raw.Index,
			Delta: &part,
		}, nil

	case "content_block_delta":
		var raw anthropic.StreamContentBlockDelta
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode anthropic stream content_block_delta: %w", err)
		}
		var delta ContentPart
		switch raw.Delta.Type {
		case "text_delta":
			text := ""
			if raw.Delta.Text != nil {
				text = *raw.Delta.Text
			}
			delta = ContentPart{
				Type: ContentTypeText,
				Text: &TextContent{Text: text},
			}
		case "input_json_delta":
			delta = ContentPart{
				Type: ContentTypeToolUse,
				ToolUse: &ToolUseContent{
					Arguments: func() json.RawMessage {
						if raw.Delta.PartialJSON != nil {
							return json.RawMessage(*raw.Delta.PartialJSON)
						}
						return nil
					}(),
				},
			}
		case "thinking_delta":
			thinking := ""
			if raw.Delta.Thinking != nil {
				thinking = *raw.Delta.Thinking
			}
			delta = ContentPart{
				Type: ContentTypeThinking,
				Thinking: &ThinkingContent{
					Thinking: thinking,
				},
			}
		case "signature_delta":
			var sig string
			if raw.Delta.Signature != nil {
				sig = *raw.Delta.Signature
			}
			delta = ContentPart{
				Type: ContentTypeThinking,
				Thinking: &ThinkingContent{
					Signature: sig,
				},
			}
		default:
			delta = ContentPart{Type: ContentType(raw.Delta.Type)}
		}
		return &StreamEvent{
			Type:  StreamEventDelta,
			Index: raw.Index,
			Delta: &delta,
		}, nil

	case "content_block_stop":
		var raw anthropic.StreamContentBlockStop
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode anthropic stream content_block_stop: %w", err)
		}
		return &StreamEvent{
			Type:  StreamEventContentBlockStop,
			Index: raw.Index,
		}, nil

	case "message_delta":
		var raw anthropic.StreamMessageDelta
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode anthropic stream message_delta: %w", err)
		}
		var stopReason StopReason
		switch raw.Delta.StopReason {
		case "end_turn":
			stopReason = StopReasonEndTurn
		case "max_tokens":
			stopReason = StopReasonMaxTokens
		case "stop_sequence":
			stopReason = StopReasonStopSequence
		case "tool_use":
			stopReason = StopReasonToolUse
		case "pause_turn":
			stopReason = StopReasonPauseTurn
		default:
			stopReason = StopReason(raw.Delta.StopReason)
		}
		usage := Usage{
			OutputTokens: raw.Usage.OutputTokens,
		}
		return &StreamEvent{
			Type:       StreamEventDelta,
			StopReason: &stopReason,
			Usage:      &usage,
		}, nil

	case "message_stop":
		return &StreamEvent{
			Type: StreamEventStop,
		}, nil

	case "error":
		// Anthropic SSE error event: {"type":"error","error":{"type":"...","message":"..."}}
		var errPayload struct {
			Type  string `json:"type"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(data, &errPayload); err != nil {
			return nil, fmt.Errorf("decode anthropic stream error: %w", err)
		}
		return &StreamEvent{
			Type: StreamEventError,
			Error: &StreamError{
				Type:    errPayload.Error.Type,
				Message: errPayload.Error.Message,
			},
		}, nil

	default:
		return nil, fmt.Errorf("unknown anthropic stream event type: %q", eventType)
	}
}

// EncodeAnthropicStreamEvent encodes a unified IR StreamEvent into an Anthropic SSE event.
// Returns the SSE event type string and the JSON data bytes.
func EncodeAnthropicStreamEvent(event *StreamEvent) (string, []byte, error) {
	switch event.Type {
	case StreamEventStart:
		msg := anthropic.StreamMessageStart{
			Type: "message_start",
		}
		if event.Response != nil {
			msg.Message = anthropic.StreamMessage{
				ID:    event.Response.ID,
				Type:  "message",
				Model: event.Response.Model,
				Role:  "assistant",
				Usage: anthropic.Usage{
					InputTokens:              event.Response.Usage.InputTokens,
					OutputTokens:             event.Response.Usage.OutputTokens,
					CacheCreationInputTokens: event.Response.Usage.CacheCreationTokens,
					CacheReadInputTokens:     event.Response.Usage.CacheReadTokens,
				},
			}
		}
		data, err := json.Marshal(msg)
		if err != nil {
			return "", nil, fmt.Errorf("encode anthropic stream message_start: %w", err)
		}
		return "message_start", data, nil

	case StreamEventContentBlockStart:
		if event.Delta == nil {
			raw := anthropic.StreamContentBlockStart{
				Type:  "content_block_start",
				Index: event.Index,
			}
			data, err := json.Marshal(raw)
			if err != nil {
				return "", nil, fmt.Errorf("encode anthropic stream content_block_start: %w", err)
			}
			return "content_block_start", data, nil
		}
		block, err := encodeAnthropicContentPart(*event.Delta)
		if err != nil {
			return "", nil, fmt.Errorf("encode anthropic stream content_block_start block: %w", err)
		}
		raw := anthropic.StreamContentBlockStart{
			Type:         "content_block_start",
			Index:        event.Index,
			ContentBlock: block,
		}
		data, err := json.Marshal(raw)
		if err != nil {
			return "", nil, fmt.Errorf("encode anthropic stream content_block_start: %w", err)
		}
		return "content_block_start", data, nil

	case StreamEventDelta:
		// message_delta (stop reason) takes priority over content delta
		if event.StopReason != nil {
			var stopReason string
			switch *event.StopReason {
			case StopReasonEndTurn:
				stopReason = "end_turn"
			case StopReasonMaxTokens:
				stopReason = "max_tokens"
			case StopReasonStopSequence:
				stopReason = "stop_sequence"
			case StopReasonToolUse:
				stopReason = "tool_use"
			case StopReasonPauseTurn:
				stopReason = "pause_turn"
			default:
				stopReason = string(*event.StopReason)
			}
			raw := anthropic.StreamMessageDelta{
				Type: "message_delta",
				Delta: anthropic.StreamMessageDeltaInner{
					StopReason: stopReason,
				},
			}
			if event.Usage != nil {
				raw.Usage = anthropic.Usage{
					InputTokens:              event.Usage.InputTokens,
					OutputTokens:             event.Usage.OutputTokens,
					CacheCreationInputTokens: event.Usage.CacheCreationTokens,
					CacheReadInputTokens:     event.Usage.CacheReadTokens,
				}
			}
			data, err := json.Marshal(raw)
			if err != nil {
				return "", nil, fmt.Errorf("encode anthropic stream message_delta: %w", err)
			}
			return "message_delta", data, nil
		}

		// content_block_delta
		if event.Delta == nil {
			return "", nil, fmt.Errorf("encode anthropic stream delta: nil Delta and nil StopReason")
		}
		var delta anthropic.StreamDelta
		switch event.Delta.Type {
		case ContentTypeText:
			delta.Type = "text_delta"
			text := ""
			if event.Delta.Text != nil {
				text = event.Delta.Text.Text
			}
			delta.Text = &text
		case ContentTypeToolUse:
			delta.Type = "input_json_delta"
			if event.Delta.ToolUse != nil {
				pj := string(event.Delta.ToolUse.Arguments)
				delta.PartialJSON = &pj
			}
		case ContentTypeRefusal:
			// Weak mapping: refusal delta → text_delta (Anthropic has no native refusal type)
			delta.Type = "text_delta"
			refusal := ""
			if event.Delta.Refusal != nil {
				refusal = event.Delta.Refusal.Refusal
			}
			delta.Text = &refusal
		case ContentTypeThinking:
			if event.Delta.Thinking != nil && event.Delta.Thinking.Signature != "" {
				delta.Type = "signature_delta"
				sig := event.Delta.Thinking.Signature
				delta.Signature = &sig
			} else {
				delta.Type = "thinking_delta"
				thinking := ""
				if event.Delta.Thinking != nil {
					thinking = event.Delta.Thinking.Thinking
				}
				delta.Thinking = &thinking
			}
		default:
			delta.Type = string(event.Delta.Type)
		}
		raw := anthropic.StreamContentBlockDelta{
			Type:  "content_block_delta",
			Index: event.Index,
			Delta: delta,
		}
		data, err := json.Marshal(raw)
		if err != nil {
			return "", nil, fmt.Errorf("encode anthropic stream content_block_delta: %w", err)
		}
		return "content_block_delta", data, nil

	case StreamEventContentBlockStop:
		raw := anthropic.StreamContentBlockStop{
			Type:  "content_block_stop",
			Index: event.Index,
		}
		data, err := json.Marshal(raw)
		if err != nil {
			return "", nil, fmt.Errorf("encode anthropic stream content_block_stop: %w", err)
		}
		return "content_block_stop", data, nil

	case StreamEventStop:
		data, err := json.Marshal(map[string]string{"type": "message_stop"})
		if err != nil {
			return "", nil, fmt.Errorf("encode anthropic stream message_stop: %w", err)
		}
		return "message_stop", data, nil

	case StreamEventError:
		errPayload := struct {
			Type  string `json:"type"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}{
			Type: "error",
		}
		if event.Error != nil {
			errPayload.Error.Type = event.Error.Type
			errPayload.Error.Message = event.Error.Message
		}
		data, err := json.Marshal(errPayload)
		if err != nil {
			return "", nil, fmt.Errorf("encode anthropic stream error: %w", err)
		}
		return "error", data, nil

	default:
		return "", nil, fmt.Errorf("unknown IR stream event type: %q", event.Type)
	}
}

// EncodeAnthropicResponse encodes a unified IR Response into an Anthropic Messages API JSON body.
func EncodeAnthropicResponse(resp *Response) ([]byte, error) {
	raw := anthropic.Response{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: resp.Model,
		Usage: anthropic.Usage{
			InputTokens:              resp.Usage.InputTokens,
			OutputTokens:             resp.Usage.OutputTokens,
			CacheCreationInputTokens: resp.Usage.CacheCreationTokens,
			CacheReadInputTokens:     resp.Usage.CacheReadTokens,
		},
	}

	// Content — always include (Anthropic API requires it, even as empty array)
	blocks, err := encodeAnthropicContentParts(resp.Content)
	if err != nil {
		return nil, fmt.Errorf("encode anthropic response content: %w", err)
	}
	raw.Content = blocks

	// Stop reason (reverse mapping)
	switch resp.StopReason {
	case StopReasonEndTurn:
		raw.StopReason = "end_turn"
	case StopReasonMaxTokens:
		raw.StopReason = "max_tokens"
	case StopReasonStopSequence:
		raw.StopReason = "stop_sequence"
	case StopReasonToolUse:
		raw.StopReason = "tool_use"
	case StopReasonPauseTurn:
		raw.StopReason = "pause_turn"
	default:
		raw.StopReason = string(resp.StopReason)
	}

	if resp.StopSequence != "" {
		raw.StopSequence = &resp.StopSequence
	}

	return json.Marshal(raw)
}
