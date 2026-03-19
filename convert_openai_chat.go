package llmapimux

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/llmapimux/llmapimux/protocol/openaichat"
)

// decodeOpenAIChatFinishReason maps an OpenAI Chat finish_reason string to an IR StopReason.
func decodeOpenAIChatFinishReason(s string) StopReason {
	switch s {
	case "stop":
		return StopReasonEndTurn
	case "length":
		return StopReasonMaxTokens
	case "tool_calls":
		return StopReasonToolUse
	case "content_filter":
		return StopReasonContentFilter
	default:
		return StopReason(s)
	}
}

// encodeOpenAIChatFinishReason maps an IR StopReason to an OpenAI Chat finish_reason string.
func encodeOpenAIChatFinishReason(r StopReason) string {
	switch r {
	case StopReasonEndTurn:
		return "stop"
	case StopReasonMaxTokens:
		return "length"
	case StopReasonToolUse:
		return "tool_calls"
	case StopReasonContentFilter:
		return "content_filter"
	case StopReasonStopSequence:
		return "stop"
	case StopReasonPauseTurn:
		return "stop"
	default:
		return string(r)
	}
}

func decodeOpenAIChatUsage(u *openaichat.ChatUsage) Usage {
	usage := Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
	}
	if u.PromptTokensDetails != nil {
		usage.CacheReadTokens = u.PromptTokensDetails.CachedTokens
	}
	if u.CompletionTokensDetails != nil {
		usage.ThinkingTokens = u.CompletionTokensDetails.ReasoningTokens
	}
	return usage
}

func encodeOpenAIChatUsage(u *Usage) *openaichat.ChatUsage {
	raw := &openaichat.ChatUsage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.CacheReadTokens != 0 {
		raw.PromptTokensDetails = &openaichat.ChatPromptDetails{CachedTokens: u.CacheReadTokens}
	}
	if u.ThinkingTokens != 0 {
		raw.CompletionTokensDetails = &openaichat.ChatCompletionDetails{ReasoningTokens: u.ThinkingTokens}
	}
	return raw
}

// --- Decode functions ---

// DecodeOpenAIChatRequest decodes an OpenAI Chat Completions API JSON request body
// into the unified IR Request type.
func DecodeOpenAIChatRequest(body []byte) (*Request, error) {
	var raw openaichat.ChatRequest
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode openai chat request: %w", err)
	}

	req := &Request{
		Model:       raw.Model,
		Temperature: raw.Temperature,
		TopP:        raw.TopP,
		Stream:      raw.Stream,
	}

	// max_completion_tokens takes precedence over max_tokens
	if raw.MaxCompletionTokens != nil {
		req.MaxTokens = *raw.MaxCompletionTokens
	} else if raw.MaxTokens != nil {
		req.MaxTokens = *raw.MaxTokens
	}

	// Stop sequences: can be string or array of strings
	if len(raw.Stop) > 0 {
		stops, err := decodeOpenAIStop(raw.Stop)
		if err != nil {
			return nil, fmt.Errorf("decode openai chat request stop: %w", err)
		}
		req.StopSequences = stops
	}

	// Messages
	var systemParts []ContentPart
	var messages []Message
	for i, m := range raw.Messages {
		switch m.Role {
		case "system", "developer":
			// Accumulate into SystemPrompt
			parts, err := decodeOpenAIChatMessageContent(m.Content)
			if err != nil {
				return nil, fmt.Errorf("decode openai chat request messages[%d]: %w", i, err)
			}
			systemParts = append(systemParts, parts...)

		case "user":
			parts, err := decodeOpenAIChatMessageContent(m.Content)
			if err != nil {
				return nil, fmt.Errorf("decode openai chat request messages[%d]: %w", i, err)
			}
			messages = append(messages, Message{
				Role:    RoleUser,
				Content: parts,
			})

		case "assistant":
			parts, err := decodeOpenAIChatAssistantMessage(m)
			if err != nil {
				return nil, fmt.Errorf("decode openai chat request messages[%d]: %w", i, err)
			}
			messages = append(messages, Message{
				Role:    RoleAssistant,
				Content: parts,
			})

		case "tool":
			parts := []ContentPart{
				{
					Type: ContentTypeToolResult,
					ToolResult: &ToolResultContent{
						ToolUseID: m.ToolCallID,
						Content: []ContentPart{
							{Type: ContentTypeText, Text: &TextContent{Text: decodeOpenAIChatStringContent(m.Content)}},
						},
						IsError: false,
					},
				},
			}
			messages = append(messages, Message{
				Role:    RoleTool,
				Content: parts,
			})

		default:
			// Unknown role — pass through
			parts, err := decodeOpenAIChatMessageContent(m.Content)
			if err != nil {
				return nil, fmt.Errorf("decode openai chat request messages[%d]: %w", i, err)
			}
			messages = append(messages, Message{
				Role:    Role(m.Role),
				Content: parts,
			})
		}
	}
	if len(systemParts) > 0 {
		req.SystemPrompt = systemParts
	}
	if len(messages) > 0 {
		req.Messages = messages
	}

	// Tools
	if len(raw.Tools) > 0 {
		tools := make([]Tool, 0, len(raw.Tools))
		for _, t := range raw.Tools {
			if t.Type != "function" {
				continue
			}
			tools = append(tools, Tool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
				Strict:      t.Function.Strict,
			})
		}
		if len(tools) > 0 {
			req.Tools = tools
		}
	}

	// Tool choice
	if len(raw.ToolChoice) > 0 {
		tc, err := decodeOpenAIChatToolChoice(raw.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("decode openai chat request tool_choice: %w", err)
		}
		req.ToolChoice = tc
	}

	// parallel_tool_calls → IR AllowParallelCalls
	if raw.ParallelToolCalls != nil {
		if req.ToolChoice == nil {
			req.ToolChoice = &ToolChoice{}
		}
		req.ToolChoice.AllowParallelCalls = raw.ParallelToolCalls
	}

	// Response format
	if raw.ResponseFormat != nil {
		rf := &ResponseFormat{
			Type: raw.ResponseFormat.Type,
		}
		if raw.ResponseFormat.JSONSchema != nil {
			rf.JSONSchema = raw.ResponseFormat.JSONSchema.Schema
		}
		req.ResponseFormat = rf
	}

	// Reasoning effort
	if raw.ReasoningEffort != "" {
		req.Thinking = &ThinkingConfig{
			Mode:   "enabled",
			Effort: raw.ReasoningEffort,
		}
	}

	return req, nil
}


// decodeOpenAIChatToolChoice decodes the tool_choice field which can be a string or object.
func decodeOpenAIChatToolChoice(raw json.RawMessage) (*ToolChoice, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	// Try string first
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		return &ToolChoice{Type: s}, nil
	}
	// Object form
	var obj openaichat.ChatToolChoiceObj
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, err
	}
	tc := &ToolChoice{Type: "tool"}
	if obj.Function != nil {
		tc.ToolName = obj.Function.Name
	}
	return tc, nil
}

// decodeOpenAIChatMessageContent decodes the content field of a message
// which can be a string or array of content parts.
func decodeOpenAIChatMessageContent(raw json.RawMessage) ([]ContentPart, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	// Try string shorthand
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, fmt.Errorf("unmarshal string content: %w", err)
		}
		return []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: text}},
		}, nil
	}
	// Array form
	var parts []openaichat.ChatContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("unmarshal content parts: %w", err)
	}
	result := make([]ContentPart, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case "text":
			result = append(result, ContentPart{
				Type: ContentTypeText,
				Text: &TextContent{Text: p.Text},
			})
		case "image_url":
			if p.ImageURL == nil {
				continue
			}
			img, err := decodeOpenAIChatImageURL(p.ImageURL)
			if err != nil {
				return nil, fmt.Errorf("decode image_url: %w", err)
			}
			result = append(result, ContentPart{
				Type:  ContentTypeImage,
				Image: img,
			})
		default:
			// Unknown type — pass through as-is
			result = append(result, ContentPart{Type: ContentType(p.Type)})
		}
	}
	return result, nil
}

// decodeOpenAIChatStringContent decodes the content field as a plain string.
func decodeOpenAIChatStringContent(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return ""
	}
	return s
}

// decodeOpenAIChatImageURL decodes an image_url content part into an IR ImageContent.
func decodeOpenAIChatImageURL(img *openaichat.ChatImageURL) (*ImageContent, error) {
	if strings.HasPrefix(img.URL, "data:") {
		// Parse data URI: data:<media_type>;base64,<data>
		mediaType, b64Data, err := parseDataURI(img.URL)
		if err != nil {
			return nil, err
		}
		data, err := base64.StdEncoding.DecodeString(b64Data)
		if err != nil {
			return nil, fmt.Errorf("decode base64 image data: %w", err)
		}
		return &ImageContent{
			Data:      data,
			MediaType: mediaType,
			Detail:    img.Detail,
		}, nil
	}
	return &ImageContent{
		URL:    img.URL,
		Detail: img.Detail,
	}, nil
}

// parseDataURI parses a data URI of the form "data:<mediatype>;base64,<data>".
func parseDataURI(uri string) (mediaType string, data string, err error) {
	// Remove "data:" prefix
	rest := strings.TrimPrefix(uri, "data:")
	// Split on comma
	parts := strings.SplitN(rest, ",", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("invalid data URI: missing comma")
	}
	// The first part is "mediatype;base64"
	meta := parts[0]
	data = parts[1]
	// Extract media type (strip ";base64")
	mediaType = strings.TrimSuffix(meta, ";base64")
	return mediaType, data, nil
}

// decodeOpenAIChatAssistantMessage decodes an assistant message which may have
// text content and/or tool_calls.
func decodeOpenAIChatAssistantMessage(m openaichat.ChatMessage) ([]ContentPart, error) {
	var parts []ContentPart

	// Text content — support both string shorthand and array format.
	if len(m.Content) > 0 && string(m.Content) != "null" {
		contentParts, err := decodeOpenAIChatMessageContent(m.Content)
		if err != nil {
			return nil, fmt.Errorf("decode assistant content: %w", err)
		}
		parts = append(parts, contentParts...)
	}

	// Tool calls
	for _, tc := range m.ToolCalls {
		parts = append(parts, ContentPart{
			Type: ContentTypeToolUse,
			ToolUse: &ToolUseContent{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: json.RawMessage(tc.Function.Arguments),
			},
		})
	}

	return parts, nil
}

// --- Encode functions ---

// EncodeOpenAIChatRequest encodes a unified IR Request into an OpenAI Chat Completions API JSON body.
func EncodeOpenAIChatRequest(req *Request) ([]byte, error) {
	raw := openaichat.ChatRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}

	// MaxTokens → max_completion_tokens
	if req.MaxTokens > 0 {
		mt := req.MaxTokens
		raw.MaxCompletionTokens = &mt
	}

	// Stop sequences
	if len(req.StopSequences) > 0 {
		data, err := json.Marshal(req.StopSequences)
		if err != nil {
			return nil, fmt.Errorf("encode openai chat request stop: %w", err)
		}
		raw.Stop = data
	}

	// System prompt → developer role message
	if len(req.SystemPrompt) > 0 {
		content := encodeOpenAIChatContentParts(req.SystemPrompt)
		contentJSON, err := json.Marshal(content)
		if err != nil {
			return nil, fmt.Errorf("encode openai chat request system: %w", err)
		}
		raw.Messages = append(raw.Messages, openaichat.ChatMessage{
			Role:    "developer",
			Content: contentJSON,
		})
	}

	// Messages
	for i, m := range req.Messages {
		msgs, err := encodeOpenAIChatMessages(m)
		if err != nil {
			return nil, fmt.Errorf("encode openai chat request messages[%d]: %w", i, err)
		}
		raw.Messages = append(raw.Messages, msgs...)
	}

	// Tools
	if len(req.Tools) > 0 {
		tools := make([]openaichat.ChatTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, openaichat.ChatTool{
				Type: "function",
				Function: openaichat.ChatFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
					Strict:      t.Strict,
				},
			})
		}
		raw.Tools = tools
	}

	// Tool choice
	if req.ToolChoice != nil {
		tc, err := encodeOpenAIChatToolChoice(req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("encode openai chat request tool_choice: %w", err)
		}
		raw.ToolChoice = tc

		// AllowParallelCalls → parallel_tool_calls (top-level)
		if req.ToolChoice.AllowParallelCalls != nil {
			raw.ParallelToolCalls = req.ToolChoice.AllowParallelCalls
		}
	}

	// Response format
	if req.ResponseFormat != nil {
		rf := &openaichat.ChatResponseFormat{
			Type: req.ResponseFormat.Type,
		}
		if req.ResponseFormat.Type == "json_schema" && len(req.ResponseFormat.JSONSchema) > 0 {
			rf.JSONSchema = &openaichat.ChatResponseJSONSchema{
				Schema: req.ResponseFormat.JSONSchema,
			}
		}
		raw.ResponseFormat = rf
	}

	// Thinking config → reasoning_effort
	if req.Thinking != nil {
		if req.Thinking.Effort != "" {
			raw.ReasoningEffort = req.Thinking.Effort
		} else if req.Thinking.Mode == "enabled" {
			raw.ReasoningEffort = "medium"
		}
	}

	return json.Marshal(raw)
}

// encodeOpenAIChatMessages converts an IR Message to one or more openaichat.ChatMessages.
// A single IR message may produce multiple Chat messages when a RoleUser message
// contains mixed content (e.g. tool_result + text from Anthropic inbound).
func encodeOpenAIChatMessages(m Message) ([]openaichat.ChatMessage, error) {
	switch m.Role {
	case RoleUser:
		// Split tool_result parts into separate "tool" messages;
		// remaining parts become a user message.
		var msgs []openaichat.ChatMessage
		var userParts []ContentPart
		for _, p := range m.Content {
			if p.Type == ContentTypeToolResult && p.ToolResult != nil {
				toolMsg, err := encodeOpenAIChatToolResultPart(p)
				if err != nil {
					return nil, err
				}
				msgs = append(msgs, toolMsg)
			} else {
				userParts = append(userParts, p)
			}
		}
		if len(userParts) > 0 {
			content := encodeOpenAIChatContentParts(userParts)
			contentJSON, err := json.Marshal(content)
			if err != nil {
				return nil, fmt.Errorf("marshal user content: %w", err)
			}
			msgs = append(msgs, openaichat.ChatMessage{
				Role:    "user",
				Content: contentJSON,
			})
		}
		return msgs, nil

	case RoleAssistant:
		msg, err := encodeOpenAIChatAssistantMessage(m)
		if err != nil {
			return nil, err
		}
		return []openaichat.ChatMessage{msg}, nil

	case RoleTool:
		msg, err := encodeOpenAIChatToolMessage(m)
		if err != nil {
			return nil, err
		}
		return []openaichat.ChatMessage{msg}, nil

	default:
		content := encodeOpenAIChatContentParts(m.Content)
		contentJSON, err := json.Marshal(content)
		if err != nil {
			return nil, fmt.Errorf("marshal content: %w", err)
		}
		return []openaichat.ChatMessage{
			{
				Role:    string(m.Role),
				Content: contentJSON,
			},
		}, nil
	}
}

// encodeOpenAIChatToolResultPart converts a single ContentTypeToolResult part
// to an OpenAI Chat tool message.
func encodeOpenAIChatToolResultPart(p ContentPart) (openaichat.ChatMessage, error) {
	msg := openaichat.ChatMessage{
		Role:       "tool",
		ToolCallID: p.ToolResult.ToolUseID,
	}
	contentJSON, err := json.Marshal(toolResultText(p.ToolResult))
	if err != nil {
		return openaichat.ChatMessage{}, fmt.Errorf("marshal tool content: %w", err)
	}
	msg.Content = contentJSON
	return msg, nil
}

// encodeOpenAIChatAssistantMessage encodes an assistant message, splitting text, refusal and tool_calls.
func encodeOpenAIChatAssistantMessage(m Message) (openaichat.ChatMessage, error) {
	msg := openaichat.ChatMessage{
		Role: "assistant",
	}

	var textParts []string
	var toolCalls []openaichat.ToolCall

	for _, p := range m.Content {
		switch p.Type {
		case ContentTypeText:
			if p.Text != nil {
				textParts = append(textParts, p.Text.Text)
			}
		case ContentTypeRefusal:
			// Refusal in request message history — degrade to text
			if p.Refusal != nil {
				textParts = append(textParts, p.Refusal.Refusal)
			}
		case ContentTypeToolUse:
			if p.ToolUse != nil {
				toolCalls = append(toolCalls, openaichat.ToolCall{
					ID:   p.ToolUse.ID,
					Type: "function",
					Function: openaichat.ToolCallFunction{
						Name:      p.ToolUse.Name,
						Arguments: string(p.ToolUse.Arguments),
					},
				})
			}
		}
	}

	if len(textParts) > 0 {
		text := strings.Join(textParts, "")
		contentJSON, err := json.Marshal(text)
		if err != nil {
			return openaichat.ChatMessage{}, fmt.Errorf("marshal assistant content: %w", err)
		}
		msg.Content = contentJSON
	}

	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	return msg, nil
}

// encodeOpenAIChatToolMessage encodes a tool role message.
func encodeOpenAIChatToolMessage(m Message) (openaichat.ChatMessage, error) {
	msg := openaichat.ChatMessage{
		Role: "tool",
	}

	// Extract tool result
	for _, p := range m.Content {
		if p.Type == ContentTypeToolResult && p.ToolResult != nil {
			msg.ToolCallID = p.ToolResult.ToolUseID
			contentJSON, err := json.Marshal(toolResultText(p.ToolResult))
			if err != nil {
				return openaichat.ChatMessage{}, fmt.Errorf("marshal tool content: %w", err)
			}
			msg.Content = contentJSON
			break
		}
	}

	return msg, nil
}

// encodeOpenAIChatContentParts converts IR ContentParts to OpenAI Chat content parts.
func encodeOpenAIChatContentParts(parts []ContentPart) []openaichat.ChatContentPart {
	result := make([]openaichat.ChatContentPart, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case ContentTypeText:
			cp := openaichat.ChatContentPart{Type: "text"}
			if p.Text != nil {
				cp.Text = p.Text.Text
			}
			result = append(result, cp)
		case ContentTypeImage:
			if p.Image != nil {
				cp := openaichat.ChatContentPart{
					Type: "image_url",
					ImageURL: &openaichat.ChatImageURL{
						Detail: p.Image.Detail,
					},
				}
				if len(p.Image.Data) > 0 {
					// Build data URI
					cp.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s",
						p.Image.MediaType,
						base64.StdEncoding.EncodeToString(p.Image.Data))
				} else if p.Image.URL != "" {
					cp.ImageURL.URL = p.Image.URL
				}
				result = append(result, cp)
			}
		}
	}
	return result
}

// encodeOpenAIChatToolChoice converts an IR ToolChoice to the OpenAI tool_choice JSON.
// Returns nil, nil when tc.Type is empty (e.g. ToolChoice was created only to carry
// AllowParallelCalls with no tool_choice type set). The caller must not emit tool_choice
// in that case; parallel_tool_calls is already emitted separately as a top-level field.
func encodeOpenAIChatToolChoice(tc *ToolChoice) (json.RawMessage, error) {
	switch tc.Type {
	case "":
		return nil, nil
	case "auto", "none", "required":
		return json.Marshal(tc.Type)
	case "tool":
		obj := openaichat.ChatToolChoiceObj{
			Type: "function",
			Function: &openaichat.ChatToolChoiceFunc{
				Name: tc.ToolName,
			},
		}
		return json.Marshal(obj)
	default:
		return json.Marshal(tc.Type)
	}
}


// --- Response decode/encode ---

// DecodeOpenAIChatResponse decodes an OpenAI Chat Completions API JSON response body
// into the unified IR Response type.
func DecodeOpenAIChatResponse(body []byte) (*Response, error) {
	var raw openaichat.ChatResponse
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode openai chat response: %w", err)
	}

	resp := &Response{
		ID:    raw.ID,
		Model: raw.Model,
	}

	// Usage
	if raw.Usage != nil {
		resp.Usage = decodeOpenAIChatUsage(raw.Usage)
	}

	// Choices (use first choice)
	if len(raw.Choices) > 0 {
		choice := raw.Choices[0]

		if choice.Message != nil {
			// Parse annotations into IR citations (attached to first text part)
			var msgCitations []Citation
			if len(choice.Message.Annotations) > 0 {
				var err error
				msgCitations, err = parseOpenAIAnnotations(choice.Message.Annotations)
				if err != nil {
					return nil, fmt.Errorf("decode openai chat response annotations: %w", err)
				}
			}

			// Text content
			if choice.Message.Content != nil && *choice.Message.Content != "" {
				textPart := ContentPart{
					Type: ContentTypeText,
					Text: &TextContent{Text: *choice.Message.Content},
				}
				if len(msgCitations) > 0 {
					textPart.Citations = msgCitations
				}
				resp.Content = append(resp.Content, textPart)
			}

			// Refusal content
			if choice.Message.Refusal != nil && *choice.Message.Refusal != "" {
				resp.Content = append(resp.Content, ContentPart{
					Type:    ContentTypeRefusal,
					Refusal: &RefusalContent{Refusal: *choice.Message.Refusal},
				})
			}

			// Tool calls
			for _, tc := range choice.Message.ToolCalls {
				resp.Content = append(resp.Content, ContentPart{
					Type: ContentTypeToolUse,
					ToolUse: &ToolUseContent{
						ID:        tc.ID,
						Name:      tc.Function.Name,
						Arguments: json.RawMessage(tc.Function.Arguments),
					},
				})
			}
		}

		// Finish reason
		if choice.FinishReason != nil {
			resp.StopReason = decodeOpenAIChatFinishReason(*choice.FinishReason)
		}
		// Some providers return finish_reason "stop" even when tool calls are present.
		// Normalize to tool_use so downstream consumers can rely on stop reason.
		if resp.StopReason == StopReasonEndTurn && choice.Message != nil && len(choice.Message.ToolCalls) > 0 {
			resp.StopReason = StopReasonToolUse
		}
	}

	return resp, nil
}

// EncodeOpenAIChatResponse encodes a unified IR Response into an OpenAI Chat Completions API JSON body.
func EncodeOpenAIChatResponse(resp *Response) ([]byte, error) {
	raw := openaichat.ChatResponse{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  resp.Model,
	}

	// Build the choice message
	msg := &openaichat.ChatChoiceMessage{
		Role: "assistant",
	}

	var textParts []string
	var refusalParts []string
	var toolCalls []openaichat.ToolCall
	var allCitations []Citation

	for _, p := range resp.Content {
		switch p.Type {
		case ContentTypeText:
			if p.Text != nil {
				textParts = append(textParts, p.Text.Text)
			}
			if len(p.Citations) > 0 {
				allCitations = append(allCitations, p.Citations...)
			}
		case ContentTypeRefusal:
			if p.Refusal != nil {
				refusalParts = append(refusalParts, p.Refusal.Refusal)
			}
		case ContentTypeToolUse:
			if p.ToolUse != nil {
				toolCalls = append(toolCalls, openaichat.ToolCall{
					ID:   p.ToolUse.ID,
					Type: "function",
					Function: openaichat.ToolCallFunction{
						Name:      p.ToolUse.Name,
						Arguments: string(p.ToolUse.Arguments),
					},
				})
			}
		}
	}

	if len(textParts) > 0 {
		text := strings.Join(textParts, "")
		msg.Content = &text
	}
	if len(refusalParts) > 0 {
		refusal := strings.Join(refusalParts, "")
		msg.Refusal = &refusal
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	if len(allCitations) > 0 {
		msg.Annotations = encodeOpenAIAnnotations(allCitations)
	}

	// Finish reason
	finishReason := encodeOpenAIChatFinishReason(resp.StopReason)

	raw.Choices = []openaichat.ChatChoice{
		{
			Index:        0,
			Message:      msg,
			FinishReason: &finishReason,
		},
	}

	// Usage
	raw.Usage = encodeOpenAIChatUsage(&resp.Usage)

	return json.Marshal(raw)
}

// --- Streaming decode/encode ---

// DecodeOpenAIChatStreamChunk decodes an OpenAI Chat streaming chunk JSON (the data
// from an SSE "data:" line) into the unified IR StreamEvent.
func DecodeOpenAIChatStreamChunk(data []byte) (*StreamEvent, error) {
	var raw openaichat.ChatStreamChunk
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("decode openai chat stream chunk: %w", err)
	}

	// No choices — could be a usage-only chunk
	if len(raw.Choices) == 0 {
		if raw.Usage != nil {
			u := decodeOpenAIChatUsage(raw.Usage)
			return &StreamEvent{
				Type:  StreamEventDelta,
				Usage: &u,
			}, nil
		}
		// Start event with just ID/Model
		return &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    raw.ID,
				Model: raw.Model,
			},
		}, nil
	}

	choice := raw.Choices[0]
	delta := choice.Delta

	// Check for finish_reason
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		stopReason := decodeOpenAIChatFinishReason(*choice.FinishReason)
		event := &StreamEvent{
			Type:       StreamEventStop,
			StopReason: &stopReason,
		}
		if raw.Usage != nil {
			u := decodeOpenAIChatUsage(raw.Usage)
			event.Usage = &u
		}
		return event, nil
	}

	// First chunk with role
	if delta != nil && delta.Role != "" && delta.Content == nil && delta.Refusal == nil && len(delta.ToolCalls) == 0 {
		return &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    raw.ID,
				Model: raw.Model,
			},
		}, nil
	}

	// Refusal delta
	if delta != nil && delta.Refusal != nil {
		return &StreamEvent{
			Type: StreamEventDelta,
			Delta: &ContentPart{
				Type:    ContentTypeRefusal,
				Refusal: &RefusalContent{Refusal: *delta.Refusal},
			},
		}, nil
	}

	// Content delta
	if delta != nil && delta.Content != nil {
		return &StreamEvent{
			Type: StreamEventDelta,
			Delta: &ContentPart{
				Type: ContentTypeText,
				Text: &TextContent{Text: *delta.Content},
			},
		}, nil
	}

	// Tool calls delta
	if delta != nil && len(delta.ToolCalls) > 0 {
		tc := delta.ToolCalls[0]
		return &StreamEvent{
			Type:  StreamEventDelta,
			Index: tc.Index,
			Delta: &ContentPart{
				Type: ContentTypeToolUse,
				ToolUse: &ToolUseContent{
					ID:        tc.ID,
					Name:      tc.Function.Name,
					Arguments: json.RawMessage(tc.Function.Arguments),
				},
			},
		}, nil
	}

	// Default: start event
	return &StreamEvent{
		Type: StreamEventStart,
		Response: &Response{
			ID:    raw.ID,
			Model: raw.Model,
		},
	}, nil
}

// EncodeOpenAIChatStreamChunk encodes a unified IR StreamEvent into an OpenAI Chat
// streaming chunk JSON (suitable for an SSE "data:" line).
func EncodeOpenAIChatStreamChunk(event *StreamEvent) ([]byte, error) {
	raw := openaichat.ChatStreamChunk{
		Object: "chat.completion.chunk",
	}

	switch event.Type {
	case StreamEventStart:
		if event.Response != nil {
			raw.ID = event.Response.ID
			raw.Model = event.Response.Model
		}
		role := "assistant"
		raw.Choices = []openaichat.ChatChoice{
			{
				Index: 0,
				Delta: &openaichat.ChatChoiceMessage{
					Role: role,
				},
				FinishReason: nil,
			},
		}

	case StreamEventDelta:
		if event.Delta != nil {
			switch event.Delta.Type {
			case ContentTypeText:
				var text string
				if event.Delta.Text != nil {
					text = event.Delta.Text.Text
				}
				raw.Choices = []openaichat.ChatChoice{
					{
						Index: event.Index,
						Delta: &openaichat.ChatChoiceMessage{
							Content: &text,
						},
						FinishReason: nil,
					},
				}
			case ContentTypeRefusal:
				var refusal string
				if event.Delta.Refusal != nil {
					refusal = event.Delta.Refusal.Refusal
				}
				raw.Choices = []openaichat.ChatChoice{
					{
						Index: event.Index,
						Delta: &openaichat.ChatChoiceMessage{
							Refusal: &refusal,
						},
						FinishReason: nil,
					},
				}
			case ContentTypeToolUse:
				if event.Delta.ToolUse != nil {
					tc := openaichat.ToolCall{
						Index: event.Index,
						ID:    event.Delta.ToolUse.ID,
						Type:  "function",
						Function: openaichat.ToolCallFunction{
							Name:      event.Delta.ToolUse.Name,
							Arguments: string(event.Delta.ToolUse.Arguments),
						},
					}
					raw.Choices = []openaichat.ChatChoice{
						{
							Index: 0,
							Delta: &openaichat.ChatChoiceMessage{
								ToolCalls: []openaichat.ToolCall{tc},
							},
							FinishReason: nil,
						},
					}
				}
			}
		}
		// Usage-only delta
		if event.Usage != nil {
			raw.Usage = encodeOpenAIChatUsage(event.Usage)
		}

	case StreamEventStop:
		finishReason := "stop"
		if event.StopReason != nil {
			finishReason = encodeOpenAIChatFinishReason(*event.StopReason)
		}
		raw.Choices = []openaichat.ChatChoice{
			{
				Index:        0,
				Delta:        &openaichat.ChatChoiceMessage{},
				FinishReason: &finishReason,
			},
		}
		if event.Usage != nil {
			raw.Usage = encodeOpenAIChatUsage(event.Usage)
		}

	case StreamEventContentBlockStart, StreamEventContentBlockStop:
		// These lifecycle events have no equivalent in OpenAI Chat streaming; skip silently.
		return nil, nil

	case StreamEventError:
		// Error events have no direct equivalent in OpenAI Chat streaming chunks; skip silently.
		return nil, nil

	default:
		return nil, fmt.Errorf("unknown IR stream event type: %q", event.Type)
	}

	return json.Marshal(raw)
}
