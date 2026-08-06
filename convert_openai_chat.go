package llmapimux

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mahoushoujoarale/llmapimux/protocol/openaichat"
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
//
// The result is always one of OpenAI's documented finish_reason values. Unknown IR
// reasons (protocol-specific values such as Anthropic's "refusal" or
// "model_context_window_exceeded", which reach the IR verbatim) are bucketed onto
// the closest valid value instead of being passed through, because clients and
// SDKs parse finish_reason as a closed enum.
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
		return "length"
	case "":
		return ""
	default:
		return openAIFinishReasonFallback(string(r))
	}
}

// openAIFinishReasonFallback buckets an unrecognised stop reason onto a valid
// OpenAI finish_reason value using substring heuristics over the vocabularies used
// by Anthropic, Gemini and OpenAI-compatible providers.
func openAIFinishReasonFallback(raw string) string {
	s := strings.ToLower(raw)
	switch {
	case strings.Contains(s, "tool") || strings.Contains(s, "function"):
		return "tool_calls"
	case strings.Contains(s, "refus") || strings.Contains(s, "safety") ||
		strings.Contains(s, "filter") || strings.Contains(s, "block") ||
		strings.Contains(s, "recitation") || strings.Contains(s, "prohibited") ||
		strings.Contains(s, "blocklist"):
		return "content_filter"
	case strings.Contains(s, "length") || strings.Contains(s, "token") ||
		strings.Contains(s, "max") || strings.Contains(s, "window") ||
		strings.Contains(s, "truncat"):
		return "length"
	default:
		return "stop"
	}
}

func decodeOpenAIChatUsage(u *openaichat.ChatUsage) Usage {
	usage := Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.PromptTokensDetails != nil {
		usage.PromptCacheHitTokens = u.PromptTokensDetails.CachedTokens
		usage.PromptAudioTokens = u.PromptTokensDetails.AudioTokens
	}
	if u.CompletionTokensDetails != nil {
		usage.CompletionReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
		usage.CompletionAudioTokens = u.CompletionTokensDetails.AudioTokens
		usage.CompletionAcceptedPrediction = u.CompletionTokensDetails.AcceptedPredictionTokens
		usage.CompletionRejectedPrediction = u.CompletionTokensDetails.RejectedPredictionTokens
	}
	return usage
}

func encodeOpenAIChatUsage(u *Usage) *openaichat.ChatUsage {
	raw := &openaichat.ChatUsage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.PromptCacheHitTokens != 0 || u.PromptAudioTokens != 0 {
		raw.PromptTokensDetails = &openaichat.ChatPromptDetails{
			CachedTokens: u.PromptCacheHitTokens,
			AudioTokens:  u.PromptAudioTokens,
		}
	}
	if u.CompletionReasoningTokens != 0 || u.CompletionAudioTokens != 0 || u.CompletionAcceptedPrediction != 0 || u.CompletionRejectedPrediction != 0 {
		raw.CompletionTokensDetails = &openaichat.ChatCompletionDetails{
			ReasoningTokens:          u.CompletionReasoningTokens,
			AudioTokens:              u.CompletionAudioTokens,
			AcceptedPredictionTokens: u.CompletionAcceptedPrediction,
			RejectedPredictionTokens: u.CompletionRejectedPrediction,
		}
	}
	return raw
}

// --- Decode functions ---

// DecodeOpenAIChatRequest decodes an OpenAI Chat Completions API JSON request body
// into the unified IR Request type.
func DecodeOpenAIChatRequest(body []byte) (*Request, error) {
	if err := requireJSONObject(body); err != nil {
		return nil, fmt.Errorf("decode request: %w", err)
	}
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
			// content may be a plain string or an array of content parts; both
			// forms are used in practice (the Responses-style SDKs emit arrays).
			toolParts, err := decodeOpenAIChatMessageContent(m.Content)
			if err != nil {
				return nil, fmt.Errorf("decode openai chat request messages[%d]: %w", i, err)
			}
			if len(toolParts) == 0 {
				toolParts = []ContentPart{
					{Type: ContentTypeText, Text: &TextContent{Text: ""}},
				}
			}
			parts := []ContentPart{
				{
					Type: ContentTypeToolResult,
					ToolResult: &ToolResultContent{
						ToolUseID: m.ToolCallID,
						Content:   toolParts,
						IsError:   false,
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
			rf.Name = raw.ResponseFormat.JSONSchema.Name
			rf.Strict = raw.ResponseFormat.JSONSchema.Strict
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
		case "video_url":
			if p.VideoURL == nil {
				continue
			}
			vid, err := decodeOpenAIChatVideoURL(p.VideoURL)
			if err != nil {
				return nil, fmt.Errorf("decode video_url: %w", err)
			}
			result = append(result, ContentPart{
				Type:  ContentTypeVideo,
				Video: vid,
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

// decodeOpenAIChatVideoURL decodes a video_url content part into an IR VideoContent.
func decodeOpenAIChatVideoURL(vid *openaichat.ChatVideoURL) (*VideoContent, error) {
	if strings.HasPrefix(vid.URL, "data:") {
		mediaType, b64Data, err := parseDataURI(vid.URL)
		if err != nil {
			return nil, err
		}
		data, err := base64.StdEncoding.DecodeString(b64Data)
		if err != nil {
			return nil, fmt.Errorf("decode base64 video data: %w", err)
		}
		return &VideoContent{
			Data:      data,
			MediaType: mediaType,
		}, nil
	}
	return &VideoContent{
		URL: vid.URL,
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

	// Reasoning/thinking content. reasoning_signature / reasoning_redacted are
	// non-standard companions emitted by this gateway so that Anthropic thinking
	// blocks survive a trip through the Chat Completions history: Anthropic
	// rejects a replayed thinking block whose signature is missing.
	if m.ReasoningRedacted != nil && *m.ReasoningRedacted != "" {
		parts = append(parts, ContentPart{
			Type:             ContentTypeRedactedThinking,
			RedactedThinking: &RedactedThinkingContent{Data: *m.ReasoningRedacted},
		})
	}
	if m.ReasoningContent != nil && *m.ReasoningContent != "" {
		thinking := &ThinkingContent{Thinking: *m.ReasoningContent}
		if m.ReasoningSignature != nil {
			thinking.Signature = *m.ReasoningSignature
		}
		parts = append(parts, ContentPart{
			Type:     ContentTypeThinking,
			Thinking: thinking,
		})
	}

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

	// Tools — OpenAI Chat Completions only supports function tools; Anthropic
	// server-side tools (web_search, bash, computer, text_editor) have no chat
	// representation and must be dropped before the request is sent. Forwarding
	// them as type:"function" with no parameters used to reach the provider
	// and produce a 400.
	encodedToolNames := map[string]bool{}
	if len(req.Tools) > 0 {
		tools := make([]openaichat.ChatTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			if !isOpenAIChatSupportedToolType(t.Type) {
				continue
			}
			tools = append(tools, openaichat.ChatTool{
				Type: "function",
				Function: openaichat.ChatFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
					Strict:      t.Strict,
				},
			})
			if t.Name != "" {
				encodedToolNames[t.Name] = true
			}
		}
		if len(tools) > 0 {
			raw.Tools = tools
		}
	}

	// Tool choice — sanitize against the surviving tool set so a named
	// selector pointing at a dropped Anthropic server tool degrades to auto
	// instead of reproducing the original 400.
	if req.ToolChoice != nil {
		effective := sanitizeToolChoiceForEncode(req.ToolChoice, encodedToolNames, len(raw.Tools), len(req.Tools))
		if effective != nil {
			tc, err := encodeOpenAIChatToolChoice(effective)
			if err != nil {
				return nil, fmt.Errorf("encode openai chat request tool_choice: %w", err)
			}
			raw.ToolChoice = tc
		}

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
			// OpenAI requires json_schema.name; synthesise one when the inbound
			// protocol had no equivalent field (Gemini, Anthropic) so the request
			// is not rejected with "missing required parameter name".
			name := req.ResponseFormat.Name
			if name == "" {
				name = defaultJSONSchemaName
			}
			rf.JSONSchema = &openaichat.ChatResponseJSONSchema{
				Name:   name,
				Schema: req.ResponseFormat.JSONSchema,
				Strict: req.ResponseFormat.Strict,
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
			content := ensureNonEmptyChatContent(encodeOpenAIChatContentParts(userParts))
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
		return encodeOpenAIChatToolMessages(m)

	default:
		content := ensureNonEmptyChatContent(encodeOpenAIChatContentParts(m.Content))
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

// ensureNonEmptyChatContent guarantees at least one content part. OpenAI rejects
// a message whose content array is empty, which would otherwise happen when every
// part of a message had no Chat Completions representation.
func ensureNonEmptyChatContent(parts []openaichat.ChatContentPart) []openaichat.ChatContentPart {
	if len(parts) > 0 {
		return parts
	}
	return []openaichat.ChatContentPart{{Type: "text", Text: ""}}
}

// encodeOpenAIChatToolResultPart converts a single ContentTypeToolResult part
// to an OpenAI Chat tool message.
func encodeOpenAIChatToolResultPart(p ContentPart) (openaichat.ChatMessage, error) {
	msg := openaichat.ChatMessage{
		Role:       "tool",
		ToolCallID: p.ToolResult.ToolUseID,
	}
	contentJSON, err := json.Marshal(toolResultTextWithError(p.ToolResult))
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
	var reasoningParts []string
	var reasoningSignature string
	var reasoningRedacted string
	var toolCalls []openaichat.ToolCall

	for _, p := range m.Content {
		switch p.Type {
		case ContentTypeText:
			if p.Text != nil {
				textParts = append(textParts, p.Text.Text)
			}
		case ContentTypeThinking:
			if p.Thinking != nil {
				reasoningParts = append(reasoningParts, p.Thinking.Thinking)
				// Keep the signature so an Anthropic target can validate the block
				// if this history is later replayed there.
				if p.Thinking.Signature != "" {
					reasoningSignature = p.Thinking.Signature
				}
			}
		case ContentTypeRedactedThinking:
			if p.RedactedThinking != nil && p.RedactedThinking.Data != "" {
				reasoningRedacted = p.RedactedThinking.Data
			}
		case ContentTypeRefusal:
			// Refusal in request message history — degrade to text
			if p.Refusal != nil {
				textParts = append(textParts, p.Refusal.Refusal)
			}
		case ContentTypeToolUse:
			if p.ToolUse != nil {
				toolCalls = append(toolCalls, openaichat.ToolCall{
					Index: len(toolCalls),
					ID:    p.ToolUse.ID,
					Type:  "function",
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

	if len(reasoningParts) > 0 {
		reasoning := strings.Join(reasoningParts, "")
		msg.ReasoningContent = &reasoning
	}
	if reasoningSignature != "" {
		msg.ReasoningSignature = &reasoningSignature
	}
	if reasoningRedacted != "" {
		msg.ReasoningRedacted = &reasoningRedacted
	}

	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	return msg, nil
}

// encodeOpenAIChatToolMessages encodes a tool role message. OpenAI Chat requires
// one "tool" message per tool_call_id, so an IR message carrying several tool
// results (Anthropic packs parallel results into a single user turn) must fan out
// into several messages. Emitting only the first would leave the remaining
// tool_call_ids unanswered, which providers reject.
func encodeOpenAIChatToolMessages(m Message) ([]openaichat.ChatMessage, error) {
	var msgs []openaichat.ChatMessage
	for _, p := range m.Content {
		if p.Type != ContentTypeToolResult || p.ToolResult == nil {
			continue
		}
		msg, err := encodeOpenAIChatToolResultPart(p)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, msg)
	}
	return msgs, nil
}

// encodeOpenAIChatContentParts converts IR ContentParts to OpenAI Chat content parts.
//
// Content types with no Chat Completions representation (documents, thinking,
// server tool results, ...) are replaced by a short text placeholder rather than
// being skipped outright. Dropping them silently can yield an empty content array,
// which providers reject with a 400, and it leaves the model unaware that content
// was present.
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
		case ContentTypeVideo:
			if p.Video != nil {
				cp := openaichat.ChatContentPart{
					Type: "video_url",
					VideoURL: &openaichat.ChatVideoURL{},
				}
				if len(p.Video.Data) > 0 {
					mediaType := p.Video.MediaType
					if mediaType == "" {
						mediaType = "application/octet-stream"
					}
					cp.VideoURL.URL = fmt.Sprintf("data:%s;base64,%s",
						mediaType,
						base64.StdEncoding.EncodeToString(p.Video.Data))
				} else if p.Video.URL != "" {
					cp.VideoURL.URL = p.Video.URL
				}
				result = append(result, cp)
			}
		case ContentTypeDocument:
			// OpenAI Chat Completions has no document/file content part.
			if text := documentPlaceholderText(p.Document); text != "" {
				result = append(result, openaichat.ChatContentPart{Type: "text", Text: text})
			}
		case ContentTypeThinking:
			// Thinking is carried on the message-level reasoning_content field,
			// not as a content part; handled by the caller.
		default:
			if text := unrepresentablePlaceholderText(p); text != "" {
				result = append(result, openaichat.ChatContentPart{Type: "text", Text: text})
			}
		}
	}
	return result
}

// documentPlaceholderText renders a document part as text for protocols with no
// document content type.
func documentPlaceholderText(doc *DocumentContent) string {
	if doc == nil {
		return "[document omitted]"
	}
	switch {
	case doc.URL != "":
		if doc.Title != "" {
			return fmt.Sprintf("[document: %s (%s)]", doc.Title, doc.URL)
		}
		return fmt.Sprintf("[document: %s]", doc.URL)
	case doc.Title != "":
		return fmt.Sprintf("[document: %s (content omitted: not supported by this provider)]", doc.Title)
	default:
		return "[document omitted: not supported by this provider]"
	}
}

// videoPlaceholderText renders a video part as text for protocols with no
// video content type.
func videoPlaceholderText(vid *VideoContent) string {
	if vid == nil {
		return "[video omitted]"
	}
	if vid.URL != "" {
		return fmt.Sprintf("[video: %s]", vid.URL)
	}
	return "[video omitted: not supported by this provider]"
}

// unrepresentablePlaceholderText renders content parts that have no equivalent in
// the target protocol as a short text note, so the message never encodes to an
// empty content array and the model is told something was dropped.
func unrepresentablePlaceholderText(p ContentPart) string {
	switch p.Type {
	case ContentTypeVideo:
		return videoPlaceholderText(p.Video)
	case ContentTypeRefusal:
		if p.Refusal != nil {
			return p.Refusal.Refusal
		}
		return ""
	case ContentTypeRedactedThinking:
		// Deliberately not rendered: the payload is opaque ciphertext and is
		// preserved separately for same-protocol round-trips.
		return ""
	case ContentTypeServerToolUse:
		if p.ServerToolUse != nil && p.ServerToolUse.Name != "" {
			return fmt.Sprintf("[server tool call: %s]", p.ServerToolUse.Name)
		}
		return "[server tool call]"
	case ContentTypeWebSearchToolResult:
		return webSearchResultPlaceholderText(p.WebSearchToolResult)
	default:
		return ""
	}
}

// webSearchResultPlaceholderText renders built-in web search results as text for
// protocols that cannot express them structurally, so the citations the model
// relied on are not lost entirely.
func webSearchResultPlaceholderText(r *WebSearchToolResultContent) string {
	if r == nil {
		return ""
	}
	if r.IsError {
		if r.ErrorCode != "" {
			return fmt.Sprintf("[web search failed: %s]", r.ErrorCode)
		}
		return "[web search failed]"
	}
	if len(r.Content) == 0 {
		return "[web search returned no results]"
	}
	var b strings.Builder
	b.WriteString("[web search results:")
	for _, hit := range r.Content {
		b.WriteString("\n- ")
		if hit.Title != "" {
			b.WriteString(hit.Title)
			b.WriteString(": ")
		}
		b.WriteString(hit.URL)
	}
	b.WriteString("]")
	return b.String()
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
		ID:      raw.ID,
		Model:   raw.Model,
		Created: raw.Created,
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

			// Reasoning/thinking content (emitted first, matching natural order)
			if choice.Message.ReasoningContent != nil && *choice.Message.ReasoningContent != "" {
				resp.Content = append(resp.Content, ContentPart{
					Type:     ContentTypeThinking,
					Thinking: &ThinkingContent{Thinking: *choice.Message.ReasoningContent},
				})
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
					Type:       ContentTypeRefusal,
					Refusal:    &RefusalContent{Refusal: *choice.Message.Refusal},
					SourceType: ContentTypeRefusal,
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
		ID:      resp.ID,
		Object:  "chat.completion",
		Model:   resp.Model,
		Created: resp.Created,
	}

	// Build the choice message
	msg := &openaichat.ChatChoiceMessage{
		Role: "assistant",
	}

	var textParts []string
	var reasoningParts []string
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
		case ContentTypeThinking:
			if p.Thinking != nil {
				reasoningParts = append(reasoningParts, p.Thinking.Thinking)
			}
		case ContentTypeRefusal:
			if p.Refusal != nil {
				refusalParts = append(refusalParts, p.Refusal.Refusal)
			}
		case ContentTypeToolUse:
			if p.ToolUse != nil {
				toolCalls = append(toolCalls, openaichat.ToolCall{
					Index: len(toolCalls),
					ID:    p.ToolUse.ID,
					Type:  "function",
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
	} else if len(toolCalls) > 0 || len(reasoningParts) > 0 {
		empty := ""
		msg.Content = &empty
	}
	if len(reasoningParts) > 0 {
		reasoning := strings.Join(reasoningParts, "")
		msg.ReasoningContent = &reasoning
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
// from an SSE "data:" line) into a single unified IR StreamEvent.
//
// Deprecated: a single OpenAI Chat chunk can legitimately carry several IR events
// (e.g. content plus finish_reason, or several parallel tool_calls). Prefer
// DecodeOpenAIChatStreamChunks, which returns all of them. This wrapper returns
// only the first event and is kept for backwards compatibility.
func DecodeOpenAIChatStreamChunk(data []byte) (*StreamEvent, error) {
	events, err := DecodeOpenAIChatStreamChunks(data)
	if err != nil {
		return nil, err
	}
	if len(events) == 0 {
		return nil, nil
	}
	return events[0], nil
}

// DecodeOpenAIChatStreamChunks decodes an OpenAI Chat streaming chunk JSON (the data
// from an SSE "data:" line) into zero or more unified IR StreamEvents.
//
// A single chunk may fan out into several IR events:
//   - Providers such as vLLM, DeepSeek and Azure emit content and finish_reason in
//     the same chunk; both the content delta and the stop event must be produced or
//     the final token is lost.
//   - Parallel tool calls may be batched into one delta.tool_calls array; each entry
//     becomes its own IR delta keyed by its own index.
func DecodeOpenAIChatStreamChunks(data []byte) ([]*StreamEvent, error) {
	var raw openaichat.ChatStreamChunk
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("decode openai chat stream chunk: %w", err)
	}

	startEvent := func() *StreamEvent {
		return &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:      raw.ID,
				Model:   raw.Model,
				Created: raw.Created,
			},
		}
	}

	// No choices — could be a usage-only chunk.
	if len(raw.Choices) == 0 {
		if raw.Usage != nil {
			u := decodeOpenAIChatUsage(raw.Usage)
			return []*StreamEvent{{
				Type:  StreamEventDelta,
				Usage: &u,
			}}, nil
		}
		return []*StreamEvent{startEvent()}, nil
	}

	choice := raw.Choices[0]
	delta := choice.Delta
	// Multi-choice (n>1) streaming: keep the choice index so downstream encoders
	// can keep the candidates in distinct content blocks instead of interleaving
	// them into one.
	baseIndex := choice.Index

	// First chunk with role only.
	if delta != nil && delta.Role != "" && delta.Content == nil && delta.ReasoningContent == nil &&
		delta.Refusal == nil && len(delta.ToolCalls) == 0 &&
		(choice.FinishReason == nil || *choice.FinishReason == "") {
		return []*StreamEvent{startEvent()}, nil
	}

	var events []*StreamEvent

	if delta != nil {
		// Reasoning/thinking content delta (emitted first, matching natural order).
		if delta.ReasoningContent != nil {
			events = append(events, &StreamEvent{
				Type:  StreamEventDelta,
				Index: baseIndex,
				Delta: &ContentPart{
					Type:     ContentTypeThinking,
					Thinking: &ThinkingContent{Thinking: *delta.ReasoningContent},
				},
			})
		}

		// Content delta.
		if delta.Content != nil {
			events = append(events, &StreamEvent{
				Type:  StreamEventDelta,
				Index: baseIndex,
				Delta: &ContentPart{
					Type: ContentTypeText,
					Text: &TextContent{Text: *delta.Content},
				},
			})
		}

		// Refusal delta.
		if delta.Refusal != nil {
			events = append(events, &StreamEvent{
				Type:  StreamEventDelta,
				Index: baseIndex,
				Delta: &ContentPart{
					Type:       ContentTypeRefusal,
					Refusal:    &RefusalContent{Refusal: *delta.Refusal},
					SourceType: ContentTypeRefusal,
				},
			})
		}

		// Tool call deltas — every entry in the array, not just the first.
		for _, tc := range delta.ToolCalls {
			events = append(events, &StreamEvent{
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
			})
		}
	}

	// finish_reason — emitted after any content in the same chunk so the final
	// token is not dropped.
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		stopReason := decodeOpenAIChatFinishReason(*choice.FinishReason)
		stop := &StreamEvent{
			Type:       StreamEventStop,
			StopReason: &stopReason,
		}
		if raw.Usage != nil {
			u := decodeOpenAIChatUsage(raw.Usage)
			stop.Usage = &u
		}
		events = append(events, stop)
		return events, nil
	}

	// A usage payload alongside a content delta (some providers do this) must not
	// be dropped — attach it to the last emitted event.
	if raw.Usage != nil && len(events) > 0 {
		u := decodeOpenAIChatUsage(raw.Usage)
		events[len(events)-1].Usage = &u
	}

	if len(events) == 0 {
		return []*StreamEvent{startEvent()}, nil
	}
	return events, nil
}

// EncodeOpenAIChatStreamChunk encodes a unified IR StreamEvent into an OpenAI Chat
// streaming chunk JSON (suitable for an SSE "data:" line).
func EncodeOpenAIChatStreamChunk(event *StreamEvent) ([]byte, error) {
	// A nil event is a "nothing to emit" signal from an upstream decoder that saw a
	// chunk it does not map. Treat it as a skip rather than dereferencing it.
	if event == nil {
		return nil, nil
	}
	raw := openaichat.ChatStreamChunk{
		Object: "chat.completion.chunk",
	}

	switch event.Type {
	case StreamEventStart:
		if event.Response != nil {
			raw.ID = event.Response.ID
			raw.Model = event.Response.Model
			raw.Created = event.Response.Created
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
		if event.Response != nil {
			raw.ID = event.Response.ID
			raw.Model = event.Response.Model
			raw.Created = event.Response.Created
		}
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
			case ContentTypeThinking:
				var reasoning string
				if event.Delta.Thinking != nil {
					reasoning = event.Delta.Thinking.Thinking
				}
				raw.Choices = []openaichat.ChatChoice{
					{
						Index: event.Index,
						Delta: &openaichat.ChatChoiceMessage{
							ReasoningContent: &reasoning,
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
		if event.Response != nil {
			raw.ID = event.Response.ID
			raw.Model = event.Response.Model
			raw.Created = event.Response.Created
		}
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

	case StreamEventContentBlockStart:
		if event.Response != nil {
			raw.ID = event.Response.ID
			raw.Model = event.Response.Model
			raw.Created = event.Response.Created
		}
		// Most block-start events carry no payload OpenAI Chat can express. A
		// tool_use block start is the exception and must not be skipped: for
		// Anthropic-style sources the tool's id and name arrive *only* here, with
		// the deltas carrying nothing but incremental argument JSON. Dropping it
		// leaves the client with an unnamed tool call.
		if event.Delta != nil && event.Delta.Type == ContentTypeToolUse && event.Delta.ToolUse != nil {
			raw.Choices = []openaichat.ChatChoice{
				{
					Index: 0,
					Delta: &openaichat.ChatChoiceMessage{
						ToolCalls: []openaichat.ToolCall{{
							Index: event.Index,
							ID:    event.Delta.ToolUse.ID,
							Type:  "function",
							Function: openaichat.ToolCallFunction{
								Name:      event.Delta.ToolUse.Name,
								Arguments: string(event.Delta.ToolUse.Arguments),
							},
						}},
					},
				},
			}
			break
		}
		return nil, nil

	case StreamEventContentBlockStop:
		// No equivalent in OpenAI Chat streaming; skip silently.
		return nil, nil

	case StreamEventError:
		// Error events have no direct equivalent in OpenAI Chat streaming chunks; skip silently.
		return nil, nil

	default:
		return nil, fmt.Errorf("unknown IR stream event type: %q", event.Type)
	}

	return json.Marshal(raw)
}
