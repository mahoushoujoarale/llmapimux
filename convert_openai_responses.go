package llmapimux

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/llmapimux/llmapimux/protocol/openairesponses"
)

func decodeOaiRespUsage(u *openairesponses.Usage) Usage {
	usage := Usage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
	}
	if u.InputTokensDetails != nil {
		usage.CacheReadTokens = u.InputTokensDetails.CachedTokens
	}
	if u.OutputTokensDetails != nil {
		usage.ThinkingTokens = u.OutputTokensDetails.ReasoningTokens
	}
	return usage
}

func encodeOaiRespUsage(u *Usage) *openairesponses.Usage {
	raw := &openairesponses.Usage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
	}
	if u.CacheReadTokens != 0 {
		raw.InputTokensDetails = &openairesponses.InputDetails{CachedTokens: u.CacheReadTokens}
	}
	if u.ThinkingTokens != 0 {
		raw.OutputTokensDetails = &openairesponses.OutputDetails{ReasoningTokens: u.ThinkingTokens}
	}
	return raw
}

// intPtr returns a pointer to an int value.
func intPtr(v int) *int { return &v }

// derefIntPtr returns the int value from a pointer, defaulting to 0.
func derefIntPtr(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// --- Decode functions ---

// DecodeOpenAIResponsesRequest decodes an OpenAI Responses API JSON request body
// into the unified IR Request type.
func DecodeOpenAIResponsesRequest(body []byte) (*Request, error) {
	var raw openairesponses.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode openai responses request: %w", err)
	}

	req := &Request{
		Model:       raw.Model,
		Temperature: raw.Temperature,
		TopP:        raw.TopP,
		Stream:      raw.Stream,
	}

	// max_output_tokens → MaxTokens
	if raw.MaxOutputTokens != nil {
		req.MaxTokens = *raw.MaxOutputTokens
	}

	// Stop sequences: can be string or array of strings
	if len(raw.Stop) > 0 {
		stops, err := decodeOpenAIStop(raw.Stop)
		if err != nil {
			return nil, fmt.Errorf("decode openai responses request stop: %w", err)
		}
		req.StopSequences = stops
	}

	// Instructions → SystemPrompt
	var systemParts []ContentPart
	if raw.Instructions != "" {
		systemParts = append(systemParts, ContentPart{
			Type: ContentTypeText,
			Text: &TextContent{Text: raw.Instructions},
		})
	}

	// Input: can be a string or an array of items
	if len(raw.Input) > 0 {
		messages, devParts, err := decodeOaiRespInput(raw.Input)
		if err != nil {
			return nil, fmt.Errorf("decode openai responses request input: %w", err)
		}
		systemParts = append(systemParts, devParts...)
		req.Messages = messages
	}

	if len(systemParts) > 0 {
		req.SystemPrompt = systemParts
	}

	// Tools
	if len(raw.Tools) > 0 {
		tools := make([]Tool, 0, len(raw.Tools))
		for i, t := range raw.Tools {
			extraFields, err := decodeOpenAIResponsesToolExtraFields(t.Type, t.ExtraFields)
			if err != nil {
				return nil, fmt.Errorf("decode openai responses request tools[%d]: %w", i, err)
			}
			tool := Tool{
				Type:        normalizeOpenAIResponsesToolType(t.Type),
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
				Strict:      t.Strict,
				ExtraFields: extraFields,
			}
			if !isFunctionToolType(tool.Type) && tool.Name == "" {
				tool.Name = defaultToolNameForType(tool.Type)
			}
			tools = append(tools, tool)
		}
		if len(tools) > 0 {
			req.Tools = tools
		}
	}

	// Tool choice
	if len(raw.ToolChoice) > 0 {
		tc, err := decodeOaiRespToolChoice(raw.ToolChoice, req.Tools)
		if err != nil {
			return nil, fmt.Errorf("decode openai responses request tool_choice: %w", err)
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

	// Reasoning → ThinkingConfig
	if raw.Reasoning != nil && raw.Reasoning.Effort != "" {
		req.Thinking = &ThinkingConfig{
			Mode:   "enabled",
			Effort: raw.Reasoning.Effort,
		}
	}

	// text.format → ResponseFormat
	if raw.Text != nil && raw.Text.Format != nil {
		rf := &ResponseFormat{
			Type: raw.Text.Format.Type,
		}
		if raw.Text.Format.Type == "json_schema" && len(raw.Text.Format.Schema) > 0 {
			rf.JSONSchema = raw.Text.Format.Schema
		}
		req.ResponseFormat = rf
	}

	return req, nil
}

// decodeOaiRespToolChoice decodes the tool_choice field which can be a string or object.
func decodeOaiRespToolChoice(raw json.RawMessage, tools []Tool) (*ToolChoice, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, err
		}
		return &ToolChoice{Type: s}, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, err
	}

	// A tool_choice object without a type field is malformed; degrade to "auto"
	// rather than returning an error — strictness here would leak upstream
	// sloppy payloads back to our callers as 400s.
	var kind string
	if typeRaw, ok := obj["type"]; ok && len(typeRaw) > 0 {
		if err := json.Unmarshal(typeRaw, &kind); err != nil {
			return nil, err
		}
	}
	switch kind {
	case "function":
		tc := &ToolChoice{Type: "tool"}
		if nameRaw, ok := obj["name"]; ok {
			if err := json.Unmarshal(nameRaw, &tc.ToolName); err != nil {
				return nil, err
			}
		}
		return tc, nil
	case "allowed_tools":
		tc := &ToolChoice{}
		if modeRaw, ok := obj["mode"]; ok {
			if err := json.Unmarshal(modeRaw, &tc.Type); err != nil {
				return nil, err
			}
		}
		if toolsRaw, ok := obj["tools"]; ok {
			var allowed []map[string]json.RawMessage
			if err := json.Unmarshal(toolsRaw, &allowed); err != nil {
				return nil, err
			}
			for _, tool := range allowed {
				// Preserve both function and built-in entries — built-in tools
				// use their type string as the logical name (e.g. "web_search"),
				// which aligns with how DecodeAnthropicRequest names server-side
				// tools by defaultToolNameForType and keeps round-trips honest.
				var toolType, name string
				if typeRaw, ok := tool["type"]; ok && len(typeRaw) > 0 {
					if err := json.Unmarshal(typeRaw, &toolType); err != nil {
						return nil, err
					}
				}
				if nameRaw, ok := tool["name"]; ok && len(nameRaw) > 0 {
					if err := json.Unmarshal(nameRaw, &name); err != nil {
						return nil, err
					}
				}
				normalized := normalizeOpenAIResponsesToolType(toolType)
				if name == "" && !isFunctionToolType(normalized) {
					name = defaultToolNameForType(normalized)
				}
				if name != "" {
					tc.AllowedToolNames = append(tc.AllowedToolNames, name)
				}
			}
		}
		if tc.Type == "" {
			tc.Type = "auto"
		}
		return tc, nil
	case "auto", "none", "required", "":
		// Empty kind falls through here (see defensive parse above).
		if kind == "" {
			kind = "auto"
		}
		return &ToolChoice{Type: kind}, nil
	default:
		// Named built-in selector like {"type":"web_search"}. Try to resolve a
		// matching tool from the request, fall back to the type-as-name.
		normalized := normalizeOpenAIResponsesToolType(kind)
		tc := &ToolChoice{Type: "tool"}
		if tool := findToolByType(tools, normalized); tool != nil && tool.Name != "" {
			tc.ToolName = tool.Name
		} else {
			tc.ToolName = defaultToolNameForType(normalized)
		}
		return tc, nil
	}
}

// decodeOaiRespInput decodes the input field which can be a string or array of items.
// Returns messages and any developer-role content parts (to be added to SystemPrompt).
func decodeOaiRespInput(raw json.RawMessage) ([]Message, []ContentPart, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil, nil
	}

	// Try string shorthand first
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return nil, nil, fmt.Errorf("unmarshal string input: %w", err)
		}
		return []Message{
			{
				Role: RoleUser,
				Content: []ContentPart{
					{Type: ContentTypeText, Text: &TextContent{Text: text}},
				},
			},
		}, nil, nil
	}

	// Array form
	var items []openairesponses.InputItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, nil, fmt.Errorf("unmarshal input items: %w", err)
	}

	var messages []Message
	var devParts []ContentPart

	for i, item := range items {
		// EasyInputMessageParam omits "type" when using omitzero; infer "message" from role.
		itemType := item.Type
		if itemType == "" && item.Role != "" {
			itemType = "message"
		}
		switch itemType {
		case "message":
			switch item.Role {
			case "developer":
				parts, err := decodeOaiRespMessageContent(item.Content)
				if err != nil {
					return nil, nil, fmt.Errorf("input[%d] developer content: %w", i, err)
				}
				devParts = append(devParts, parts...)

			case "user":
				parts, err := decodeOaiRespMessageContent(item.Content)
				if err != nil {
					return nil, nil, fmt.Errorf("input[%d] user content: %w", i, err)
				}
				messages = append(messages, Message{
					Role:    RoleUser,
					Content: parts,
				})

			case "assistant":
				parts, err := decodeOaiRespMessageContent(item.Content)
				if err != nil {
					return nil, nil, fmt.Errorf("input[%d] assistant content: %w", i, err)
				}
				messages = append(messages, Message{
					Role:    RoleAssistant,
					Content: parts,
				})

			default:
				parts, err := decodeOaiRespMessageContent(item.Content)
				if err != nil {
					return nil, nil, fmt.Errorf("input[%d] %s content: %w", i, item.Role, err)
				}
				messages = append(messages, Message{
					Role:    Role(item.Role),
					Content: parts,
				})
			}

		case "function_call":
			// function_call → ToolUseContent in assistant message
			messages = append(messages, Message{
				Role: RoleAssistant,
				Content: []ContentPart{
					{
						Type: ContentTypeToolUse,
						ToolUse: &ToolUseContent{
							ID:        item.CallID,
							Name:      item.Name,
							Arguments: json.RawMessage(item.Arguments),
						},
					},
				},
			})

		case "function_call_output":
			// function_call_output → ToolResultContent in tool message
			messages = append(messages, Message{
				Role: RoleTool,
				Content: []ContentPart{
					{
						Type: ContentTypeToolResult,
						ToolResult: &ToolResultContent{
							ToolUseID: item.CallID,
							Content: []ContentPart{
								{Type: ContentTypeText, Text: &TextContent{Text: item.Output}},
							},
						},
					},
				},
			})

		default:
			// Unknown item type — skip silently
		}
	}

	return messages, devParts, nil
}

// decodeOaiRespMessageContent decodes message content which can be a string or array of content parts.
func decodeOaiRespMessageContent(raw json.RawMessage) ([]ContentPart, error) {
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
	var parts []openairesponses.ContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("unmarshal content parts: %w", err)
	}

	result := make([]ContentPart, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case "input_text":
			result = append(result, ContentPart{
				Type: ContentTypeText,
				Text: &TextContent{Text: p.Text},
			})
		case "output_text":
			result = append(result, ContentPart{
				Type: ContentTypeText,
				Text: &TextContent{Text: p.Text},
			})
		case "input_image":
			img := &ImageContent{}
			if p.ImageURL != "" {
				img.URL = p.ImageURL
			}
			result = append(result, ContentPart{
				Type:  ContentTypeImage,
				Image: img,
			})
		case "input_file":
			doc := &DocumentContent{}
			if p.FileData != "" {
				data, err := base64.StdEncoding.DecodeString(p.FileData)
				if err != nil {
					return nil, fmt.Errorf("decode input_file file_data: %w", err)
				}
				doc.Data = data
			}
			if p.FileURL != "" {
				doc.URL = p.FileURL
			}
			if p.Filename != "" {
				doc.Title = p.Filename
			}
			result = append(result, ContentPart{
				Type:     ContentTypeDocument,
				Document: doc,
			})
		default:
			// Unknown content type — pass through
			result = append(result, ContentPart{Type: ContentType(p.Type)})
		}
	}
	return result, nil
}

// --- Encode functions ---

// EncodeOpenAIResponsesRequest encodes a unified IR Request into an OpenAI Responses API JSON body.
func EncodeOpenAIResponsesRequest(req *Request) ([]byte, error) {
	raw := openairesponses.Request{
		Model:  req.Model,
		Stream: req.Stream,
	}

	// Temperature
	if req.Temperature != nil {
		raw.Temperature = req.Temperature
	}

	// TopP
	if req.TopP != nil {
		raw.TopP = req.TopP
	}

	// MaxTokens → max_output_tokens
	if req.MaxTokens > 0 {
		mt := req.MaxTokens
		raw.MaxOutputTokens = &mt
	}

	// Stop sequences
	if len(req.StopSequences) > 0 {
		data, err := json.Marshal(req.StopSequences)
		if err != nil {
			return nil, fmt.Errorf("encode openai responses request stop: %w", err)
		}
		raw.Stop = data
	}

	// SystemPrompt → instructions (join text parts)
	if len(req.SystemPrompt) > 0 {
		var texts []string
		for _, p := range req.SystemPrompt {
			if p.Type == ContentTypeText && p.Text != nil {
				texts = append(texts, p.Text.Text)
			}
		}
		if len(texts) > 0 {
			raw.Instructions = strings.Join(texts, "\n")
		}
	}

	// Messages → input array
	if len(req.Messages) > 0 {
		inputItems := make([]openairesponses.InputItem, 0, len(req.Messages))
		for _, m := range req.Messages {
			items := encodeOaiRespMessage(m)
			inputItems = append(inputItems, items...)
		}
		data, err := json.Marshal(inputItems)
		if err != nil {
			return nil, fmt.Errorf("encode openai responses request input: %w", err)
		}
		raw.Input = data
	}

	// Tools — drop IR tool types with no OpenAI Responses representation
	// (e.g. Anthropic bash_*, computer_*, text_editor_*). Forwarding them as
	// raw type strings would trigger upstream 400 "unknown tool type" errors.
	// The sanitized name set is passed to tool_choice sanitization below so a
	// named selector pointing at a dropped tool can degrade to "auto" instead
	// of reaching the provider and being rejected.
	encodedToolNames := map[string]bool{}
	survivingTools := make([]Tool, 0, len(req.Tools))
	if len(req.Tools) > 0 {
		tools := make([]openairesponses.Tool, 0, len(req.Tools))
		for i, t := range req.Tools {
			if !isOpenAIResponsesSupportedToolType(t.Type) {
				continue
			}
			extraFields, err := encodeOpenAIResponsesToolExtraFields(t.Type, t.ExtraFields)
			if err != nil {
				return nil, fmt.Errorf("encode openai responses request tools[%d]: %w", i, err)
			}
			tool := openairesponses.Tool{
				Type:        openAIResponsesToolTypeFromIR(t.Type),
				ExtraFields: extraFields,
			}
			if isFunctionToolType(t.Type) {
				tool.Name = t.Name
				tool.Description = t.Description
				tool.Parameters = t.Parameters
				tool.Strict = t.Strict
			}
			tools = append(tools, tool)
			survivingTools = append(survivingTools, t)
			if t.Name != "" {
				encodedToolNames[t.Name] = true
			}
		}
		if len(tools) > 0 {
			raw.Tools = tools
		}
	}

	// Tool choice — sanitize against the surviving tool set. When tools were
	// emptied entirely or the named selector points at a dropped tool, the
	// sanitizer degrades or drops the choice so the outbound request stays
	// valid for strict providers.
	if req.ToolChoice != nil {
		effective := sanitizeToolChoiceForEncode(req.ToolChoice, encodedToolNames, len(raw.Tools))
		if effective != nil {
			tc, err := encodeOaiRespToolChoice(&Request{Tools: survivingTools, ToolChoice: effective})
			if err != nil {
				return nil, fmt.Errorf("encode openai responses request tool_choice: %w", err)
			}
			raw.ToolChoice = tc
		}

		// AllowParallelCalls → parallel_tool_calls (top-level)
		if req.ToolChoice.AllowParallelCalls != nil {
			raw.ParallelToolCalls = req.ToolChoice.AllowParallelCalls
		}
	}

	// ThinkingConfig → reasoning.effort
	if req.Thinking != nil {
		if req.Thinking.Effort != "" {
			raw.Reasoning = &openairesponses.Reasoning{Effort: req.Thinking.Effort}
		} else if req.Thinking.Mode == "enabled" {
			raw.Reasoning = &openairesponses.Reasoning{Effort: "medium"}
		}
	}

	// ResponseFormat → text.format
	if req.ResponseFormat != nil {
		raw.Text = &openairesponses.Text{
			Format: &openairesponses.TextFormat{
				Type: req.ResponseFormat.Type,
			},
		}
		if req.ResponseFormat.Type == "json_schema" && len(req.ResponseFormat.JSONSchema) > 0 {
			raw.Text.Format.Schema = req.ResponseFormat.JSONSchema
		}
	}

	return json.Marshal(raw)
}

// encodeOaiRespMessage converts an IR Message into one or more oaiRespInputItems.
func encodeOaiRespMessage(m Message) []openairesponses.InputItem {
	switch m.Role {
	case RoleUser:
		// Split tool_result parts into function_call_output items;
		// remaining parts become a user message.
		var msgParts []ContentPart
		var items []openairesponses.InputItem
		for _, p := range m.Content {
			if p.Type == ContentTypeToolResult && p.ToolResult != nil {
				items = append(items, openairesponses.InputItem{
					Type:   "function_call_output",
					CallID: p.ToolResult.ToolUseID,
					Output: toolResultText(p.ToolResult),
				})
			} else {
				msgParts = append(msgParts, p)
			}
		}
		if len(msgParts) > 0 {
			content := encodeOaiRespContentParts(msgParts, "input_text")
			data, _ := json.Marshal(content)
			items = append(items, openairesponses.InputItem{
				Type:    "message",
				Role:    "user",
				Content: data,
			})
		}
		return items

	case RoleAssistant:
		// Split into message content and function_call items
		var textParts []openairesponses.ContentPart
		var funcCalls []openairesponses.InputItem
		for _, p := range m.Content {
			switch p.Type {
			case ContentTypeText:
				if p.Text != nil {
					textParts = append(textParts, openairesponses.ContentPart{
						Type: "output_text",
						Text: p.Text.Text,
					})
				}
			case ContentTypeToolUse:
				if p.ToolUse != nil {
					funcCalls = append(funcCalls, openairesponses.InputItem{
						Type:      "function_call",
						CallID:    p.ToolUse.ID,
						Name:      p.ToolUse.Name,
						Arguments: string(p.ToolUse.Arguments),
					})
				}
			}
		}
		var items []openairesponses.InputItem
		if len(textParts) > 0 {
			data, _ := json.Marshal(textParts)
			items = append(items, openairesponses.InputItem{
				Type:    "message",
				Role:    "assistant",
				Content: data,
			})
		}
		items = append(items, funcCalls...)
		return items

	case RoleTool:
		// Tool result → function_call_output
		var items []openairesponses.InputItem
		for _, p := range m.Content {
			if p.Type == ContentTypeToolResult && p.ToolResult != nil {
				items = append(items, openairesponses.InputItem{
					Type:   "function_call_output",
					CallID: p.ToolResult.ToolUseID,
					Output: toolResultText(p.ToolResult),
				})
			}
		}
		return items

	default:
		content := encodeOaiRespContentParts(m.Content, "input_text")
		data, _ := json.Marshal(content)
		return []openairesponses.InputItem{
			{
				Type:    "message",
				Role:    string(m.Role),
				Content: data,
			},
		}
	}
}

// encodeOaiRespContentParts converts IR ContentParts to openairesponses.ContentPart slices.
func encodeOaiRespContentParts(parts []ContentPart, textType string) []openairesponses.ContentPart {
	result := make([]openairesponses.ContentPart, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case ContentTypeText:
			if p.Text != nil {
				result = append(result, openairesponses.ContentPart{
					Type: textType,
					Text: p.Text.Text,
				})
			}
		case ContentTypeImage:
			if p.Image != nil {
				cp := openairesponses.ContentPart{Type: "input_image"}
				if p.Image.URL != "" {
					cp.ImageURL = p.Image.URL
				} else if len(p.Image.Data) > 0 {
					mediaType := p.Image.MediaType
					if mediaType == "" {
						mediaType = "application/octet-stream"
					}
					cp.ImageURL = "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(p.Image.Data)
				}
				result = append(result, cp)
			}
		case ContentTypeDocument:
			if p.Document != nil {
				cp := openairesponses.ContentPart{Type: "input_file"}
				if p.Document.URL != "" {
					cp.FileURL = p.Document.URL
				}
				if len(p.Document.Data) > 0 {
					cp.FileData = base64.StdEncoding.EncodeToString(p.Document.Data)
				}
				if p.Document.Title != "" {
					cp.Filename = p.Document.Title
				}
				result = append(result, cp)
			}
		}
	}
	return result
}

// encodeOaiRespToolChoice converts an IR ToolChoice to Responses API tool_choice JSON.
// Returns nil, nil when tc.Type is empty (e.g. ToolChoice was created only to carry
// AllowParallelCalls with no tool_choice type set). The caller must not emit tool_choice
// in that case; parallel_tool_calls is already emitted separately as a top-level field.
func encodeOaiRespToolChoice(req *Request) (json.RawMessage, error) {
	tc := req.ToolChoice
	if tc == nil {
		return nil, nil
	}
	if len(tc.AllowedToolNames) > 0 {
		allowedTools := selectToolsByName(req.Tools, tc.AllowedToolNames)
		if len(allowedTools) == 0 {
			return nil, fmt.Errorf("encode allowed_tools: no allowed tools matched request tools")
		}
		mode := tc.Type
		if mode == "" || mode == "tool" {
			mode = "required"
		}
		payload := map[string]any{
			"type":  "allowed_tools",
			"mode":  mode,
			"tools": make([]map[string]any, 0, len(allowedTools)),
		}
		for _, tool := range allowedTools {
			selector := map[string]any{
				"type": openAIResponsesToolTypeFromIR(tool.Type),
			}
			if isFunctionToolType(tool.Type) {
				selector["name"] = tool.Name
			}
			payload["tools"] = append(payload["tools"].([]map[string]any), selector)
		}
		return json.Marshal(payload)
	}
	switch tc.Type {
	case "":
		return nil, nil
	case "auto", "none", "required":
		return json.Marshal(tc.Type)
	case "tool":
		if tool := findToolByName(req.Tools, tc.ToolName); tool != nil && !isFunctionToolType(tool.Type) {
			return json.Marshal(map[string]string{
				"type": openAIResponsesToolTypeFromIR(tool.Type),
			})
		}
		obj := openairesponses.ToolChoiceObj{Type: "function", Name: tc.ToolName}
		return json.Marshal(obj)
	default:
		if !isFunctionToolType(tc.Type) {
			return json.Marshal(map[string]string{
				"type": openAIResponsesToolTypeFromIR(tc.Type),
			})
		}
		return json.Marshal(tc.Type)
	}
}

const webSearchToolResultIndexOffset = 1000000

func webSearchToolResultStreamIndex(base int) int {
	return base + webSearchToolResultIndexOffset
}

type oaiRespWebSearchAction struct {
	Type    string   `json:"type"`
	Query   string   `json:"query,omitempty"`
	Queries []string `json:"queries,omitempty"`
	URL     string   `json:"url,omitempty"`
	Pattern string   `json:"pattern,omitempty"`
	Sources []struct {
		Title string `json:"title,omitempty"`
		URL   string `json:"url,omitempty"`
	} `json:"sources,omitempty"`
}

func decodeOaiRespWebSearchAction(actionRaw json.RawMessage) (*oaiRespWebSearchAction, error) {
	if len(actionRaw) == 0 || string(actionRaw) == "null" {
		return nil, nil
	}
	var action oaiRespWebSearchAction
	if err := json.Unmarshal(actionRaw, &action); err != nil {
		return nil, fmt.Errorf("unmarshal web_search action: %w", err)
	}
	return &action, nil
}

func webSearchQueryArguments(action *oaiRespWebSearchAction) (json.RawMessage, error) {
	if action == nil {
		return nil, nil
	}
	query := action.Query
	if query == "" && len(action.Queries) > 0 {
		query = action.Queries[0]
	}
	if query == "" {
		switch action.Type {
		case "open_page":
			query = action.URL
		case "find_in_page":
			query = action.Pattern
		}
	}
	if query == "" {
		return nil, nil
	}
	payload, err := json.Marshal(map[string]string{"query": query})
	if err != nil {
		return nil, fmt.Errorf("marshal web_search query: %w", err)
	}
	return payload, nil
}

func webSearchResultsFromAction(action *oaiRespWebSearchAction) []WebSearchResult {
	if action == nil {
		return nil
	}
	results := make([]WebSearchResult, 0, len(action.Sources))
	for _, source := range action.Sources {
		if source.URL == "" {
			continue
		}
		title := source.Title
		if title == "" {
			title = source.URL
		}
		results = append(results, WebSearchResult{
			Title: title,
			URL:   source.URL,
		})
	}
	if len(results) == 0 && action.URL != "" {
		results = append(results, WebSearchResult{
			Title: action.URL,
			URL:   action.URL,
		})
	}
	return results
}

func decodeOaiRespWebSearchCallParts(item openairesponses.OutputItem) ([]ContentPart, error) {
	action, err := decodeOaiRespWebSearchAction(item.Action)
	if err != nil {
		return nil, err
	}
	args, err := webSearchQueryArguments(action)
	if err != nil {
		return nil, err
	}
	result := []ContentPart{
		{
			Type: ContentTypeServerToolUse,
			ServerToolUse: &ServerToolUseContent{
				ID:        item.ID,
				Name:      "web_search",
				Arguments: args,
			},
		},
	}
	webSearchResult := &WebSearchToolResultContent{
		ToolUseID: item.ID,
		Content:   webSearchResultsFromAction(action),
	}
	if item.Status == "failed" && len(webSearchResult.Content) == 0 {
		webSearchResult.IsError = true
		webSearchResult.ErrorCode = "failed"
	}
	result = append(result, ContentPart{
		Type:                ContentTypeWebSearchToolResult,
		WebSearchToolResult: webSearchResult,
	})
	return result, nil
}

// decodeOaiRespWebSearchCallStreamEvents expands an OpenAI Responses
// `response.output_item.done` event carrying a web_search_call into the
// Anthropic-compatible sequence:
//
//	[content_block_start(server_tool_use), content_block_stop(server_tool_use),
//	 content_block_start(web_search_tool_result), content_block_stop(web_search_tool_result)]
//
// The two blocks use distinct indices — server_tool_use keeps the original
// OpenAI output index, and the tool result uses webSearchToolResultStreamIndex
// to avoid collision — and the outbound layer remaps them to sequential
// Anthropic block indices.
func decodeOaiRespWebSearchCallStreamEvents(item openairesponses.OutputItem, outputIndex int) ([]*StreamEvent, error) {
	parts, err := decodeOaiRespWebSearchCallParts(item)
	if err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, nil
	}
	serverIdx := outputIndex
	resultIdx := webSearchToolResultStreamIndex(outputIndex)
	events := make([]*StreamEvent, 0, 4)
	events = append(events,
		&StreamEvent{Type: StreamEventContentBlockStart, Index: serverIdx, Delta: &parts[0]},
		&StreamEvent{Type: StreamEventContentBlockStop, Index: serverIdx},
	)
	if len(parts) > 1 {
		events = append(events,
			&StreamEvent{Type: StreamEventContentBlockStart, Index: resultIdx, Delta: &parts[1]},
			&StreamEvent{Type: StreamEventContentBlockStop, Index: resultIdx},
		)
	}
	return events, nil
}

// --- Response decode/encode ---

// DecodeOpenAIResponsesResponse decodes an OpenAI Responses API JSON response body
// into the unified IR Response type.
func DecodeOpenAIResponsesResponse(body []byte) (*Response, error) {
	var raw openairesponses.Response
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode openai responses response: %w", err)
	}

	resp := &Response{
		ID:    raw.ID,
		Model: raw.Model,
	}

	// Usage
	if raw.Usage != nil {
		resp.Usage = decodeOaiRespUsage(raw.Usage)
	}

	// Output items → Content
	hasFunctionCall := false
	for _, item := range raw.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					textPart := ContentPart{
						Type: ContentTypeText,
						Text: &TextContent{Text: c.Text},
					}
					if len(c.Annotations) > 0 {
						citations, err := parseOpenAIAnnotations(c.Annotations)
						if err != nil {
							return nil, fmt.Errorf("decode openai responses response annotations: %w", err)
						}
						textPart.Citations = citations
					}
					resp.Content = append(resp.Content, textPart)
				case "refusal":
					resp.Content = append(resp.Content, ContentPart{
						Type:    ContentTypeRefusal,
						Refusal: &RefusalContent{Refusal: c.Refusal},
					})
				}
			}
		case "function_call":
			hasFunctionCall = true
			resp.Content = append(resp.Content, ContentPart{
				Type: ContentTypeToolUse,
				ToolUse: &ToolUseContent{
					ID:        item.CallID,
					Name:      item.Name,
					Arguments: json.RawMessage(item.Arguments),
				},
			})
		case "web_search_call":
			parts, err := decodeOaiRespWebSearchCallParts(item)
			if err != nil {
				return nil, fmt.Errorf("decode openai responses response web_search_call: %w", err)
			}
			resp.Content = append(resp.Content, parts...)
		}
	}

	// Status → StopReason
	switch raw.Status {
	case "completed":
		if hasFunctionCall {
			resp.StopReason = StopReasonToolUse
		} else {
			resp.StopReason = StopReasonEndTurn
		}
	case "incomplete":
		resp.StopReason = StopReasonMaxTokens
	case "failed":
		resp.StopReason = StopReasonContentFilter
	default:
		if raw.Status != "" {
			resp.StopReason = StopReason(raw.Status)
		}
	}

	return resp, nil
}

// EncodeOpenAIResponsesResponse encodes a unified IR Response into an OpenAI Responses API JSON body.
func EncodeOpenAIResponsesResponse(resp *Response) ([]byte, error) {
	raw := openairesponses.Response{
		ID:     resp.ID,
		Object: "response",
		Model:  resp.Model,
	}

	// Build output items
	var msgContent []openairesponses.OutputContent
	var funcCalls []openairesponses.OutputItem
	hasFunctionCall := false

	for _, p := range resp.Content {
		switch p.Type {
		case ContentTypeText:
			if p.Text != nil {
				oc := openairesponses.OutputContent{
					Type: "output_text",
					Text: p.Text.Text,
				}
				if len(p.Citations) > 0 {
					oc.Annotations = encodeOpenAIAnnotations(p.Citations)
				}
				msgContent = append(msgContent, oc)
			}
		case ContentTypeRefusal:
			if p.Refusal != nil {
				msgContent = append(msgContent, openairesponses.OutputContent{
					Type:    "refusal",
					Refusal: p.Refusal.Refusal,
				})
			}
		case ContentTypeToolUse:
			hasFunctionCall = true
			if p.ToolUse != nil {
				funcCalls = append(funcCalls, openairesponses.OutputItem{
					Type:      "function_call",
					CallID:    p.ToolUse.ID,
					Name:      p.ToolUse.Name,
					Arguments: string(p.ToolUse.Arguments),
				})
			}
		}
	}

	if len(msgContent) > 0 {
		raw.Output = append(raw.Output, openairesponses.OutputItem{
			Type:    "message",
			Role:    "assistant",
			Content: msgContent,
		})
	}
	raw.Output = append(raw.Output, funcCalls...)

	// StopReason → status
	switch resp.StopReason {
	case StopReasonEndTurn, StopReasonStopSequence:
		raw.Status = "completed"
	case StopReasonToolUse:
		raw.Status = "completed"
	case StopReasonMaxTokens:
		raw.Status = "incomplete"
	case StopReasonContentFilter:
		raw.Status = "failed"
	case StopReasonPauseTurn:
		raw.Status = "completed"
	default:
		if resp.StopReason != "" {
			raw.Status = string(resp.StopReason)
		} else if hasFunctionCall {
			raw.Status = "completed"
		} else {
			raw.Status = "completed"
		}
	}

	// Usage
	raw.Usage = encodeOaiRespUsage(&resp.Usage)

	return json.Marshal(raw)
}

// --- Streaming decode/encode ---

// DecodeOpenAIResponsesStreamEvent decodes an OpenAI Responses API SSE event into
// zero or more unified IR StreamEvents. Returns ([]*StreamEvent, nil) on success
// (the slice may be nil for events that should be skipped), or (nil, err) on a
// protocol-level decode failure. A single OpenAI event may fan out into several
// IR events — for example, a completed web_search_call produces both the
// server_tool_use and the web_search_tool_result blocks.
//
// eventType is from the SSE "event:" line, data is the JSON from the SSE "data:" line.
func DecodeOpenAIResponsesStreamEvent(eventType string, data []byte) ([]*StreamEvent, error) {
	switch eventType {
	case "response.created":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream response.created: %w", err)
		}
		event := &StreamEvent{
			Type: StreamEventStart,
		}
		if raw.Response != nil {
			event.Response = &Response{
				ID:    raw.Response.ID,
				Model: raw.Response.Model,
			}
		}
		return []*StreamEvent{event}, nil

	case "response.output_item.added":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream output_item.added: %w", err)
		}
		event := &StreamEvent{
			Type:  StreamEventContentBlockStart,
			Index: derefIntPtr(raw.OutputIndex),
		}
		if raw.Item != nil {
			switch raw.Item.Type {
			case "message":
				event.Delta = &ContentPart{
					Type: ContentTypeText,
					Text: &TextContent{},
				}
			case "function_call":
				event.Delta = &ContentPart{
					Type: ContentTypeToolUse,
					ToolUse: &ToolUseContent{
						ID:   raw.Item.CallID,
						Name: raw.Item.Name,
					},
				}
			case "web_search_call":
				// Defer emission until response.output_item.done, when the
				// action payload (query, sources) is available. Emitting a
				// start with an empty ServerToolUse here would leave Anthropic
				// clients waiting for input_json_delta events that never arrive.
				return nil, nil
			}
		}
		return []*StreamEvent{event}, nil

	case "response.content_part.added":
		// Skipped — response.output_item.added is the canonical block start.
		return nil, nil

	case "response.output_text.delta":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream output_text.delta: %w", err)
		}
		return []*StreamEvent{{
			Type:  StreamEventDelta,
			Index: derefIntPtr(raw.OutputIndex),
			Delta: &ContentPart{
				Type: ContentTypeText,
				Text: &TextContent{Text: raw.Delta},
			},
		}}, nil

	case "response.function_call_arguments.delta":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream function_call_arguments.delta: %w", err)
		}
		return []*StreamEvent{{
			Type:  StreamEventDelta,
			Index: derefIntPtr(raw.OutputIndex),
			Delta: &ContentPart{
				Type: ContentTypeToolUse,
				ToolUse: &ToolUseContent{
					Arguments: json.RawMessage(raw.Delta),
				},
			},
		}}, nil

	case "response.output_item.done":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream output_item.done: %w", err)
		}
		if raw.Item != nil && raw.Item.Type == "web_search_call" {
			events, err := decodeOaiRespWebSearchCallStreamEvents(*raw.Item, derefIntPtr(raw.OutputIndex))
			if err != nil {
				return nil, fmt.Errorf("decode openai responses stream web_search_call done: %w", err)
			}
			return events, nil
		}
		return []*StreamEvent{{
			Type:  StreamEventContentBlockStop,
			Index: derefIntPtr(raw.OutputIndex),
		}}, nil

	case "response.content_part.done":
		// Skipped — response.output_item.done is the canonical block stop.
		return nil, nil

	case "response.web_search_call.in_progress",
		"response.web_search_call.searching",
		"response.web_search_call.completed":
		// Progress hints — Anthropic's streaming protocol has no equivalent
		// intermediate event for server tools, so drop them here. The full
		// server_tool_use / web_search_tool_result pair is emitted from
		// response.output_item.done once the action payload is known.
		return nil, nil

	case "response.completed":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream response.completed: %w", err)
		}
		event := &StreamEvent{
			Type: StreamEventStop,
		}
		if raw.Response != nil {
			if raw.Response.Usage != nil {
				u := decodeOaiRespUsage(raw.Response.Usage)
				event.Usage = &u
			}
			// Stop reason from status
			switch raw.Response.Status {
			case "completed":
				sr := StopReasonEndTurn
				event.StopReason = &sr
			case "incomplete":
				sr := StopReasonMaxTokens
				event.StopReason = &sr
			}
		}
		return []*StreamEvent{event}, nil

	case "response.output_text.done", "response.function_call_arguments.done":
		// Skipped — response.output_item.done is the canonical block stop.
		return nil, nil

	case "response.failed":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream response.failed: %w", err)
		}
		event := &StreamEvent{
			Type: StreamEventError,
			Error: &StreamError{
				Type:    "server_error",
				Message: "response failed",
			},
		}
		if raw.Response != nil {
			if raw.Response.Usage != nil {
				u := decodeOaiRespUsage(raw.Response.Usage)
				event.Usage = &u
			}
		}
		return []*StreamEvent{event}, nil

	case "response.incomplete":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream response.incomplete: %w", err)
		}
		sr := StopReasonMaxTokens
		event := &StreamEvent{
			Type:              StreamEventStop,
			StopReason:        &sr,
			IncompleteDetails: &IncompleteDetails{Reason: "max_output_tokens"},
		}
		if raw.Response != nil && raw.Response.Usage != nil {
			u := decodeOaiRespUsage(raw.Response.Usage)
			event.Usage = &u
		}
		return []*StreamEvent{event}, nil

	case "error":
		// Top-level error event
		var errPayload struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
			Param   string `json:"param"`
		}
		if err := json.Unmarshal(data, &errPayload); err != nil {
			return nil, fmt.Errorf("decode openai responses stream error: %w", err)
		}
		return []*StreamEvent{{
			Type: StreamEventError,
			Error: &StreamError{
				Type:    errPayload.Type,
				Code:    errPayload.Code,
				Message: errPayload.Message,
				Param:   errPayload.Param,
			},
		}}, nil

	case "response.refusal.delta":
		var raw openairesponses.StreamEvent
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, fmt.Errorf("decode openai responses stream response.refusal.delta: %w", err)
		}
		return []*StreamEvent{{
			Type:  StreamEventDelta,
			Index: derefIntPtr(raw.OutputIndex),
			Delta: &ContentPart{
				Type:    ContentTypeRefusal,
				Refusal: &RefusalContent{Refusal: raw.Delta},
			},
		}}, nil

	default:
		// Unknown event types (response.in_progress, etc.) — return nil to skip
		return nil, nil
	}
}

// EncodeOpenAIResponsesStreamEvent encodes a unified IR StreamEvent into an OpenAI
// Responses API SSE event type and JSON data.
func EncodeOpenAIResponsesStreamEvent(event *StreamEvent) (string, []byte, error) {
	switch event.Type {
	case StreamEventStart:
		raw := openairesponses.StreamEvent{Type: "response.created"}
		if event.Response != nil {
			raw.Response = &openairesponses.Response{
				ID:     event.Response.ID,
				Object: "response",
				Model:  event.Response.Model,
				Status: "in_progress",
			}
		}
		data, err := json.Marshal(raw)
		if err != nil {
			return "", nil, fmt.Errorf("encode openai responses stream response.created: %w", err)
		}
		return "response.created", data, nil

	case StreamEventContentBlockStart:
		raw := openairesponses.StreamEvent{
			OutputIndex: intPtr(event.Index),
		}
		if event.Delta != nil {
			switch event.Delta.Type {
			case ContentTypeText:
				raw.Type = "response.output_item.added"
				raw.Item = &openairesponses.OutputItem{
					Type:    "message",
					Role:    "assistant",
					Content: []openairesponses.OutputContent{},
				}
				data, err := json.Marshal(raw)
				if err != nil {
					return "", nil, fmt.Errorf("encode openai responses stream output_item.added: %w", err)
				}
				return "response.output_item.added", data, nil
			case ContentTypeToolUse:
				item := &openairesponses.OutputItem{
					Type: "function_call",
				}
				if event.Delta.ToolUse != nil {
					item.CallID = event.Delta.ToolUse.ID
					item.Name = event.Delta.ToolUse.Name
				}
				raw.Item = item
			}
		}
		raw.Type = "response.output_item.added"
		data, err := json.Marshal(raw)
		if err != nil {
			return "", nil, fmt.Errorf("encode openai responses stream output_item.added: %w", err)
		}
		return "response.output_item.added", data, nil

	case StreamEventDelta:
		if event.Delta != nil {
			switch event.Delta.Type {
			case ContentTypeText:
				var text string
				if event.Delta.Text != nil {
					text = event.Delta.Text.Text
				}
				raw := openairesponses.StreamEvent{
					Type:        "response.output_text.delta",
					OutputIndex: intPtr(event.Index),
					Delta:       text,
				}
				data, err := json.Marshal(raw)
				if err != nil {
					return "", nil, fmt.Errorf("encode openai responses stream output_text.delta: %w", err)
				}
				return "response.output_text.delta", data, nil

			case ContentTypeRefusal:
				var refusal string
				if event.Delta.Refusal != nil {
					refusal = event.Delta.Refusal.Refusal
				}
				raw := openairesponses.StreamEvent{
					Type:        "response.refusal.delta",
					OutputIndex: intPtr(event.Index),
					Delta:       refusal,
				}
				data, err := json.Marshal(raw)
				if err != nil {
					return "", nil, fmt.Errorf("encode openai responses stream response.refusal.delta: %w", err)
				}
				return "response.refusal.delta", data, nil

			case ContentTypeToolUse:
				var args string
				if event.Delta.ToolUse != nil {
					args = string(event.Delta.ToolUse.Arguments)
				}
				raw := openairesponses.StreamEvent{
					Type:        "response.function_call_arguments.delta",
					OutputIndex: intPtr(event.Index),
					Delta:       args,
				}
				data, err := json.Marshal(raw)
				if err != nil {
					return "", nil, fmt.Errorf("encode openai responses stream function_call_arguments.delta: %w", err)
				}
				return "response.function_call_arguments.delta", data, nil
			}
		}
		return "", nil, fmt.Errorf("encode openai responses stream delta: nil or unsupported delta type")

	case StreamEventContentBlockStop:
		raw := openairesponses.StreamEvent{
			Type:        "response.output_item.done",
			OutputIndex: intPtr(event.Index),
		}
		data, err := json.Marshal(raw)
		if err != nil {
			return "", nil, fmt.Errorf("encode openai responses stream output_item.done: %w", err)
		}
		return "response.output_item.done", data, nil

	case StreamEventStop:
		resp := &openairesponses.Response{
			Object: "response",
			Status: "completed",
		}
		if event.StopReason != nil {
			switch *event.StopReason {
			case StopReasonMaxTokens:
				resp.Status = "incomplete"
			case StopReasonContentFilter:
				resp.Status = "failed"
			}
		}
		if event.Usage != nil {
			resp.Usage = encodeOaiRespUsage(event.Usage)
		}
		raw := openairesponses.StreamEvent{
			Type:     "response.completed",
			Response: resp,
		}
		data, err := json.Marshal(raw)
		if err != nil {
			return "", nil, fmt.Errorf("encode openai responses stream response.completed: %w", err)
		}
		return "response.completed", data, nil

	case StreamEventError:
		errPayload := struct {
			Type    string `json:"type"`
			Code    string `json:"code,omitempty"`
			Message string `json:"message,omitempty"`
			Param   string `json:"param,omitempty"`
		}{
			Type: "error",
		}
		if event.Error != nil {
			errPayload.Code = event.Error.Code
			errPayload.Message = event.Error.Message
			errPayload.Param = event.Error.Param
		}
		data, err := json.Marshal(errPayload)
		if err != nil {
			return "", nil, fmt.Errorf("encode openai responses stream error: %w", err)
		}
		return "error", data, nil

	default:
		return "", nil, fmt.Errorf("unknown IR stream event type: %q", event.Type)
	}
}
