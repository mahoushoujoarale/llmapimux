package llmapimux

import (
	"encoding/json"
	"fmt"
	"strings"
)

// This file provides the shared infrastructure for the cross-protocol conformance
// matrix in conformance_matrix_test.go:
//
//   - inboundFixture: builds a semantically equivalent request for each of the 4
//     inbound protocols, so one logical scenario can be driven through all 16
//     inbound × outbound combinations.
//   - validateOutboundBody: asserts that the JSON body the gateway is about to
//     send upstream is actually *valid* for the target protocol.
//
// The validators encode provider-side rules the gateway must never violate,
// regardless of which protocol the request arrived on. They exist because unit
// tests that only assert "field X was mapped to field Y" cannot catch whole-body
// invariants such as "tool_choice must not appear without tools" or "a message's
// content array must not be empty" — the exact class of bug that produces an
// upstream 400 in production.

// protocolName renders a Protocol for test names.
func protocolName(p Protocol) string {
	switch p {
	case ProtocolAnthropic:
		return "anthropic"
	case ProtocolOpenAIChat:
		return "openai_chat"
	case ProtocolOpenAIResponses:
		return "openai_responses"
	case ProtocolGemini:
		return "gemini"
	default:
		return string(p)
	}
}

// allProtocols is the full set used to build the 4×4 matrix.
var allProtocols = []Protocol{
	ProtocolAnthropic,
	ProtocolOpenAIChat,
	ProtocolOpenAIResponses,
	ProtocolGemini,
}

// --- Outbound body validation ---

// validateOutboundBody checks that body is a legal request for the given
// protocol. It returns every violation found so a single failure reports all
// problems rather than just the first.
func validateOutboundBody(protocol Protocol, body []byte) []string {
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	var generic map[string]json.RawMessage
	if err := json.Unmarshal(body, &generic); err != nil {
		return []string{fmt.Sprintf("body is not a JSON object: %v", err)}
	}

	// No protocol accepts an explicit null for an optional object/array field;
	// strict gateways in front of the real APIs reject them.
	for _, field := range []string{"tools", "tool_choice", "system", "thinking",
		"messages", "input", "contents", "response_format", "text", "generationConfig"} {
		if raw, ok := generic[field]; ok && string(raw) == "null" {
			add("field %q is explicit null; omit it instead", field)
		}
	}

	switch protocol {
	case ProtocolAnthropic:
		problems = append(problems, validateAnthropicOutbound(body)...)
	case ProtocolOpenAIChat:
		problems = append(problems, validateOpenAIChatOutbound(body)...)
	case ProtocolOpenAIResponses:
		problems = append(problems, validateOpenAIResponsesOutbound(body)...)
	case ProtocolGemini:
		problems = append(problems, validateGeminiOutbound(body)...)
	}
	return problems
}

func validateAnthropicOutbound(body []byte) []string {
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	var req struct {
		Model     string `json:"model"`
		MaxTokens int    `json:"max_tokens"`
		Messages  []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
		Tools []struct {
			Name string `json:"name"`
			Type string `json:"type"`
		} `json:"tools"`
		ToolChoice *struct {
			Type string `json:"type"`
			Name string `json:"name"`
		} `json:"tool_choice"`
		Thinking *struct {
			Type         string `json:"type"`
			BudgetTokens int    `json:"budget_tokens"`
		} `json:"thinking"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return []string{fmt.Sprintf("unmarshal anthropic body: %v", err)}
	}

	if req.Model == "" {
		add("model is required")
	}
	// Anthropic requires max_tokens on every request.
	if req.MaxTokens <= 0 {
		add("max_tokens must be > 0, got %d", req.MaxTokens)
	}
	if len(req.Messages) == 0 {
		add("messages must be non-empty")
	}
	for i, m := range req.Messages {
		if m.Role != "user" && m.Role != "assistant" {
			add("messages[%d].role = %q, want user or assistant", i, m.Role)
		}
		if len(m.Content) == 0 || string(m.Content) == "null" {
			add("messages[%d].content is missing", i)
			continue
		}
		if string(m.Content) == "[]" {
			add("messages[%d].content is an empty array", i)
		}
	}
	// A tool_choice with no tools is rejected.
	if req.ToolChoice != nil && len(req.Tools) == 0 {
		add("tool_choice present but tools is empty")
	}
	// A named tool_choice must reference a declared tool.
	if req.ToolChoice != nil && req.ToolChoice.Type == "tool" {
		if req.ToolChoice.Name == "" {
			add("tool_choice.type=tool requires a name")
		} else {
			found := false
			for _, t := range req.Tools {
				if t.Name == req.ToolChoice.Name {
					found = true
					break
				}
			}
			if !found {
				add("tool_choice.name=%q does not match any tool", req.ToolChoice.Name)
			}
		}
	}
	if req.ToolChoice != nil {
		switch req.ToolChoice.Type {
		case "auto", "any", "tool", "none":
		default:
			add("tool_choice.type = %q is not a valid Anthropic value", req.ToolChoice.Type)
		}
	}
	// budget_tokens is only valid when thinking is enabled.
	if req.Thinking != nil {
		switch req.Thinking.Type {
		case "enabled":
			if req.Thinking.BudgetTokens <= 0 {
				add("thinking.type=enabled requires budget_tokens > 0")
			}
		case "disabled":
			if req.Thinking.BudgetTokens != 0 {
				add("thinking.type=disabled must not carry budget_tokens")
			}
		default:
			add("thinking.type = %q is not valid", req.Thinking.Type)
		}
	}
	// Every function tool needs an input_schema.
	var toolsRaw struct {
		Tools []map[string]json.RawMessage `json:"tools"`
	}
	_ = json.Unmarshal(body, &toolsRaw)
	for i, t := range toolsRaw.Tools {
		typ := ""
		if raw, ok := t["type"]; ok {
			_ = json.Unmarshal(raw, &typ)
		}
		if typ == "" || typ == "custom" {
			if _, ok := t["input_schema"]; !ok {
				add("tools[%d] is a custom tool with no input_schema", i)
			}
		}
	}
	return problems
}

func validateOpenAIChatOutbound(body []byte) []string {
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	var req struct {
		Model    string `json:"model"`
		Messages []struct {
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			ToolCallID string          `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
		ToolChoice     json.RawMessage `json:"tool_choice"`
		ResponseFormat *struct {
			Type       string `json:"type"`
			JSONSchema *struct {
				Name   string          `json:"name"`
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
		MaxCompletionTokens *int `json:"max_completion_tokens"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return []string{fmt.Sprintf("unmarshal openai chat body: %v", err)}
	}

	if req.Model == "" {
		add("model is required")
	}
	if len(req.Messages) == 0 {
		add("messages must be non-empty")
	}

	// Collect declared tool names for tool_choice validation.
	toolNames := map[string]bool{}
	for i, t := range req.Tools {
		if t.Type != "function" {
			add("tools[%d].type = %q, Chat Completions only supports \"function\"", i, t.Type)
		}
		if t.Function.Name == "" {
			add("tools[%d].function.name is empty", i)
		}
		toolNames[t.Function.Name] = true
	}

	// Every tool message must answer a preceding assistant tool_call, and every
	// tool_call must be answered. An unanswered tool_call_id is a hard 400.
	pendingCalls := map[string]bool{}
	for i, m := range req.Messages {
		switch m.Role {
		case "system", "developer", "user", "assistant", "tool":
		default:
			add("messages[%d].role = %q is not a valid Chat role", i, m.Role)
		}
		if m.Role == "tool" {
			if m.ToolCallID == "" {
				add("messages[%d] is a tool message with no tool_call_id", i)
			} else if !pendingCalls[m.ToolCallID] {
				add("messages[%d].tool_call_id=%q does not answer any preceding tool_call", i, m.ToolCallID)
			}
			delete(pendingCalls, m.ToolCallID)
		}
		for _, tc := range m.ToolCalls {
			if tc.ID != "" {
				pendingCalls[tc.ID] = true
			}
		}
		// content may legitimately be absent on an assistant message that only
		// makes tool calls; everywhere else an empty array is invalid.
		if string(m.Content) == "[]" {
			add("messages[%d].content is an empty array", i)
		}
		if m.Role == "user" && (len(m.Content) == 0 || string(m.Content) == "null") {
			add("messages[%d] is a user message with no content", i)
		}
	}
	for id := range pendingCalls {
		add("tool_call id=%q was never answered by a tool message", id)
	}

	// tool_choice must be a valid value and, when named, reference a real tool.
	if len(req.ToolChoice) > 0 {
		if len(req.Tools) == 0 {
			add("tool_choice present but tools is empty")
		}
		if req.ToolChoice[0] == '"' {
			var s string
			_ = json.Unmarshal(req.ToolChoice, &s)
			switch s {
			case "auto", "none", "required":
			default:
				add("tool_choice = %q is not a valid Chat value", s)
			}
		} else {
			var obj struct {
				Type     string `json:"type"`
				Function *struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			if err := json.Unmarshal(req.ToolChoice, &obj); err != nil {
				add("tool_choice is neither a string nor an object: %v", err)
			} else {
				if obj.Type != "function" {
					add("tool_choice.type = %q, want \"function\"", obj.Type)
				}
				if obj.Function == nil || obj.Function.Name == "" {
					add("tool_choice object has no function.name")
				} else if !toolNames[obj.Function.Name] {
					add("tool_choice.function.name=%q does not match any tool", obj.Function.Name)
				}
			}
		}
	}

	// json_schema response_format requires a name.
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_schema" {
		if req.ResponseFormat.JSONSchema == nil {
			add("response_format.type=json_schema requires json_schema")
		} else {
			if req.ResponseFormat.JSONSchema.Name == "" {
				add("response_format.json_schema.name is required")
			}
			if len(req.ResponseFormat.JSONSchema.Schema) == 0 {
				add("response_format.json_schema.schema is required")
			}
		}
	}
	if req.MaxCompletionTokens != nil && *req.MaxCompletionTokens <= 0 {
		add("max_completion_tokens must be > 0")
	}
	return problems
}

func validateOpenAIResponsesOutbound(body []byte) []string {
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	var req struct {
		Model string          `json:"model"`
		Input json.RawMessage `json:"input"`
		Tools []struct {
			Type string `json:"type"`
			Name string `json:"name"`
		} `json:"tools"`
		ToolChoice json.RawMessage `json:"tool_choice"`
		Text       *struct {
			Format *struct {
				Type   string          `json:"type"`
				Name   string          `json:"name"`
				Schema json.RawMessage `json:"schema"`
			} `json:"format"`
		} `json:"text"`
		MaxOutputTokens *int `json:"max_output_tokens"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return []string{fmt.Sprintf("unmarshal openai responses body: %v", err)}
	}

	if req.Model == "" {
		add("model is required")
	}

	toolNames := map[string]bool{}
	for i, t := range req.Tools {
		if t.Type == "" {
			add("tools[%d].type is empty", i)
		}
		if t.Type == "function" && t.Name == "" {
			add("tools[%d] is a function tool with no name", i)
		}
		if t.Name != "" {
			toolNames[t.Name] = true
		}
	}

	// Validate the input array: every item needs a known type, and
	// function_call_output items must answer a preceding function_call.
	if len(req.Input) > 0 && req.Input[0] == '[' {
		var items []struct {
			Type    string          `json:"type"`
			Role    string          `json:"role"`
			CallID  string          `json:"call_id"`
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(req.Input, &items); err != nil {
			add("input is not a valid item array: %v", err)
		} else {
			pending := map[string]bool{}
			for i, it := range items {
				switch it.Type {
				case "message":
					switch it.Role {
					case "developer", "system", "user", "assistant":
					default:
						add("input[%d].role = %q is not valid", i, it.Role)
					}
					if string(it.Content) == "[]" {
						add("input[%d].content is an empty array", i)
					}
				case "function_call":
					if it.CallID == "" {
						add("input[%d] function_call has no call_id", i)
					}
					pending[it.CallID] = true
				case "function_call_output":
					if it.CallID == "" {
						add("input[%d] function_call_output has no call_id", i)
					} else if !pending[it.CallID] {
						add("input[%d].call_id=%q does not answer any preceding function_call", i, it.CallID)
					}
					delete(pending, it.CallID)
				case "reasoning":
					// Reasoning items carry no call correlation.
				default:
					add("input[%d].type = %q is not a known Responses item type", i, it.Type)
				}
			}
			for id := range pending {
				add("function_call id=%q was never answered", id)
			}
		}
	}

	if len(req.ToolChoice) > 0 {
		if len(req.Tools) == 0 {
			add("tool_choice present but tools is empty")
		}
		if req.ToolChoice[0] == '"' {
			var s string
			_ = json.Unmarshal(req.ToolChoice, &s)
			switch s {
			case "auto", "none", "required":
			default:
				add("tool_choice = %q is not valid", s)
			}
		} else {
			var obj struct {
				Type string `json:"type"`
				Name string `json:"name"`
			}
			if err := json.Unmarshal(req.ToolChoice, &obj); err != nil {
				add("tool_choice object invalid: %v", err)
			} else if obj.Type == "function" && obj.Name == "" {
				add("tool_choice function selector has no name")
			}
		}
	}

	if req.Text != nil && req.Text.Format != nil && req.Text.Format.Type == "json_schema" {
		if req.Text.Format.Name == "" {
			add("text.format.name is required for json_schema")
		}
		if len(req.Text.Format.Schema) == 0 {
			add("text.format.schema is required for json_schema")
		}
	}
	if req.MaxOutputTokens != nil && *req.MaxOutputTokens <= 0 {
		add("max_output_tokens must be > 0")
	}
	return problems
}

func validateGeminiOutbound(body []byte) []string {
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	var req struct {
		Contents []struct {
			Role  string            `json:"role"`
			Parts []map[string]any  `json:"parts"`
		} `json:"contents"`
		SystemInstruction *struct {
			Role  string           `json:"role"`
			Parts []map[string]any `json:"parts"`
		} `json:"systemInstruction"`
		Tools []struct {
			FunctionDeclarations []struct {
				Name string `json:"name"`
			} `json:"functionDeclarations"`
		} `json:"tools"`
		ToolConfig *struct {
			FunctionCallingConfig *struct {
				Mode                 string   `json:"mode"`
				AllowedFunctionNames []string `json:"allowedFunctionNames"`
			} `json:"functionCallingConfig"`
		} `json:"toolConfig"`
		GenerationConfig *struct {
			MaxOutputTokens  *int   `json:"maxOutputTokens"`
			ResponseMimeType string `json:"responseMimeType"`
		} `json:"generationConfig"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return []string{fmt.Sprintf("unmarshal gemini body: %v", err)}
	}

	if len(req.Contents) == 0 {
		add("contents must be non-empty")
	}
	for i, c := range req.Contents {
		if c.Role != "" && c.Role != "user" && c.Role != "model" {
			add("contents[%d].role = %q, want user or model", i, c.Role)
		}
		// Gemini returns 400 INVALID_ARGUMENT for an empty parts array.
		if len(c.Parts) == 0 {
			add("contents[%d].parts is empty", i)
		}
		for j, p := range c.Parts {
			if len(p) == 0 {
				add("contents[%d].parts[%d] is an empty object", i, j)
			}
			// functionResponse.response must be a JSON object (protobuf Struct).
			if fr, ok := p["functionResponse"].(map[string]any); ok {
				if resp, ok := fr["response"]; ok {
					if _, isObj := resp.(map[string]any); !isObj {
						add("contents[%d].parts[%d].functionResponse.response must be an object, got %T", i, j, resp)
					}
				}
			}
		}
	}
	// systemInstruction must not carry a role.
	if req.SystemInstruction != nil {
		if req.SystemInstruction.Role != "" {
			add("systemInstruction must not set a role, got %q", req.SystemInstruction.Role)
		}
		if len(req.SystemInstruction.Parts) == 0 {
			add("systemInstruction.parts is empty")
		}
	}

	declared := map[string]bool{}
	for _, td := range req.Tools {
		for i, fd := range td.FunctionDeclarations {
			if fd.Name == "" {
				add("functionDeclarations[%d].name is empty", i)
			}
			declared[fd.Name] = true
		}
	}
	if req.ToolConfig != nil && req.ToolConfig.FunctionCallingConfig != nil {
		fcc := req.ToolConfig.FunctionCallingConfig
		switch fcc.Mode {
		case "AUTO", "NONE", "ANY", "VALIDATED":
		default:
			add("functionCallingConfig.mode = %q is not valid", fcc.Mode)
		}
		if len(declared) == 0 {
			add("toolConfig present but no functionDeclarations")
		}
		for _, name := range fcc.AllowedFunctionNames {
			if !declared[name] {
				add("allowedFunctionNames contains %q which is not declared", name)
			}
		}
	}
	if req.GenerationConfig != nil {
		if req.GenerationConfig.MaxOutputTokens != nil && *req.GenerationConfig.MaxOutputTokens <= 0 {
			add("maxOutputTokens must be > 0")
		}
		if mt := req.GenerationConfig.ResponseMimeType; mt != "" &&
			mt != "text/plain" && mt != "application/json" && mt != "text/x.enum" {
			add("responseMimeType = %q is not supported", mt)
		}
	}

	// Gemini schemas must not contain JSON-Schema-only keywords, which the API
	// rejects as unknown fields.
	for _, forbidden := range []string{`"$ref"`, `"$defs"`, `"additionalProperties"`, `"allOf"`, `"not"`, `"const"`} {
		if strings.Contains(string(body), forbidden) {
			add("body contains JSON-Schema-only keyword %s which Gemini rejects", forbidden)
		}
	}
	return problems
}

// --- Inbound fixtures ---

// inboundFixture describes one logical request rendered for a specific inbound
// protocol: the HTTP path, body, and headers needed to drive the matching
// Handler.
type inboundFixture struct {
	path   string
	body   string
	header map[string]string
}
