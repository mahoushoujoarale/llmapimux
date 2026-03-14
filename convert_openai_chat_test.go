package llmapimux

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/llmapimux/llmapimux/protocol/openaichat"
)

func TestDecodeOpenAIChatRequest_Basic(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": [{"type": "text", "text": "Hello"}]},
			{"role": "assistant", "content": "Hi there!"}
		],
		"temperature": 0.7,
		"stream": false
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Model != "gpt-4o" {
		t.Errorf("Model = %q, want %q", req.Model, "gpt-4o")
	}
	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", req.Temperature)
	}
	if req.Stream {
		t.Error("Stream = true, want false")
	}

	// System prompt from system role
	if len(req.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(req.SystemPrompt))
	}
	if req.SystemPrompt[0].Type != ContentTypeText || req.SystemPrompt[0].Text.Text != "You are helpful." {
		t.Errorf("SystemPrompt[0] = %+v, want text 'You are helpful.'", req.SystemPrompt[0])
	}

	// Messages: user + assistant
	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
	}
	if req.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", req.Messages[0].Role, RoleUser)
	}
	if len(req.Messages[0].Content) != 1 || req.Messages[0].Content[0].Text.Text != "Hello" {
		t.Errorf("Messages[0].Content = %+v, want text 'Hello'", req.Messages[0].Content)
	}
	if req.Messages[1].Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", req.Messages[1].Role, RoleAssistant)
	}
	if len(req.Messages[1].Content) != 1 || req.Messages[1].Content[0].Text.Text != "Hi there!" {
		t.Errorf("Messages[1].Content = %+v, want text 'Hi there!'", req.Messages[1].Content)
	}
}

func TestDecodeOpenAIChatRequest_DeveloperRole(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "developer", "content": "You are a coding assistant."},
			{"role": "user", "content": "Write hello world"}
		]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(req.SystemPrompt))
	}
	if req.SystemPrompt[0].Text.Text != "You are a coding assistant." {
		t.Errorf("SystemPrompt[0].Text = %q, want %q", req.SystemPrompt[0].Text.Text, "You are a coding assistant.")
	}

	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	if req.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", req.Messages[0].Role, RoleUser)
	}
}

func TestDecodeOpenAIChatRequest_Tools(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "What's the weather?"}],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get the weather",
					"parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
					"strict": true
				}
			}
		],
		"tool_choice": "auto"
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(req.Tools))
	}
	tool := req.Tools[0]
	if tool.Name != "get_weather" {
		t.Errorf("Tool.Name = %q, want %q", tool.Name, "get_weather")
	}
	if tool.Description != "Get the weather" {
		t.Errorf("Tool.Description = %q, want %q", tool.Description, "Get the weather")
	}
	if !tool.Strict {
		t.Error("Tool.Strict = false, want true")
	}
	if tool.Parameters == nil {
		t.Error("Tool.Parameters is nil")
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "auto" {
		t.Errorf("ToolChoice.Type = %q, want %q", req.ToolChoice.Type, "auto")
	}
}

func TestDecodeOpenAIChatRequest_ToolChoiceRequired(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hi"}],
		"tool_choice": "required"
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.ToolChoice == nil || req.ToolChoice.Type != "required" {
		t.Errorf("ToolChoice = %+v, want type 'required'", req.ToolChoice)
	}
}

func TestDecodeOpenAIChatRequest_ToolChoiceFunction(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hi"}],
		"tool_choice": {"type": "function", "function": {"name": "read_file"}}
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "tool" {
		t.Errorf("ToolChoice.Type = %q, want %q", req.ToolChoice.Type, "tool")
	}
	if req.ToolChoice.ToolName != "read_file" {
		t.Errorf("ToolChoice.ToolName = %q, want %q", req.ToolChoice.ToolName, "read_file")
	}
}

func TestDecodeOpenAIChatRequest_Image(t *testing.T) {
	imgData := []byte{0x89, 0x50, 0x4E, 0x47} // PNG header bytes
	b64 := base64.StdEncoding.EncodeToString(imgData)
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "What is this?"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,` + b64 + `", "detail": "high"}}
			]
		}]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	content := req.Messages[0].Content
	if len(content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(content))
	}
	if content[0].Type != ContentTypeText || content[0].Text.Text != "What is this?" {
		t.Errorf("Content[0] = %+v, want text 'What is this?'", content[0])
	}
	if content[1].Type != ContentTypeImage {
		t.Fatalf("Content[1].Type = %q, want %q", content[1].Type, ContentTypeImage)
	}
	img := content[1].Image
	if img == nil {
		t.Fatal("Image is nil")
	}
	if img.MediaType != "image/png" {
		t.Errorf("Image.MediaType = %q, want %q", img.MediaType, "image/png")
	}
	if len(img.Data) != len(imgData) {
		t.Errorf("Image.Data len = %d, want %d", len(img.Data), len(imgData))
	}
	if img.Detail != "high" {
		t.Errorf("Image.Detail = %q, want %q", img.Detail, "high")
	}
}

func TestDecodeOpenAIChatRequest_ImageURL(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
			]
		}]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	img := req.Messages[0].Content[0].Image
	if img.URL != "https://example.com/img.png" {
		t.Errorf("Image.URL = %q, want %q", img.URL, "https://example.com/img.png")
	}
	if len(img.Data) != 0 {
		t.Errorf("Image.Data should be empty for URL images, got %d bytes", len(img.Data))
	}
}

func TestDecodeOpenAIChatRequest_ToolCalls(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "What's the weather in Tokyo?"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_abc123",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"city\":\"Tokyo\"}"
					}
				}]
			},
			{
				"role": "tool",
				"tool_call_id": "call_abc123",
				"content": "Sunny, 25°C"
			}
		]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(req.Messages))
	}

	// Assistant message with tool calls
	assistant := req.Messages[1]
	if assistant.Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", assistant.Role, RoleAssistant)
	}
	if len(assistant.Content) != 1 {
		t.Fatalf("Assistant content len = %d, want 1", len(assistant.Content))
	}
	tu := assistant.Content[0]
	if tu.Type != ContentTypeToolUse {
		t.Fatalf("Content[0].Type = %q, want %q", tu.Type, ContentTypeToolUse)
	}
	if tu.ToolUse.ID != "call_abc123" {
		t.Errorf("ToolUse.ID = %q, want %q", tu.ToolUse.ID, "call_abc123")
	}
	if tu.ToolUse.Name != "get_weather" {
		t.Errorf("ToolUse.Name = %q, want %q", tu.ToolUse.Name, "get_weather")
	}

	// Tool result message
	toolMsg := req.Messages[2]
	if toolMsg.Role != RoleTool {
		t.Errorf("Messages[2].Role = %q, want %q", toolMsg.Role, RoleTool)
	}
	if len(toolMsg.Content) != 1 {
		t.Fatalf("Tool content len = %d, want 1", len(toolMsg.Content))
	}
	tr := toolMsg.Content[0]
	if tr.Type != ContentTypeToolResult {
		t.Fatalf("Content[0].Type = %q, want %q", tr.Type, ContentTypeToolResult)
	}
	if tr.ToolResult.ToolUseID != "call_abc123" {
		t.Errorf("ToolResult.ToolUseID = %q, want %q", tr.ToolResult.ToolUseID, "call_abc123")
	}
	if len(tr.ToolResult.Content) != 1 || tr.ToolResult.Content[0].Text.Text != "Sunny, 25°C" {
		t.Errorf("ToolResult.Content = %+v, want text 'Sunny, 25°C'", tr.ToolResult.Content)
	}
}

func TestDecodeOpenAIChatRequest_MaxCompletionTokens(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"max_tokens": 500,
		"max_completion_tokens": 1000
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.MaxTokens != 1000 {
		t.Errorf("MaxTokens = %d, want 1000 (max_completion_tokens should take precedence)", req.MaxTokens)
	}
}

func TestDecodeOpenAIChatRequest_MaxTokensOnly(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"max_tokens": 500
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.MaxTokens != 500 {
		t.Errorf("MaxTokens = %d, want 500", req.MaxTokens)
	}
}

func TestDecodeOpenAIChatRequest_StringContent(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello world"}]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	if len(req.Messages[0].Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(req.Messages[0].Content))
	}
	if req.Messages[0].Content[0].Text.Text != "Hello world" {
		t.Errorf("Content[0].Text = %q, want %q", req.Messages[0].Content[0].Text.Text, "Hello world")
	}
}

func TestDecodeOpenAIChatRequest_StopString(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"stop": "END"
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.StopSequences) != 1 || req.StopSequences[0] != "END" {
		t.Errorf("StopSequences = %v, want [END]", req.StopSequences)
	}
}

func TestDecodeOpenAIChatRequest_StopArray(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"stop": ["END", "STOP"]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.StopSequences) != 2 {
		t.Fatalf("StopSequences len = %d, want 2", len(req.StopSequences))
	}
	if req.StopSequences[0] != "END" || req.StopSequences[1] != "STOP" {
		t.Errorf("StopSequences = %v, want [END STOP]", req.StopSequences)
	}
}

func TestDecodeOpenAIChatRequest_ResponseFormat(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"response_format": {
			"type": "json_schema",
			"json_schema": {
				"name": "response",
				"schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
			}
		}
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ResponseFormat == nil {
		t.Fatal("ResponseFormat is nil")
	}
	if req.ResponseFormat.Type != "json_schema" {
		t.Errorf("ResponseFormat.Type = %q, want %q", req.ResponseFormat.Type, "json_schema")
	}
	if req.ResponseFormat.JSONSchema == nil {
		t.Fatal("ResponseFormat.JSONSchema is nil")
	}

	// Verify the schema content
	var schema map[string]interface{}
	if err := json.Unmarshal(req.ResponseFormat.JSONSchema, &schema); err != nil {
		t.Fatalf("Failed to unmarshal JSONSchema: %v", err)
	}
	if schema["type"] != "object" {
		t.Errorf("schema.type = %v, want 'object'", schema["type"])
	}
}

func TestDecodeOpenAIChatRequest_ResponseFormatText(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"response_format": {"type": "text"}
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ResponseFormat == nil {
		t.Fatal("ResponseFormat is nil")
	}
	if req.ResponseFormat.Type != "text" {
		t.Errorf("ResponseFormat.Type = %q, want %q", req.ResponseFormat.Type, "text")
	}
}

func TestDecodeOpenAIChatRequest_ResponseFormatJSONObject(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"response_format": {"type": "json_object"}
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ResponseFormat == nil {
		t.Fatal("ResponseFormat is nil")
	}
	if req.ResponseFormat.Type != "json_object" {
		t.Errorf("ResponseFormat.Type = %q, want %q", req.ResponseFormat.Type, "json_object")
	}
}

func TestDecodeOpenAIChatRequest_ReasoningEffort(t *testing.T) {
	body := []byte(`{
		"model": "o1",
		"messages": [{"role": "user", "content": "Think hard about this"}],
		"reasoning_effort": "high"
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Thinking == nil {
		t.Fatal("Thinking is nil")
	}
	if req.Thinking.Mode != "enabled" {
		t.Errorf("Thinking.Mode = %q, want %q", req.Thinking.Mode, "enabled")
	}
	if req.Thinking.Effort != "high" {
		t.Errorf("Thinking.Effort = %q, want %q", req.Thinking.Effort, "high")
	}
}

func TestEncodeOpenAIChatRequest_Basic(t *testing.T) {
	temp := 0.5
	topP := 0.9
	req := &Request{
		Model: "gpt-4o",
		SystemPrompt: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Be helpful."}},
		},
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []ContentPart{
					{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}},
				},
			},
		},
		MaxTokens:   1024,
		Temperature: &temp,
		TopP:        &topP,
		Stream:      true,
		StopSequences: []string{"END"},
	}

	data, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if raw["model"] != "gpt-4o" {
		t.Errorf("model = %v, want gpt-4o", raw["model"])
	}
	if raw["stream"] != true {
		t.Errorf("stream = %v, want true", raw["stream"])
	}

	// max_completion_tokens should be used (not max_tokens)
	if raw["max_completion_tokens"] != float64(1024) {
		t.Errorf("max_completion_tokens = %v, want 1024", raw["max_completion_tokens"])
	}
	if _, ok := raw["max_tokens"]; ok {
		t.Error("max_tokens should not be present; should use max_completion_tokens")
	}

	// Messages: first should be developer role (system prompt)
	msgs := raw["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("messages len = %d, want 2", len(msgs))
	}
	devMsg := msgs[0].(map[string]interface{})
	if devMsg["role"] != "developer" {
		t.Errorf("messages[0].role = %v, want developer", devMsg["role"])
	}

	userMsg := msgs[1].(map[string]interface{})
	if userMsg["role"] != "user" {
		t.Errorf("messages[1].role = %v, want user", userMsg["role"])
	}
}

func TestEncodeOpenAIChatRequest_ToolChoice(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
		ToolChoice: &ToolChoice{Type: "tool", ToolName: "read_file"},
	}

	data, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	var tc map[string]interface{}
	if err := json.Unmarshal(raw["tool_choice"], &tc); err != nil {
		t.Fatalf("failed to unmarshal tool_choice: %v", err)
	}
	if tc["type"] != "function" {
		t.Errorf("tool_choice.type = %v, want function", tc["type"])
	}
	fn := tc["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Errorf("tool_choice.function.name = %v, want read_file", fn["name"])
	}
}

func TestEncodeOpenAIChatRequest_ThinkingConfig(t *testing.T) {
	req := &Request{
		Model: "o1",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
		Thinking: &ThinkingConfig{Mode: "enabled", Effort: "high"},
	}

	data, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if raw["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want high", raw["reasoning_effort"])
	}
}

func TestEncodeOpenAIChatRequest_ThinkingConfigDefaultEffort(t *testing.T) {
	req := &Request{
		Model: "o1",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
		Thinking: &ThinkingConfig{Mode: "enabled"},
	}

	data, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if raw["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort = %v, want medium (default when enabled without effort)", raw["reasoning_effort"])
	}
}

func TestDecodeOpenAIChatResponse_Basic(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Hello!"},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID != "chatcmpl-123" {
		t.Errorf("ID = %q, want %q", resp.ID, "chatcmpl-123")
	}
	if resp.Model != "gpt-4o" {
		t.Errorf("Model = %q, want %q", resp.Model, "gpt-4o")
	}
	if resp.StopReason != StopReasonEndTurn {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonEndTurn)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if resp.Content[0].Text.Text != "Hello!" {
		t.Errorf("Content[0].Text = %q, want %q", resp.Content[0].Text.Text, "Hello!")
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("Usage.InputTokens = %d, want 10", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("Usage.OutputTokens = %d, want 5", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("Usage.TotalTokens = %d, want 15", resp.Usage.TotalTokens)
	}
}

func TestDecodeOpenAIChatResponse_ToolCalls(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-456",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"tool_calls": [{
					"id": "call_abc",
					"type": "function",
					"function": {
						"name": "get_weather",
						"arguments": "{\"city\":\"London\"}"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.StopReason != StopReasonToolUse {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonToolUse)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	tc := resp.Content[0]
	if tc.Type != ContentTypeToolUse {
		t.Fatalf("Content[0].Type = %q, want %q", tc.Type, ContentTypeToolUse)
	}
	if tc.ToolUse.ID != "call_abc" {
		t.Errorf("ToolUse.ID = %q, want %q", tc.ToolUse.ID, "call_abc")
	}
	if tc.ToolUse.Name != "get_weather" {
		t.Errorf("ToolUse.Name = %q, want %q", tc.ToolUse.Name, "get_weather")
	}
	if string(tc.ToolUse.Arguments) != `{"city":"London"}` {
		t.Errorf("ToolUse.Arguments = %s, want {\"city\":\"London\"}", string(tc.ToolUse.Arguments))
	}
}

func TestDecodeOpenAIChatResponse_LengthFinishReason(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-789",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Truncated..."},
			"finish_reason": "length"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.StopReason != StopReasonMaxTokens {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonMaxTokens)
	}
}

func TestDecodeOpenAIChatResponse_ContentFilterFinishReason(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-cf",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": ""},
			"finish_reason": "content_filter"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.StopReason != StopReasonContentFilter {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonContentFilter)
	}
}

func TestEncodeOpenAIChatResponse_Basic(t *testing.T) {
	resp := &Response{
		ID:         "chatcmpl-123",
		Model:      "gpt-4o",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Hello!"}},
		},
		Usage: Usage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
		},
	}

	data, err := EncodeOpenAIChatResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if raw["id"] != "chatcmpl-123" {
		t.Errorf("id = %v, want chatcmpl-123", raw["id"])
	}
	if raw["object"] != "chat.completion" {
		t.Errorf("object = %v, want chat.completion", raw["object"])
	}
	if raw["model"] != "gpt-4o" {
		t.Errorf("model = %v, want gpt-4o", raw["model"])
	}

	choices := raw["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("choices len = %d, want 1", len(choices))
	}
	choice := choices[0].(map[string]interface{})
	if choice["finish_reason"] != "stop" {
		t.Errorf("finish_reason = %v, want stop", choice["finish_reason"])
	}
	msg := choice["message"].(map[string]interface{})
	if msg["role"] != "assistant" {
		t.Errorf("message.role = %v, want assistant", msg["role"])
	}
	if msg["content"] != "Hello!" {
		t.Errorf("message.content = %v, want Hello!", msg["content"])
	}

	usage := raw["usage"].(map[string]interface{})
	if usage["prompt_tokens"] != float64(10) {
		t.Errorf("usage.prompt_tokens = %v, want 10", usage["prompt_tokens"])
	}
	if usage["completion_tokens"] != float64(5) {
		t.Errorf("usage.completion_tokens = %v, want 5", usage["completion_tokens"])
	}
}

func TestEncodeOpenAIChatResponse_ToolCalls(t *testing.T) {
	resp := &Response{
		ID:         "chatcmpl-tc",
		Model:      "gpt-4o",
		StopReason: StopReasonToolUse,
		Content: []ContentPart{
			{
				Type: ContentTypeToolUse,
				ToolUse: &ToolUseContent{
					ID:        "call_abc",
					Name:      "get_weather",
					Arguments: json.RawMessage(`{"city":"Paris"}`),
				},
			},
		},
		Usage: Usage{InputTokens: 20, OutputTokens: 10, TotalTokens: 30},
	}

	data, err := EncodeOpenAIChatResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	choices := raw["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", choice["finish_reason"])
	}

	msg := choice["message"].(map[string]interface{})
	tcs := msg["tool_calls"].([]interface{})
	if len(tcs) != 1 {
		t.Fatalf("tool_calls len = %d, want 1", len(tcs))
	}
	tc := tcs[0].(map[string]interface{})
	if tc["id"] != "call_abc" {
		t.Errorf("tool_call.id = %v, want call_abc", tc["id"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_call.type = %v, want function", tc["type"])
	}
	fn := tc["function"].(map[string]interface{})
	if fn["name"] != "get_weather" {
		t.Errorf("tool_call.function.name = %v, want get_weather", fn["name"])
	}
}

func TestEncodeOpenAIChatResponse_PauseTurnDowngradesToStop(t *testing.T) {
	resp := &Response{
		ID:         "chatcmpl-pause",
		Model:      "gpt-4o",
		StopReason: StopReasonPauseTurn,
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Need more input."}},
		},
	}

	data, err := EncodeOpenAIChatResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	choices := raw["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason = %v, want stop", choice["finish_reason"])
	}
}

func TestEncodeOpenAIChatStreamChunk_PauseTurnDowngradesToStop(t *testing.T) {
	stopReason := StopReasonPauseTurn
	event := &StreamEvent{
		Type:       StreamEventStop,
		StopReason: &stopReason,
	}

	data, err := EncodeOpenAIChatStreamChunk(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	choices := raw["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason = %v, want stop", choice["finish_reason"])
	}
}

func TestDecodeOpenAIChatStreamChunk_Text(t *testing.T) {
	// First chunk: role only (start event)
	startData := []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)
	event, err := DecodeOpenAIChatStreamChunk(startData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event.Type != StreamEventStart {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventStart)
	}
	if event.Response == nil || event.Response.ID != "chatcmpl-1" {
		t.Errorf("Response = %+v, want ID chatcmpl-1", event.Response)
	}

	// Content chunk
	contentData := []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`)
	event, err = DecodeOpenAIChatStreamChunk(contentData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventDelta)
	}
	if event.Delta == nil || event.Delta.Text == nil || event.Delta.Text.Text != "Hello" {
		t.Errorf("Delta = %+v, want text 'Hello'", event.Delta)
	}

	// Finish chunk
	finishData := []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	event, err = DecodeOpenAIChatStreamChunk(finishData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event.Type != StreamEventStop {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventStop)
	}
	if event.StopReason == nil || *event.StopReason != StopReasonEndTurn {
		t.Errorf("StopReason = %v, want %q", event.StopReason, StopReasonEndTurn)
	}
}

func TestDecodeOpenAIChatStreamChunk_ToolCalls(t *testing.T) {
	// Tool call start: id + name
	tcStartData := []byte(`{"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`)
	event, err := DecodeOpenAIChatStreamChunk(tcStartData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventDelta)
	}
	if event.Delta == nil || event.Delta.ToolUse == nil {
		t.Fatal("Delta.ToolUse is nil")
	}
	if event.Delta.ToolUse.ID != "call_xyz" {
		t.Errorf("ToolUse.ID = %q, want %q", event.Delta.ToolUse.ID, "call_xyz")
	}
	if event.Delta.ToolUse.Name != "get_weather" {
		t.Errorf("ToolUse.Name = %q, want %q", event.Delta.ToolUse.Name, "get_weather")
	}

	// Tool call argument chunk
	tcArgData := []byte(`{"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}`)
	event, err = DecodeOpenAIChatStreamChunk(tcArgData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventDelta)
	}
	if string(event.Delta.ToolUse.Arguments) != `{"city":` {
		t.Errorf("ToolUse.Arguments = %s, want {\"city\":", string(event.Delta.ToolUse.Arguments))
	}

	// Finish with tool_calls
	finishData := []byte(`{"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
	event, err = DecodeOpenAIChatStreamChunk(finishData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event.Type != StreamEventStop {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventStop)
	}
	if event.StopReason == nil || *event.StopReason != StopReasonToolUse {
		t.Errorf("StopReason = %v, want %q", event.StopReason, StopReasonToolUse)
	}
}

func TestEncodeOpenAIChatStreamChunk_Text(t *testing.T) {
	// Encode start event
	startEvent := &StreamEvent{
		Type: StreamEventStart,
		Response: &Response{
			ID:    "chatcmpl-enc",
			Model: "gpt-4o",
		},
	}
	data, err := EncodeOpenAIChatStreamChunk(startEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if raw["id"] != "chatcmpl-enc" {
		t.Errorf("id = %v, want chatcmpl-enc", raw["id"])
	}
	if raw["object"] != "chat.completion.chunk" {
		t.Errorf("object = %v, want chat.completion.chunk", raw["object"])
	}

	// Encode text delta
	textDelta := &StreamEvent{
		Type: StreamEventDelta,
		Delta: &ContentPart{
			Type: ContentTypeText,
			Text: &TextContent{Text: "Hello"},
		},
	}
	data, err = EncodeOpenAIChatStreamChunk(textDelta)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	choices := raw["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	delta := choice["delta"].(map[string]interface{})
	if delta["content"] != "Hello" {
		t.Errorf("delta.content = %v, want Hello", delta["content"])
	}

	// Encode stop event
	stopReason := StopReasonEndTurn
	stopEvent := &StreamEvent{
		Type:       StreamEventStop,
		StopReason: &stopReason,
	}
	data, err = EncodeOpenAIChatStreamChunk(stopEvent)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	choices = raw["choices"].([]interface{})
	choice = choices[0].(map[string]interface{})
	if choice["finish_reason"] != "stop" {
		t.Errorf("finish_reason = %v, want stop", choice["finish_reason"])
	}
}

func TestEncodeOpenAIChatStreamChunk_ToolCalls(t *testing.T) {
	event := &StreamEvent{
		Type: StreamEventDelta,
		Delta: &ContentPart{
			Type: ContentTypeToolUse,
			ToolUse: &ToolUseContent{
				ID:        "call_xyz",
				Name:      "get_weather",
				Arguments: json.RawMessage(`{"city":"Tokyo"}`),
			},
		},
	}

	data, err := EncodeOpenAIChatStreamChunk(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	choices := raw["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	delta := choice["delta"].(map[string]interface{})
	tcs := delta["tool_calls"].([]interface{})
	if len(tcs) != 1 {
		t.Fatalf("tool_calls len = %d, want 1", len(tcs))
	}
	tc := tcs[0].(map[string]interface{})
	if tc["id"] != "call_xyz" {
		t.Errorf("tool_call.id = %v, want call_xyz", tc["id"])
	}
	fn := tc["function"].(map[string]interface{})
	if fn["name"] != "get_weather" {
		t.Errorf("tool_call.function.name = %v, want get_weather", fn["name"])
	}
}

func TestOpenAIChatRequestRoundTrip(t *testing.T) {
	temp := 0.7
	topP := 0.9
	original := &Request{
		Model: "gpt-4o",
		SystemPrompt: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Be helpful."}},
		},
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []ContentPart{
					{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}},
				},
			},
			{
				Role: RoleAssistant,
				Content: []ContentPart{
					{Type: ContentTypeText, Text: &TextContent{Text: "Hi!"}},
				},
			},
		},
		Tools: []Tool{
			{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters:  json.RawMessage(`{"type":"object"}`),
				Strict:      true,
			},
		},
		ToolChoice:    &ToolChoice{Type: "auto"},
		MaxTokens:     1024,
		Temperature:   &temp,
		TopP:          &topP,
		StopSequences: []string{"END", "STOP"},
		Stream:        true,
	}

	// Encode
	data, err := EncodeOpenAIChatRequest(original)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	// Decode back
	decoded, err := DecodeOpenAIChatRequest(data)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	// Verify key fields
	if decoded.Model != original.Model {
		t.Errorf("Model = %q, want %q", decoded.Model, original.Model)
	}
	if decoded.MaxTokens != original.MaxTokens {
		t.Errorf("MaxTokens = %d, want %d", decoded.MaxTokens, original.MaxTokens)
	}
	if decoded.Temperature == nil || *decoded.Temperature != *original.Temperature {
		t.Errorf("Temperature = %v, want %v", decoded.Temperature, *original.Temperature)
	}
	if decoded.TopP == nil || *decoded.TopP != *original.TopP {
		t.Errorf("TopP = %v, want %v", decoded.TopP, *original.TopP)
	}
	if decoded.Stream != original.Stream {
		t.Errorf("Stream = %v, want %v", decoded.Stream, original.Stream)
	}
	if len(decoded.StopSequences) != len(original.StopSequences) {
		t.Errorf("StopSequences len = %d, want %d", len(decoded.StopSequences), len(original.StopSequences))
	}

	// System prompt preserved
	if len(decoded.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(decoded.SystemPrompt))
	}
	if decoded.SystemPrompt[0].Text.Text != "Be helpful." {
		t.Errorf("SystemPrompt text = %q, want %q", decoded.SystemPrompt[0].Text.Text, "Be helpful.")
	}

	// Messages preserved (developer role in encoded form becomes system prompt again)
	if len(decoded.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(decoded.Messages))
	}
	if decoded.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", decoded.Messages[0].Role, RoleUser)
	}
	if decoded.Messages[1].Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", decoded.Messages[1].Role, RoleAssistant)
	}

	// Tools preserved
	if len(decoded.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(decoded.Tools))
	}
	if decoded.Tools[0].Name != "get_weather" {
		t.Errorf("Tool.Name = %q, want %q", decoded.Tools[0].Name, "get_weather")
	}
	if !decoded.Tools[0].Strict {
		t.Error("Tool.Strict = false, want true")
	}

	// Tool choice preserved
	if decoded.ToolChoice == nil || decoded.ToolChoice.Type != "auto" {
		t.Errorf("ToolChoice = %+v, want type 'auto'", decoded.ToolChoice)
	}
}

func TestOpenAIChatResponseRoundTrip(t *testing.T) {
	original := &Response{
		ID:         "chatcmpl-rt",
		Model:      "gpt-4o",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Hello world!"}},
		},
		Usage: Usage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
		},
	}

	// Encode
	data, err := EncodeOpenAIChatResponse(original)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	// Decode back
	decoded, err := DecodeOpenAIChatResponse(data)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	// Verify
	if decoded.ID != original.ID {
		t.Errorf("ID = %q, want %q", decoded.ID, original.ID)
	}
	if decoded.Model != original.Model {
		t.Errorf("Model = %q, want %q", decoded.Model, original.Model)
	}
	if decoded.StopReason != original.StopReason {
		t.Errorf("StopReason = %q, want %q", decoded.StopReason, original.StopReason)
	}
	if len(decoded.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(decoded.Content))
	}
	if decoded.Content[0].Text.Text != "Hello world!" {
		t.Errorf("Content[0].Text = %q, want %q", decoded.Content[0].Text.Text, "Hello world!")
	}
	if decoded.Usage.InputTokens != original.Usage.InputTokens {
		t.Errorf("Usage.InputTokens = %d, want %d", decoded.Usage.InputTokens, original.Usage.InputTokens)
	}
	if decoded.Usage.OutputTokens != original.Usage.OutputTokens {
		t.Errorf("Usage.OutputTokens = %d, want %d", decoded.Usage.OutputTokens, original.Usage.OutputTokens)
	}
	if decoded.Usage.TotalTokens != original.Usage.TotalTokens {
		t.Errorf("Usage.TotalTokens = %d, want %d", decoded.Usage.TotalTokens, original.Usage.TotalTokens)
	}
}

func TestOpenAIChatResponseRoundTrip_ToolCalls(t *testing.T) {
	original := &Response{
		ID:         "chatcmpl-tc-rt",
		Model:      "gpt-4o",
		StopReason: StopReasonToolUse,
		Content: []ContentPart{
			{
				Type: ContentTypeToolUse,
				ToolUse: &ToolUseContent{
					ID:        "call_rt",
					Name:      "search",
					Arguments: json.RawMessage(`{"query":"test"}`),
				},
			},
		},
		Usage: Usage{InputTokens: 5, OutputTokens: 3, TotalTokens: 8},
	}

	data, err := EncodeOpenAIChatResponse(original)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	decoded, err := DecodeOpenAIChatResponse(data)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if decoded.StopReason != StopReasonToolUse {
		t.Errorf("StopReason = %q, want %q", decoded.StopReason, StopReasonToolUse)
	}
	if len(decoded.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(decoded.Content))
	}
	tc := decoded.Content[0]
	if tc.Type != ContentTypeToolUse {
		t.Errorf("Content[0].Type = %q, want %q", tc.Type, ContentTypeToolUse)
	}
	if tc.ToolUse.ID != "call_rt" {
		t.Errorf("ToolUse.ID = %q, want %q", tc.ToolUse.ID, "call_rt")
	}
	if tc.ToolUse.Name != "search" {
		t.Errorf("ToolUse.Name = %q, want %q", tc.ToolUse.Name, "search")
	}
}

func TestDecodeOpenAIChatStreamChunk_UsageOnly(t *testing.T) {
	data := []byte(`{"id":"chatcmpl-u","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
	event, err := DecodeOpenAIChatStreamChunk(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventDelta)
	}
	if event.Usage == nil {
		t.Fatal("Usage is nil")
	}
	if event.Usage.InputTokens != 10 {
		t.Errorf("Usage.InputTokens = %d, want 10", event.Usage.InputTokens)
	}
	if event.Usage.OutputTokens != 5 {
		t.Errorf("Usage.OutputTokens = %d, want 5", event.Usage.OutputTokens)
	}
}

func TestEncodeOpenAIChatStreamChunk_StopWithUsage(t *testing.T) {
	stopReason := StopReasonEndTurn
	event := &StreamEvent{
		Type:       StreamEventStop,
		StopReason: &stopReason,
		Usage: &Usage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
		},
	}

	data, err := EncodeOpenAIChatStreamChunk(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	choices := raw["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	if choice["finish_reason"] != "stop" {
		t.Errorf("finish_reason = %v, want stop", choice["finish_reason"])
	}

	usage := raw["usage"].(map[string]interface{})
	if usage["prompt_tokens"] != float64(10) {
		t.Errorf("usage.prompt_tokens = %v, want 10", usage["prompt_tokens"])
	}
}

func TestEncodeOpenAIChatStreamChunk_ToolCallIndex(t *testing.T) {
	// When encoding a tool_call delta with index=2, the encoded chunk must carry index=2.
	event := &StreamEvent{
		Type:  StreamEventDelta,
		Index: 2,
		Delta: &ContentPart{
			Type: ContentTypeToolUse,
			ToolUse: &ToolUseContent{
				ID:        "call_abc",
				Name:      "my_tool",
				Arguments: json.RawMessage(`{"x":1}`),
			},
		},
	}

	data, err := EncodeOpenAIChatStreamChunk(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chunk struct {
		Choices []struct {
			Delta struct {
				ToolCalls []struct {
					Index int    `json:"index"`
					ID    string `json:"id"`
				} `json:"tool_calls"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &chunk); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(chunk.Choices) == 0 || len(chunk.Choices[0].Delta.ToolCalls) == 0 {
		t.Fatal("no tool_calls in encoded chunk")
	}
	got := chunk.Choices[0].Delta.ToolCalls[0].Index
	if got != 2 {
		t.Errorf("tool_call index = %d, want 2 (index was lost)", got)
	}
}

func TestDecodeOpenAIChatRequest_MultipleSystemMessages(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "developer", "content": "Be concise."},
			{"role": "user", "content": "Hello"}
		]
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Both system and developer messages should be accumulated
	if len(req.SystemPrompt) != 2 {
		t.Fatalf("SystemPrompt len = %d, want 2", len(req.SystemPrompt))
	}
	if req.SystemPrompt[0].Text.Text != "You are helpful." {
		t.Errorf("SystemPrompt[0] = %q, want %q", req.SystemPrompt[0].Text.Text, "You are helpful.")
	}
	if req.SystemPrompt[1].Text.Text != "Be concise." {
		t.Errorf("SystemPrompt[1] = %q, want %q", req.SystemPrompt[1].Text.Text, "Be concise.")
	}
}

func TestDecodeOpenAIChatRequest_AssistantArrayContent(t *testing.T) {
	// assistant message with content as array (multimodal/multi-part format)
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": [{"type": "text", "text": "I am the assistant response"}]}
		],
		"stream": false
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
	}

	assistant := req.Messages[1]
	if assistant.Role != RoleAssistant {
		t.Errorf("role = %q, want assistant", assistant.Role)
	}
	if len(assistant.Content) != 1 {
		t.Fatalf("assistant content parts = %d, want 1 (array content was discarded)", len(assistant.Content))
	}
	if assistant.Content[0].Type != ContentTypeText {
		t.Errorf("content type = %q, want text", assistant.Content[0].Type)
	}
	if assistant.Content[0].Text == nil || assistant.Content[0].Text.Text != "I am the assistant response" {
		t.Errorf("content text = %v, want 'I am the assistant response'", assistant.Content[0].Text)
	}
}

func TestEncodeOpenAIChatRequest_ImageRoundTrip(t *testing.T) {
	imgData := []byte{0x89, 0x50, 0x4E, 0x47}
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []ContentPart{
					{Type: ContentTypeText, Text: &TextContent{Text: "What is this?"}},
					{
						Type: ContentTypeImage,
						Image: &ImageContent{
							Data:      imgData,
							MediaType: "image/png",
							Detail:    "high",
						},
					},
				},
			},
		},
	}

	data, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	decoded, err := DecodeOpenAIChatRequest(data)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if len(decoded.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(decoded.Messages))
	}
	content := decoded.Messages[0].Content
	if len(content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(content))
	}
	img := content[1].Image
	if img == nil {
		t.Fatal("Image is nil")
	}
	if img.MediaType != "image/png" {
		t.Errorf("Image.MediaType = %q, want %q", img.MediaType, "image/png")
	}
	if len(img.Data) != len(imgData) {
		t.Errorf("Image.Data len = %d, want %d", len(img.Data), len(imgData))
	}
	if img.Detail != "high" {
		t.Errorf("Image.Detail = %q, want %q", img.Detail, "high")
	}
}

// TestEncodeOpenAIChatRequest_MixedUserToolResult tests that a RoleUser message
// containing both ContentTypeToolResult and ContentTypeText parts correctly
// splits into separate "tool" messages + a "user" message.
func TestEncodeOpenAIChatRequest_MixedUserToolResult(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "Use the tool"}},
			}},
			{
				Role: RoleAssistant,
				Content: []ContentPart{
					{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{
						ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Tokyo"}`),
					}},
				},
			},
			// Mixed user message: tool_result + text
			{
				Role: RoleUser,
				Content: []ContentPart{
					{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{
						ToolUseID: "call_1",
						Content:   []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Sunny, 25C"}}},
					}},
					{Type: ContentTypeText, Text: &TextContent{Text: "Please summarize"}},
				},
			},
		},
	}

	data, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw struct {
		Messages []json.RawMessage `json:"messages"`
	}
	json.Unmarshal(data, &raw)

	// Expect 4 messages: user, assistant(tool_calls), tool, user
	if len(raw.Messages) != 4 {
		t.Fatalf("messages len = %d, want 4; got: %s", len(raw.Messages), data)
	}

	// Check msg[2] is a "tool" message with correct tool_call_id
	var toolMsg map[string]interface{}
	json.Unmarshal(raw.Messages[2], &toolMsg)
	if toolMsg["role"] != "tool" {
		t.Errorf("msg[2].role = %v, want 'tool'", toolMsg["role"])
	}
	if toolMsg["tool_call_id"] != "call_1" {
		t.Errorf("msg[2].tool_call_id = %v, want 'call_1'", toolMsg["tool_call_id"])
	}

	// Check msg[3] is a "user" message
	var userMsg map[string]interface{}
	json.Unmarshal(raw.Messages[3], &userMsg)
	if userMsg["role"] != "user" {
		t.Errorf("msg[3].role = %v, want 'user'", userMsg["role"])
	}
}

// TestAnthropicToOpenAIChat_MixedToolResult tests the full conversion path:
// Anthropic request with mixed user message -> IR -> OpenAI Chat encoded request.
func TestAnthropicToOpenAIChat_MixedToolResult(t *testing.T) {
	body := []byte(`{
		"model": "claude-3-5-sonnet-20241022",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "Call the tool"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "toolu_xyz", "name": "read_file", "input": {"path": "/tmp/test"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_xyz", "content": "file content here"},
				{"type": "text", "text": "Now explain this file"}
			]}
		],
		"tools": [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]
	}`)

	irReq, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("decode anthropic: %v", err)
	}

	irReq.Model = "gpt-4o"
	data, err := EncodeOpenAIChatRequest(irReq)
	if err != nil {
		t.Fatalf("encode openai chat: %v", err)
	}

	var raw struct {
		Messages []json.RawMessage `json:"messages"`
	}
	json.Unmarshal(data, &raw)

	// Must have a "tool" message with tool_call_id = "toolu_xyz"
	foundTool := false
	for i, msgJSON := range raw.Messages {
		var msg map[string]interface{}
		json.Unmarshal(msgJSON, &msg)
		if msg["role"] == "tool" && msg["tool_call_id"] == "toolu_xyz" {
			foundTool = true
			t.Logf("found tool message at index %d", i)
		}
	}
	if !foundTool {
		t.Errorf("no tool message found for toolu_xyz in messages: %s", data)
	}
}

// TestDecodeOpenAIChatResponse_ReasoningTokens verifies that reasoning_tokens
// inside completion_tokens_details is mapped to Usage.ThinkingTokens.
func TestDecodeOpenAIChatResponse_ReasoningTokens(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-reasoning",
		"model": "o1-mini",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "The answer is 42."},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 100,
			"completion_tokens": 200,
			"total_tokens": 300,
			"completion_tokens_details": {
				"reasoning_tokens": 150
			}
		}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Usage.InputTokens != 100 {
		t.Errorf("Usage.InputTokens = %d, want 100", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 200 {
		t.Errorf("Usage.OutputTokens = %d, want 200", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens != 300 {
		t.Errorf("Usage.TotalTokens = %d, want 300", resp.Usage.TotalTokens)
	}
	if resp.Usage.ThinkingTokens != 150 {
		t.Errorf("Usage.ThinkingTokens = %d, want 150 (from reasoning_tokens)", resp.Usage.ThinkingTokens)
	}
}

// TestDecodeOpenAIChatResponse_CachedAndReasoningTokens verifies that both
// cache_read_tokens (from prompt_tokens_details.cached_tokens) and
// ThinkingTokens (from completion_tokens_details.reasoning_tokens) are decoded.
func TestDecodeOpenAIChatResponse_CachedAndReasoningTokens(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-mixed",
		"model": "o1",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Done."},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 500,
			"completion_tokens": 100,
			"total_tokens": 600,
			"prompt_tokens_details": {
				"cached_tokens": 400
			},
			"completion_tokens_details": {
				"reasoning_tokens": 60
			}
		}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Usage.CacheReadTokens != 400 {
		t.Errorf("Usage.CacheReadTokens = %d, want 400", resp.Usage.CacheReadTokens)
	}
	if resp.Usage.ThinkingTokens != 60 {
		t.Errorf("Usage.ThinkingTokens = %d, want 60", resp.Usage.ThinkingTokens)
	}
}

// TestEncodeOpenAIChatResponse_ReasoningTokensRoundTrip verifies that ThinkingTokens
// is encoded back as reasoning_tokens in completion_tokens_details and survives a
// full Decode → Encode → raw JSON inspection round-trip.
func TestEncodeOpenAIChatResponse_ReasoningTokensRoundTrip(t *testing.T) {
	ir := &Response{
		ID:         "chatcmpl-rt",
		Model:      "o1-mini",
		StopReason: StopReasonEndTurn,
		Content:    []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}},
		Usage: Usage{
			InputTokens:    10,
			OutputTokens:   20,
			TotalTokens:    30,
			ThinkingTokens: 12,
		},
	}

	body, err := EncodeOpenAIChatResponse(ir)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Inspect the raw JSON to confirm the wire key is correct.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	usageRaw, ok := raw["usage"]
	if !ok {
		t.Fatal("usage missing from encoded OpenAI Chat response")
	}
	var usageMap map[string]json.RawMessage
	if err := json.Unmarshal(usageRaw, &usageMap); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}
	detailsRaw, ok := usageMap["completion_tokens_details"]
	if !ok {
		t.Fatal("completion_tokens_details missing from usage when ThinkingTokens > 0")
	}
	var details map[string]json.RawMessage
	if err := json.Unmarshal(detailsRaw, &details); err != nil {
		t.Fatalf("unmarshal completion_tokens_details: %v", err)
	}
	if _, ok := details["reasoning_tokens"]; !ok {
		t.Error("reasoning_tokens missing from completion_tokens_details")
	}

	// Also verify round-trip via Decode.
	decoded, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.Usage.ThinkingTokens != 12 {
		t.Errorf("ThinkingTokens round-trip: got %d, want 12", decoded.Usage.ThinkingTokens)
	}
}

// --- Tests for parallel_tool_calls ---

func TestDecodeOpenAIChatRequest_ParallelToolCallsTrue(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"parallel_tool_calls": true
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil, expected non-nil for parallel_tool_calls")
	}
	if req.ToolChoice.AllowParallelCalls == nil {
		t.Fatal("AllowParallelCalls is nil, expected non-nil")
	}
	if !*req.ToolChoice.AllowParallelCalls {
		t.Errorf("AllowParallelCalls = false, want true")
	}
}

func TestDecodeOpenAIChatRequest_ParallelToolCallsFalse(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"parallel_tool_calls": false
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil, expected non-nil for parallel_tool_calls")
	}
	if req.ToolChoice.AllowParallelCalls == nil {
		t.Fatal("AllowParallelCalls is nil, expected non-nil")
	}
	if *req.ToolChoice.AllowParallelCalls {
		t.Errorf("AllowParallelCalls = true, want false")
	}
}

func TestDecodeOpenAIChatRequest_ParallelToolCallsWithToolChoice(t *testing.T) {
	// parallel_tool_calls should be merged with existing tool_choice
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"tool_choice": "auto",
		"parallel_tool_calls": false
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "auto" {
		t.Errorf("ToolChoice.Type = %q, want auto", req.ToolChoice.Type)
	}
	if req.ToolChoice.AllowParallelCalls == nil {
		t.Fatal("AllowParallelCalls is nil")
	}
	if *req.ToolChoice.AllowParallelCalls {
		t.Errorf("AllowParallelCalls = true, want false")
	}
}

func TestDecodeOpenAIChatRequest_NoParallelToolCalls(t *testing.T) {
	// Absent parallel_tool_calls should leave AllowParallelCalls nil
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"tool_choice": "auto"
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.AllowParallelCalls != nil {
		t.Errorf("AllowParallelCalls = %v, want nil", req.ToolChoice.AllowParallelCalls)
	}
}

func TestEncodeOpenAIChatRequest_ParallelToolCallsTrue(t *testing.T) {
	allow := true
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type:               "auto",
			AllowParallelCalls: &allow,
		},
	}

	body, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	ptcRaw, ok := m["parallel_tool_calls"]
	if !ok {
		t.Fatal("parallel_tool_calls missing from encoded request")
	}
	var ptc bool
	if err := json.Unmarshal(ptcRaw, &ptc); err != nil {
		t.Fatalf("unmarshal parallel_tool_calls: %v", err)
	}
	if !ptc {
		t.Errorf("parallel_tool_calls = false, want true")
	}
}

func TestEncodeOpenAIChatRequest_ParallelToolCallsFalse(t *testing.T) {
	allow := false
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type:               "auto",
			AllowParallelCalls: &allow,
		},
	}

	body, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	ptcRaw, ok := m["parallel_tool_calls"]
	if !ok {
		t.Fatal("parallel_tool_calls missing from encoded request")
	}
	var ptc bool
	if err := json.Unmarshal(ptcRaw, &ptc); err != nil {
		t.Fatalf("unmarshal parallel_tool_calls: %v", err)
	}
	if ptc {
		t.Errorf("parallel_tool_calls = true, want false")
	}
}

func TestEncodeOpenAIChatRequest_NoParallelToolCalls(t *testing.T) {
	// When AllowParallelCalls is nil, parallel_tool_calls should not appear
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type: "auto",
		},
	}

	body, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if _, ok := m["parallel_tool_calls"]; ok {
		t.Error("parallel_tool_calls present in encoded request, want absent")
	}
}

func TestDecodeEncodeOpenAIChatRequest_ParallelToolCallsRoundTrip(t *testing.T) {
	// Verify that parallel_tool_calls round-trips faithfully through Decode/Encode
	original := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"tool_choice":"auto","parallel_tool_calls":false}`)

	req, err := DecodeOpenAIChatRequest(original)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	encoded, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	ptcRaw, ok := m["parallel_tool_calls"]
	if !ok {
		t.Fatal("parallel_tool_calls missing after round-trip")
	}
	var ptc bool
	if err := json.Unmarshal(ptcRaw, &ptc); err != nil {
		t.Fatalf("unmarshal parallel_tool_calls: %v", err)
	}
	if ptc {
		t.Errorf("parallel_tool_calls = true after round-trip, want false")
	}
}

// TestEncodeOpenAIChatRequest_ParallelToolCallsNoToolChoice verifies that when
// parallel_tool_calls is set but tool_choice is absent (Type == ""), the encoder
// does not emit a tool_choice field (which would be invalid) but does preserve
// parallel_tool_calls as a top-level field.
func TestEncodeOpenAIChatRequest_ParallelToolCallsNoToolChoice(t *testing.T) {
	trueVal := true
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}},
		},
		// ToolChoice with empty Type but AllowParallelCalls set — as produced by
		// DecodeOpenAIChatRequest when only parallel_tool_calls is present.
		ToolChoice: &ToolChoice{
			Type:               "",
			AllowParallelCalls: &trueVal,
		},
	}

	encoded, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// tool_choice must NOT be present
	if _, ok := m["tool_choice"]; ok {
		t.Errorf("tool_choice should not be emitted when ToolChoice.Type is empty, got: %s", m["tool_choice"])
	}

	// parallel_tool_calls must be present and true
	ptcRaw, ok := m["parallel_tool_calls"]
	if !ok {
		t.Fatal("parallel_tool_calls missing")
	}
	var ptc bool
	if err := json.Unmarshal(ptcRaw, &ptc); err != nil {
		t.Fatalf("unmarshal parallel_tool_calls: %v", err)
	}
	if !ptc {
		t.Errorf("parallel_tool_calls = false, want true")
	}
}

// TestDecodeEncodeOpenAIChatRequest_ParallelToolCallsNoToolChoice is a full
// round-trip test: decode an OpenAI Chat request that has parallel_tool_calls
// but no tool_choice, then encode back and verify no tool_choice is emitted.
func TestDecodeEncodeOpenAIChatRequest_ParallelToolCallsNoToolChoice(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "hi"}],
		"parallel_tool_calls": true
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// Sanity: ToolChoice should have been created with empty Type
	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil after decode")
	}
	if req.ToolChoice.Type != "" {
		t.Errorf("ToolChoice.Type = %q after decode, want empty string", req.ToolChoice.Type)
	}
	if req.ToolChoice.AllowParallelCalls == nil || !*req.ToolChoice.AllowParallelCalls {
		t.Error("ToolChoice.AllowParallelCalls not set to true after decode")
	}

	encoded, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// tool_choice must NOT be present
	if _, ok := m["tool_choice"]; ok {
		t.Errorf("tool_choice should not be emitted, got: %s", m["tool_choice"])
	}

	// parallel_tool_calls must be present and true
	ptcRaw, ok := m["parallel_tool_calls"]
	if !ok {
		t.Fatal("parallel_tool_calls missing from encoded output")
	}
	var ptc bool
	if err := json.Unmarshal(ptcRaw, &ptc); err != nil {
		t.Fatalf("unmarshal parallel_tool_calls: %v", err)
	}
	if !ptc {
		t.Errorf("parallel_tool_calls = false, want true")
	}
}

// --- Refusal tests ---

func TestDecodeOpenAIChatResponse_Refusal(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"refusal": "I cannot help with that request."
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have one refusal content part
	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeRefusal {
		t.Errorf("Content[0].Type = %q, want %q", resp.Content[0].Type, ContentTypeRefusal)
	}
	if resp.Content[0].Refusal == nil {
		t.Fatal("Content[0].Refusal is nil")
	}
	if resp.Content[0].Refusal.Refusal != "I cannot help with that request." {
		t.Errorf("Refusal = %q, want %q", resp.Content[0].Refusal.Refusal, "I cannot help with that request.")
	}
}

func TestDecodeOpenAIChatResponse_RefusalWithContent(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Some text",
				"refusal": "I cannot help with that."
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have both text and refusal content parts
	if len(resp.Content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q, want %q", resp.Content[0].Type, ContentTypeText)
	}
	if resp.Content[1].Type != ContentTypeRefusal {
		t.Errorf("Content[1].Type = %q, want %q", resp.Content[1].Type, ContentTypeRefusal)
	}
}

func TestEncodeOpenAIChatResponse_Refusal(t *testing.T) {
	resp := &Response{
		ID:         "chatcmpl-123",
		Model:      "gpt-4o",
		StopReason: StopReasonContentFilter,
		Content: []ContentPart{
			{Type: ContentTypeRefusal, Refusal: &RefusalContent{Refusal: "I cannot help with that."}},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	data, err := EncodeOpenAIChatResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	var choices []struct {
		Message struct {
			Refusal *string `json:"refusal"`
		} `json:"message"`
	}
	if err := json.Unmarshal(raw["choices"], &choices); err != nil {
		t.Fatalf("unmarshal choices: %v", err)
	}
	if len(choices) == 0 {
		t.Fatal("no choices")
	}
	if choices[0].Message.Refusal == nil {
		t.Fatal("refusal is nil")
	}
	if *choices[0].Message.Refusal != "I cannot help with that." {
		t.Errorf("refusal = %q, want %q", *choices[0].Message.Refusal, "I cannot help with that.")
	}
}

func TestDecodeOpenAIChatStreamChunk_Refusal(t *testing.T) {
	refusalText := "I cannot help"
	chunk := openaichat.ChatStreamChunk{
		ID:     "chatcmpl-123",
		Object: "chat.completion.chunk",
		Model:  "gpt-4o",
		Choices: []openaichat.ChatChoice{
			{
				Index: 0,
				Delta: &openaichat.ChatChoiceMessage{
					Refusal: &refusalText,
				},
			},
		},
	}
	data, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	event, err := DecodeOpenAIChatStreamChunk(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if event.Type != StreamEventDelta {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventDelta)
	}
	if event.Delta == nil {
		t.Fatal("Delta is nil")
	}
	if event.Delta.Type != ContentTypeRefusal {
		t.Errorf("Delta.Type = %q, want %q", event.Delta.Type, ContentTypeRefusal)
	}
	if event.Delta.Refusal == nil {
		t.Fatal("Delta.Refusal is nil")
	}
	if event.Delta.Refusal.Refusal != "I cannot help" {
		t.Errorf("Delta.Refusal.Refusal = %q, want %q", event.Delta.Refusal.Refusal, "I cannot help")
	}
}

func TestEncodeOpenAIChatStreamChunk_Refusal(t *testing.T) {
	event := &StreamEvent{
		Type: StreamEventDelta,
		Delta: &ContentPart{
			Type:    ContentTypeRefusal,
			Refusal: &RefusalContent{Refusal: "I cannot"},
		},
	}

	data, err := EncodeOpenAIChatStreamChunk(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw openaichat.ChatStreamChunk
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(raw.Choices) == 0 {
		t.Fatal("no choices")
	}
	if raw.Choices[0].Delta == nil {
		t.Fatal("delta is nil")
	}
	if raw.Choices[0].Delta.Refusal == nil {
		t.Fatal("delta.refusal is nil")
	}
	if *raw.Choices[0].Delta.Refusal != "I cannot" {
		t.Errorf("delta.refusal = %q, want %q", *raw.Choices[0].Delta.Refusal, "I cannot")
	}
}

func TestEncodeOpenAIChatStreamChunk_ErrorSkipped(t *testing.T) {
	event := &StreamEvent{
		Type: StreamEventError,
		Error: &StreamError{
			Type:    "server_error",
			Message: "overloaded",
		},
	}

	data, err := EncodeOpenAIChatStreamChunk(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Error events are silently skipped in OpenAI Chat format
	if data != nil {
		t.Errorf("expected nil data for error event, got %s", data)
	}
}

func TestDecodeOpenAIChatResponse_RefusalRoundTrip(t *testing.T) {
	// Decode -> Encode -> Decode should preserve refusal
	body := []byte(`{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"refusal": "Cannot assist."
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	encoded, err := EncodeOpenAIChatResponse(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	resp2, err := DecodeOpenAIChatResponse(encoded)
	if err != nil {
		t.Fatalf("decode2: %v", err)
	}

	if len(resp2.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp2.Content))
	}
	if resp2.Content[0].Type != ContentTypeRefusal {
		t.Errorf("Content[0].Type = %q, want %q", resp2.Content[0].Type, ContentTypeRefusal)
	}
	if resp2.Content[0].Refusal.Refusal != "Cannot assist." {
		t.Errorf("Refusal = %q, want %q", resp2.Content[0].Refusal.Refusal, "Cannot assist.")
	}
}

// TestEncodeOpenAIChatRequest_ThinkingConfig_Phase2_Degradation verifies that
// Phase 2 thinking fields (IncludeThoughts, Level) are silently dropped when
// encoding to OpenAI Chat, which has no native equivalent.
func TestEncodeOpenAIChatRequest_ThinkingConfig_Phase2_Degradation(t *testing.T) {
	inclTrue := true
	req := &Request{
		Model: "o1",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
		Thinking: &ThinkingConfig{
			Mode:            "enabled",
			Effort:          "high",
			IncludeThoughts: &inclTrue,
			Level:           "HIGH",
		},
	}

	data, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// reasoning_effort should still be set from Effort
	if raw["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want high", raw["reasoning_effort"])
	}

	// IncludeThoughts and Level have no OpenAI Chat equivalent — should be absent
	if _, ok := raw["include_thoughts"]; ok {
		t.Error("include_thoughts should not appear in OpenAI Chat request (silently dropped)")
	}
	if _, ok := raw["includeThoughts"]; ok {
		t.Error("includeThoughts should not appear in OpenAI Chat request (silently dropped)")
	}
	if _, ok := raw["level"]; ok {
		t.Error("level should not appear in OpenAI Chat request (silently dropped)")
	}
	if _, ok := raw["thinkingLevel"]; ok {
		t.Error("thinkingLevel should not appear in OpenAI Chat request (silently dropped)")
	}
}
