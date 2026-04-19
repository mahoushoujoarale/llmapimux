package llmapimux

import (
	"encoding/json"
	"testing"

	"github.com/llmapimux/llmapimux/protocol/openairesponses"
)

func TestDecodeOpenAIResponsesRequest_StringInput(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"temperature": 0.7,
		"stream": true
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Model != "gpt-4o" {
		t.Errorf("Model = %q, want %q", req.Model, "gpt-4o")
	}
	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", req.Temperature)
	}
	if !req.Stream {
		t.Error("Stream = false, want true")
	}

	// String input → single user message
	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	if req.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", req.Messages[0].Role, RoleUser)
	}
	if len(req.Messages[0].Content) != 1 {
		t.Fatalf("Messages[0].Content len = %d, want 1", len(req.Messages[0].Content))
	}
	if req.Messages[0].Content[0].Type != ContentTypeText {
		t.Errorf("Messages[0].Content[0].Type = %q, want %q", req.Messages[0].Content[0].Type, ContentTypeText)
	}
	if req.Messages[0].Content[0].Text.Text != "Hello" {
		t.Errorf("Messages[0].Content[0].Text = %q, want %q", req.Messages[0].Content[0].Text.Text, "Hello")
	}
}

func TestDecodeOpenAIResponsesRequest_MessageArray(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi there!"}]}
		]
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
	}

	// User message
	if req.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", req.Messages[0].Role, RoleUser)
	}
	if len(req.Messages[0].Content) != 1 || req.Messages[0].Content[0].Text.Text != "Hello" {
		t.Errorf("Messages[0].Content = %+v, want text 'Hello'", req.Messages[0].Content)
	}

	// Assistant message
	if req.Messages[1].Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", req.Messages[1].Role, RoleAssistant)
	}
	if len(req.Messages[1].Content) != 1 || req.Messages[1].Content[0].Text.Text != "Hi there!" {
		t.Errorf("Messages[1].Content = %+v, want text 'Hi there!'", req.Messages[1].Content)
	}
}

func TestDecodeOpenAIResponsesRequest_DeveloperRole(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be brief"}]},
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
		]
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Developer messages → SystemPrompt
	if len(req.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(req.SystemPrompt))
	}
	if req.SystemPrompt[0].Type != ContentTypeText || req.SystemPrompt[0].Text.Text != "Be brief" {
		t.Errorf("SystemPrompt[0] = %+v, want text 'Be brief'", req.SystemPrompt[0])
	}

	// Only user message in Messages
	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	if req.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", req.Messages[0].Role, RoleUser)
	}
}

func TestDecodeOpenAIResponsesRequest_Instructions(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"instructions": "Be helpful"
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Instructions → SystemPrompt (prepended)
	if len(req.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(req.SystemPrompt))
	}
	if req.SystemPrompt[0].Text.Text != "Be helpful" {
		t.Errorf("SystemPrompt[0].Text = %q, want %q", req.SystemPrompt[0].Text.Text, "Be helpful")
	}
}

func TestDecodeOpenAIResponsesRequest_InstructionsAndDeveloper(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"instructions": "Be helpful",
		"input": [
			{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Also be brief"}]},
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
		]
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Instructions prepended, then developer message
	if len(req.SystemPrompt) != 2 {
		t.Fatalf("SystemPrompt len = %d, want 2", len(req.SystemPrompt))
	}
	if req.SystemPrompt[0].Text.Text != "Be helpful" {
		t.Errorf("SystemPrompt[0].Text = %q, want %q", req.SystemPrompt[0].Text.Text, "Be helpful")
	}
	if req.SystemPrompt[1].Text.Text != "Also be brief" {
		t.Errorf("SystemPrompt[1].Text = %q, want %q", req.SystemPrompt[1].Text.Text, "Also be brief")
	}
}

func TestDecodeOpenAIResponsesRequest_FunctionCalls(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Read the file"}]},
			{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\":\"/tmp\"}"},
			{"type": "function_call_output", "call_id": "call_1", "output": "file contents here"}
		]
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(req.Messages))
	}

	// User message
	if req.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", req.Messages[0].Role, RoleUser)
	}

	// function_call → assistant message with ToolUse
	if req.Messages[1].Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", req.Messages[1].Role, RoleAssistant)
	}
	if len(req.Messages[1].Content) != 1 {
		t.Fatalf("Messages[1].Content len = %d, want 1", len(req.Messages[1].Content))
	}
	tu := req.Messages[1].Content[0]
	if tu.Type != ContentTypeToolUse {
		t.Errorf("Messages[1].Content[0].Type = %q, want %q", tu.Type, ContentTypeToolUse)
	}
	if tu.ToolUse.ID != "call_1" {
		t.Errorf("ToolUse.ID = %q, want %q", tu.ToolUse.ID, "call_1")
	}
	if tu.ToolUse.Name != "read_file" {
		t.Errorf("ToolUse.Name = %q, want %q", tu.ToolUse.Name, "read_file")
	}
	if string(tu.ToolUse.Arguments) != `{"path":"/tmp"}` {
		t.Errorf("ToolUse.Arguments = %s, want %s", tu.ToolUse.Arguments, `{"path":"/tmp"}`)
	}

	// function_call_output → tool message with ToolResult
	if req.Messages[2].Role != RoleTool {
		t.Errorf("Messages[2].Role = %q, want %q", req.Messages[2].Role, RoleTool)
	}
	if len(req.Messages[2].Content) != 1 {
		t.Fatalf("Messages[2].Content len = %d, want 1", len(req.Messages[2].Content))
	}
	tr := req.Messages[2].Content[0]
	if tr.Type != ContentTypeToolResult {
		t.Errorf("Messages[2].Content[0].Type = %q, want %q", tr.Type, ContentTypeToolResult)
	}
	if tr.ToolResult.ToolUseID != "call_1" {
		t.Errorf("ToolResult.ToolUseID = %q, want %q", tr.ToolResult.ToolUseID, "call_1")
	}
	if len(tr.ToolResult.Content) != 1 || tr.ToolResult.Content[0].Text.Text != "file contents here" {
		t.Errorf("ToolResult.Content = %+v, want text 'file contents here'", tr.ToolResult.Content)
	}
}

func TestDecodeOpenAIResponsesRequest_Tools(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"tools": [
			{"type": "function", "name": "read_file", "description": "Read a file", "parameters": {"type": "object"}, "strict": true},
			{"type": "file_search"},
			{"type": "web_search"},
			{"type": "code_interpreter"},
			{"type": "computer_use"},
			{"type": "mcp"}
		],
		"tool_choice": "auto"
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Tools) != 6 {
		t.Fatalf("Tools len = %d, want 6", len(req.Tools))
	}
	if req.Tools[0].Name != "read_file" {
		t.Errorf("Tools[0].Name = %q, want %q", req.Tools[0].Name, "read_file")
	}
	if req.Tools[0].Type != "function" {
		t.Errorf("Tools[0].Type = %q, want function", req.Tools[0].Type)
	}
	if req.Tools[0].Description != "Read a file" {
		t.Errorf("Tools[0].Description = %q, want %q", req.Tools[0].Description, "Read a file")
	}
	if !req.Tools[0].Strict {
		t.Error("Tools[0].Strict = false, want true")
	}
	wantTypes := []string{"function", "file_search", "web_search", "code_interpreter", "computer_use", "mcp"}
	for i, want := range wantTypes {
		if req.Tools[i].Type != want {
			t.Errorf("Tools[%d].Type = %q, want %q", i, req.Tools[i].Type, want)
		}
	}

	// Tool choice
	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "auto" {
		t.Errorf("ToolChoice.Type = %q, want %q", req.ToolChoice.Type, "auto")
	}
}

func TestDecodeOpenAIResponsesRequest_ToolChoiceObject(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"tools": [{"type": "function", "name": "read_file", "parameters": {}}],
		"tool_choice": {"type": "function", "name": "read_file"}
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
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

func TestDecodeOpenAIResponsesRequest_BuiltInToolChoiceObject(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"tools": [{"type": "web_search"}],
		"tool_choice": {"type": "web_search"}
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(req.Tools))
	}
	if req.Tools[0].Type != "web_search" {
		t.Errorf("Tools[0].Type = %q, want web_search", req.Tools[0].Type)
	}
	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "tool" {
		t.Errorf("ToolChoice.Type = %q, want tool", req.ToolChoice.Type)
	}
	if req.ToolChoice.ToolName != "web_search" {
		t.Errorf("ToolChoice.ToolName = %q, want web_search", req.ToolChoice.ToolName)
	}
}

func TestDecodeOpenAIResponsesRequest_WebSearchToolFilters(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"tools": [{
			"type": "web_search",
			"filters": {"allowed_domains": ["openai.com"]},
			"search_context_size": "high"
		}]
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(req.Tools))
	}
	if req.Tools[0].Type != "web_search" {
		t.Errorf("Tools[0].Type = %q, want web_search", req.Tools[0].Type)
	}
	if req.Tools[0].ExtraFields == nil {
		t.Fatal("Tools[0].ExtraFields is nil")
	}

	var allowedDomains []string
	if err := json.Unmarshal(req.Tools[0].ExtraFields["allowed_domains"], &allowedDomains); err != nil {
		t.Fatalf("unmarshal Tools[0].ExtraFields[allowed_domains]: %v", err)
	}
	if len(allowedDomains) != 1 || allowedDomains[0] != "openai.com" {
		t.Errorf("Tools[0].ExtraFields[allowed_domains] = %v, want [openai.com]", allowedDomains)
	}

	var searchContextSize string
	if err := json.Unmarshal(req.Tools[0].ExtraFields["search_context_size"], &searchContextSize); err != nil {
		t.Fatalf("unmarshal Tools[0].ExtraFields[search_context_size]: %v", err)
	}
	if searchContextSize != "high" {
		t.Errorf("Tools[0].ExtraFields[search_context_size] = %q, want high", searchContextSize)
	}
}

func TestDecodeOpenAIResponsesRequest_AllowedTools(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"tools": [
			{"type": "function", "name": "read_file", "parameters": {}},
			{"type": "function", "name": "write_file", "parameters": {}}
		],
		"tool_choice": {
			"type": "allowed_tools",
			"mode": "required",
			"tools": [
				{"type": "function", "name": "read_file"},
				{"type": "function", "name": "write_file"}
			]
		}
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "required" {
		t.Errorf("ToolChoice.Type = %q, want required", req.ToolChoice.Type)
	}
	if len(req.ToolChoice.AllowedToolNames) != 2 {
		t.Fatalf("AllowedToolNames len = %d, want 2", len(req.ToolChoice.AllowedToolNames))
	}
	if req.ToolChoice.AllowedToolNames[0] != "read_file" {
		t.Errorf("AllowedToolNames[0] = %q, want read_file", req.ToolChoice.AllowedToolNames[0])
	}
	if req.ToolChoice.AllowedToolNames[1] != "write_file" {
		t.Errorf("AllowedToolNames[1] = %q, want write_file", req.ToolChoice.AllowedToolNames[1])
	}
}

func TestDecodeOpenAIResponsesRequest_Reasoning(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"reasoning": {"effort": "medium"}
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Thinking == nil {
		t.Fatal("Thinking is nil")
	}
	if req.Thinking.Mode != "enabled" {
		t.Errorf("Thinking.Mode = %q, want %q", req.Thinking.Mode, "enabled")
	}
	if req.Thinking.Effort != "medium" {
		t.Errorf("Thinking.Effort = %q, want %q", req.Thinking.Effort, "medium")
	}
}

func TestDecodeOpenAIResponsesRequest_ResponseFormat(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"text": {
			"format": {
				"type": "json_schema",
				"name": "my_schema",
				"schema": {"type": "object", "properties": {"name": {"type": "string"}}}
			}
		}
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ResponseFormat == nil {
		t.Fatal("ResponseFormat is nil")
	}
	if req.ResponseFormat.Type != "json_schema" {
		t.Errorf("ResponseFormat.Type = %q, want %q", req.ResponseFormat.Type, "json_schema")
	}
	if len(req.ResponseFormat.JSONSchema) == 0 {
		t.Error("ResponseFormat.JSONSchema is empty")
	}

	// Verify schema content
	var schema map[string]interface{}
	if err := json.Unmarshal(req.ResponseFormat.JSONSchema, &schema); err != nil {
		t.Fatalf("unmarshal JSONSchema: %v", err)
	}
	if schema["type"] != "object" {
		t.Errorf("JSONSchema type = %v, want 'object'", schema["type"])
	}
}

func TestDecodeOpenAIResponsesRequest_MaxOutputTokens(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"max_output_tokens": 2048
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.MaxTokens != 2048 {
		t.Errorf("MaxTokens = %d, want 2048", req.MaxTokens)
	}
}

func TestDecodeOpenAIResponsesRequest_StopSequences(t *testing.T) {
	// String form
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"stop": "END"
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error (string): %v", err)
	}
	if len(req.StopSequences) != 1 || req.StopSequences[0] != "END" {
		t.Errorf("StopSequences = %v, want [END]", req.StopSequences)
	}

	// Array form
	body = []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"stop": ["END", "STOP"]
	}`)

	req, err = DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error (array): %v", err)
	}
	if len(req.StopSequences) != 2 || req.StopSequences[0] != "END" || req.StopSequences[1] != "STOP" {
		t.Errorf("StopSequences = %v, want [END STOP]", req.StopSequences)
	}
}

func TestDecodeOpenAIResponsesRequest_ImageAndFile(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": [
				{"type": "input_text", "text": "What is this?"},
				{"type": "input_image", "image_url": "https://example.com/img.png"},
				{"type": "input_file", "filename": "doc.pdf"}
			]}
		]
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}

	content := req.Messages[0].Content
	if len(content) != 3 {
		t.Fatalf("Content len = %d, want 3", len(content))
	}

	// Text
	if content[0].Type != ContentTypeText || content[0].Text.Text != "What is this?" {
		t.Errorf("Content[0] = %+v, want text 'What is this?'", content[0])
	}

	// Image
	if content[1].Type != ContentTypeImage {
		t.Errorf("Content[1].Type = %q, want %q", content[1].Type, ContentTypeImage)
	}
	if content[1].Image.URL != "https://example.com/img.png" {
		t.Errorf("Content[1].Image.URL = %q, want %q", content[1].Image.URL, "https://example.com/img.png")
	}

	// Document
	if content[2].Type != ContentTypeDocument {
		t.Errorf("Content[2].Type = %q, want %q", content[2].Type, ContentTypeDocument)
	}
	if content[2].Document.Title != "doc.pdf" {
		t.Errorf("Content[2].Document.Title = %q, want %q", content[2].Document.Title, "doc.pdf")
	}
}

func TestDecodeOpenAIResponsesRequest_MultimodalPreservesPayloads(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": [
				{"type": "input_image", "image_url": "data:image/png;base64,ZmFrZS1pbWFnZQ=="},
				{"type": "input_file", "file_data": "ZmFrZS1wZGY=", "filename": "doc.pdf"},
				{"type": "input_file", "file_url": "https://example.com/doc.pdf", "filename": "remote.pdf"}
			]}
		]
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	content := req.Messages[0].Content
	if len(content) != 3 {
		t.Fatalf("Content len = %d, want 3", len(content))
	}
	if content[0].Image == nil || content[0].Image.URL != "data:image/png;base64,ZmFrZS1pbWFnZQ==" {
		t.Fatalf("image URL = %+v, want data URL preserved", content[0].Image)
	}
	if content[1].Document == nil || string(content[1].Document.Data) != "fake-pdf" || content[1].Document.Title != "doc.pdf" {
		t.Fatalf("inline document = %+v, want decoded data and title", content[1].Document)
	}
	if content[2].Document == nil || content[2].Document.URL != "https://example.com/doc.pdf" || content[2].Document.Title != "remote.pdf" {
		t.Fatalf("remote document = %+v, want url and title", content[2].Document)
	}
}

func TestEncodeOpenAIResponsesRequest_MultimodalPreservesPayloads(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []ContentPart{
					{Type: ContentTypeImage, Image: &ImageContent{URL: "https://example.com/img.png"}},
					{Type: ContentTypeImage, Image: &ImageContent{Data: []byte("fake-image"), MediaType: "image/png"}},
					{Type: ContentTypeDocument, Document: &DocumentContent{URL: "https://example.com/doc.pdf", Title: "doc.pdf"}},
					{Type: ContentTypeDocument, Document: &DocumentContent{Data: []byte("fake-pdf"), MediaType: "application/pdf", Title: "inline.pdf"}},
				},
			},
		},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	var input []map[string]any
	if err := json.Unmarshal(raw["input"], &input); err != nil {
		t.Fatalf("unmarshal input: %v", err)
	}
	content, _ := input[0]["content"].([]any)
	if len(content) != 4 {
		t.Fatalf("content len = %d, want 4", len(content))
	}

	first := content[0].(map[string]any)
	if first["type"] != "input_image" || first["image_url"] != "https://example.com/img.png" {
		t.Fatalf("first content = %+v, want URL image", first)
	}
	second := content[1].(map[string]any)
	if second["type"] != "input_image" || second["image_url"] == "" {
		t.Fatalf("second content = %+v, want inline image encoded as data URL", second)
	}
	third := content[2].(map[string]any)
	if third["type"] != "input_file" || third["file_url"] != "https://example.com/doc.pdf" || third["filename"] != "doc.pdf" {
		t.Fatalf("third content = %+v, want URL file", third)
	}
	fourth := content[3].(map[string]any)
	if fourth["type"] != "input_file" || fourth["file_data"] != "ZmFrZS1wZGY=" || fourth["filename"] != "inline.pdf" {
		t.Fatalf("fourth content = %+v, want inline file", fourth)
	}
}

func TestDecodeOpenAIResponsesRequest_PreviousResponseIDIgnored(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"previous_response_id": "resp_abc123"
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// previous_response_id is silently ignored — no error
	if req.Model != "gpt-4o" {
		t.Errorf("Model = %q, want %q", req.Model, "gpt-4o")
	}
}

func TestEncodeOpenAIResponsesRequest_Basic(t *testing.T) {
	temp := 0.7
	topP := 0.9
	req := &Request{
		Model: "gpt-4o",
		SystemPrompt: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Be helpful"}},
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
		Tools: []Tool{
			{
				Name:        "read_file",
				Description: "Read a file",
				Parameters:  json.RawMessage(`{"type":"object"}`),
				Strict:      true,
			},
		},
		ToolChoice: &ToolChoice{Type: "auto"},
		Thinking:   &ThinkingConfig{Mode: "enabled", Effort: "high"},
		ResponseFormat: &ResponseFormat{
			Type:       "json_schema",
			JSONSchema: json.RawMessage(`{"type":"object"}`),
		},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}

	// model
	var model string
	json.Unmarshal(raw["model"], &model)
	if model != "gpt-4o" {
		t.Errorf("model = %q, want %q", model, "gpt-4o")
	}

	// instructions
	var instructions string
	json.Unmarshal(raw["instructions"], &instructions)
	if instructions != "Be helpful" {
		t.Errorf("instructions = %q, want %q", instructions, "Be helpful")
	}

	// max_output_tokens
	var maxTokens int
	json.Unmarshal(raw["max_output_tokens"], &maxTokens)
	if maxTokens != 1024 {
		t.Errorf("max_output_tokens = %d, want 1024", maxTokens)
	}

	// stream
	var stream bool
	json.Unmarshal(raw["stream"], &stream)
	if !stream {
		t.Error("stream = false, want true")
	}

	// tools
	var tools []map[string]interface{}
	json.Unmarshal(raw["tools"], &tools)
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(tools))
	}
	if tools[0]["type"] != "function" {
		t.Errorf("tools[0].type = %v, want 'function'", tools[0]["type"])
	}
	if tools[0]["name"] != "read_file" {
		t.Errorf("tools[0].name = %v, want 'read_file'", tools[0]["name"])
	}

	// tool_choice
	var toolChoice string
	json.Unmarshal(raw["tool_choice"], &toolChoice)
	if toolChoice != "auto" {
		t.Errorf("tool_choice = %q, want %q", toolChoice, "auto")
	}

	// reasoning
	var reasoning map[string]string
	json.Unmarshal(raw["reasoning"], &reasoning)
	if reasoning["effort"] != "high" {
		t.Errorf("reasoning.effort = %q, want %q", reasoning["effort"], "high")
	}

	// text.format
	var text map[string]json.RawMessage
	json.Unmarshal(raw["text"], &text)
	var format map[string]json.RawMessage
	json.Unmarshal(text["format"], &format)
	var fmtType string
	json.Unmarshal(format["type"], &fmtType)
	if fmtType != "json_schema" {
		t.Errorf("text.format.type = %q, want %q", fmtType, "json_schema")
	}
}

func TestEncodeOpenAIResponsesRequest_ToolChoiceObject(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{Type: "tool", ToolName: "read_file"},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	var tc map[string]string
	json.Unmarshal(raw["tool_choice"], &tc)
	if tc["type"] != "function" {
		t.Errorf("tool_choice.type = %q, want %q", tc["type"], "function")
	}
	if tc["name"] != "read_file" {
		t.Errorf("tool_choice.name = %q, want %q", tc["name"], "read_file")
	}
}

func TestEncodeOpenAIResponsesRequest_BuiltInToolChoiceObject(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		Tools: []Tool{
			{Type: "web_search", Name: "web_search"},
		},
		ToolChoice: &ToolChoice{Type: "tool", ToolName: "web_search"},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	var tc map[string]string
	json.Unmarshal(raw["tool_choice"], &tc)
	if tc["type"] != "web_search" {
		t.Errorf("tool_choice.type = %q, want web_search", tc["type"])
	}
}

func TestEncodeOpenAIResponsesRequest_WebSearchToolFilters(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		Tools: []Tool{
			{
				Type: "web_search",
				Name: "web_search",
				ExtraFields: map[string]json.RawMessage{
					"allowed_domains":     json.RawMessage(`["openai.com"]`),
					"blocked_domains":     json.RawMessage(`["example.com"]`),
					"max_uses":            json.RawMessage(`3`),
					"search_context_size": json.RawMessage(`"high"`),
				},
			},
		},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}

	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(raw["tools"], &tools); err != nil {
		t.Fatalf("unmarshal tools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(tools))
	}

	if _, ok := tools[0]["allowed_domains"]; ok {
		t.Fatal("tools[0].allowed_domains present, want nested under filters")
	}
	if _, ok := tools[0]["blocked_domains"]; ok {
		t.Fatal("tools[0].blocked_domains present, want dropped")
	}
	if _, ok := tools[0]["max_uses"]; ok {
		t.Fatal("tools[0].max_uses present, want dropped")
	}

	var filters map[string]json.RawMessage
	if err := json.Unmarshal(tools[0]["filters"], &filters); err != nil {
		t.Fatalf("unmarshal tools[0].filters: %v", err)
	}

	var allowedDomains []string
	if err := json.Unmarshal(filters["allowed_domains"], &allowedDomains); err != nil {
		t.Fatalf("unmarshal tools[0].filters.allowed_domains: %v", err)
	}
	if len(allowedDomains) != 1 || allowedDomains[0] != "openai.com" {
		t.Errorf("tools[0].filters.allowed_domains = %v, want [openai.com]", allowedDomains)
	}

	var searchContextSize string
	if err := json.Unmarshal(tools[0]["search_context_size"], &searchContextSize); err != nil {
		t.Fatalf("unmarshal tools[0].search_context_size: %v", err)
	}
	if searchContextSize != "high" {
		t.Errorf("tools[0].search_context_size = %q, want high", searchContextSize)
	}
}

func TestEncodeOpenAIResponsesRequest_AllowedTools(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		Tools: []Tool{
			{Name: "read_file", Parameters: json.RawMessage(`{"type":"object"}`)},
			{Name: "write_file", Parameters: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: &ToolChoice{
			Type:             "required",
			AllowedToolNames: []string{"read_file", "write_file"},
		},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	var tc map[string]json.RawMessage
	if err := json.Unmarshal(raw["tool_choice"], &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}

	var tcType string
	if err := json.Unmarshal(tc["type"], &tcType); err != nil {
		t.Fatalf("unmarshal tool_choice.type: %v", err)
	}
	if tcType != "allowed_tools" {
		t.Errorf("tool_choice.type = %q, want allowed_tools", tcType)
	}

	var mode string
	if err := json.Unmarshal(tc["mode"], &mode); err != nil {
		t.Fatalf("unmarshal tool_choice.mode: %v", err)
	}
	if mode != "required" {
		t.Errorf("tool_choice.mode = %q, want required", mode)
	}

	var allowedTools []map[string]json.RawMessage
	if err := json.Unmarshal(tc["tools"], &allowedTools); err != nil {
		t.Fatalf("unmarshal tool_choice.tools: %v", err)
	}
	if len(allowedTools) != 2 {
		t.Fatalf("tool_choice.tools len = %d, want 2", len(allowedTools))
	}
}

func TestEncodeOpenAIResponsesRequest_FunctionCalls(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{
				Role: RoleAssistant,
				Content: []ContentPart{
					{Type: ContentTypeText, Text: &TextContent{Text: "I'll read the file"}},
					{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{
						ID:        "call_1",
						Name:      "read_file",
						Arguments: json.RawMessage(`{"path":"/tmp"}`),
					}},
				},
			},
			{
				Role: RoleTool,
				Content: []ContentPart{
					{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{
						ToolUseID: "call_1",
						Content:   []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "file contents"}}},
					}},
				},
			},
		},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	var input []map[string]interface{}
	json.Unmarshal(raw["input"], &input)

	if len(input) != 3 {
		t.Fatalf("input len = %d, want 3", len(input))
	}

	// First: assistant message with text
	if input[0]["type"] != "message" {
		t.Errorf("input[0].type = %v, want 'message'", input[0]["type"])
	}
	if input[0]["role"] != "assistant" {
		t.Errorf("input[0].role = %v, want 'assistant'", input[0]["role"])
	}

	// Second: function_call
	if input[1]["type"] != "function_call" {
		t.Errorf("input[1].type = %v, want 'function_call'", input[1]["type"])
	}
	if input[1]["call_id"] != "call_1" {
		t.Errorf("input[1].call_id = %v, want 'call_1'", input[1]["call_id"])
	}
	if input[1]["name"] != "read_file" {
		t.Errorf("input[1].name = %v, want 'read_file'", input[1]["name"])
	}

	// Third: function_call_output
	if input[2]["type"] != "function_call_output" {
		t.Errorf("input[2].type = %v, want 'function_call_output'", input[2]["type"])
	}
	if input[2]["call_id"] != "call_1" {
		t.Errorf("input[2].call_id = %v, want 'call_1'", input[2]["call_id"])
	}
	if input[2]["output"] != "file contents" {
		t.Errorf("input[2].output = %v, want 'file contents'", input[2]["output"])
	}
}

func TestDecodeOpenAIResponsesResponse_Basic(t *testing.T) {
	body := []byte(`{
		"id": "resp_1",
		"model": "gpt-4o",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Hello!"}]
			}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
	}`)

	resp, err := DecodeOpenAIResponsesResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID != "resp_1" {
		t.Errorf("ID = %q, want %q", resp.ID, "resp_1")
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
	if resp.Content[0].Type != ContentTypeText || resp.Content[0].Text.Text != "Hello!" {
		t.Errorf("Content[0] = %+v, want text 'Hello!'", resp.Content[0])
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

func TestDecodeOpenAIResponsesResponse_FunctionCall(t *testing.T) {
	body := []byte(`{
		"id": "resp_2",
		"model": "gpt-4o",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Let me check."}]
			},
			{
				"type": "function_call",
				"id": "fc_1",
				"call_id": "call_1",
				"name": "read_file",
				"arguments": "{\"path\":\"/tmp\"}"
			}
		],
		"usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
	}`)

	resp, err := DecodeOpenAIResponsesResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Status completed with function_call → ToolUse stop reason
	if resp.StopReason != StopReasonToolUse {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonToolUse)
	}

	if len(resp.Content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(resp.Content))
	}

	// Text content
	if resp.Content[0].Type != ContentTypeText || resp.Content[0].Text.Text != "Let me check." {
		t.Errorf("Content[0] = %+v, want text 'Let me check.'", resp.Content[0])
	}

	// Tool use content
	if resp.Content[1].Type != ContentTypeToolUse {
		t.Errorf("Content[1].Type = %q, want %q", resp.Content[1].Type, ContentTypeToolUse)
	}
	if resp.Content[1].ToolUse.ID != "call_1" {
		t.Errorf("ToolUse.ID = %q, want %q", resp.Content[1].ToolUse.ID, "call_1")
	}
	if resp.Content[1].ToolUse.Name != "read_file" {
		t.Errorf("ToolUse.Name = %q, want %q", resp.Content[1].ToolUse.Name, "read_file")
	}
}

func TestDecodeOpenAIResponsesResponse_WebSearchCall(t *testing.T) {
	body := []byte(`{
		"id": "resp_ws",
		"model": "gpt-4o",
		"status": "completed",
		"output": [
			{
				"type": "web_search_call",
				"id": "ws_1",
				"status": "completed",
				"action": {
					"type": "search",
					"query": "qwer1234",
					"sources": [{"type":"url","url":"https://openai.com"}]
				}
			},
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Found a source."}]
			}
		]
	}`)

	resp, err := DecodeOpenAIResponsesResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 3 {
		t.Fatalf("Content len = %d, want 3", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeServerToolUse {
		t.Fatalf("Content[0].Type = %q, want %q", resp.Content[0].Type, ContentTypeServerToolUse)
	}
	if resp.Content[0].ServerToolUse == nil {
		t.Fatal("Content[0].ServerToolUse is nil")
	}
	if resp.Content[0].ServerToolUse.ID != "ws_1" {
		t.Errorf("ServerToolUse.ID = %q, want ws_1", resp.Content[0].ServerToolUse.ID)
	}

	var args map[string]string
	if err := json.Unmarshal(resp.Content[0].ServerToolUse.Arguments, &args); err != nil {
		t.Fatalf("unmarshal ServerToolUse.Arguments: %v", err)
	}
	if args["query"] != "qwer1234" {
		t.Errorf("ServerToolUse.Arguments[query] = %q, want qwer1234", args["query"])
	}

	if resp.Content[1].Type != ContentTypeWebSearchToolResult {
		t.Fatalf("Content[1].Type = %q, want %q", resp.Content[1].Type, ContentTypeWebSearchToolResult)
	}
	if resp.Content[1].WebSearchToolResult == nil {
		t.Fatal("Content[1].WebSearchToolResult is nil")
	}
	if resp.Content[1].WebSearchToolResult.ToolUseID != "ws_1" {
		t.Errorf("WebSearchToolResult.ToolUseID = %q, want ws_1", resp.Content[1].WebSearchToolResult.ToolUseID)
	}
	if len(resp.Content[1].WebSearchToolResult.Content) != 1 {
		t.Fatalf("WebSearchToolResult.Content len = %d, want 1", len(resp.Content[1].WebSearchToolResult.Content))
	}
	if resp.Content[1].WebSearchToolResult.Content[0].URL != "https://openai.com" {
		t.Errorf("WebSearchToolResult.Content[0].URL = %q, want https://openai.com", resp.Content[1].WebSearchToolResult.Content[0].URL)
	}
	if resp.Content[2].Type != ContentTypeText || resp.Content[2].Text == nil || resp.Content[2].Text.Text != "Found a source." {
		t.Errorf("Content[2] = %+v, want text 'Found a source.'", resp.Content[2])
	}
	if resp.StopReason != StopReasonEndTurn {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonEndTurn)
	}
}

func TestDecodeOpenAIResponsesResponse_Incomplete(t *testing.T) {
	body := []byte(`{
		"id": "resp_3",
		"model": "gpt-4o",
		"status": "incomplete",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Partial..."}]
			}
		],
		"usage": {"input_tokens": 10, "output_tokens": 100, "total_tokens": 110}
	}`)

	resp, err := DecodeOpenAIResponsesResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.StopReason != StopReasonMaxTokens {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonMaxTokens)
	}
}

func TestEncodeOpenAIResponsesResponse_Basic(t *testing.T) {
	resp := &Response{
		ID:         "resp_1",
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

	data, err := EncodeOpenAIResponsesResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}

	var id string
	json.Unmarshal(raw["id"], &id)
	if id != "resp_1" {
		t.Errorf("id = %q, want %q", id, "resp_1")
	}

	var status string
	json.Unmarshal(raw["status"], &status)
	if status != "completed" {
		t.Errorf("status = %q, want %q", status, "completed")
	}

	var output []map[string]json.RawMessage
	json.Unmarshal(raw["output"], &output)
	if len(output) != 1 {
		t.Fatalf("output len = %d, want 1", len(output))
	}

	var itemType string
	json.Unmarshal(output[0]["type"], &itemType)
	if itemType != "message" {
		t.Errorf("output[0].type = %q, want %q", itemType, "message")
	}
}

func TestEncodeOpenAIResponsesResponse_FunctionCall(t *testing.T) {
	resp := &Response{
		ID:         "resp_2",
		Model:      "gpt-4o",
		StopReason: StopReasonToolUse,
		Content: []ContentPart{
			{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{
				ID:        "call_1",
				Name:      "read_file",
				Arguments: json.RawMessage(`{"path":"/tmp"}`),
			}},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	data, err := EncodeOpenAIResponsesResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	var status string
	json.Unmarshal(raw["status"], &status)
	if status != "completed" {
		t.Errorf("status = %q, want %q", status, "completed")
	}

	var output []map[string]interface{}
	json.Unmarshal(raw["output"], &output)
	if len(output) != 1 {
		t.Fatalf("output len = %d, want 1", len(output))
	}
	if output[0]["type"] != "function_call" {
		t.Errorf("output[0].type = %v, want 'function_call'", output[0]["type"])
	}
	if output[0]["call_id"] != "call_1" {
		t.Errorf("output[0].call_id = %v, want 'call_1'", output[0]["call_id"])
	}
}

func TestEncodeOpenAIResponsesResponse_PauseTurnDowngradesToCompleted(t *testing.T) {
	resp := &Response{
		ID:         "resp_pause",
		Model:      "gpt-4o",
		StopReason: StopReasonPauseTurn,
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Need more input."}},
		},
	}

	data, err := EncodeOpenAIResponsesResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}

	var status string
	if err := json.Unmarshal(raw["status"], &status); err != nil {
		t.Fatalf("unmarshal status: %v", err)
	}
	if status != "completed" {
		t.Fatalf("status = %q, want %q", status, "completed")
	}
}

func TestEncodeOpenAIResponsesStreamEvent_PauseTurnDowngradesToCompleted(t *testing.T) {
	stopReason := StopReasonPauseTurn
	event := &StreamEvent{
		Type:       StreamEventStop,
		StopReason: &stopReason,
	}

	eventType, data, err := EncodeOpenAIResponsesStreamEvent(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if eventType != "response.completed" {
		t.Fatalf("eventType = %q, want %q", eventType, "response.completed")
	}

	var raw openairesponses.StreamEvent
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw.Response == nil {
		t.Fatal("Response is nil")
	}
	if raw.Response.Status != "completed" {
		t.Fatalf("status = %q, want %q", raw.Response.Status, "completed")
	}
}

func TestEncodeOpenAIResponsesStreamEvent_TextLifecycleBoundaries(t *testing.T) {
	start := &StreamEvent{
		Type:  StreamEventContentBlockStart,
		Index: 0,
		Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{}},
	}

	eventType, data, err := EncodeOpenAIResponsesStreamEvent(start)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if eventType != "response.output_item.added" {
		t.Fatalf("eventType = %q, want %q", eventType, "response.output_item.added")
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal output_item.added: %v", err)
	}
	if _, ok := raw["item"]; !ok {
		t.Fatal("output_item.added missing item")
	}
	if _, ok := raw["part"]; ok {
		t.Fatal("output_item.added unexpectedly included part")
	}

	delta := &StreamEvent{
		Type:  StreamEventDelta,
		Index: 0,
		Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}},
	}
	deltaType, _, err := EncodeOpenAIResponsesStreamEvent(delta)
	if err != nil {
		t.Fatalf("unexpected delta error: %v", err)
	}
	if deltaType != "response.output_text.delta" {
		t.Fatalf("deltaType = %q, want %q", deltaType, "response.output_text.delta")
	}

	stop := &StreamEvent{Type: StreamEventContentBlockStop, Index: 0}
	stopType, _, err := EncodeOpenAIResponsesStreamEvent(stop)
	if err != nil {
		t.Fatalf("unexpected stop error: %v", err)
	}
	if stopType != "response.output_item.done" {
		t.Fatalf("stopType = %q, want %q", stopType, "response.output_item.done")
	}
}

func TestDecodeOpenAIResponsesStreamEvent(t *testing.T) {
	tests := []struct {
		name      string
		eventType string
		data      string
		wantType  StreamEventType
		check     func(t *testing.T, e *StreamEvent)
	}{
		{
			name:      "response.created",
			eventType: "response.created",
			data:      `{"response":{"id":"resp_1","model":"gpt-4o","status":"in_progress"}}`,
			wantType:  StreamEventStart,
			check: func(t *testing.T, e *StreamEvent) {
				if e.Response == nil {
					t.Fatal("Response is nil")
				}
				if e.Response.ID != "resp_1" {
					t.Errorf("Response.ID = %q, want %q", e.Response.ID, "resp_1")
				}
				if e.Response.Model != "gpt-4o" {
					t.Errorf("Response.Model = %q, want %q", e.Response.Model, "gpt-4o")
				}
			},
		},
		{
			name:      "response.output_item.added (message)",
			eventType: "response.output_item.added",
			data:      `{"output_index":0,"item":{"type":"message","role":"assistant"}}`,
			wantType:  StreamEventContentBlockStart,
			check: func(t *testing.T, e *StreamEvent) {
				if e.Index != 0 {
					t.Errorf("Index = %d, want 0", e.Index)
				}
				if e.Delta == nil {
					t.Fatal("Delta is nil")
				}
				if e.Delta.Type != ContentTypeText {
					t.Errorf("Delta.Type = %q, want %q", e.Delta.Type, ContentTypeText)
				}
			},
		},
		{
			name:      "response.output_item.added (function_call)",
			eventType: "response.output_item.added",
			data:      `{"output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"read_file"}}`,
			wantType:  StreamEventContentBlockStart,
			check: func(t *testing.T, e *StreamEvent) {
				if e.Index != 1 {
					t.Errorf("Index = %d, want 1", e.Index)
				}
				if e.Delta == nil {
					t.Fatal("Delta is nil")
				}
				if e.Delta.Type != ContentTypeToolUse {
					t.Errorf("Delta.Type = %q, want %q", e.Delta.Type, ContentTypeToolUse)
				}
				if e.Delta.ToolUse.ID != "call_1" {
					t.Errorf("Delta.ToolUse.ID = %q, want %q", e.Delta.ToolUse.ID, "call_1")
				}
				if e.Delta.ToolUse.Name != "read_file" {
					t.Errorf("Delta.ToolUse.Name = %q, want %q", e.Delta.ToolUse.Name, "read_file")
				}
			},
		},
		{
			name:      "response.output_text.delta",
			eventType: "response.output_text.delta",
			data:      `{"output_index":0,"delta":"Hello"}`,
			wantType:  StreamEventDelta,
			check: func(t *testing.T, e *StreamEvent) {
				if e.Delta == nil {
					t.Fatal("Delta is nil")
				}
				if e.Delta.Type != ContentTypeText {
					t.Errorf("Delta.Type = %q, want %q", e.Delta.Type, ContentTypeText)
				}
				if e.Delta.Text.Text != "Hello" {
					t.Errorf("Delta.Text = %q, want %q", e.Delta.Text.Text, "Hello")
				}
			},
		},
		{
			name:      "response.function_call_arguments.delta",
			eventType: "response.function_call_arguments.delta",
			data:      `{"output_index":1,"delta":"{\"path\""}`,
			wantType:  StreamEventDelta,
			check: func(t *testing.T, e *StreamEvent) {
				if e.Delta == nil {
					t.Fatal("Delta is nil")
				}
				if e.Delta.Type != ContentTypeToolUse {
					t.Errorf("Delta.Type = %q, want %q", e.Delta.Type, ContentTypeToolUse)
				}
				if string(e.Delta.ToolUse.Arguments) != `{"path"` {
					t.Errorf("Delta.ToolUse.Arguments = %s, want %s", e.Delta.ToolUse.Arguments, `{"path"`)
				}
			},
		},
		{
			name:      "response.output_item.done",
			eventType: "response.output_item.done",
			data:      `{"output_index":0}`,
			wantType:  StreamEventContentBlockStop,
			check: func(t *testing.T, e *StreamEvent) {
				if e.Index != 0 {
					t.Errorf("Index = %d, want 0", e.Index)
				}
			},
		},
		{
			name:      "response.completed",
			eventType: "response.completed",
			data:      `{"response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`,
			wantType:  StreamEventStop,
			check: func(t *testing.T, e *StreamEvent) {
				if e.Usage == nil {
					t.Fatal("Usage is nil")
				}
				if e.Usage.InputTokens != 10 {
					t.Errorf("Usage.InputTokens = %d, want 10", e.Usage.InputTokens)
				}
				if e.Usage.OutputTokens != 5 {
					t.Errorf("Usage.OutputTokens = %d, want 5", e.Usage.OutputTokens)
				}
				if e.StopReason == nil {
					t.Fatal("StopReason is nil")
				}
				if *e.StopReason != StopReasonEndTurn {
					t.Errorf("StopReason = %q, want %q", *e.StopReason, StopReasonEndTurn)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			event, err := DecodeOpenAIResponsesStreamEvent(tt.eventType, []byte(tt.data))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if event == nil {
				t.Fatal("event is nil")
			}
			if event.Type != tt.wantType {
				t.Errorf("Type = %q, want %q", event.Type, tt.wantType)
			}
			tt.check(t, event)
		})
	}
}

func TestDecodeOpenAIResponsesStreamEvent_WebSearchCallDone(t *testing.T) {
	data := []byte(`{
		"type":"response.output_item.done",
		"output_index":0,
		"item":{
			"type":"web_search_call",
			"id":"ws_1",
			"status":"completed",
			"action":{
				"type":"search",
				"query":"qwer1234",
				"sources":[{"type":"url","url":"https://openai.com"}]
			}
		}
	}`)

	event, err := DecodeOpenAIResponsesStreamEvent("response.output_item.done", data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event == nil {
		t.Fatal("event is nil")
	}
	if event.Type != StreamEventContentBlockStart {
		t.Fatalf("Type = %q, want %q", event.Type, StreamEventContentBlockStart)
	}
	if event.Index != webSearchToolResultStreamIndex(0) {
		t.Fatalf("Index = %d, want %d", event.Index, webSearchToolResultStreamIndex(0))
	}
	if event.Delta == nil || event.Delta.Type != ContentTypeWebSearchToolResult {
		t.Fatalf("Delta = %+v, want web_search_tool_result", event.Delta)
	}
	if event.Delta.WebSearchToolResult == nil {
		t.Fatal("Delta.WebSearchToolResult is nil")
	}
	if event.Delta.WebSearchToolResult.ToolUseID != "ws_1" {
		t.Errorf("ToolUseID = %q, want ws_1", event.Delta.WebSearchToolResult.ToolUseID)
	}
	if len(event.Delta.WebSearchToolResult.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(event.Delta.WebSearchToolResult.Content))
	}
	if event.Delta.WebSearchToolResult.Content[0].URL != "https://openai.com" {
		t.Errorf("Content[0].URL = %q, want https://openai.com", event.Delta.WebSearchToolResult.Content[0].URL)
	}
}

func TestDecodeOpenAIResponsesStreamEvent_Unknown(t *testing.T) {
	event, err := DecodeOpenAIResponsesStreamEvent("response.in_progress", []byte(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if event != nil {
		t.Errorf("expected nil for unknown event, got %+v", event)
	}
}

func TestDecodeOpenAIResponsesStreamEvent_SkippedEvents(t *testing.T) {
	// These events are skipped (return nil) because response.output_item.added /
	// response.output_item.done are the canonical lifecycle events.
	skipped := []struct {
		eventType string
		data      string
	}{
		{"response.content_part.added", `{"output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}`},
		{"response.content_part.done", `{"output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hello"}}`},
		{"response.output_text.done", `{"output_index":0,"content_index":0,"item_id":"item_1"}`},
		{"response.function_call_arguments.done", `{"output_index":0,"item_id":"item_1"}`},
	}
	for _, s := range skipped {
		t.Run(s.eventType, func(t *testing.T) {
			event, err := DecodeOpenAIResponsesStreamEvent(s.eventType, []byte(s.data))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if event != nil {
				t.Errorf("expected nil (skipped) for %q, got %+v", s.eventType, event)
			}
		})
	}
}

func TestEncodeOpenAIResponsesStreamEvent(t *testing.T) {
	tests := []struct {
		name          string
		event         *StreamEvent
		wantEventType string
		check         func(t *testing.T, data []byte)
	}{
		{
			name: "start",
			event: &StreamEvent{
				Type: StreamEventStart,
				Response: &Response{
					ID:    "resp_1",
					Model: "gpt-4o",
				},
			},
			wantEventType: "response.created",
			check: func(t *testing.T, data []byte) {
				var raw map[string]json.RawMessage
				json.Unmarshal(data, &raw)
				var resp map[string]interface{}
				json.Unmarshal(raw["response"], &resp)
				if resp["id"] != "resp_1" {
					t.Errorf("response.id = %v, want 'resp_1'", resp["id"])
				}
			},
		},
		{
			name: "content_block_start (text)",
			event: &StreamEvent{
				Type:  StreamEventContentBlockStart,
				Index: 0,
				Delta: &ContentPart{
					Type: ContentTypeText,
					Text: &TextContent{},
				},
			},
			wantEventType: "response.output_item.added",
			check: func(t *testing.T, data []byte) {
				var raw map[string]json.RawMessage
				json.Unmarshal(data, &raw)
				var item map[string]interface{}
				json.Unmarshal(raw["item"], &item)
				if item["type"] != "message" {
					t.Errorf("item.type = %v, want 'message'", item["type"])
				}
				if item["role"] != "assistant" {
					t.Errorf("item.role = %v, want 'assistant'", item["role"])
				}
			},
		},
		{
			name: "content_block_start (tool_use)",
			event: &StreamEvent{
				Type:  StreamEventContentBlockStart,
				Index: 1,
				Delta: &ContentPart{
					Type: ContentTypeToolUse,
					ToolUse: &ToolUseContent{
						ID:   "call_1",
						Name: "read_file",
					},
				},
			},
			wantEventType: "response.output_item.added",
			check: func(t *testing.T, data []byte) {
				var raw map[string]json.RawMessage
				json.Unmarshal(data, &raw)
				var item map[string]interface{}
				json.Unmarshal(raw["item"], &item)
				if item["type"] != "function_call" {
					t.Errorf("item.type = %v, want 'function_call'", item["type"])
				}
				if item["call_id"] != "call_1" {
					t.Errorf("item.call_id = %v, want 'call_1'", item["call_id"])
				}
			},
		},
		{
			name: "delta (text)",
			event: &StreamEvent{
				Type:  StreamEventDelta,
				Index: 0,
				Delta: &ContentPart{
					Type: ContentTypeText,
					Text: &TextContent{Text: "Hello"},
				},
			},
			wantEventType: "response.output_text.delta",
			check: func(t *testing.T, data []byte) {
				var raw map[string]interface{}
				json.Unmarshal(data, &raw)
				if raw["delta"] != "Hello" {
					t.Errorf("delta = %v, want 'Hello'", raw["delta"])
				}
			},
		},
		{
			name: "delta (tool_use)",
			event: &StreamEvent{
				Type:  StreamEventDelta,
				Index: 1,
				Delta: &ContentPart{
					Type: ContentTypeToolUse,
					ToolUse: &ToolUseContent{
						Arguments: json.RawMessage(`{"path"`),
					},
				},
			},
			wantEventType: "response.function_call_arguments.delta",
			check: func(t *testing.T, data []byte) {
				var raw map[string]interface{}
				json.Unmarshal(data, &raw)
				if raw["delta"] != `{"path"` {
					t.Errorf("delta = %v, want '{\"path\"'", raw["delta"])
				}
			},
		},
		{
			name: "content_block_stop",
			event: &StreamEvent{
				Type:  StreamEventContentBlockStop,
				Index: 0,
			},
			wantEventType: "response.output_item.done",
			check: func(t *testing.T, data []byte) {
				var raw map[string]interface{}
				json.Unmarshal(data, &raw)
				if raw["output_index"] != float64(0) {
					t.Errorf("output_index = %v, want 0", raw["output_index"])
				}
			},
		},
		{
			name: "stop",
			event: &StreamEvent{
				Type: StreamEventStop,
				Usage: &Usage{
					InputTokens:  10,
					OutputTokens: 5,
					TotalTokens:  15,
				},
			},
			wantEventType: "response.completed",
			check: func(t *testing.T, data []byte) {
				var raw map[string]json.RawMessage
				json.Unmarshal(data, &raw)
				var resp map[string]json.RawMessage
				json.Unmarshal(raw["response"], &resp)
				var status string
				json.Unmarshal(resp["status"], &status)
				if status != "completed" {
					t.Errorf("response.status = %q, want %q", status, "completed")
				}
				var usage map[string]interface{}
				json.Unmarshal(resp["usage"], &usage)
				if usage["input_tokens"] != float64(10) {
					t.Errorf("usage.input_tokens = %v, want 10", usage["input_tokens"])
				}
			},
		},
		{
			name: "stop with max_tokens",
			event: &StreamEvent{
				Type:       StreamEventStop,
				StopReason: stopReasonPtr(StopReasonMaxTokens),
			},
			wantEventType: "response.completed",
			check: func(t *testing.T, data []byte) {
				var raw map[string]json.RawMessage
				json.Unmarshal(data, &raw)
				var resp map[string]interface{}
				json.Unmarshal(raw["response"], &resp)
				if resp["status"] != "incomplete" {
					t.Errorf("response.status = %v, want 'incomplete'", resp["status"])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eventType, data, err := EncodeOpenAIResponsesStreamEvent(tt.event)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if eventType != tt.wantEventType {
				t.Errorf("eventType = %q, want %q", eventType, tt.wantEventType)
			}
			tt.check(t, data)
		})
	}
}

func TestOpenAIResponsesRequestRoundTrip(t *testing.T) {
	temp := 0.7
	topP := 0.9
	original := &Request{
		Model: "gpt-4o",
		SystemPrompt: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Be helpful"}},
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
					{Type: ContentTypeText, Text: &TextContent{Text: "Hi there!"}},
				},
			},
		},
		MaxTokens:   1024,
		Temperature: &temp,
		TopP:        &topP,
		Stream:      true,
		Tools: []Tool{
			{
				Name:        "read_file",
				Description: "Read a file",
				Parameters:  json.RawMessage(`{"type":"object"}`),
				Strict:      true,
			},
		},
		ToolChoice: &ToolChoice{Type: "auto"},
		Thinking:   &ThinkingConfig{Mode: "enabled", Effort: "high"},
	}

	// Encode
	data, err := EncodeOpenAIResponsesRequest(original)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	// Decode
	roundTripped, err := DecodeOpenAIResponsesRequest(data)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	// Verify key fields
	if roundTripped.Model != original.Model {
		t.Errorf("Model = %q, want %q", roundTripped.Model, original.Model)
	}
	if roundTripped.MaxTokens != original.MaxTokens {
		t.Errorf("MaxTokens = %d, want %d", roundTripped.MaxTokens, original.MaxTokens)
	}
	if roundTripped.Temperature == nil || *roundTripped.Temperature != *original.Temperature {
		t.Errorf("Temperature = %v, want %v", roundTripped.Temperature, original.Temperature)
	}
	if roundTripped.TopP == nil || *roundTripped.TopP != *original.TopP {
		t.Errorf("TopP = %v, want %v", roundTripped.TopP, original.TopP)
	}
	if roundTripped.Stream != original.Stream {
		t.Errorf("Stream = %v, want %v", roundTripped.Stream, original.Stream)
	}

	// SystemPrompt (instructions round-trip)
	if len(roundTripped.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(roundTripped.SystemPrompt))
	}
	if roundTripped.SystemPrompt[0].Text.Text != "Be helpful" {
		t.Errorf("SystemPrompt = %q, want %q", roundTripped.SystemPrompt[0].Text.Text, "Be helpful")
	}

	// Messages
	if len(roundTripped.Messages) != len(original.Messages) {
		t.Fatalf("Messages len = %d, want %d", len(roundTripped.Messages), len(original.Messages))
	}
	for i, m := range roundTripped.Messages {
		if m.Role != original.Messages[i].Role {
			t.Errorf("Messages[%d].Role = %q, want %q", i, m.Role, original.Messages[i].Role)
		}
		if len(m.Content) != len(original.Messages[i].Content) {
			t.Errorf("Messages[%d].Content len = %d, want %d", i, len(m.Content), len(original.Messages[i].Content))
			continue
		}
		for j, c := range m.Content {
			if c.Type != original.Messages[i].Content[j].Type {
				t.Errorf("Messages[%d].Content[%d].Type = %q, want %q", i, j, c.Type, original.Messages[i].Content[j].Type)
			}
			if c.Text != nil && original.Messages[i].Content[j].Text != nil {
				if c.Text.Text != original.Messages[i].Content[j].Text.Text {
					t.Errorf("Messages[%d].Content[%d].Text = %q, want %q", i, j, c.Text.Text, original.Messages[i].Content[j].Text.Text)
				}
			}
		}
	}

	// Tools
	if len(roundTripped.Tools) != len(original.Tools) {
		t.Fatalf("Tools len = %d, want %d", len(roundTripped.Tools), len(original.Tools))
	}
	if roundTripped.Tools[0].Name != original.Tools[0].Name {
		t.Errorf("Tools[0].Name = %q, want %q", roundTripped.Tools[0].Name, original.Tools[0].Name)
	}
	if roundTripped.Tools[0].Strict != original.Tools[0].Strict {
		t.Errorf("Tools[0].Strict = %v, want %v", roundTripped.Tools[0].Strict, original.Tools[0].Strict)
	}

	// ToolChoice
	if roundTripped.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if roundTripped.ToolChoice.Type != original.ToolChoice.Type {
		t.Errorf("ToolChoice.Type = %q, want %q", roundTripped.ToolChoice.Type, original.ToolChoice.Type)
	}

	// Thinking
	if roundTripped.Thinking == nil {
		t.Fatal("Thinking is nil")
	}
	if roundTripped.Thinking.Effort != original.Thinking.Effort {
		t.Errorf("Thinking.Effort = %q, want %q", roundTripped.Thinking.Effort, original.Thinking.Effort)
	}
}

func TestOpenAIResponsesResponseRoundTrip(t *testing.T) {
	original := &Response{
		ID:         "resp_1",
		Model:      "gpt-4o",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Hello!"}},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	data, err := EncodeOpenAIResponsesResponse(original)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	roundTripped, err := DecodeOpenAIResponsesResponse(data)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if roundTripped.ID != original.ID {
		t.Errorf("ID = %q, want %q", roundTripped.ID, original.ID)
	}
	if roundTripped.Model != original.Model {
		t.Errorf("Model = %q, want %q", roundTripped.Model, original.Model)
	}
	if roundTripped.StopReason != original.StopReason {
		t.Errorf("StopReason = %q, want %q", roundTripped.StopReason, original.StopReason)
	}
	if len(roundTripped.Content) != 1 || roundTripped.Content[0].Text.Text != "Hello!" {
		t.Errorf("Content = %+v, want text 'Hello!'", roundTripped.Content)
	}
	if roundTripped.Usage.InputTokens != 10 {
		t.Errorf("Usage.InputTokens = %d, want 10", roundTripped.Usage.InputTokens)
	}
}

// TestEncodeOpenAIResponsesRequest_MixedUserToolResult tests that a RoleUser
// message containing both ContentTypeToolResult and ContentTypeText parts
// (as produced by the Anthropic decoder for mixed user messages) correctly
// splits into function_call_output items + a user message.
func TestEncodeOpenAIResponsesRequest_MixedUserToolResult(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "What's the weather?"}},
			}},
			{
				Role: RoleAssistant,
				Content: []ContentPart{
					{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{
						ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`),
					}},
				},
			},
			// Mixed user message: tool_result + text (common Anthropic pattern)
			{
				Role: RoleUser,
				Content: []ContentPart{
					{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{
						ToolUseID: "call_1",
						Content:   []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "sunny, 22C"}}},
					}},
					{Type: ContentTypeText, Text: &TextContent{Text: "Now summarize the result"}},
				},
			},
		},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	var input []map[string]interface{}
	json.Unmarshal(raw["input"], &input)

	// Expect 4 items: user message, function_call, function_call_output, user message
	if len(input) != 4 {
		t.Fatalf("input len = %d, want 4; got: %s", len(input), raw["input"])
	}

	// input[0]: user message
	if input[0]["type"] != "message" || input[0]["role"] != "user" {
		t.Errorf("input[0] = %v, want user message", input[0])
	}

	// input[1]: function_call
	if input[1]["type"] != "function_call" {
		t.Errorf("input[1].type = %v, want 'function_call'", input[1]["type"])
	}
	if input[1]["call_id"] != "call_1" {
		t.Errorf("input[1].call_id = %v, want 'call_1'", input[1]["call_id"])
	}

	// input[2]: function_call_output (extracted from mixed user message)
	if input[2]["type"] != "function_call_output" {
		t.Errorf("input[2].type = %v, want 'function_call_output'", input[2]["type"])
	}
	if input[2]["call_id"] != "call_1" {
		t.Errorf("input[2].call_id = %v, want 'call_1'", input[2]["call_id"])
	}
	if input[2]["output"] != "sunny, 22C" {
		t.Errorf("input[2].output = %v, want 'sunny, 22C'", input[2]["output"])
	}

	// input[3]: remaining user message with text
	if input[3]["type"] != "message" || input[3]["role"] != "user" {
		t.Errorf("input[3] = %v, want user message", input[3])
	}
}

// TestEncodeOpenAIResponsesRequest_MultipleToolResultsMixed tests multiple
// tool_result parts interleaved with text in a single user message.
func TestEncodeOpenAIResponsesRequest_MultipleToolResultsMixed(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []ContentPart{
					{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{
						ToolUseID: "call_1",
						Content:   []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "result 1"}}},
					}},
					{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{
						ToolUseID: "call_2",
						Content:   []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "result 2"}}},
					}},
					{Type: ContentTypeText, Text: &TextContent{Text: "continue"}},
				},
			},
		},
	}

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	var input []map[string]interface{}
	json.Unmarshal(raw["input"], &input)

	// Expect 3 items: 2 function_call_output + 1 user message
	if len(input) != 3 {
		t.Fatalf("input len = %d, want 3; got: %s", len(input), raw["input"])
	}

	if input[0]["type"] != "function_call_output" || input[0]["call_id"] != "call_1" {
		t.Errorf("input[0] = %v, want function_call_output call_1", input[0])
	}
	if input[1]["type"] != "function_call_output" || input[1]["call_id"] != "call_2" {
		t.Errorf("input[1] = %v, want function_call_output call_2", input[1])
	}
	if input[2]["type"] != "message" || input[2]["role"] != "user" {
		t.Errorf("input[2] = %v, want user message", input[2])
	}
}

// TestAnthropicToOpenAIResponses_MixedToolResult tests the full conversion path:
// Anthropic request with mixed user message → IR → OpenAI Responses encoded request.
func TestAnthropicToOpenAIResponses_MixedToolResult(t *testing.T) {
	// Anthropic request with tool_result + text in same user message (Claude CLI pattern)
	body := []byte(`{
		"model": "claude-3-5-sonnet-20241022",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "What is the weather?"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"city": "Tokyo"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "Sunny, 25°C"},
				{"type": "text", "text": "Please summarize this"}
			]}
		],
		"tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
	}`)

	// Step 1: Decode Anthropic request to IR
	irReq, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("decode anthropic: %v", err)
	}

	// Verify IR: message[2] should be RoleUser with mixed content
	if len(irReq.Messages) != 3 {
		t.Fatalf("IR messages len = %d, want 3", len(irReq.Messages))
	}
	msg2 := irReq.Messages[2]
	if msg2.Role != RoleUser {
		t.Fatalf("msg[2].Role = %q, want %q", msg2.Role, RoleUser)
	}
	if len(msg2.Content) != 2 {
		t.Fatalf("msg[2].Content len = %d, want 2", len(msg2.Content))
	}

	// Step 2: Re-target to OpenAI Responses
	irReq.Model = "gpt-4o"
	data, err := EncodeOpenAIResponsesRequest(irReq)
	if err != nil {
		t.Fatalf("encode openai responses: %v", err)
	}

	var raw map[string]json.RawMessage
	json.Unmarshal(data, &raw)

	var input []map[string]interface{}
	json.Unmarshal(raw["input"], &input)

	// Must have function_call_output for toolu_abc
	foundOutput := false
	for _, item := range input {
		if item["type"] == "function_call_output" && item["call_id"] == "toolu_abc" {
			foundOutput = true
			if item["output"] != "Sunny, 25°C" {
				t.Errorf("function_call_output.output = %v, want 'Sunny, 25°C'", item["output"])
			}
		}
	}
	if !foundOutput {
		t.Errorf("no function_call_output found for toolu_abc in input: %s", raw["input"])
	}

	// Must also have function_call for toolu_abc
	foundCall := false
	for _, item := range input {
		if item["type"] == "function_call" && item["call_id"] == "toolu_abc" {
			foundCall = true
		}
	}
	if !foundCall {
		t.Errorf("no function_call found for toolu_abc in input: %s", raw["input"])
	}
}

// stopReasonPtr is a helper to create a pointer to a StopReason.
func stopReasonPtr(sr StopReason) *StopReason {
	return &sr
}

// TestDecodeOpenAIResponsesResponse_ReasoningTokens verifies that reasoning_tokens
// inside output_tokens_details is mapped to Usage.ThinkingTokens.
func TestDecodeOpenAIResponsesResponse_ReasoningTokens(t *testing.T) {
	body := []byte(`{
		"id": "resp_reasoning",
		"model": "o1",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "42."}]
			}
		],
		"usage": {
			"input_tokens": 100,
			"output_tokens": 200,
			"total_tokens": 300,
			"output_tokens_details": {
				"reasoning_tokens": 150
			}
		}
	}`)

	resp, err := DecodeOpenAIResponsesResponse(body)
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

// TestDecodeOpenAIResponsesResponse_CachedAndReasoningTokens verifies that both
// CacheReadTokens (from input_tokens_details.cached_tokens) and ThinkingTokens
// (from output_tokens_details.reasoning_tokens) are decoded correctly.
func TestDecodeOpenAIResponsesResponse_CachedAndReasoningTokens(t *testing.T) {
	body := []byte(`{
		"id": "resp_mixed",
		"model": "o1",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Done."}]
			}
		],
		"usage": {
			"input_tokens": 500,
			"output_tokens": 100,
			"total_tokens": 600,
			"input_tokens_details": {
				"cached_tokens": 400
			},
			"output_tokens_details": {
				"reasoning_tokens": 60
			}
		}
	}`)

	resp, err := DecodeOpenAIResponsesResponse(body)
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

// TestEncodeOpenAIResponsesResponse_ReasoningTokensRoundTrip verifies that ThinkingTokens
// is encoded back as reasoning_tokens in output_tokens_details and survives a
// full round-trip.
func TestEncodeOpenAIResponsesResponse_ReasoningTokensRoundTrip(t *testing.T) {
	ir := &Response{
		ID:         "resp_rt",
		Model:      "o1",
		StopReason: StopReasonEndTurn,
		Content:    []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}},
		Usage: Usage{
			InputTokens:    10,
			OutputTokens:   20,
			TotalTokens:    30,
			ThinkingTokens: 12,
		},
	}

	body, err := EncodeOpenAIResponsesResponse(ir)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Inspect raw JSON for the correct wire key.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	usageRaw, ok := raw["usage"]
	if !ok {
		t.Fatal("usage missing from encoded OpenAI Responses response")
	}
	var usageMap map[string]json.RawMessage
	if err := json.Unmarshal(usageRaw, &usageMap); err != nil {
		t.Fatalf("unmarshal usage: %v", err)
	}
	detailsRaw, ok := usageMap["output_tokens_details"]
	if !ok {
		t.Fatal("output_tokens_details missing from usage when ThinkingTokens > 0")
	}
	var details map[string]json.RawMessage
	if err := json.Unmarshal(detailsRaw, &details); err != nil {
		t.Fatalf("unmarshal output_tokens_details: %v", err)
	}
	if _, ok := details["reasoning_tokens"]; !ok {
		t.Error("reasoning_tokens missing from output_tokens_details")
	}

	// Also verify round-trip via Decode.
	decoded, err := DecodeOpenAIResponsesResponse(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.Usage.ThinkingTokens != 12 {
		t.Errorf("ThinkingTokens round-trip: got %d, want 12", decoded.Usage.ThinkingTokens)
	}
}

// --- Tests for parallel_tool_calls ---

func TestDecodeOpenAIResponsesRequest_ParallelToolCallsTrue(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"parallel_tool_calls": true
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
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

func TestDecodeOpenAIResponsesRequest_ParallelToolCallsFalse(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"parallel_tool_calls": false
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
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

func TestDecodeOpenAIResponsesRequest_ParallelToolCallsWithToolChoice(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"tool_choice": "auto",
		"parallel_tool_calls": false
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
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

func TestDecodeOpenAIResponsesRequest_NoParallelToolCalls(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"input": "Hello",
		"tool_choice": "auto"
	}`)

	req, err := DecodeOpenAIResponsesRequest(body)
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

func TestEncodeOpenAIResponsesRequest_ParallelToolCallsTrue(t *testing.T) {
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

	body, err := EncodeOpenAIResponsesRequest(req)
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

func TestEncodeOpenAIResponsesRequest_ParallelToolCallsFalse(t *testing.T) {
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

	body, err := EncodeOpenAIResponsesRequest(req)
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

func TestEncodeOpenAIResponsesRequest_NoParallelToolCalls(t *testing.T) {
	req := &Request{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type: "auto",
		},
	}

	body, err := EncodeOpenAIResponsesRequest(req)
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

func TestDecodeEncodeOpenAIResponsesRequest_ParallelToolCallsRoundTrip(t *testing.T) {
	original := []byte(`{"model":"gpt-4o","input":"hi","tool_choice":"auto","parallel_tool_calls":false}`)

	req, err := DecodeOpenAIResponsesRequest(original)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	encoded, err := EncodeOpenAIResponsesRequest(req)
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

// --- Refusal and stream error/incomplete tests ---

func TestDecodeOpenAIResponsesResponse_Refusal(t *testing.T) {
	body := []byte(`{
		"id": "resp_123",
		"model": "gpt-4o",
		"status": "completed",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "refusal", "refusal": "I cannot help with that."}]
			}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
	}`)

	resp, err := DecodeOpenAIResponsesResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeRefusal {
		t.Errorf("Content[0].Type = %q, want %q", resp.Content[0].Type, ContentTypeRefusal)
	}
	if resp.Content[0].Refusal == nil {
		t.Fatal("Content[0].Refusal is nil")
	}
	if resp.Content[0].Refusal.Refusal != "I cannot help with that." {
		t.Errorf("Refusal = %q, want %q", resp.Content[0].Refusal.Refusal, "I cannot help with that.")
	}
}

func TestEncodeOpenAIResponsesResponse_Refusal(t *testing.T) {
	resp := &Response{
		ID:         "resp_123",
		Model:      "gpt-4o",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{Type: ContentTypeRefusal, Refusal: &RefusalContent{Refusal: "Cannot do that."}},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	data, err := EncodeOpenAIResponsesResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Type    string `json:"type"`
				Refusal string `json:"refusal"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(raw.Output) == 0 {
		t.Fatal("no output items")
	}
	if len(raw.Output[0].Content) == 0 {
		t.Fatal("no content in output")
	}
	if raw.Output[0].Content[0].Type != "refusal" {
		t.Errorf("content type = %q, want %q", raw.Output[0].Content[0].Type, "refusal")
	}
	if raw.Output[0].Content[0].Refusal != "Cannot do that." {
		t.Errorf("refusal = %q, want %q", raw.Output[0].Content[0].Refusal, "Cannot do that.")
	}
}

func TestDecodeOpenAIResponsesStreamEvent_ResponseFailed(t *testing.T) {
	data := []byte(`{
		"type": "response.failed",
		"response": {
			"id": "resp_123",
			"status": "failed",
			"usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}
		}
	}`)

	event, err := DecodeOpenAIResponsesStreamEvent("response.failed", data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if event == nil {
		t.Fatal("event is nil — response.failed should not be silently dropped")
	}
	if event.Type != StreamEventError {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventError)
	}
	if event.Error == nil {
		t.Fatal("Error is nil")
	}
	if event.Usage == nil {
		t.Fatal("Usage is nil")
	}
	if event.Usage.InputTokens != 10 {
		t.Errorf("Usage.InputTokens = %d, want 10", event.Usage.InputTokens)
	}
}

func TestDecodeOpenAIResponsesStreamEvent_ResponseIncomplete(t *testing.T) {
	data := []byte(`{
		"type": "response.incomplete",
		"response": {
			"id": "resp_123",
			"status": "incomplete",
			"usage": {"input_tokens": 10, "output_tokens": 100, "total_tokens": 110}
		}
	}`)

	event, err := DecodeOpenAIResponsesStreamEvent("response.incomplete", data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if event == nil {
		t.Fatal("event is nil — response.incomplete should not be silently dropped")
	}
	if event.Type != StreamEventStop {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventStop)
	}
	if event.StopReason == nil {
		t.Fatal("StopReason is nil")
	}
	if *event.StopReason != StopReasonMaxTokens {
		t.Errorf("StopReason = %q, want %q", *event.StopReason, StopReasonMaxTokens)
	}
	if event.IncompleteDetails == nil {
		t.Fatal("IncompleteDetails is nil")
	}
	if event.IncompleteDetails.Reason != "max_output_tokens" {
		t.Errorf("IncompleteDetails.Reason = %q, want %q", event.IncompleteDetails.Reason, "max_output_tokens")
	}
	if event.Usage == nil {
		t.Fatal("Usage is nil")
	}
}

func TestDecodeOpenAIResponsesStreamEvent_Error(t *testing.T) {
	data := []byte(`{
		"type": "error",
		"code": "rate_limit_exceeded",
		"message": "Rate limit exceeded",
		"param": "model"
	}`)

	event, err := DecodeOpenAIResponsesStreamEvent("error", data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if event == nil {
		t.Fatal("event is nil — error event should not be silently dropped")
	}
	if event.Type != StreamEventError {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventError)
	}
	if event.Error == nil {
		t.Fatal("Error is nil")
	}
	if event.Error.Code != "rate_limit_exceeded" {
		t.Errorf("Error.Code = %q, want %q", event.Error.Code, "rate_limit_exceeded")
	}
	if event.Error.Message != "Rate limit exceeded" {
		t.Errorf("Error.Message = %q, want %q", event.Error.Message, "Rate limit exceeded")
	}
	if event.Error.Param != "model" {
		t.Errorf("Error.Param = %q, want %q", event.Error.Param, "model")
	}
}

func TestDecodeOpenAIResponsesStreamEvent_RefusalDelta(t *testing.T) {
	data := []byte(`{
		"type": "response.refusal.delta",
		"output_index": 0,
		"delta": "I cannot"
	}`)

	event, err := DecodeOpenAIResponsesStreamEvent("response.refusal.delta", data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if event == nil {
		t.Fatal("event is nil")
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
	if event.Delta.Refusal.Refusal != "I cannot" {
		t.Errorf("Delta.Refusal = %q, want %q", event.Delta.Refusal.Refusal, "I cannot")
	}
}

func TestEncodeOpenAIResponsesStreamEvent_Error(t *testing.T) {
	event := &StreamEvent{
		Type: StreamEventError,
		Error: &StreamError{
			Type:    "error",
			Code:    "server_error",
			Message: "Internal server error",
		},
	}

	eventType, data, err := EncodeOpenAIResponsesStreamEvent(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if eventType != "error" {
		t.Errorf("eventType = %q, want %q", eventType, "error")
	}

	var raw struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw.Type != "error" {
		t.Errorf("type = %q, want %q", raw.Type, "error")
	}
	if raw.Code != "server_error" {
		t.Errorf("code = %q, want %q", raw.Code, "server_error")
	}
}

func TestEncodeOpenAIResponsesStreamEvent_RefusalDelta(t *testing.T) {
	event := &StreamEvent{
		Type: StreamEventDelta,
		Delta: &ContentPart{
			Type:    ContentTypeRefusal,
			Refusal: &RefusalContent{Refusal: "I refuse"},
		},
	}

	eventType, data, err := EncodeOpenAIResponsesStreamEvent(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if eventType != "response.refusal.delta" {
		t.Errorf("eventType = %q, want %q", eventType, "response.refusal.delta")
	}

	var raw struct {
		Delta string `json:"delta"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw.Delta != "I refuse" {
		t.Errorf("delta = %q, want %q", raw.Delta, "I refuse")
	}
}

// TestEncodeOpenAIResponsesRequest_ThinkingConfig_Phase2_Degradation verifies that
// Phase 2 thinking fields (IncludeThoughts, Level) are silently dropped when
// encoding to OpenAI Responses, which has no native equivalent.
func TestEncodeOpenAIResponsesRequest_ThinkingConfig_Phase2_Degradation(t *testing.T) {
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

	data, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// reasoning.effort should be set from Effort
	reasoningRaw, ok := raw["reasoning"]
	if !ok {
		t.Fatal("reasoning field missing from OpenAI Responses request")
	}
	reasoningMap, ok := reasoningRaw.(map[string]interface{})
	if !ok {
		t.Fatalf("reasoning is not an object: %T", reasoningRaw)
	}
	if reasoningMap["effort"] != "high" {
		t.Errorf("reasoning.effort = %v, want high", reasoningMap["effort"])
	}

	// IncludeThoughts and Level have no OpenAI Responses equivalent — should be absent
	if _, ok := raw["include_thoughts"]; ok {
		t.Error("include_thoughts should not appear in OpenAI Responses request (silently dropped)")
	}
	if _, ok := raw["includeThoughts"]; ok {
		t.Error("includeThoughts should not appear in OpenAI Responses request (silently dropped)")
	}
	if _, ok := raw["level"]; ok {
		t.Error("level should not appear in OpenAI Responses request (silently dropped)")
	}
	if _, ok := raw["thinkingLevel"]; ok {
		t.Error("thinkingLevel should not appear in OpenAI Responses request (silently dropped)")
	}
}
