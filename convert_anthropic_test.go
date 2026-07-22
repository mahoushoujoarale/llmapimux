package llmapimux

import (
	"encoding/base64"
	"encoding/json"
	"testing"
)

func TestDecodeAnthropicRequest_Basic(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"system": [{"type": "text", "text": "You are helpful."}],
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Hello"}]},
			{"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}
		],
		"temperature": 0.7,
		"top_p": 0.9,
		"stop_sequences": ["END"],
		"stream": true
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Model != "claude-sonnet-4-20250514" {
		t.Errorf("Model = %q, want %q", req.Model, "claude-sonnet-4-20250514")
	}
	if req.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %d, want 1024", req.MaxTokens)
	}
	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", req.Temperature)
	}
	if req.TopP == nil || *req.TopP != 0.9 {
		t.Errorf("TopP = %v, want 0.9", req.TopP)
	}
	if !req.Stream {
		t.Error("Stream = false, want true")
	}
	if len(req.StopSequences) != 1 || req.StopSequences[0] != "END" {
		t.Errorf("StopSequences = %v, want [END]", req.StopSequences)
	}

	// system prompt
	if len(req.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(req.SystemPrompt))
	}
	sp := req.SystemPrompt[0]
	if sp.Type != ContentTypeText {
		t.Errorf("SystemPrompt[0].Type = %q, want %q", sp.Type, ContentTypeText)
	}
	if sp.Text == nil || sp.Text.Text != "You are helpful." {
		t.Errorf("SystemPrompt[0].Text = %v, want 'You are helpful.'", sp.Text)
	}

	// messages
	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
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
	if req.Messages[0].Content[0].Text == nil || req.Messages[0].Content[0].Text.Text != "Hello" {
		t.Errorf("Messages[0].Content[0].Text = %v, want 'Hello'", req.Messages[0].Content[0].Text)
	}

	if req.Messages[1].Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", req.Messages[1].Role, RoleAssistant)
	}
}

func TestDecodeAnthropicRequest_Tools(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-5",
		"max_tokens": 512,
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Use the tool"}]}],
		"tools": [
			{
				"name": "read_file",
				"description": "Read a file",
				"input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
				"type": "custom"
			},
			{
				"name": "web_search",
				"type": "web_search_20250305"
			}
		],
		"tool_choice": {"type": "auto"}
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Tools) != 2 {
		t.Fatalf("Tools len = %d, want 2", len(req.Tools))
	}
	customTool := req.Tools[0]
	if customTool.Type != "function" {
		t.Errorf("Tools[0].Type = %q, want function", customTool.Type)
	}
	if customTool.Name != "read_file" {
		t.Errorf("Tools[0].Name = %q, want %q", customTool.Name, "read_file")
	}
	if customTool.Description != "Read a file" {
		t.Errorf("Tools[0].Description = %q, want %q", customTool.Description, "Read a file")
	}
	if customTool.Parameters == nil {
		t.Error("Tools[0].Parameters is nil")
	}

	serverTool := req.Tools[1]
	if serverTool.Type != "web_search" {
		t.Errorf("Tools[1].Type = %q, want web_search", serverTool.Type)
	}
	if serverTool.Name != "web_search" {
		t.Errorf("Tools[1].Name = %q, want web_search", serverTool.Name)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "auto" {
		t.Errorf("ToolChoice.Type = %q, want %q", req.ToolChoice.Type, "auto")
	}
}

func TestDecodeAnthropicRequest_Image(t *testing.T) {
	imgData := []byte("fake-png-data")
	encoded := base64.StdEncoding.EncodeToString(imgData)

	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 256,
		"messages": [
			{"role": "user", "content": [
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "` + encoded + `"}},
				{"type": "text", "text": "What is this?"},
				{"type": "image", "source": {"type": "url", "url": "https://example.com/img.jpg"}}
			]}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
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

	// base64 image
	img := content[0]
	if img.Type != ContentTypeImage {
		t.Errorf("content[0].Type = %q, want %q", img.Type, ContentTypeImage)
	}
	if img.Image == nil {
		t.Fatal("content[0].Image is nil")
	}
	if img.Image.MediaType != "image/png" {
		t.Errorf("Image.MediaType = %q, want %q", img.Image.MediaType, "image/png")
	}
	if string(img.Image.Data) != string(imgData) {
		t.Errorf("Image.Data = %v, want %v", img.Image.Data, imgData)
	}
	if img.Image.URL != "" {
		t.Errorf("Image.URL = %q, want empty", img.Image.URL)
	}

	// text
	if content[1].Type != ContentTypeText {
		t.Errorf("content[1].Type = %q, want %q", content[1].Type, ContentTypeText)
	}

	// url image
	urlImg := content[2]
	if urlImg.Type != ContentTypeImage {
		t.Errorf("content[2].Type = %q, want %q", urlImg.Type, ContentTypeImage)
	}
	if urlImg.Image == nil {
		t.Fatal("content[2].Image is nil")
	}
	if urlImg.Image.URL != "https://example.com/img.jpg" {
		t.Errorf("Image.URL = %q, want %q", urlImg.Image.URL, "https://example.com/img.jpg")
	}
	if len(urlImg.Image.Data) != 0 {
		t.Errorf("Image.Data should be empty for URL image, got %v", urlImg.Image.Data)
	}
}

func TestDecodeAnthropicRequest_ToolUse(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 512,
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Read the file"}]},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "tu_123", "name": "read_file", "input": {"path": "/etc/hosts"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_123", "content": [{"type": "text", "text": "127.0.0.1 localhost"}], "is_error": false}
			]}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(req.Messages))
	}

	// assistant tool_use
	assistantMsg := req.Messages[1]
	if len(assistantMsg.Content) != 1 {
		t.Fatalf("assistant Content len = %d, want 1", len(assistantMsg.Content))
	}
	tu := assistantMsg.Content[0]
	if tu.Type != ContentTypeToolUse {
		t.Errorf("tool_use content.Type = %q, want %q", tu.Type, ContentTypeToolUse)
	}
	if tu.ToolUse == nil {
		t.Fatal("ToolUse is nil")
	}
	if tu.ToolUse.ID != "tu_123" {
		t.Errorf("ToolUse.ID = %q, want %q", tu.ToolUse.ID, "tu_123")
	}
	if tu.ToolUse.Name != "read_file" {
		t.Errorf("ToolUse.Name = %q, want %q", tu.ToolUse.Name, "read_file")
	}
	if tu.ToolUse.Arguments == nil {
		t.Error("ToolUse.Arguments is nil")
	}
	// Verify the Arguments JSON contains the path
	var args map[string]string
	if err := json.Unmarshal(tu.ToolUse.Arguments, &args); err != nil {
		t.Errorf("failed to unmarshal Arguments: %v", err)
	} else if args["path"] != "/etc/hosts" {
		t.Errorf("Arguments[path] = %q, want %q", args["path"], "/etc/hosts")
	}

	// user tool_result — should be promoted to RoleTool
	userMsg := req.Messages[2]
	if userMsg.Role != RoleTool {
		t.Errorf("tool_result message Role = %q, want %q", userMsg.Role, RoleTool)
	}
	if len(userMsg.Content) != 1 {
		t.Fatalf("user Content len = %d, want 1", len(userMsg.Content))
	}
	tr := userMsg.Content[0]
	if tr.Type != ContentTypeToolResult {
		t.Errorf("tool_result content.Type = %q, want %q", tr.Type, ContentTypeToolResult)
	}
	if tr.ToolResult == nil {
		t.Fatal("ToolResult is nil")
	}
	if tr.ToolResult.ToolUseID != "tu_123" {
		t.Errorf("ToolResult.ToolUseID = %q, want %q", tr.ToolResult.ToolUseID, "tu_123")
	}
	if tr.ToolResult.IsError {
		t.Error("ToolResult.IsError = true, want false")
	}
	if len(tr.ToolResult.Content) != 1 {
		t.Fatalf("ToolResult.Content len = %d, want 1", len(tr.ToolResult.Content))
	}
	if tr.ToolResult.Content[0].Type != ContentTypeText {
		t.Errorf("ToolResult.Content[0].Type = %q, want %q", tr.ToolResult.Content[0].Type, ContentTypeText)
	}
	if tr.ToolResult.Content[0].Text == nil || tr.ToolResult.Content[0].Text.Text != "127.0.0.1 localhost" {
		t.Errorf("ToolResult.Content[0].Text = %v", tr.ToolResult.Content[0].Text)
	}
}

func TestDecodeAnthropicRequest_Thinking(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 2048,
		"thinking": {"type": "enabled", "budget_tokens": 4096},
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Solve this problem"}]},
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "Let me think step by step...", "signature": "sig_abc"},
				{"type": "redacted_thinking", "data": "redacted_blob_xyz"},
				{"type": "text", "text": "The answer is 42."}
			]}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Thinking == nil {
		t.Fatal("Thinking is nil")
	}
	if req.Thinking.Mode != "enabled" {
		t.Errorf("Thinking.Mode = %q, want %q", req.Thinking.Mode, "enabled")
	}
	if req.Thinking.BudgetTokens != 4096 {
		t.Errorf("Thinking.BudgetTokens = %d, want 4096", req.Thinking.BudgetTokens)
	}

	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
	}
	assistantContent := req.Messages[1].Content
	if len(assistantContent) != 3 {
		t.Fatalf("assistant Content len = %d, want 3", len(assistantContent))
	}

	// thinking block
	thk := assistantContent[0]
	if thk.Type != ContentTypeThinking {
		t.Errorf("thinking block Type = %q, want %q", thk.Type, ContentTypeThinking)
	}
	if thk.Thinking == nil {
		t.Fatal("Thinking is nil")
	}
	if thk.Thinking.Thinking != "Let me think step by step..." {
		t.Errorf("Thinking.Thinking = %q", thk.Thinking.Thinking)
	}
	if thk.Thinking.Signature != "sig_abc" {
		t.Errorf("Thinking.Signature = %q, want %q", thk.Thinking.Signature, "sig_abc")
	}

	// redacted_thinking block
	rt := assistantContent[1]
	if rt.Type != ContentTypeRedactedThinking {
		t.Errorf("redacted_thinking block Type = %q, want %q", rt.Type, ContentTypeRedactedThinking)
	}
	if rt.RedactedThinking == nil {
		t.Fatal("RedactedThinking is nil")
	}
	if rt.RedactedThinking.Data != "redacted_blob_xyz" {
		t.Errorf("RedactedThinking.Data = %q, want %q", rt.RedactedThinking.Data, "redacted_blob_xyz")
	}

	// text block
	txt := assistantContent[2]
	if txt.Type != ContentTypeText {
		t.Errorf("text block Type = %q, want %q", txt.Type, ContentTypeText)
	}
}

func TestDecodeAnthropicRequest_ThinkingDisabled(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 512,
		"thinking": {"type": "disabled"},
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Thinking == nil {
		t.Fatal("Thinking is nil")
	}
	if req.Thinking.Mode != "disabled" {
		t.Errorf("Thinking.Mode = %q, want %q", req.Thinking.Mode, "disabled")
	}
}

func TestDecodeAnthropicRequest_StringContent(t *testing.T) {
	body := []byte(`{
		"model": "claude-haiku-4-5",
		"max_tokens": 128,
		"messages": [
			{"role": "user", "content": "Hello, world!"},
			{"role": "assistant", "content": "Hi there!"}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
	}

	userMsg := req.Messages[0]
	if len(userMsg.Content) != 1 {
		t.Fatalf("user Content len = %d, want 1", len(userMsg.Content))
	}
	if userMsg.Content[0].Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q, want %q", userMsg.Content[0].Type, ContentTypeText)
	}
	if userMsg.Content[0].Text == nil || userMsg.Content[0].Text.Text != "Hello, world!" {
		t.Errorf("Content[0].Text = %v, want 'Hello, world!'", userMsg.Content[0].Text)
	}

	assistantMsg := req.Messages[1]
	if len(assistantMsg.Content) != 1 {
		t.Fatalf("assistant Content len = %d, want 1", len(assistantMsg.Content))
	}
	if assistantMsg.Content[0].Text == nil || assistantMsg.Content[0].Text.Text != "Hi there!" {
		t.Errorf("assistant Content[0].Text = %v", assistantMsg.Content[0].Text)
	}
}

func TestDecodeAnthropicRequest_ToolChoiceAny(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 256,
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Use any tool"}]}],
		"tools": [{"name": "calc", "description": "Calculator", "input_schema": {"type": "object"}, "type": "custom"}],
		"tool_choice": {"type": "any"}
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "required" {
		t.Errorf("ToolChoice.Type = %q, want %q (any maps to required)", req.ToolChoice.Type, "required")
	}
}

func TestDecodeAnthropicRequest_ToolChoiceTool(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 256,
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Read the file"}]}],
		"tools": [{"name": "read_file", "description": "Read", "input_schema": {"type": "object"}, "type": "custom"}],
		"tool_choice": {"type": "tool", "name": "read_file"}
	}`)

	req, err := DecodeAnthropicRequest(body)
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

func TestDecodeAnthropicRequest_ToolChoiceNone(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 256,
		"messages": [{"role": "user", "content": [{"type": "text", "text": "No tools please"}]}],
		"tool_choice": {"type": "none"}
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.Type != "none" {
		t.Errorf("ToolChoice.Type = %q, want %q", req.ToolChoice.Type, "none")
	}
}

func TestDecodeAnthropicRequest_Document(t *testing.T) {
	pdfData := []byte("fake-pdf-data")
	encoded := base64.StdEncoding.EncodeToString(pdfData)

	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 512,
		"messages": [
			{"role": "user", "content": [
				{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "` + encoded + `"}},
				{"type": "text", "text": "Summarize this document"}
			]}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
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

	doc := content[0]
	if doc.Type != ContentTypeDocument {
		t.Errorf("doc.Type = %q, want %q", doc.Type, ContentTypeDocument)
	}
	if doc.Document == nil {
		t.Fatal("Document is nil")
	}
	if doc.Document.MediaType != "application/pdf" {
		t.Errorf("Document.MediaType = %q, want %q", doc.Document.MediaType, "application/pdf")
	}
	if string(doc.Document.Data) != string(pdfData) {
		t.Errorf("Document.Data mismatch")
	}
}

func TestDecodeAnthropicRequest_TopK(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 512,
		"top_k": 40,
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.TopK == nil || *req.TopK != 40 {
		t.Errorf("TopK = %v, want 40", req.TopK)
	}
}

func TestDecodeAnthropicRequest_ToolWithoutType(t *testing.T) {
	// Tools without an explicit type field should default to custom
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 256,
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Use the tool"}]}],
		"tools": [
			{
				"name": "list_files",
				"description": "List files",
				"input_schema": {"type": "object"}
			}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(req.Tools))
	}
	if req.Tools[0].Name != "list_files" {
		t.Errorf("Tools[0].Name = %q, want %q", req.Tools[0].Name, "list_files")
	}
}

func TestDecodeAnthropicRequest_InvalidJSON(t *testing.T) {
	body := []byte(`{invalid json`)
	_, err := DecodeAnthropicRequest(body)
	if err == nil {
		t.Error("expected error for invalid JSON, got nil")
	}
}

func TestDecodeAnthropicRequest_ToolResultIsError(t *testing.T) {
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 256,
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Do something"}]},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "tu_err", "name": "risky_op", "input": {}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_err", "content": [{"type": "text", "text": "Error: permission denied"}], "is_error": true}
			]}
		]
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	userMsg := req.Messages[2]
	tr := userMsg.Content[0]
	if !tr.ToolResult.IsError {
		t.Error("ToolResult.IsError = false, want true")
	}
}

func TestEncodeAnthropicRequest_Basic(t *testing.T) {
	temp := 0.7
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		SystemPrompt: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "You are helpful."}},
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
		Temperature: &temp,
		Stream:      true,
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Round-trip decode to verify
	decoded, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("failed to round-trip: %v", err)
	}

	if decoded.Model != "claude-sonnet-4-20250514" {
		t.Errorf("Model = %q, want %q", decoded.Model, "claude-sonnet-4-20250514")
	}
	if decoded.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %d, want 1024", decoded.MaxTokens)
	}
	if decoded.Temperature == nil || *decoded.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", decoded.Temperature)
	}
	if !decoded.Stream {
		t.Error("Stream = false, want true")
	}
	if len(decoded.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(decoded.SystemPrompt))
	}
	if decoded.SystemPrompt[0].Text == nil || decoded.SystemPrompt[0].Text.Text != "You are helpful." {
		t.Errorf("SystemPrompt[0].Text = %v", decoded.SystemPrompt[0].Text)
	}
	if len(decoded.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(decoded.Messages))
	}
	if decoded.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q", decoded.Messages[0].Role)
	}
	if decoded.Messages[1].Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q", decoded.Messages[1].Role)
	}
}

func TestEncodeAnthropicRequest_Tools(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)
	req := &Request{
		Model:     "claude-opus-4-5",
		MaxTokens: 512,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Use the tool"}}}},
		},
		Tools: []Tool{
			{Name: "read_file", Description: "Read a file", Parameters: schema},
		},
		ToolChoice: &ToolChoice{Type: "required"},
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	decoded, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("failed to round-trip: %v", err)
	}

	if len(decoded.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(decoded.Tools))
	}
	if decoded.Tools[0].Name != "read_file" {
		t.Errorf("Tools[0].Name = %q, want %q", decoded.Tools[0].Name, "read_file")
	}
	if decoded.Tools[0].Description != "Read a file" {
		t.Errorf("Tools[0].Description = %q", decoded.Tools[0].Description)
	}
	if decoded.ToolChoice == nil {
		t.Fatal("ToolChoice is nil after round-trip")
	}
	// "required" IR → "any" Anthropic → "required" IR
	if decoded.ToolChoice.Type != "required" {
		t.Errorf("ToolChoice.Type = %q, want %q", decoded.ToolChoice.Type, "required")
	}
}

func TestEncodeAnthropicRequest_ToolChoiceTool(t *testing.T) {
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 256,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "use read_file"}}}},
		},
		Tools: []Tool{
			{Name: "read_file", Description: "Read", Parameters: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: &ToolChoice{Type: "tool", ToolName: "read_file"},
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	decoded, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("failed to round-trip: %v", err)
	}

	if decoded.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if decoded.ToolChoice.Type != "tool" {
		t.Errorf("ToolChoice.Type = %q, want tool", decoded.ToolChoice.Type)
	}
	if decoded.ToolChoice.ToolName != "read_file" {
		t.Errorf("ToolChoice.ToolName = %q, want read_file", decoded.ToolChoice.ToolName)
	}
}

func TestDecodeAnthropicResponse_Basic(t *testing.T) {
	body := []byte(`{
		"id": "msg_abc123",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{"type": "text", "text": "Hello!"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 10,
			"output_tokens": 5,
			"cache_creation_input_tokens": 2,
			"cache_read_input_tokens": 3
		}
	}`)

	resp, err := DecodeAnthropicResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.ID != "msg_abc123" {
		t.Errorf("ID = %q, want %q", resp.ID, "msg_abc123")
	}
	if resp.Model != "claude-sonnet-4-20250514" {
		t.Errorf("Model = %q", resp.Model)
	}
	if resp.StopReason != StopReasonEndTurn {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonEndTurn)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q", resp.Content[0].Type)
	}
	if resp.Content[0].Text == nil || resp.Content[0].Text.Text != "Hello!" {
		t.Errorf("Content[0].Text = %v", resp.Content[0].Text)
	}
	if resp.Usage.PromptTokens != 15 {
		t.Errorf("Usage.PromptTokens = %d, want 15 (input_tokens + cache_creation + cache_read)", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("Usage.CompletionTokens = %d, want 5", resp.Usage.CompletionTokens)
	}
	if resp.Usage.PromptCacheWriteTokens != 2 {
		t.Errorf("Usage.PromptCacheWriteTokens = %d, want 2", resp.Usage.PromptCacheWriteTokens)
	}
	if resp.Usage.PromptCacheHitTokens != 3 {
		t.Errorf("Usage.PromptCacheHitTokens = %d, want 3", resp.Usage.PromptCacheHitTokens)
	}
}

func TestDecodeAnthropicResponse_ToolUse(t *testing.T) {
	body := []byte(`{
		"id": "msg_tool1",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [
			{"type": "text", "text": "I'll read the file."},
			{"type": "tool_use", "id": "tu_001", "name": "read_file", "input": {"path": "/etc/hosts"}}
		],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 20, "output_tokens": 15, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
	}`)

	resp, err := DecodeAnthropicResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.StopReason != StopReasonToolUse {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonToolUse)
	}
	if len(resp.Content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(resp.Content))
	}

	text := resp.Content[0]
	if text.Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q", text.Type)
	}

	tu := resp.Content[1]
	if tu.Type != ContentTypeToolUse {
		t.Errorf("Content[1].Type = %q, want tool_use", tu.Type)
	}
	if tu.ToolUse == nil {
		t.Fatal("ToolUse is nil")
	}
	if tu.ToolUse.ID != "tu_001" {
		t.Errorf("ToolUse.ID = %q, want tu_001", tu.ToolUse.ID)
	}
	if tu.ToolUse.Name != "read_file" {
		t.Errorf("ToolUse.Name = %q, want read_file", tu.ToolUse.Name)
	}
	var args map[string]string
	if err := json.Unmarshal(tu.ToolUse.Arguments, &args); err != nil {
		t.Errorf("failed to unmarshal arguments: %v", err)
	} else if args["path"] != "/etc/hosts" {
		t.Errorf("Arguments[path] = %q, want /etc/hosts", args["path"])
	}
}

func TestDecodeAnthropicResponse_WebSearchBlocks(t *testing.T) {
	body := []byte(`{
		"id": "msg_ws1",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [
			{"type": "server_tool_use", "id": "ws_1", "name": "web_search", "input": {"query": "qwer1234"}},
			{"type": "web_search_tool_result", "tool_use_id": "ws_1", "content": [{"title":"OpenAI","url":"https://openai.com"}]},
			{"type": "text", "text": "Found a source."}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 20, "output_tokens": 15, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
	}`)

	resp, err := DecodeAnthropicResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 3 {
		t.Fatalf("Content len = %d, want 3", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeServerToolUse || resp.Content[0].ServerToolUse == nil {
		t.Fatalf("Content[0] = %+v, want server_tool_use", resp.Content[0])
	}
	if resp.Content[0].ServerToolUse.ID != "ws_1" {
		t.Errorf("ServerToolUse.ID = %q, want ws_1", resp.Content[0].ServerToolUse.ID)
	}
	if resp.Content[1].Type != ContentTypeWebSearchToolResult || resp.Content[1].WebSearchToolResult == nil {
		t.Fatalf("Content[1] = %+v, want web_search_tool_result", resp.Content[1])
	}
	if resp.Content[1].WebSearchToolResult.ToolUseID != "ws_1" {
		t.Errorf("WebSearchToolResult.ToolUseID = %q, want ws_1", resp.Content[1].WebSearchToolResult.ToolUseID)
	}
	if len(resp.Content[1].WebSearchToolResult.Content) != 1 {
		t.Fatalf("WebSearchToolResult.Content len = %d, want 1", len(resp.Content[1].WebSearchToolResult.Content))
	}
	if resp.Content[1].WebSearchToolResult.Content[0].Title != "OpenAI" {
		t.Errorf("WebSearchToolResult.Content[0].Title = %q, want OpenAI", resp.Content[1].WebSearchToolResult.Content[0].Title)
	}
}

func TestDecodeAnthropicResponse_StopReasons(t *testing.T) {
	cases := []struct {
		raw  string
		want StopReason
	}{
		{"end_turn", StopReasonEndTurn},
		{"max_tokens", StopReasonMaxTokens},
		{"stop_sequence", StopReasonStopSequence},
		{"tool_use", StopReasonToolUse},
		{"pause_turn", StopReasonPauseTurn},
	}

	for _, tc := range cases {
		body := []byte(`{"id":"x","model":"m","content":[],"stop_reason":"` + tc.raw + `","usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`)
		resp, err := DecodeAnthropicResponse(body)
		if err != nil {
			t.Fatalf("stop_reason=%q: unexpected error: %v", tc.raw, err)
		}
		if resp.StopReason != tc.want {
			t.Errorf("stop_reason=%q: got %q, want %q", tc.raw, resp.StopReason, tc.want)
		}
	}
}

func TestEncodeAnthropicResponse_Basic(t *testing.T) {
	resp := &Response{
		ID:         "msg_xyz",
		Model:      "claude-sonnet-4-20250514",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Hello!"}},
		},
		Usage: Usage{
			PromptTokens:         10,
			CompletionTokens:        5,
			PromptCacheWriteTokens: 2,
			PromptCacheHitTokens:     3,
		},
	}

	body, err := EncodeAnthropicResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Parse raw JSON to verify fields
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("failed to unmarshal encoded response: %v", err)
	}

	if raw["id"] != "msg_xyz" {
		t.Errorf("id = %v, want msg_xyz", raw["id"])
	}
	if raw["type"] != "message" {
		t.Errorf("type = %v, want message", raw["type"])
	}
	if raw["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", raw["role"])
	}
	if raw["model"] != "claude-sonnet-4-20250514" {
		t.Errorf("model = %v", raw["model"])
	}
	if raw["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v, want end_turn", raw["stop_reason"])
	}

	usage, ok := raw["usage"].(map[string]interface{})
	if !ok {
		t.Fatal("usage is not an object")
	}
	if usage["input_tokens"].(float64) != 5 {
		t.Errorf("usage.input_tokens = %v, want 5 (PromptTokens - PromptCacheWriteTokens - PromptCacheHitTokens)", usage["input_tokens"])
	}
	if usage["output_tokens"].(float64) != 5 {
		t.Errorf("usage.output_tokens = %v, want 5", usage["output_tokens"])
	}
	if usage["cache_creation_input_tokens"].(float64) != 2 {
		t.Errorf("usage.cache_creation_input_tokens = %v, want 2", usage["cache_creation_input_tokens"])
	}
	if usage["cache_read_input_tokens"].(float64) != 3 {
		t.Errorf("usage.cache_read_input_tokens = %v, want 3", usage["cache_read_input_tokens"])
	}

	content, ok := raw["content"].([]interface{})
	if !ok {
		t.Fatal("content is not an array")
	}
	if len(content) != 1 {
		t.Fatalf("content len = %d, want 1", len(content))
	}
	block := content[0].(map[string]interface{})
	if block["type"] != "text" {
		t.Errorf("content[0].type = %v, want text", block["type"])
	}
	if block["text"] != "Hello!" {
		t.Errorf("content[0].text = %v, want Hello!", block["text"])
	}
}

func TestEncodeAnthropicResponse_WebSearchBlocks(t *testing.T) {
	resp := &Response{
		ID:         "msg_ws",
		Model:      "claude-sonnet-4-20250514",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{
				Type: ContentTypeServerToolUse,
				ServerToolUse: &ServerToolUseContent{
					ID:        "ws_1",
					Name:      "web_search",
					Arguments: json.RawMessage(`{"query":"qwer1234"}`),
				},
			},
			{
				Type: ContentTypeWebSearchToolResult,
				WebSearchToolResult: &WebSearchToolResultContent{
					ToolUseID: "ws_1",
					Content: []WebSearchResult{
						{Title: "OpenAI", URL: "https://openai.com"},
					},
				},
			},
		},
	}

	body, err := EncodeAnthropicResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	var content []map[string]json.RawMessage
	if err := json.Unmarshal(raw["content"], &content); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}
	if len(content) != 2 {
		t.Fatalf("content len = %d, want 2", len(content))
	}

	var firstType string
	if err := json.Unmarshal(content[0]["type"], &firstType); err != nil {
		t.Fatalf("unmarshal content[0].type: %v", err)
	}
	if firstType != "server_tool_use" {
		t.Errorf("content[0].type = %q, want server_tool_use", firstType)
	}

	var secondType string
	if err := json.Unmarshal(content[1]["type"], &secondType); err != nil {
		t.Fatalf("unmarshal content[1].type: %v", err)
	}
	if secondType != "web_search_tool_result" {
		t.Errorf("content[1].type = %q, want web_search_tool_result", secondType)
	}
}

func TestEncodeAnthropicResponse_WebSearchBlocks_EmptySuccessIncludesContentArray(t *testing.T) {
	resp := &Response{
		ID:         "msg_ws_empty",
		Model:      "claude-sonnet-4-20250514",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{
				Type: ContentTypeWebSearchToolResult,
				WebSearchToolResult: &WebSearchToolResultContent{
					ToolUseID: "ws_1",
				},
			},
		},
	}

	body, err := EncodeAnthropicResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	var content []map[string]json.RawMessage
	if err := json.Unmarshal(raw["content"], &content); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}
	if len(content) != 1 {
		t.Fatalf("content len = %d, want 1", len(content))
	}

	contentRaw, ok := content[0]["content"]
	if !ok {
		t.Fatal("content[0].content missing, want empty array")
	}
	// Strict byte-level check: nil slice marshals to "null" by default in Go,
	// but Anthropic clients expect an array. Distinguish the two explicitly.
	if string(contentRaw) != "[]" {
		t.Fatalf("content[0].content = %s, want %q", string(contentRaw), "[]")
	}
	var hits []map[string]json.RawMessage
	if err := json.Unmarshal(contentRaw, &hits); err != nil {
		t.Fatalf("unmarshal content[0].content: %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("content[0].content len = %d, want 0", len(hits))
	}
}

func TestEncodeAnthropicResponse_WebSearchBlocks_ErrorBlockNormalized(t *testing.T) {
	// Anthropic's wire format for a web_search_tool_result error is
	// content = {"type":"web_search_tool_result_error", "error_code":"..."}
	// with NO top-level is_error / error_code on the block itself.
	resp := &Response{
		ID:         "msg_ws_err",
		Model:      "claude-sonnet-4-20250514",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{
				Type: ContentTypeWebSearchToolResult,
				WebSearchToolResult: &WebSearchToolResultContent{
					ToolUseID: "ws_1",
					IsError:   true,
					ErrorCode: "max_uses_exceeded",
				},
			},
		},
	}

	body, err := EncodeAnthropicResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	var content []map[string]json.RawMessage
	if err := json.Unmarshal(raw["content"], &content); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}
	if len(content) != 1 {
		t.Fatalf("content len = %d, want 1", len(content))
	}
	block := content[0]

	if _, has := block["is_error"]; has {
		t.Errorf("web_search_tool_result block has top-level is_error, should nest error in content: %s", raw["content"])
	}
	if _, has := block["error_code"]; has {
		t.Errorf("web_search_tool_result block has top-level error_code, should nest error in content: %s", raw["content"])
	}

	var errContent map[string]string
	if err := json.Unmarshal(block["content"], &errContent); err != nil {
		t.Fatalf("unmarshal nested error content: %v (raw=%s)", err, block["content"])
	}
	if errContent["type"] != "web_search_tool_result_error" {
		t.Errorf("content.type = %q, want %q", errContent["type"], "web_search_tool_result_error")
	}
	if errContent["error_code"] != "max_uses_exceeded" {
		t.Errorf("content.error_code = %q, want max_uses_exceeded", errContent["error_code"])
	}
}

func TestEncodeAnthropicResponse_PauseTurn(t *testing.T) {
	resp := &Response{
		ID:         "msg_pause",
		Model:      "claude-sonnet-4-20250514",
		StopReason: StopReasonPauseTurn,
		Content:    []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Paused."}}},
		Usage:      Usage{PromptTokens: 5, CompletionTokens: 3},
	}

	body, err := EncodeAnthropicResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	if raw["stop_reason"] != "pause_turn" {
		t.Errorf("stop_reason = %v, want pause_turn", raw["stop_reason"])
	}
}

func TestAnthropicRequestRoundTrip(t *testing.T) {
	original := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 2048,
		"system": [{"type": "text", "text": "You are a helpful assistant."}],
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
			{"role": "assistant", "content": [{"type": "text", "text": "4"}]},
			{"role": "user", "content": [{"type": "text", "text": "Thanks!"}]}
		],
		"temperature": 0.5,
		"top_p": 0.95,
		"stop_sequences": ["STOP", "END"],
		"stream": false
	}`)

	decoded, err := DecodeAnthropicRequest(original)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	encoded, err := EncodeAnthropicRequest(decoded)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	redecoded, err := DecodeAnthropicRequest(encoded)
	if err != nil {
		t.Fatalf("re-decode error: %v", err)
	}

	if redecoded.Model != "claude-sonnet-4-20250514" {
		t.Errorf("Model = %q", redecoded.Model)
	}
	if redecoded.MaxTokens != 2048 {
		t.Errorf("MaxTokens = %d, want 2048", redecoded.MaxTokens)
	}
	if len(redecoded.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(redecoded.SystemPrompt))
	}
	if redecoded.SystemPrompt[0].Text.Text != "You are a helpful assistant." {
		t.Errorf("SystemPrompt[0] = %q", redecoded.SystemPrompt[0].Text.Text)
	}
	if len(redecoded.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(redecoded.Messages))
	}
	if redecoded.Temperature == nil || *redecoded.Temperature != 0.5 {
		t.Errorf("Temperature = %v, want 0.5", redecoded.Temperature)
	}
	if redecoded.TopP == nil || *redecoded.TopP != 0.95 {
		t.Errorf("TopP = %v, want 0.95", redecoded.TopP)
	}
	if len(redecoded.StopSequences) != 2 {
		t.Errorf("StopSequences = %v, want 2 elements", redecoded.StopSequences)
	}
}

// ---- Streaming tests ----

func TestDecodeAnthropicStreamEvent_Ping(t *testing.T) {
	event, err := DecodeAnthropicStreamEvent("ping", []byte(`{"type":"ping"}`))
	if err != nil {
		t.Fatal(err)
	}
	if event != nil {
		t.Errorf("expected nil event for ping, got %+v", event)
	}
}

func TestDecodeAnthropicStreamEvent_MessageStart(t *testing.T) {
	data := `{"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-20250514","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`
	event, err := DecodeAnthropicStreamEvent("message_start", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventStart {
		t.Errorf("type = %q, want %q", event.Type, StreamEventStart)
	}
	if event.Response == nil {
		t.Fatal("Response is nil")
	}
	if event.Response.ID != "msg_1" {
		t.Errorf("id = %q, want msg_1", event.Response.ID)
	}
	if event.Response.Model != "claude-sonnet-4-20250514" {
		t.Errorf("model = %q", event.Response.Model)
	}
	if event.Response.Usage.PromptTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", event.Response.Usage.PromptTokens)
	}
}

func TestDecodeAnthropicStreamEvent_ContentBlockStart_Text(t *testing.T) {
	data := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	event, err := DecodeAnthropicStreamEvent("content_block_start", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventContentBlockStart {
		t.Errorf("type = %q, want %q", event.Type, StreamEventContentBlockStart)
	}
	if event.Index != 0 {
		t.Errorf("index = %d, want 0", event.Index)
	}
	if event.Delta == nil {
		t.Fatal("Delta is nil")
	}
	if event.Delta.Type != ContentTypeText {
		t.Errorf("Delta.Type = %q, want text", event.Delta.Type)
	}
}

func TestDecodeAnthropicStreamEvent_ContentBlockStart_ToolUse(t *testing.T) {
	data := `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_123","name":"read_file","input":{}}}`
	event, err := DecodeAnthropicStreamEvent("content_block_start", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventContentBlockStart {
		t.Errorf("type = %q, want %q", event.Type, StreamEventContentBlockStart)
	}
	if event.Index != 1 {
		t.Errorf("index = %d, want 1", event.Index)
	}
	if event.Delta == nil || event.Delta.Type != ContentTypeToolUse {
		t.Fatalf("Delta.Type = %v, want tool_use", event.Delta)
	}
	if event.Delta.ToolUse == nil {
		t.Fatal("ToolUse is nil")
	}
	if event.Delta.ToolUse.ID != "toolu_123" {
		t.Errorf("ToolUse.ID = %q, want toolu_123", event.Delta.ToolUse.ID)
	}
	if event.Delta.ToolUse.Name != "read_file" {
		t.Errorf("ToolUse.Name = %q, want read_file", event.Delta.ToolUse.Name)
	}
}

func TestDecodeAnthropicStreamEvent_ContentBlockDelta_TextDelta(t *testing.T) {
	data := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`
	event, err := DecodeAnthropicStreamEvent("content_block_delta", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("type = %q, want %q", event.Type, StreamEventDelta)
	}
	if event.Index != 0 {
		t.Errorf("index = %d, want 0", event.Index)
	}
	if event.Delta == nil {
		t.Fatal("Delta is nil")
	}
	if event.Delta.Type != ContentTypeText {
		t.Errorf("Delta.Type = %q, want text", event.Delta.Type)
	}
	if event.Delta.Text == nil || event.Delta.Text.Text != "Hello" {
		t.Errorf("Delta.Text = %v, want Hello", event.Delta.Text)
	}
}

func TestDecodeAnthropicStreamEvent_InputJsonDelta(t *testing.T) {
	data := `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}`
	event, err := DecodeAnthropicStreamEvent("content_block_delta", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("type = %q, want delta", event.Type)
	}
	if event.Index != 1 {
		t.Errorf("index = %d, want 1", event.Index)
	}
	if event.Delta == nil || event.Delta.Type != ContentTypeToolUse {
		t.Fatalf("Delta.Type = %v, want tool_use", event.Delta)
	}
	if event.Delta.ToolUse == nil {
		t.Fatal("ToolUse is nil")
	}
	if string(event.Delta.ToolUse.Arguments) != `{"path":` {
		t.Errorf("Arguments = %q, want {\"path\":", string(event.Delta.ToolUse.Arguments))
	}
}

func TestDecodeAnthropicStreamEvent_ThinkingDelta(t *testing.T) {
	data := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}`
	event, err := DecodeAnthropicStreamEvent("content_block_delta", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("type = %q, want delta", event.Type)
	}
	if event.Delta == nil || event.Delta.Type != ContentTypeThinking {
		t.Fatalf("Delta.Type = %v, want thinking", event.Delta)
	}
	if event.Delta.Thinking == nil || event.Delta.Thinking.Thinking != "Let me think..." {
		t.Errorf("Thinking = %v, want 'Let me think...'", event.Delta.Thinking)
	}
	if event.Delta.Thinking.Signature != "" {
		t.Errorf("Signature should be empty, got %q", event.Delta.Thinking.Signature)
	}
}

func TestDecodeAnthropicStreamEvent_SignatureDelta(t *testing.T) {
	data := `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc"}}`
	event, err := DecodeAnthropicStreamEvent("content_block_delta", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("type = %q, want delta", event.Type)
	}
	if event.Delta == nil || event.Delta.Type != ContentTypeThinking {
		t.Fatalf("Delta.Type = %v, want thinking", event.Delta)
	}
	if event.Delta.Thinking == nil || event.Delta.Thinking.Signature != "sig_abc" {
		t.Errorf("Signature = %v, want sig_abc", event.Delta.Thinking)
	}
	if event.Delta.Thinking.Thinking != "" {
		t.Errorf("Thinking field should be empty, got %q", event.Delta.Thinking.Thinking)
	}
}

func TestDecodeAnthropicStreamEvent_ContentBlockStop(t *testing.T) {
	data := `{"type":"content_block_stop","index":0}`
	event, err := DecodeAnthropicStreamEvent("content_block_stop", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventContentBlockStop {
		t.Errorf("type = %q, want %q", event.Type, StreamEventContentBlockStop)
	}
	if event.Index != 0 {
		t.Errorf("index = %d, want 0", event.Index)
	}
}

func TestDecodeAnthropicStreamEvent_MessageDelta(t *testing.T) {
	data := `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}`
	event, err := DecodeAnthropicStreamEvent("message_delta", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventDelta {
		t.Errorf("type = %q, want delta", event.Type)
	}
	if event.StopReason == nil {
		t.Fatal("StopReason is nil")
	}
	if *event.StopReason != StopReasonEndTurn {
		t.Errorf("StopReason = %q, want end_turn", *event.StopReason)
	}
	if event.Usage == nil {
		t.Fatal("Usage is nil")
	}
	if event.Usage.CompletionTokens != 15 {
		t.Errorf("OutputTokens = %d, want 15", event.Usage.CompletionTokens)
	}
}

func TestDecodeAnthropicStreamEvent_MessageDelta_AllStopReasons(t *testing.T) {
	cases := []struct {
		raw  string
		want StopReason
	}{
		{"end_turn", StopReasonEndTurn},
		{"max_tokens", StopReasonMaxTokens},
		{"stop_sequence", StopReasonStopSequence},
		{"tool_use", StopReasonToolUse},
		{"pause_turn", StopReasonPauseTurn},
	}
	for _, tc := range cases {
		data := `{"type":"message_delta","delta":{"stop_reason":"` + tc.raw + `"},"usage":{"output_tokens":0}}`
		event, err := DecodeAnthropicStreamEvent("message_delta", []byte(data))
		if err != nil {
			t.Fatalf("stop_reason=%q: %v", tc.raw, err)
		}
		if event.StopReason == nil || *event.StopReason != tc.want {
			t.Errorf("stop_reason=%q: got %v, want %v", tc.raw, event.StopReason, tc.want)
		}
	}
}

func TestDecodeAnthropicStreamEvent_MessageStop(t *testing.T) {
	data := `{"type":"message_stop"}`
	event, err := DecodeAnthropicStreamEvent("message_stop", []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if event.Type != StreamEventStop {
		t.Errorf("type = %q, want %q", event.Type, StreamEventStop)
	}
}

func TestDecodeAnthropicStreamEvent_UnknownType(t *testing.T) {
	_, err := DecodeAnthropicStreamEvent("unknown_event", []byte(`{"type":"unknown_event"}`))
	if err == nil {
		t.Error("expected error for unknown event type, got nil")
	}
}

func TestEncodeAnthropicStreamEvent_MessageStart(t *testing.T) {
	event := &StreamEvent{
		Type: StreamEventStart,
		Response: &Response{
			ID:    "msg_1",
			Model: "claude-sonnet-4-20250514",
			Usage: Usage{PromptTokens: 10},
		},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "message_start" {
		t.Errorf("eventType = %q, want message_start", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw["type"] != "message_start" {
		t.Errorf("type = %v, want message_start", raw["type"])
	}
	msg, ok := raw["message"].(map[string]interface{})
	if !ok {
		t.Fatal("message field missing or not object")
	}
	if msg["id"] != "msg_1" {
		t.Errorf("message.id = %v, want msg_1", msg["id"])
	}
	if msg["model"] != "claude-sonnet-4-20250514" {
		t.Errorf("message.model = %v", msg["model"])
	}
}

func TestEncodeAnthropicStreamEvent_ContentBlockStart_Text(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventContentBlockStart,
		Index: 0,
		Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: ""}},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_start" {
		t.Errorf("eventType = %q, want content_block_start", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw["type"] != "content_block_start" {
		t.Errorf("type = %v", raw["type"])
	}
	if raw["index"].(float64) != 0 {
		t.Errorf("index = %v, want 0", raw["index"])
	}
	block, ok := raw["content_block"].(map[string]interface{})
	if !ok {
		t.Fatal("content_block missing or not object")
	}
	if block["type"] != "text" {
		t.Errorf("content_block.type = %v, want text", block["type"])
	}
}

func TestEncodeAnthropicStreamEvent_ContentBlockStart_ToolUse(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventContentBlockStart,
		Index: 1,
		Delta: &ContentPart{
			Type: ContentTypeToolUse,
			ToolUse: &ToolUseContent{
				ID:   "toolu_123",
				Name: "read_file",
			},
		},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_start" {
		t.Errorf("eventType = %q, want content_block_start", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw["index"].(float64) != 1 {
		t.Errorf("index = %v, want 1", raw["index"])
	}
	block := raw["content_block"].(map[string]interface{})
	if block["type"] != "tool_use" {
		t.Errorf("content_block.type = %v, want tool_use", block["type"])
	}
	if block["id"] != "toolu_123" {
		t.Errorf("content_block.id = %v, want toolu_123", block["id"])
	}
}

func TestEncodeAnthropicStreamEvent_ContentBlockStart_WebSearchToolResult(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventContentBlockStart,
		Index: 2,
		Delta: &ContentPart{
			Type: ContentTypeWebSearchToolResult,
			WebSearchToolResult: &WebSearchToolResultContent{
				ToolUseID: "ws_1",
				Content: []WebSearchResult{
					{Title: "OpenAI", URL: "https://openai.com"},
				},
			},
		},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_start" {
		t.Errorf("eventType = %q, want content_block_start", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	block := raw["content_block"].(map[string]interface{})
	if block["type"] != "web_search_tool_result" {
		t.Errorf("content_block.type = %v, want web_search_tool_result", block["type"])
	}
	if block["tool_use_id"] != "ws_1" {
		t.Errorf("content_block.tool_use_id = %v, want ws_1", block["tool_use_id"])
	}
}

func TestEncodeAnthropicStreamEvent_ContentBlockStart_WebSearchToolResult_EmptySuccessIncludesContentArray(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventContentBlockStart,
		Index: 2,
		Delta: &ContentPart{
			Type: ContentTypeWebSearchToolResult,
			WebSearchToolResult: &WebSearchToolResultContent{
				ToolUseID: "ws_1",
			},
		},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_start" {
		t.Errorf("eventType = %q, want content_block_start", eventType)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var block map[string]json.RawMessage
	if err := json.Unmarshal(raw["content_block"], &block); err != nil {
		t.Fatalf("unmarshal content_block: %v", err)
	}
	contentRaw, ok := block["content"]
	if !ok {
		t.Fatal("content_block.content missing, want empty array")
	}
	if string(contentRaw) != "[]" {
		t.Fatalf("content_block.content = %s, want %q", string(contentRaw), "[]")
	}
	var hits []map[string]json.RawMessage
	if err := json.Unmarshal(contentRaw, &hits); err != nil {
		t.Fatalf("unmarshal content_block.content: %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("content_block.content len = %d, want 0", len(hits))
	}
}

func TestEncodeAnthropicStreamEvent_TextDelta(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventDelta,
		Index: 0,
		Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_delta" {
		t.Errorf("eventType = %q, want content_block_delta", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw["type"] != "content_block_delta" {
		t.Errorf("type = %v", raw["type"])
	}
	if raw["index"].(float64) != 0 {
		t.Errorf("index = %v, want 0", raw["index"])
	}
	delta := raw["delta"].(map[string]interface{})
	if delta["type"] != "text_delta" {
		t.Errorf("delta.type = %v, want text_delta", delta["type"])
	}
	if delta["text"] != "Hi" {
		t.Errorf("delta.text = %v, want Hi", delta["text"])
	}
}

func TestEncodeAnthropicStreamEvent_InputJsonDelta(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventDelta,
		Index: 1,
		Delta: &ContentPart{
			Type: ContentTypeToolUse,
			ToolUse: &ToolUseContent{
				Arguments: json.RawMessage(`{"path":`),
			},
		},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_delta" {
		t.Errorf("eventType = %q, want content_block_delta", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	delta := raw["delta"].(map[string]interface{})
	if delta["type"] != "input_json_delta" {
		t.Errorf("delta.type = %v, want input_json_delta", delta["type"])
	}
	if delta["partial_json"] != `{"path":` {
		t.Errorf("delta.partial_json = %v", delta["partial_json"])
	}
}

func TestEncodeAnthropicStreamEvent_ServerToolUseInputJsonDelta(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventDelta,
		Index: 1,
		Delta: &ContentPart{
			Type: ContentTypeServerToolUse,
			ServerToolUse: &ServerToolUseContent{
				Arguments: json.RawMessage(`{"query":"qwer1234"}`),
			},
		},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_delta" {
		t.Errorf("eventType = %q, want content_block_delta", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	delta := raw["delta"].(map[string]interface{})
	if delta["type"] != "input_json_delta" {
		t.Errorf("delta.type = %v, want input_json_delta", delta["type"])
	}
	if delta["partial_json"] != `{"query":"qwer1234"}` {
		t.Errorf("delta.partial_json = %v, want {\"query\":\"qwer1234\"}", delta["partial_json"])
	}
}

func TestEncodeAnthropicStreamEvent_ThinkingDelta(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventDelta,
		Index: 0,
		Delta: &ContentPart{
			Type:     ContentTypeThinking,
			Thinking: &ThinkingContent{Thinking: "Let me think..."},
		},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_delta" {
		t.Errorf("eventType = %q, want content_block_delta", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	delta := raw["delta"].(map[string]interface{})
	if delta["type"] != "thinking_delta" {
		t.Errorf("delta.type = %v, want thinking_delta", delta["type"])
	}
	if delta["thinking"] != "Let me think..." {
		t.Errorf("delta.thinking = %v", delta["thinking"])
	}
}

func TestEncodeAnthropicStreamEvent_SignatureDelta(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventDelta,
		Index: 0,
		Delta: &ContentPart{
			Type:     ContentTypeThinking,
			Thinking: &ThinkingContent{Signature: "sig_abc"},
		},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_delta" {
		t.Errorf("eventType = %q, want content_block_delta", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	delta := raw["delta"].(map[string]interface{})
	if delta["type"] != "signature_delta" {
		t.Errorf("delta.type = %v, want signature_delta", delta["type"])
	}
	if delta["signature"] != "sig_abc" {
		t.Errorf("delta.signature = %v", delta["signature"])
	}
}

func TestEncodeAnthropicStreamEvent_MessageDelta(t *testing.T) {
	stopReason := StopReasonEndTurn
	event := &StreamEvent{
		Type:       StreamEventDelta,
		StopReason: &stopReason,
		Usage:      &Usage{CompletionTokens: 15},
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "message_delta" {
		t.Errorf("eventType = %q, want message_delta", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw["type"] != "message_delta" {
		t.Errorf("type = %v, want message_delta", raw["type"])
	}
	delta := raw["delta"].(map[string]interface{})
	if delta["stop_reason"] != "end_turn" {
		t.Errorf("delta.stop_reason = %v, want end_turn", delta["stop_reason"])
	}
	usage := raw["usage"].(map[string]interface{})
	if usage["output_tokens"].(float64) != 15 {
		t.Errorf("usage.output_tokens = %v, want 15", usage["output_tokens"])
	}
}

func TestEncodeAnthropicStreamEvent_ContentBlockStop(t *testing.T) {
	event := &StreamEvent{
		Type:  StreamEventContentBlockStop,
		Index: 2,
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "content_block_stop" {
		t.Errorf("eventType = %q, want content_block_stop", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw["type"] != "content_block_stop" {
		t.Errorf("type = %v, want content_block_stop", raw["type"])
	}
	if raw["index"].(float64) != 2 {
		t.Errorf("index = %v, want 2", raw["index"])
	}
}

func TestEncodeAnthropicStreamEvent_MessageStop(t *testing.T) {
	event := &StreamEvent{
		Type: StreamEventStop,
	}
	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	if eventType != "message_stop" {
		t.Errorf("eventType = %q, want message_stop", eventType)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw["type"] != "message_stop" {
		t.Errorf("type = %v, want message_stop", raw["type"])
	}
}

func TestAnthropicStreamRoundTrip_TextSequence(t *testing.T) {
	// Simulate a complete streaming sequence: message_start, content_block_start,
	// content_block_delta, content_block_stop, message_delta, message_stop
	events := []struct {
		eventType string
		data      string
	}{
		{"message_start", `{"type":"message_start","message":{"id":"msg_rt","model":"claude-sonnet-4-20250514","role":"assistant","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}

	for _, ev := range events {
		decoded, err := DecodeAnthropicStreamEvent(ev.eventType, []byte(ev.data))
		if err != nil {
			t.Fatalf("eventType=%q decode error: %v", ev.eventType, err)
		}
		if decoded == nil {
			t.Fatalf("eventType=%q: got nil event", ev.eventType)
		}
		// Re-encode and verify no error
		reEncType, reEncData, err := EncodeAnthropicStreamEvent(decoded)
		if err != nil {
			t.Fatalf("eventType=%q re-encode error: %v", ev.eventType, err)
		}
		// Re-decode and compare types
		reDecoded, err := DecodeAnthropicStreamEvent(reEncType, reEncData)
		if err != nil {
			t.Fatalf("eventType=%q re-decode error: %v", ev.eventType, err)
		}
		if reDecoded.Type != decoded.Type {
			t.Errorf("eventType=%q: round-trip type mismatch: got %q, want %q", ev.eventType, reDecoded.Type, decoded.Type)
		}
	}
}

func TestAnthropicResponseRoundTrip(t *testing.T) {
	original := []byte(`{
		"id": "msg_roundtrip",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [
			{"type": "thinking", "thinking": "Let me think...", "signature": "sig_xyz"},
			{"type": "text", "text": "The answer is 42."}
		],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50,
			"cache_creation_input_tokens": 10,
			"cache_read_input_tokens": 5
		}
	}`)

	decoded, err := DecodeAnthropicResponse(original)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}

	encoded, err := EncodeAnthropicResponse(decoded)
	if err != nil {
		t.Fatalf("encode error: %v", err)
	}

	redecoded, err := DecodeAnthropicResponse(encoded)
	if err != nil {
		t.Fatalf("re-decode error: %v", err)
	}

	if redecoded.ID != "msg_roundtrip" {
		t.Errorf("ID = %q", redecoded.ID)
	}
	if redecoded.Model != "claude-sonnet-4-20250514" {
		t.Errorf("Model = %q", redecoded.Model)
	}
	if redecoded.StopReason != StopReasonEndTurn {
		t.Errorf("StopReason = %q", redecoded.StopReason)
	}
	if len(redecoded.Content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(redecoded.Content))
	}

	// thinking block
	thk := redecoded.Content[0]
	if thk.Type != ContentTypeThinking {
		t.Errorf("Content[0].Type = %q, want thinking", thk.Type)
	}
	if thk.Thinking == nil || thk.Thinking.Thinking != "Let me think..." {
		t.Errorf("Content[0].Thinking = %v", thk.Thinking)
	}
	if thk.Thinking.Signature != "sig_xyz" {
		t.Errorf("Thinking.Signature = %q, want sig_xyz", thk.Thinking.Signature)
	}

	// text block
	txt := redecoded.Content[1]
	if txt.Type != ContentTypeText {
		t.Errorf("Content[1].Type = %q, want text", txt.Type)
	}
	if txt.Text == nil || txt.Text.Text != "The answer is 42." {
		t.Errorf("Content[1].Text = %v", txt.Text)
	}

	if redecoded.Usage.PromptTokens != 115 {
		t.Errorf("Usage.PromptTokens = %d, want 115 (input_tokens + cache_creation + cache_read)", redecoded.Usage.PromptTokens)
	}
	if redecoded.Usage.CompletionTokens != 50 {
		t.Errorf("Usage.CompletionTokens = %d, want 50", redecoded.Usage.CompletionTokens)
	}
	if redecoded.Usage.PromptCacheWriteTokens != 10 {
		t.Errorf("Usage.PromptCacheWriteTokens = %d, want 10", redecoded.Usage.PromptCacheWriteTokens)
	}
	if redecoded.Usage.PromptCacheHitTokens != 5 {
		t.Errorf("Usage.PromptCacheHitTokens = %d, want 5", redecoded.Usage.PromptCacheHitTokens)
	}
}

// --- Tests for disable_parallel_tool_use ---

func TestDecodeAnthropicRequest_DisableParallelToolUseTrue(t *testing.T) {
	// disable=true → AllowParallelCalls=false
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Hello"}],
		"tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.AllowParallelCalls == nil {
		t.Fatal("AllowParallelCalls is nil, expected non-nil")
	}
	if *req.ToolChoice.AllowParallelCalls {
		t.Errorf("AllowParallelCalls = true, want false (disable_parallel_tool_use=true → allow=false)")
	}
}

func TestDecodeAnthropicRequest_DisableParallelToolUseFalse(t *testing.T) {
	// disable=false → AllowParallelCalls=true
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Hello"}],
		"tool_choice": {"type": "auto", "disable_parallel_tool_use": false}
	}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if req.ToolChoice.AllowParallelCalls == nil {
		t.Fatal("AllowParallelCalls is nil, expected non-nil")
	}
	if !*req.ToolChoice.AllowParallelCalls {
		t.Errorf("AllowParallelCalls = false, want true (disable_parallel_tool_use=false → allow=true)")
	}
}

func TestDecodeAnthropicRequest_NoDisableParallelToolUse(t *testing.T) {
	// Absent disable_parallel_tool_use → AllowParallelCalls nil
	body := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Hello"}],
		"tool_choice": {"type": "auto"}
	}`)

	req, err := DecodeAnthropicRequest(body)
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

func TestEncodeAnthropicRequest_DisableParallelToolUseFromAllowFalse(t *testing.T) {
	// AllowParallelCalls=false → disable_parallel_tool_use=true
	allow := false
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type:               "auto",
			AllowParallelCalls: &allow,
		},
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	tcRaw, ok := m["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing from encoded request")
	}

	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}

	dptuRaw, ok := tc["disable_parallel_tool_use"]
	if !ok {
		t.Fatal("disable_parallel_tool_use missing from tool_choice")
	}
	var dptu bool
	if err := json.Unmarshal(dptuRaw, &dptu); err != nil {
		t.Fatalf("unmarshal disable_parallel_tool_use: %v", err)
	}
	if !dptu {
		t.Errorf("disable_parallel_tool_use = false, want true (AllowParallelCalls=false → disable=true)")
	}
}

func TestEncodeAnthropicRequest_DisableParallelToolUseFromAllowTrue(t *testing.T) {
	// AllowParallelCalls=true → disable_parallel_tool_use=false
	allow := true
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type:               "auto",
			AllowParallelCalls: &allow,
		},
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	tcRaw, ok := m["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing from encoded request")
	}

	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}

	dptuRaw, ok := tc["disable_parallel_tool_use"]
	if !ok {
		t.Fatal("disable_parallel_tool_use missing from tool_choice")
	}
	var dptu bool
	if err := json.Unmarshal(dptuRaw, &dptu); err != nil {
		t.Fatalf("unmarshal disable_parallel_tool_use: %v", err)
	}
	if dptu {
		t.Errorf("disable_parallel_tool_use = true, want false (AllowParallelCalls=true → disable=false)")
	}
}

func TestEncodeAnthropicRequest_NoDisableParallelToolUse(t *testing.T) {
	// Nil AllowParallelCalls → disable_parallel_tool_use should not appear
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type: "auto",
		},
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if tcRaw, ok := m["tool_choice"]; ok {
		var tc map[string]json.RawMessage
		if err := json.Unmarshal(tcRaw, &tc); err == nil {
			if _, ok := tc["disable_parallel_tool_use"]; ok {
				t.Error("disable_parallel_tool_use present in tool_choice, want absent")
			}
		}
	}
}

func TestDecodeEncodeAnthropicRequest_DisableParallelToolUseRoundTrip(t *testing.T) {
	// Round-trip: disable=true → AllowParallelCalls=false → encode → disable=true
	original := []byte(`{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Hello"}],
		"tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
	}`)

	req, err := DecodeAnthropicRequest(original)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if req.ToolChoice == nil || req.ToolChoice.AllowParallelCalls == nil {
		t.Fatal("AllowParallelCalls not decoded")
	}
	if *req.ToolChoice.AllowParallelCalls {
		t.Fatalf("AllowParallelCalls decoded as true, want false")
	}

	encoded, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	tcRaw, ok := m["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing after round-trip")
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}
	dptuRaw, ok := tc["disable_parallel_tool_use"]
	if !ok {
		t.Fatal("disable_parallel_tool_use missing after round-trip")
	}
	var dptu bool
	if err := json.Unmarshal(dptuRaw, &dptu); err != nil {
		t.Fatalf("unmarshal disable_parallel_tool_use: %v", err)
	}
	if !dptu {
		t.Errorf("disable_parallel_tool_use = false after round-trip, want true")
	}
}

// --- Tests for AllowedToolNames degradation in Anthropic ---

func TestEncodeAnthropicRequest_AllowedToolNamesSingleTool(t *testing.T) {
	// Single allowed tool name → tool_choice.type="tool", tool_choice.name=<name>
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type:             "auto",
			AllowedToolNames: []string{"my_tool"},
		},
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	tcRaw, ok := m["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing from encoded request")
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}

	var tcType string
	if err := json.Unmarshal(tc["type"], &tcType); err != nil {
		t.Fatalf("unmarshal tool_choice.type: %v", err)
	}
	if tcType != "tool" {
		t.Errorf("tool_choice.type = %q, want tool", tcType)
	}

	var tcName string
	if err := json.Unmarshal(tc["name"], &tcName); err != nil {
		t.Fatalf("unmarshal tool_choice.name: %v", err)
	}
	if tcName != "my_tool" {
		t.Errorf("tool_choice.name = %q, want my_tool", tcName)
	}
}

func TestEncodeAnthropicRequest_WebSearchToolDropsResponsesOnlyFields(t *testing.T) {
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		Tools: []Tool{
			{
				Type: "web_search",
				Name: "web_search",
				ExtraFields: map[string]json.RawMessage{
					"filters":             json.RawMessage(`{"allowed_domains":["openai.com"]}`),
					"search_context_size": json.RawMessage(`"high"`),
				},
			},
		},
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(m["tools"], &tools); err != nil {
		t.Fatalf("unmarshal tools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(tools))
	}

	var toolType string
	if err := json.Unmarshal(tools[0]["type"], &toolType); err != nil {
		t.Fatalf("unmarshal tools[0].type: %v", err)
	}
	if toolType != "web_search_20250305" {
		t.Errorf("tools[0].type = %q, want web_search_20250305", toolType)
	}

	var allowedDomains []string
	if err := json.Unmarshal(tools[0]["allowed_domains"], &allowedDomains); err != nil {
		t.Fatalf("unmarshal tools[0].allowed_domains: %v", err)
	}
	if len(allowedDomains) != 1 || allowedDomains[0] != "openai.com" {
		t.Errorf("tools[0].allowed_domains = %v, want [openai.com]", allowedDomains)
	}

	if _, ok := tools[0]["filters"]; ok {
		t.Fatal("tools[0].filters present, want dropped")
	}
	if _, ok := tools[0]["search_context_size"]; ok {
		t.Fatal("tools[0].search_context_size present, want dropped")
	}
}

func TestEncodeAnthropicRequest_AllowedToolNamesMultiToolDropped(t *testing.T) {
	// Multiple allowed tool names → cannot represent, silently drop (type stays "auto")
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type:             "auto",
			AllowedToolNames: []string{"tool_a", "tool_b"},
		},
	}

	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	tcRaw, ok := m["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing from encoded request")
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}

	var tcType string
	if err := json.Unmarshal(tc["type"], &tcType); err != nil {
		t.Fatalf("unmarshal tool_choice.type: %v", err)
	}
	// Should still be "auto" because multi-tool allowlist is silently dropped
	if tcType != "auto" {
		t.Errorf("tool_choice.type = %q, want auto (multi-tool allowlist silently dropped)", tcType)
	}
}

// TestEncodeAnthropicRequest_ParallelToolCallsNoToolChoice verifies that when a
// ToolChoice has an empty Type (synthesised to carry AllowParallelCalls only),
// no tool_choice field is emitted in the Anthropic request. Anthropic requires
// disable_parallel_tool_use to be nested inside a valid tool_choice, so the
// parallel info is silently dropped.
func TestEncodeAnthropicRequest_ParallelToolCallsNoToolChoice(t *testing.T) {
	trueVal := true
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}},
		},
		// ToolChoice with empty Type but AllowParallelCalls set — as produced when
		// an OpenAI inbound request has parallel_tool_calls but no tool_choice.
		ToolChoice: &ToolChoice{
			Type:               "",
			AllowParallelCalls: &trueVal,
		},
	}

	encoded, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// tool_choice must NOT be present (and must not be {"type":"",...})
	if raw, ok := m["tool_choice"]; ok {
		// null is acceptable (omitempty); a non-null value is a bug
		if string(raw) != "null" {
			t.Errorf("tool_choice should not be emitted when ToolChoice.Type is empty, got: %s", raw)
		}
	}
}

// TestDecodeOpenAIChatThenEncodeAnthropicRequest_ParallelToolCallsNoToolChoice is a
// cross-protocol test: decode an OpenAI Chat request that has parallel_tool_calls
// but no tool_choice, then encode to Anthropic and verify no tool_choice is emitted.
func TestDecodeOpenAIChatThenEncodeAnthropicRequest_ParallelToolCallsNoToolChoice(t *testing.T) {
	body := []byte(`{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "hi"}],
		"parallel_tool_calls": true
	}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	req.Model = "claude-sonnet-4-20250514"

	encoded, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// tool_choice must NOT be present as a non-null value
	if raw, ok := m["tool_choice"]; ok && string(raw) != "null" {
		t.Errorf("tool_choice should not be emitted, got: %s", raw)
	}
}

// --- Anthropic SSE error event tests ---

func TestDecodeAnthropicStreamEvent_Error(t *testing.T) {
	data := []byte(`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)

	event, err := DecodeAnthropicStreamEvent("error", data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if event == nil {
		t.Fatal("event is nil — error event should not crash the stream")
	}
	if event.Type != StreamEventError {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventError)
	}
	if event.Error == nil {
		t.Fatal("Error is nil")
	}
	if event.Error.Type != "overloaded_error" {
		t.Errorf("Error.Type = %q, want %q", event.Error.Type, "overloaded_error")
	}
	if event.Error.Message != "Overloaded" {
		t.Errorf("Error.Message = %q, want %q", event.Error.Message, "Overloaded")
	}
}

func TestDecodeAnthropicStreamEvent_Error_NoCrash(t *testing.T) {
	// Previously, the "error" event type fell into the default case and returned
	// an error (crashing the stream). Verify it no longer does.
	data := []byte(`{"type":"error","error":{"type":"api_error","message":"Internal error"}}`)

	event, err := DecodeAnthropicStreamEvent("error", data)
	if err != nil {
		t.Fatalf("error event should not return an error, got: %v", err)
	}
	if event == nil {
		t.Fatal("event should not be nil")
	}
	if event.Type != StreamEventError {
		t.Errorf("Type = %q, want %q", event.Type, StreamEventError)
	}
}

func TestEncodeAnthropicStreamEvent_Error(t *testing.T) {
	event := &StreamEvent{
		Type: StreamEventError,
		Error: &StreamError{
			Type:    "overloaded_error",
			Message: "Overloaded",
		},
	}

	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if eventType != "error" {
		t.Errorf("eventType = %q, want %q", eventType, "error")
	}

	var raw struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw.Type != "error" {
		t.Errorf("type = %q, want %q", raw.Type, "error")
	}
	if raw.Error.Type != "overloaded_error" {
		t.Errorf("error.type = %q, want %q", raw.Error.Type, "overloaded_error")
	}
	if raw.Error.Message != "Overloaded" {
		t.Errorf("error.message = %q, want %q", raw.Error.Message, "Overloaded")
	}
}

// --- Anthropic refusal weak mapping tests ---

func TestEncodeAnthropicResponse_RefusalToText(t *testing.T) {
	// IR refusal content should be encoded as text in Anthropic (weak mapping)
	resp := &Response{
		ID:         "msg_123",
		Model:      "claude-sonnet-4-20250514",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{Type: ContentTypeRefusal, Refusal: &RefusalContent{Refusal: "I cannot help with that."}},
		},
		Usage: Usage{PromptTokens: 10, CompletionTokens: 5},
	}

	data, err := EncodeAnthropicResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(raw.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(raw.Content))
	}
	// Refusal is degraded to text
	if raw.Content[0].Type != "text" {
		t.Errorf("content[0].type = %q, want %q", raw.Content[0].Type, "text")
	}
	if raw.Content[0].Text != "I cannot help with that." {
		t.Errorf("content[0].text = %q, want %q", raw.Content[0].Text, "I cannot help with that.")
	}
}

func TestEncodeAnthropicStreamEvent_RefusalDeltaToText(t *testing.T) {
	// IR refusal delta should be encoded as text_delta in Anthropic streaming (weak mapping)
	event := &StreamEvent{
		Type: StreamEventDelta,
		Delta: &ContentPart{
			Type:    ContentTypeRefusal,
			Refusal: &RefusalContent{Refusal: "I refuse"},
		},
	}

	eventType, data, err := EncodeAnthropicStreamEvent(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if eventType != "content_block_delta" {
		t.Errorf("eventType = %q, want %q", eventType, "content_block_delta")
	}

	var raw struct {
		Delta struct {
			Type string  `json:"type"`
			Text *string `json:"text"`
		} `json:"delta"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if raw.Delta.Type != "text_delta" {
		t.Errorf("delta.type = %q, want %q", raw.Delta.Type, "text_delta")
	}
	if raw.Delta.Text == nil {
		t.Fatal("delta.text is nil")
	}
	if *raw.Delta.Text != "I refuse" {
		t.Errorf("delta.text = %q, want %q", *raw.Delta.Text, "I refuse")
	}
}

// --- Cross-protocol refusal tests ---

func TestCrossProtocol_RefusalOpenAIChatToAnthropic(t *testing.T) {
	// Decode refusal from OpenAI Chat, encode to Anthropic -> should be text
	body := []byte(`{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"refusal": "I cannot help with that."
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// Encode to Anthropic
	anthropicData, err := EncodeAnthropicResponse(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var raw struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(anthropicData, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(raw.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(raw.Content))
	}
	// Refusal degraded to text
	if raw.Content[0].Type != "text" {
		t.Errorf("content[0].type = %q, want %q (weak mapping)", raw.Content[0].Type, "text")
	}
	if raw.Content[0].Text != "I cannot help with that." {
		t.Errorf("content[0].text = %q, want %q", raw.Content[0].Text, "I cannot help with that.")
	}
}

// TestEncodeAnthropicRequest_ThinkingConfig_Phase2_Degradation verifies that
// Phase 2 thinking fields (IncludeThoughts, Level) are silently dropped when
// encoding to Anthropic, which has no native equivalent.
func TestEncodeAnthropicRequest_ThinkingConfig_Phase2_Degradation(t *testing.T) {
	inclTrue := true
	req := &Request{
		Model: "claude-sonnet-4-20250514",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
		Thinking: &ThinkingConfig{
			Mode:            "enabled",
			BudgetTokens:    4096,
			IncludeThoughts: &inclTrue,
			Level:           "HIGH",
		},
	}

	data, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// thinking config should be present with the native Anthropic shape
	thinkingRaw, ok := raw["thinking"]
	if !ok {
		t.Fatal("thinking field missing from Anthropic request")
	}
	thinkingMap, ok := thinkingRaw.(map[string]interface{})
	if !ok {
		t.Fatalf("thinking is not an object: %T", thinkingRaw)
	}
	if thinkingMap["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want enabled", thinkingMap["type"])
	}

	// IncludeThoughts and Level have no Anthropic equivalent — should be absent
	if _, ok := raw["include_thoughts"]; ok {
		t.Error("include_thoughts should not appear in Anthropic request (silently dropped)")
	}
	if _, ok := raw["includeThoughts"]; ok {
		t.Error("includeThoughts should not appear in Anthropic request (silently dropped)")
	}
	if _, ok := raw["level"]; ok {
		t.Error("level should not appear in Anthropic request (silently dropped)")
	}
	if _, ok := raw["thinkingLevel"]; ok {
		t.Error("thinkingLevel should not appear in Anthropic request (silently dropped)")
	}
	// Also should not appear nested inside the thinking object
	if _, ok := thinkingMap["include_thoughts"]; ok {
		t.Error("thinking.include_thoughts should not appear in Anthropic request (silently dropped)")
	}
	if _, ok := thinkingMap["level"]; ok {
		t.Error("thinking.level should not appear in Anthropic request (silently dropped)")
	}
}
