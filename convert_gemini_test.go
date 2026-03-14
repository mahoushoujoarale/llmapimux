package llmapimux

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	gemini "github.com/llmapimux/llmapimux/protocol/gemini"
)

func TestDecodeGeminiRequest_Basic(t *testing.T) {
	body := []byte(`{
		"contents": [
			{"role": "user", "parts": [{"text": "Hello"}]},
			{"role": "model", "parts": [{"text": "Hi there!"}]},
			{"role": "user", "parts": [{"text": "How are you?"}]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Model != "gemini-2.5-pro" {
		t.Errorf("Model = %q, want %q", req.Model, "gemini-2.5-pro")
	}
	if len(req.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(req.Messages))
	}
	if req.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", req.Messages[0].Role, RoleUser)
	}
	if req.Messages[0].Content[0].Type != ContentTypeText || req.Messages[0].Content[0].Text.Text != "Hello" {
		t.Errorf("Messages[0].Content[0] = %v, want text 'Hello'", req.Messages[0].Content[0])
	}
	if req.Messages[1].Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", req.Messages[1].Role, RoleAssistant)
	}
	if req.Messages[1].Content[0].Text.Text != "Hi there!" {
		t.Errorf("Messages[1].Content[0].Text = %q, want %q", req.Messages[1].Content[0].Text.Text, "Hi there!")
	}
	if req.Messages[2].Role != RoleUser {
		t.Errorf("Messages[2].Role = %q, want %q", req.Messages[2].Role, RoleUser)
	}
}

func TestDecodeGeminiRequest_StreamURL(t *testing.T) {
	body := []byte(`{"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.0-flash:streamGenerateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model != "gemini-2.0-flash" {
		t.Errorf("Model = %q, want %q", req.Model, "gemini-2.0-flash")
	}
}

func TestDecodeGeminiRequest_BetaURL(t *testing.T) {
	body := []byte(`{"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}`)

	req, err := DecodeGeminiRequest("/v1beta/models/gemini-2.5-pro-preview:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model != "gemini-2.5-pro-preview" {
		t.Errorf("Model = %q, want %q", req.Model, "gemini-2.5-pro-preview")
	}
}

func TestDecodeGeminiRequest_InvalidURL(t *testing.T) {
	body := []byte(`{"contents": []}`)

	_, err := DecodeGeminiRequest("/v1/invalid/path", body)
	if err == nil {
		t.Fatal("expected error for invalid URL, got nil")
	}
}

func TestDecodeGeminiRequest_SystemInstruction(t *testing.T) {
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"systemInstruction": {"parts": [{"text": "You are a helpful assistant."}]}
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.SystemPrompt) != 1 {
		t.Fatalf("SystemPrompt len = %d, want 1", len(req.SystemPrompt))
	}
	if req.SystemPrompt[0].Type != ContentTypeText {
		t.Errorf("SystemPrompt[0].Type = %q, want %q", req.SystemPrompt[0].Type, ContentTypeText)
	}
	if req.SystemPrompt[0].Text.Text != "You are a helpful assistant." {
		t.Errorf("SystemPrompt[0].Text = %q, want %q", req.SystemPrompt[0].Text.Text, "You are a helpful assistant.")
	}
}

func TestDecodeGeminiRequest_FunctionCall(t *testing.T) {
	body := []byte(`{
		"contents": [
			{"role": "user", "parts": [{"text": "Read the file"}]},
			{"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {"path": "/tmp/test.txt"}, "id": "call-123"}}]},
			{"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {"content": "file data"}, "id": "call-123"}}]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(req.Messages))
	}

	// Assistant message with FunctionCall
	assistMsg := req.Messages[1]
	if assistMsg.Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", assistMsg.Role, RoleAssistant)
	}
	if len(assistMsg.Content) != 1 {
		t.Fatalf("Messages[1].Content len = %d, want 1", len(assistMsg.Content))
	}
	if assistMsg.Content[0].Type != ContentTypeToolUse {
		t.Errorf("Messages[1].Content[0].Type = %q, want %q", assistMsg.Content[0].Type, ContentTypeToolUse)
	}
	if assistMsg.Content[0].ToolUse.ID != "call-123" {
		t.Errorf("ToolUse.ID = %q, want %q", assistMsg.Content[0].ToolUse.ID, "call-123")
	}
	if assistMsg.Content[0].ToolUse.Name != "read_file" {
		t.Errorf("ToolUse.Name = %q, want %q", assistMsg.Content[0].ToolUse.Name, "read_file")
	}

	// Tool message with FunctionResponse
	toolMsg := req.Messages[2]
	if toolMsg.Role != RoleTool {
		t.Errorf("Messages[2].Role = %q, want %q", toolMsg.Role, RoleTool)
	}
	if len(toolMsg.Content) != 1 {
		t.Fatalf("Messages[2].Content len = %d, want 1", len(toolMsg.Content))
	}
	if toolMsg.Content[0].Type != ContentTypeToolResult {
		t.Errorf("Messages[2].Content[0].Type = %q, want %q", toolMsg.Content[0].Type, ContentTypeToolResult)
	}
	if toolMsg.Content[0].ToolResult.ToolUseID != "call-123" {
		t.Errorf("ToolResult.ToolUseID = %q, want %q", toolMsg.Content[0].ToolResult.ToolUseID, "call-123")
	}
}

func TestDecodeGeminiRequest_FunctionCallNoID(t *testing.T) {
	body := []byte(`{
		"contents": [
			{"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	tu := req.Messages[0].Content[0].ToolUse
	if tu == nil {
		t.Fatal("expected ToolUse, got nil")
	}
	if tu.ID == "" {
		t.Error("expected synthetic ID for FunctionCall without ID, got empty string")
	}
	if !strings.HasPrefix(tu.ID, "call_") {
		t.Errorf("synthetic ID = %q, want prefix 'call_'", tu.ID)
	}
}

func TestDecodeGeminiRequest_FunctionCallAndResponseNoIDCorrelate(t *testing.T) {
	body := []byte(`{
		"contents": [
			{"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}]},
			{"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp": "72F"}}}]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
	}

	toolUse := req.Messages[0].Content[0].ToolUse
	if toolUse == nil {
		t.Fatal("expected ToolUse, got nil")
	}
	toolResult := req.Messages[1].Content[0].ToolResult
	if toolResult == nil {
		t.Fatal("expected ToolResult, got nil")
	}
	if toolResult.ToolUseID != toolUse.ID {
		t.Fatalf("ToolResult.ToolUseID = %q, want %q", toolResult.ToolUseID, toolUse.ID)
	}
}

func TestDecodeGeminiRequest_FunctionResponsesWithoutIDsCorrelateByNameAndOrder(t *testing.T) {
	body := []byte(`{
		"contents": [
			{"role": "model", "parts": [
				{"functionCall": {"name": "lookup_weather", "args": {"city": "NYC"}}},
				{"functionCall": {"name": "lookup_weather", "args": {"city": "SF"}}}
			]},
			{"role": "user", "parts": [
				{"functionResponse": {"name": "lookup_weather", "response": {"temp": "72F"}}},
				{"functionResponse": {"name": "lookup_weather", "response": {"temp": "65F"}}}
			]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 2 {
		t.Fatalf("Messages len = %d, want 2", len(req.Messages))
	}

	firstCall := req.Messages[0].Content[0].ToolUse
	secondCall := req.Messages[0].Content[1].ToolUse
	if firstCall == nil || secondCall == nil {
		t.Fatal("expected ToolUse parts, got nil")
	}
	firstResult := req.Messages[1].Content[0].ToolResult
	secondResult := req.Messages[1].Content[1].ToolResult
	if firstResult == nil || secondResult == nil {
		t.Fatal("expected ToolResult parts, got nil")
	}

	if firstResult.ToolUseID != firstCall.ID {
		t.Fatalf("first ToolResult.ToolUseID = %q, want %q", firstResult.ToolUseID, firstCall.ID)
	}
	if secondResult.ToolUseID != secondCall.ID {
		t.Fatalf("second ToolResult.ToolUseID = %q, want %q", secondResult.ToolUseID, secondCall.ID)
	}
}

func TestDecodeGeminiRequest_Image(t *testing.T) {
	imgData := base64.StdEncoding.EncodeToString([]byte("fake-png-data"))
	body := []byte(`{
		"contents": [
			{"role": "user", "parts": [
				{"inlineData": {"mimeType": "image/png", "data": "` + imgData + `"}},
				{"text": "What is this?"}
			]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Messages) != 1 {
		t.Fatalf("Messages len = %d, want 1", len(req.Messages))
	}
	msg := req.Messages[0]
	if len(msg.Content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(msg.Content))
	}

	// Image part
	if msg.Content[0].Type != ContentTypeImage {
		t.Errorf("Content[0].Type = %q, want %q", msg.Content[0].Type, ContentTypeImage)
	}
	if msg.Content[0].Image == nil {
		t.Fatal("Content[0].Image is nil")
	}
	if msg.Content[0].Image.MediaType != "image/png" {
		t.Errorf("Image.MediaType = %q, want %q", msg.Content[0].Image.MediaType, "image/png")
	}
	if string(msg.Content[0].Image.Data) != "fake-png-data" {
		t.Errorf("Image.Data = %q, want %q", string(msg.Content[0].Image.Data), "fake-png-data")
	}

	// Text part
	if msg.Content[1].Type != ContentTypeText {
		t.Errorf("Content[1].Type = %q, want %q", msg.Content[1].Type, ContentTypeText)
	}
	if msg.Content[1].Text.Text != "What is this?" {
		t.Errorf("Content[1].Text = %q, want %q", msg.Content[1].Text.Text, "What is this?")
	}
}

func TestDecodeGeminiRequest_FileData(t *testing.T) {
	body := []byte(`{
		"contents": [
			{"role": "user", "parts": [
				{"fileData": {"mimeType": "image/jpeg", "fileUri": "https://example.com/image.jpg"}},
				{"text": "Describe this"}
			]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	img := req.Messages[0].Content[0]
	if img.Type != ContentTypeImage {
		t.Errorf("Type = %q, want %q", img.Type, ContentTypeImage)
	}
	if img.Image.URL != "https://example.com/image.jpg" {
		t.Errorf("Image.URL = %q, want %q", img.Image.URL, "https://example.com/image.jpg")
	}
	if img.Image.MediaType != "image/jpeg" {
		t.Errorf("Image.MediaType = %q, want %q", img.Image.MediaType, "image/jpeg")
	}
}

func TestDecodeGeminiRequest_PDFInlineData(t *testing.T) {
	pdfData := base64.StdEncoding.EncodeToString([]byte("fake-pdf-data"))
	body := []byte(`{
		"contents": [
			{"role": "user", "parts": [
				{"inlineData": {"mimeType": "application/pdf", "data": "` + pdfData + `"}}
			]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	doc := req.Messages[0].Content[0]
	if doc.Type != ContentTypeDocument {
		t.Errorf("Type = %q, want %q", doc.Type, ContentTypeDocument)
	}
	if doc.Document == nil {
		t.Fatal("Document is nil")
	}
	if doc.Document.MediaType != "application/pdf" {
		t.Errorf("MediaType = %q, want %q", doc.Document.MediaType, "application/pdf")
	}
	if string(doc.Document.Data) != "fake-pdf-data" {
		t.Errorf("Data = %q, want %q", string(doc.Document.Data), "fake-pdf-data")
	}
}

func TestDecodeGeminiRequest_PDFFileData(t *testing.T) {
	body := []byte(`{
		"contents": [
			{"role": "user", "parts": [
				{"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/doc.pdf"}}
			]}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	doc := req.Messages[0].Content[0]
	if doc.Type != ContentTypeDocument {
		t.Errorf("Type = %q, want %q", doc.Type, ContentTypeDocument)
	}
	if doc.Document.URL != "gs://bucket/doc.pdf" {
		t.Errorf("URL = %q, want %q", doc.Document.URL, "gs://bucket/doc.pdf")
	}
}

func TestDecodeGeminiRequest_Tools(t *testing.T) {
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"tools": [{
			"functionDeclarations": [{
				"name": "read_file",
				"description": "Read a file",
				"parameters": {
					"type": "OBJECT",
					"properties": {
						"path": {"type": "STRING", "description": "File path"}
					},
					"required": ["path"]
				}
			}]
		}]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(req.Tools))
	}
	tool := req.Tools[0]
	if tool.Name != "read_file" {
		t.Errorf("Tool.Name = %q, want %q", tool.Name, "read_file")
	}
	if tool.Description != "Read a file" {
		t.Errorf("Tool.Description = %q, want %q", tool.Description, "Read a file")
	}

	// Verify schema was converted from Gemini to JSON Schema
	var params map[string]interface{}
	if err := json.Unmarshal(tool.Parameters, &params); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	if params["type"] != "object" {
		t.Errorf("params.type = %v, want %q", params["type"], "object")
	}
	props := params["properties"].(map[string]interface{})
	pathProp := props["path"].(map[string]interface{})
	if pathProp["type"] != "string" {
		t.Errorf("params.properties.path.type = %v, want %q", pathProp["type"], "string")
	}
}

func TestDecodeGeminiRequest_ToolConfig(t *testing.T) {
	tests := []struct {
		mode     string
		wantType string
	}{
		{"AUTO", "auto"},
		{"NONE", "none"},
		{"ANY", "required"},
	}

	for _, tt := range tests {
		t.Run(tt.mode, func(t *testing.T) {
			body := []byte(`{
				"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
				"toolConfig": {"functionCallingConfig": {"mode": "` + tt.mode + `"}}
			}`)

			req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if req.ToolChoice == nil {
				t.Fatal("ToolChoice is nil")
			}
			if req.ToolChoice.Type != tt.wantType {
				t.Errorf("ToolChoice.Type = %q, want %q", req.ToolChoice.Type, tt.wantType)
			}
		})
	}
}

func TestDecodeGeminiRequest_GenerationConfig(t *testing.T) {
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"generationConfig": {
			"temperature": 0.7,
			"topP": 0.9,
			"topK": 40,
			"maxOutputTokens": 1024,
			"stopSequences": ["END", "DONE"]
		}
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", req.Temperature)
	}
	if req.TopP == nil || *req.TopP != 0.9 {
		t.Errorf("TopP = %v, want 0.9", req.TopP)
	}
	if req.TopK == nil || *req.TopK != 40 {
		t.Errorf("TopK = %v, want 40", req.TopK)
	}
	if req.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %d, want 1024", req.MaxTokens)
	}
	if len(req.StopSequences) != 2 || req.StopSequences[0] != "END" || req.StopSequences[1] != "DONE" {
		t.Errorf("StopSequences = %v, want [END DONE]", req.StopSequences)
	}
}

func TestDecodeGeminiRequest_ResponseFormat(t *testing.T) {
	t.Run("text/plain", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
			"generationConfig": {"responseMimeType": "text/plain"}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.ResponseFormat == nil {
			t.Fatal("ResponseFormat is nil")
		}
		if req.ResponseFormat.Type != "text" {
			t.Errorf("ResponseFormat.Type = %q, want %q", req.ResponseFormat.Type, "text")
		}
	})

	t.Run("application/json without schema", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
			"generationConfig": {"responseMimeType": "application/json"}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.ResponseFormat == nil {
			t.Fatal("ResponseFormat is nil")
		}
		if req.ResponseFormat.Type != "json_object" {
			t.Errorf("ResponseFormat.Type = %q, want %q", req.ResponseFormat.Type, "json_object")
		}
	})

	t.Run("application/json with schema", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
			"generationConfig": {
				"responseMimeType": "application/json",
				"responseSchema": {
					"type": "OBJECT",
					"properties": {
						"name": {"type": "STRING"}
					}
				}
			}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.ResponseFormat == nil {
			t.Fatal("ResponseFormat is nil")
		}
		if req.ResponseFormat.Type != "json_schema" {
			t.Errorf("ResponseFormat.Type = %q, want %q", req.ResponseFormat.Type, "json_schema")
		}

		// Verify the schema was converted to JSON Schema
		var schema map[string]interface{}
		if err := json.Unmarshal(req.ResponseFormat.JSONSchema, &schema); err != nil {
			t.Fatalf("unmarshal schema: %v", err)
		}
		if schema["type"] != "object" {
			t.Errorf("schema.type = %v, want %q", schema["type"], "object")
		}
		props := schema["properties"].(map[string]interface{})
		nameProp := props["name"].(map[string]interface{})
		if nameProp["type"] != "string" {
			t.Errorf("schema.properties.name.type = %v, want %q", nameProp["type"], "string")
		}
	})
}

func TestDecodeGeminiRequest_ThinkingConfig(t *testing.T) {
	t.Run("with budget", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Think deeply"}]}],
			"generationConfig": {"thinkingConfig": {"thinkingBudget": 4096}}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
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
	})

	t.Run("zero budget adaptive", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
			"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.Thinking == nil {
			t.Fatal("Thinking is nil, want adaptive mode")
		}
		if req.Thinking.Mode != "adaptive" {
			t.Errorf("Thinking.Mode = %q, want %q", req.Thinking.Mode, "adaptive")
		}
		if req.Thinking.BudgetTokens != 0 {
			t.Errorf("Thinking.BudgetTokens = %d, want 0", req.Thinking.BudgetTokens)
		}
	})

	t.Run("absent", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.Thinking != nil {
			t.Errorf("Thinking = %v, want nil", req.Thinking)
		}
	})
}

func TestEncodeGeminiRequest_ToolResultPreservesName(t *testing.T) {
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{
				Role: RoleTool,
				Content: []ContentPart{
					{
						Type: ContentTypeToolResult,
						ToolResult: &ToolResultContent{
							ToolUseID: "call_123",
							Name:      "lookup_weather",
							Content: []ContentPart{
								{Type: ContentTypeText, Text: &TextContent{Text: `{"temp":"72F"}`}},
							},
						},
					},
				},
			},
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal encoded request: %v", err)
	}

	if len(raw.Contents) != 1 || len(raw.Contents[0].Parts) != 1 {
		t.Fatalf("encoded contents = %+v, want single tool result part", raw.Contents)
	}
	fr := raw.Contents[0].Parts[0].FunctionResponse
	if fr == nil {
		t.Fatal("expected FunctionResponse, got nil")
	}
	if fr.Name != "lookup_weather" {
		t.Fatalf("FunctionResponse.Name = %q, want %q", fr.Name, "lookup_weather")
	}
}

func TestEncodeGeminiRequest_Basic(t *testing.T) {
	temp := 0.7
	topP := 0.9
	topK := 40
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}},
			}},
			{Role: RoleAssistant, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "Hi there!"}},
			}},
		},
		Temperature: &temp,
		TopP:        &topP,
		TopK:        &topK,
		MaxTokens:   1024,
		SystemPrompt: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Be helpful"}},
		},
	}

	model, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if model != "gemini-2.5-pro" {
		t.Errorf("model = %q, want %q", model, "gemini-2.5-pro")
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	// System instruction
	if raw.SystemInstruction == nil {
		t.Fatal("SystemInstruction is nil")
	}
	if len(raw.SystemInstruction.Parts) != 1 || raw.SystemInstruction.Parts[0].Text != "Be helpful" {
		t.Errorf("SystemInstruction = %v, want [Be helpful]", raw.SystemInstruction)
	}

	// Contents
	if len(raw.Contents) != 2 {
		t.Fatalf("Contents len = %d, want 2", len(raw.Contents))
	}
	if raw.Contents[0].Role != "user" {
		t.Errorf("Contents[0].Role = %q, want %q", raw.Contents[0].Role, "user")
	}
	if raw.Contents[0].Parts[0].Text != "Hello" {
		t.Errorf("Contents[0].Parts[0].Text = %q, want %q", raw.Contents[0].Parts[0].Text, "Hello")
	}
	if raw.Contents[1].Role != "model" {
		t.Errorf("Contents[1].Role = %q, want %q", raw.Contents[1].Role, "model")
	}

	// Generation config
	if raw.GenerationConfig == nil {
		t.Fatal("GenerationConfig is nil")
	}
	if raw.GenerationConfig.Temperature == nil || *raw.GenerationConfig.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", raw.GenerationConfig.Temperature)
	}
	if raw.GenerationConfig.TopP == nil || *raw.GenerationConfig.TopP != 0.9 {
		t.Errorf("TopP = %v, want 0.9", raw.GenerationConfig.TopP)
	}
	if raw.GenerationConfig.TopK == nil || *raw.GenerationConfig.TopK != 40 {
		t.Errorf("TopK = %v, want 40", raw.GenerationConfig.TopK)
	}
	if raw.GenerationConfig.MaxOutputTokens == nil || *raw.GenerationConfig.MaxOutputTokens != 1024 {
		t.Errorf("MaxOutputTokens = %v, want 1024", raw.GenerationConfig.MaxOutputTokens)
	}
}

func TestEncodeGeminiRequest_ToolResult(t *testing.T) {
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "Read the file"}},
			}},
			{Role: RoleAssistant, Content: []ContentPart{
				{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{
					ID:        "call-123",
					Name:      "read_file",
					Arguments: json.RawMessage(`{"path":"/tmp/test.txt"}`),
				}},
			}},
			{Role: RoleTool, Content: []ContentPart{
				{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{
					ToolUseID: "call-123",
					Content: []ContentPart{
						{Type: ContentTypeText, Text: &TextContent{Text: `{"content":"file data"}`}},
					},
				}},
			}},
		},
	}

	model, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if model != "gemini-2.5-pro" {
		t.Errorf("model = %q, want %q", model, "gemini-2.5-pro")
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	if len(raw.Contents) != 3 {
		t.Fatalf("Contents len = %d, want 3", len(raw.Contents))
	}

	// Tool message should be encoded as "user" role
	toolContent := raw.Contents[2]
	if toolContent.Role != "user" {
		t.Errorf("tool content Role = %q, want %q", toolContent.Role, "user")
	}

	// Should have FunctionResponse part
	if len(toolContent.Parts) != 1 {
		t.Fatalf("tool content Parts len = %d, want 1", len(toolContent.Parts))
	}
	fr := toolContent.Parts[0].FunctionResponse
	if fr == nil {
		t.Fatal("FunctionResponse is nil")
	}
	if fr.ID != "call-123" {
		t.Errorf("FunctionResponse.ID = %q, want %q", fr.ID, "call-123")
	}

	// Assistant message should have FunctionCall
	assistContent := raw.Contents[1]
	if assistContent.Role != "model" {
		t.Errorf("assistant content Role = %q, want %q", assistContent.Role, "model")
	}
	fc := assistContent.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("FunctionCall is nil")
	}
	if fc.Name != "read_file" {
		t.Errorf("FunctionCall.Name = %q, want %q", fc.Name, "read_file")
	}
	if fc.ID != "call-123" {
		t.Errorf("FunctionCall.ID = %q, want %q", fc.ID, "call-123")
	}
}

// TestEncodeGeminiRequest_ToolResultNameResolution tests the cross-protocol path where
// ToolResult.Name is empty (e.g. from Anthropic) but is resolved from a prior ToolUse part.
func TestEncodeGeminiRequest_ToolResultNameResolution(t *testing.T) {
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleAssistant, Content: []ContentPart{
				{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{
					ID:        "tu_abc",
					Name:      "search_web",
					Arguments: json.RawMessage(`{"query":"golang"}`),
				}},
			}},
			// ToolResult with no Name set — simulates Anthropic inbound which omits the name.
			{Role: RoleTool, Content: []ContentPart{
				{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{
					ToolUseID: "tu_abc",
					Content: []ContentPart{
						{Type: ContentTypeText, Text: &TextContent{Text: "results here"}},
					},
				}},
			}},
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	if len(raw.Contents) != 2 {
		t.Fatalf("Contents len = %d, want 2", len(raw.Contents))
	}
	fr := raw.Contents[1].Parts[0].FunctionResponse
	if fr == nil {
		t.Fatal("FunctionResponse is nil")
	}
	if fr.Name != "search_web" {
		t.Errorf("FunctionResponse.Name = %q, want %q (should be resolved from prior ToolUse)", fr.Name, "search_web")
	}
	if fr.ID != "tu_abc" {
		t.Errorf("FunctionResponse.ID = %q, want %q", fr.ID, "tu_abc")
	}
}

func TestEncodeGeminiRequest_Tools(t *testing.T) {
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}},
			}},
		},
		Tools: []Tool{
			{
				Name:        "read_file",
				Description: "Read a file",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
			},
		},
		ToolChoice: &ToolChoice{Type: "auto"},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	if len(raw.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(raw.Tools))
	}
	if len(raw.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("FunctionDeclarations len = %d, want 1", len(raw.Tools[0].FunctionDeclarations))
	}
	fd := raw.Tools[0].FunctionDeclarations[0]
	if fd.Name != "read_file" {
		t.Errorf("Name = %q, want %q", fd.Name, "read_file")
	}

	// Verify schema was converted to Gemini format (uppercase types)
	var params map[string]interface{}
	if err := json.Unmarshal(fd.Parameters, &params); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	if params["type"] != "OBJECT" {
		t.Errorf("params.type = %v, want %q", params["type"], "OBJECT")
	}

	// Tool config
	if raw.ToolConfig == nil {
		t.Fatal("ToolConfig is nil")
	}
	if raw.ToolConfig.FunctionCallingConfig == nil {
		t.Fatal("FunctionCallingConfig is nil")
	}
	if raw.ToolConfig.FunctionCallingConfig.Mode != "AUTO" {
		t.Errorf("Mode = %q, want %q", raw.ToolConfig.FunctionCallingConfig.Mode, "AUTO")
	}
}

func TestEncodeGeminiRequest_ResponseFormat(t *testing.T) {
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}},
			}},
		},
		ResponseFormat: &ResponseFormat{
			Type:       "json_schema",
			JSONSchema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`),
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	if raw.GenerationConfig == nil {
		t.Fatal("GenerationConfig is nil")
	}
	if raw.GenerationConfig.ResponseMimeType != "application/json" {
		t.Errorf("ResponseMimeType = %q, want %q", raw.GenerationConfig.ResponseMimeType, "application/json")
	}

	var schema map[string]interface{}
	if err := json.Unmarshal(raw.GenerationConfig.ResponseSchema, &schema); err != nil {
		t.Fatalf("unmarshal schema: %v", err)
	}
	if schema["type"] != "OBJECT" {
		t.Errorf("schema.type = %v, want %q", schema["type"], "OBJECT")
	}
}

func TestEncodeGeminiRequest_ThinkingConfig(t *testing.T) {
	t.Run("enabled", func(t *testing.T) {
		req := &Request{
			Model: "gemini-2.5-pro",
			Messages: []Message{
				{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Think"}}}},
			},
			Thinking: &ThinkingConfig{
				Mode:         "enabled",
				BudgetTokens: 4096,
			},
		}

		_, body, err := EncodeGeminiRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Request
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}

		if raw.GenerationConfig == nil {
			t.Fatal("GenerationConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig == nil {
			t.Fatal("ThinkingConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig.ThinkingBudget != 4096 {
			t.Errorf("ThinkingBudget = %d, want 4096", raw.GenerationConfig.ThinkingConfig.ThinkingBudget)
		}
	})

	t.Run("adaptive", func(t *testing.T) {
		req := &Request{
			Model: "gemini-2.5-pro",
			Messages: []Message{
				{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Think adaptively"}}}},
			},
			Thinking: &ThinkingConfig{
				Mode:         "adaptive",
				BudgetTokens: 0,
			},
		}

		_, body, err := EncodeGeminiRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Request
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}

		if raw.GenerationConfig == nil {
			t.Fatal("GenerationConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig == nil {
			t.Fatal("ThinkingConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig.ThinkingBudget != 0 {
			t.Errorf("ThinkingBudget = %d, want 0", raw.GenerationConfig.ThinkingConfig.ThinkingBudget)
		}
	})
}

func TestEncodeGeminiRequest_Image(t *testing.T) {
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeImage, Image: &ImageContent{
					Data:      []byte("fake-png-data"),
					MediaType: "image/png",
				}},
				{Type: ContentTypeImage, Image: &ImageContent{
					URL:       "https://example.com/image.jpg",
					MediaType: "image/jpeg",
				}},
			}},
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	parts := raw.Contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("Parts len = %d, want 2", len(parts))
	}

	// InlineData
	if parts[0].InlineData == nil {
		t.Fatal("InlineData is nil")
	}
	if parts[0].InlineData.MimeType != "image/png" {
		t.Errorf("InlineData.MimeType = %q, want %q", parts[0].InlineData.MimeType, "image/png")
	}
	decoded, _ := base64.StdEncoding.DecodeString(parts[0].InlineData.Data)
	if string(decoded) != "fake-png-data" {
		t.Errorf("InlineData.Data decoded = %q, want %q", string(decoded), "fake-png-data")
	}

	// FileData
	if parts[1].FileData == nil {
		t.Fatal("FileData is nil")
	}
	if parts[1].FileData.FileURI != "https://example.com/image.jpg" {
		t.Errorf("FileData.FileURI = %q, want %q", parts[1].FileData.FileURI, "https://example.com/image.jpg")
	}
}

func TestDecodeGeminiResponse_Basic(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {"role": "model", "parts": [{"text": "Hello!"}]},
			"finishReason": "STOP"
		}],
		"usageMetadata": {
			"promptTokenCount": 10,
			"candidatesTokenCount": 5,
			"totalTokenCount": 15
		},
		"modelVersion": "gemini-2.5-pro"
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Model != "gemini-2.5-pro" {
		t.Errorf("Model = %q, want %q", resp.Model, "gemini-2.5-pro")
	}
	if resp.StopReason != StopReasonEndTurn {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonEndTurn)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q, want %q", resp.Content[0].Type, ContentTypeText)
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

func TestDecodeGeminiResponse_FunctionCall(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {
				"role": "model",
				"parts": [
					{"text": "Let me read that file."},
					{"functionCall": {"name": "read_file", "args": {"path": "/tmp/test.txt"}, "id": "call-456"}}
				]
			},
			"finishReason": "STOP"
		}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 15, "totalTokenCount": 25},
		"modelVersion": "gemini-2.5-pro"
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should infer StopReasonToolUse regardless of finishReason
	if resp.StopReason != StopReasonToolUse {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonToolUse)
	}

	if len(resp.Content) != 2 {
		t.Fatalf("Content len = %d, want 2", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q, want %q", resp.Content[0].Type, ContentTypeText)
	}
	if resp.Content[1].Type != ContentTypeToolUse {
		t.Errorf("Content[1].Type = %q, want %q", resp.Content[1].Type, ContentTypeToolUse)
	}
	if resp.Content[1].ToolUse.Name != "read_file" {
		t.Errorf("ToolUse.Name = %q, want %q", resp.Content[1].ToolUse.Name, "read_file")
	}
	if resp.Content[1].ToolUse.ID != "call-456" {
		t.Errorf("ToolUse.ID = %q, want %q", resp.Content[1].ToolUse.ID, "call-456")
	}
}

func TestDecodeGeminiResponse_MaxTokens(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {"role": "model", "parts": [{"text": "partial..."}]},
			"finishReason": "MAX_TOKENS"
		}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 100, "totalTokenCount": 110}
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.StopReason != StopReasonMaxTokens {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonMaxTokens)
	}
}

func TestDecodeGeminiResponse_Safety(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {"role": "model", "parts": [{"text": ""}]},
			"finishReason": "SAFETY"
		}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0, "totalTokenCount": 10}
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.StopReason != StopReasonContentFilter {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, StopReasonContentFilter)
	}
}

func TestEncodeGeminiResponse_Basic(t *testing.T) {
	resp := &Response{
		Model: "gemini-2.5-pro",
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Hello!"}},
		},
		StopReason: StopReasonEndTurn,
		Usage: Usage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
		},
	}

	body, err := EncodeGeminiResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Response
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if raw.ModelVersion != "gemini-2.5-pro" {
		t.Errorf("ModelVersion = %q, want %q", raw.ModelVersion, "gemini-2.5-pro")
	}
	if len(raw.Candidates) != 1 {
		t.Fatalf("Candidates len = %d, want 1", len(raw.Candidates))
	}
	if raw.Candidates[0].FinishReason != "STOP" {
		t.Errorf("FinishReason = %q, want %q", raw.Candidates[0].FinishReason, "STOP")
	}
	if len(raw.Candidates[0].Content.Parts) != 1 {
		t.Fatalf("Parts len = %d, want 1", len(raw.Candidates[0].Content.Parts))
	}
	if raw.Candidates[0].Content.Parts[0].Text != "Hello!" {
		t.Errorf("Text = %q, want %q", raw.Candidates[0].Content.Parts[0].Text, "Hello!")
	}
	if raw.UsageMetadata.PromptTokenCount != 10 {
		t.Errorf("PromptTokenCount = %d, want 10", raw.UsageMetadata.PromptTokenCount)
	}
	if raw.UsageMetadata.CandidatesTokenCount != 5 {
		t.Errorf("CandidatesTokenCount = %d, want 5", raw.UsageMetadata.CandidatesTokenCount)
	}
}

func TestEncodeGeminiResponse_ToolUse(t *testing.T) {
	resp := &Response{
		Model: "gemini-2.5-pro",
		Content: []ContentPart{
			{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{
				ID:        "call-789",
				Name:      "get_weather",
				Arguments: json.RawMessage(`{"city":"NYC"}`),
			}},
		},
		StopReason: StopReasonToolUse,
		Usage:      Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30},
	}

	body, err := EncodeGeminiResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Response
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// StopReasonToolUse maps to "STOP" in Gemini (no explicit tool_use finish reason)
	if raw.Candidates[0].FinishReason != "STOP" {
		t.Errorf("FinishReason = %q, want %q", raw.Candidates[0].FinishReason, "STOP")
	}

	fc := raw.Candidates[0].Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("FunctionCall is nil")
	}
	if fc.Name != "get_weather" {
		t.Errorf("FunctionCall.Name = %q, want %q", fc.Name, "get_weather")
	}
	if fc.ID != "call-789" {
		t.Errorf("FunctionCall.ID = %q, want %q", fc.ID, "call-789")
	}
}

func TestEncodeGeminiResponse_PauseTurnDowngradesToStop(t *testing.T) {
	resp := &Response{
		Model:      "gemini-2.5-pro",
		StopReason: StopReasonPauseTurn,
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Need more input."}},
		},
	}

	body, err := EncodeGeminiResponse(resp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Response
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if raw.Candidates[0].FinishReason != "STOP" {
		t.Fatalf("FinishReason = %q, want %q", raw.Candidates[0].FinishReason, "STOP")
	}
}

func TestEncodeGeminiStreamChunk_PauseTurnDowngradesToStop(t *testing.T) {
	stopReason := StopReasonPauseTurn
	event := &StreamEvent{
		Type:       StreamEventStop,
		StopReason: &stopReason,
	}

	body, err := EncodeGeminiStreamChunk(event)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Response
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if raw.Candidates[0].FinishReason != "STOP" {
		t.Fatalf("FinishReason = %q, want %q", raw.Candidates[0].FinishReason, "STOP")
	}
}

func TestDecodeGeminiStreamChunk(t *testing.T) {
	t.Run("text delta", func(t *testing.T) {
		data := []byte(`{
			"candidates": [{
				"content": {"role": "model", "parts": [{"text": "Hello"}]}
			}]
		}`)

		events, err := DecodeGeminiStreamChunk(data)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("got %d events, want 1", len(events))
		}
		event := events[0]

		if event.Type != StreamEventDelta {
			t.Errorf("Type = %q, want %q", event.Type, StreamEventDelta)
		}
		if event.Delta == nil {
			t.Fatal("Delta is nil")
		}
		if event.Delta.Type != ContentTypeText {
			t.Errorf("Delta.Type = %q, want %q", event.Delta.Type, ContentTypeText)
		}
		if event.Delta.Text.Text != "Hello" {
			t.Errorf("Delta.Text = %q, want %q", event.Delta.Text.Text, "Hello")
		}
	})

	t.Run("function call delta", func(t *testing.T) {
		data := []byte(`{
			"candidates": [{
				"content": {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {"path": "/tmp"}, "id": "c1"}}]}
			}]
		}`)

		events, err := DecodeGeminiStreamChunk(data)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("got %d events, want 1", len(events))
		}
		event := events[0]

		if event.Type != StreamEventDelta {
			t.Errorf("Type = %q, want %q", event.Type, StreamEventDelta)
		}
		if event.Delta.Type != ContentTypeToolUse {
			t.Errorf("Delta.Type = %q, want %q", event.Delta.Type, ContentTypeToolUse)
		}
		// Should infer tool use stop reason
		if event.StopReason == nil || *event.StopReason != StopReasonToolUse {
			t.Errorf("StopReason = %v, want %q", event.StopReason, StopReasonToolUse)
		}
	})

	t.Run("finish with STOP", func(t *testing.T) {
		data := []byte(`{
			"candidates": [{
				"content": {"role": "model", "parts": [{"text": " world!"}]},
				"finishReason": "STOP"
			}],
			"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10, "totalTokenCount": 15}
		}`)

		events, err := DecodeGeminiStreamChunk(data)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("got %d events, want 1", len(events))
		}
		event := events[0]

		if event.Type != StreamEventDelta {
			t.Errorf("Type = %q, want %q", event.Type, StreamEventDelta)
		}
		if event.StopReason == nil || *event.StopReason != StopReasonEndTurn {
			t.Errorf("StopReason = %v, want %q", event.StopReason, StopReasonEndTurn)
		}
		if event.Usage == nil {
			t.Fatal("Usage is nil")
		}
		if event.Usage.InputTokens != 5 {
			t.Errorf("Usage.InputTokens = %d, want 5", event.Usage.InputTokens)
		}
		if event.Usage.OutputTokens != 10 {
			t.Errorf("Usage.OutputTokens = %d, want 10", event.Usage.OutputTokens)
		}
	})

	t.Run("empty candidates start", func(t *testing.T) {
		data := []byte(`{
			"modelVersion": "gemini-2.5-pro"
		}`)

		events, err := DecodeGeminiStreamChunk(data)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("got %d events, want 1", len(events))
		}
		event := events[0]

		if event.Type != StreamEventStart {
			t.Errorf("Type = %q, want %q", event.Type, StreamEventStart)
		}
		if event.Response == nil {
			t.Fatal("Response is nil")
		}
		if event.Response.Model != "gemini-2.5-pro" {
			t.Errorf("Model = %q, want %q", event.Response.Model, "gemini-2.5-pro")
		}
	})

	t.Run("finish reason only (no content)", func(t *testing.T) {
		data := []byte(`{
			"candidates": [{
				"finishReason": "MAX_TOKENS"
			}],
			"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 100, "totalTokenCount": 105}
		}`)

		events, err := DecodeGeminiStreamChunk(data)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("got %d events, want 1", len(events))
		}
		event := events[0]

		if event.Type != StreamEventStop {
			t.Errorf("Type = %q, want %q", event.Type, StreamEventStop)
		}
		if event.StopReason == nil || *event.StopReason != StopReasonMaxTokens {
			t.Errorf("StopReason = %v, want %q", event.StopReason, StopReasonMaxTokens)
		}
	})
}

func TestEncodeGeminiStreamChunk(t *testing.T) {
	t.Run("start", func(t *testing.T) {
		event := &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				Model: "gemini-2.5-pro",
			},
		}

		data, err := EncodeGeminiStreamChunk(event)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Response
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		if raw.ModelVersion != "gemini-2.5-pro" {
			t.Errorf("ModelVersion = %q, want %q", raw.ModelVersion, "gemini-2.5-pro")
		}
		if len(raw.Candidates) != 1 {
			t.Fatalf("Candidates len = %d, want 1", len(raw.Candidates))
		}
	})

	t.Run("text delta", func(t *testing.T) {
		event := &StreamEvent{
			Type: StreamEventDelta,
			Delta: &ContentPart{
				Type: ContentTypeText,
				Text: &TextContent{Text: "Hello"},
			},
		}

		data, err := EncodeGeminiStreamChunk(event)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Response
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		if len(raw.Candidates) != 1 {
			t.Fatalf("Candidates len = %d, want 1", len(raw.Candidates))
		}
		if len(raw.Candidates[0].Content.Parts) != 1 {
			t.Fatalf("Parts len = %d, want 1", len(raw.Candidates[0].Content.Parts))
		}
		if raw.Candidates[0].Content.Parts[0].Text != "Hello" {
			t.Errorf("Text = %q, want %q", raw.Candidates[0].Content.Parts[0].Text, "Hello")
		}
	})

	t.Run("stop with reason", func(t *testing.T) {
		stopReason := StopReasonEndTurn
		event := &StreamEvent{
			Type:       StreamEventStop,
			StopReason: &stopReason,
			Usage: &Usage{
				InputTokens:  10,
				OutputTokens: 20,
				TotalTokens:  30,
			},
		}

		data, err := EncodeGeminiStreamChunk(event)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Response
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		if raw.Candidates[0].FinishReason != "STOP" {
			t.Errorf("FinishReason = %q, want %q", raw.Candidates[0].FinishReason, "STOP")
		}
		if raw.UsageMetadata == nil {
			t.Fatal("UsageMetadata is nil")
		}
		if raw.UsageMetadata.PromptTokenCount != 10 {
			t.Errorf("PromptTokenCount = %d, want 10", raw.UsageMetadata.PromptTokenCount)
		}
	})

	t.Run("delta with stop reason", func(t *testing.T) {
		stopReason := StopReasonToolUse
		event := &StreamEvent{
			Type: StreamEventDelta,
			Delta: &ContentPart{
				Type:    ContentTypeToolUse,
				ToolUse: &ToolUseContent{ID: "c1", Name: "fn", Arguments: json.RawMessage(`{}`)},
			},
			StopReason: &stopReason,
		}

		data, err := EncodeGeminiStreamChunk(event)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Response
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		// ToolUse maps to "STOP" in Gemini
		if raw.Candidates[0].FinishReason != "STOP" {
			t.Errorf("FinishReason = %q, want %q", raw.Candidates[0].FinishReason, "STOP")
		}
	})
}

func TestGeminiSchemaConversion(t *testing.T) {
	t.Run("Gemini to JSON Schema", func(t *testing.T) {
		geminiRaw := json.RawMessage(`{
			"type": "OBJECT",
			"description": "A person",
			"properties": {
				"name": {"type": "STRING", "description": "The name"},
				"age": {"type": "INTEGER"},
				"scores": {"type": "ARRAY", "items": {"type": "NUMBER"}},
				"active": {"type": "BOOLEAN"}
			},
			"required": ["name"]
		}`)

		jsonRaw, err := geminiSchemaToJSONSchema(geminiRaw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var schema map[string]interface{}
		if err := json.Unmarshal(jsonRaw, &schema); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		if schema["type"] != "object" {
			t.Errorf("type = %v, want %q", schema["type"], "object")
		}
		if schema["description"] != "A person" {
			t.Errorf("description = %v, want %q", schema["description"], "A person")
		}

		props := schema["properties"].(map[string]interface{})
		if props["name"].(map[string]interface{})["type"] != "string" {
			t.Errorf("properties.name.type = %v, want %q", props["name"].(map[string]interface{})["type"], "string")
		}
		if props["age"].(map[string]interface{})["type"] != "integer" {
			t.Errorf("properties.age.type = %v, want %q", props["age"].(map[string]interface{})["type"], "integer")
		}
		if props["scores"].(map[string]interface{})["type"] != "array" {
			t.Errorf("properties.scores.type = %v, want %q", props["scores"].(map[string]interface{})["type"], "array")
		}
		scoresItems := props["scores"].(map[string]interface{})["items"].(map[string]interface{})
		if scoresItems["type"] != "number" {
			t.Errorf("properties.scores.items.type = %v, want %q", scoresItems["type"], "number")
		}
		if props["active"].(map[string]interface{})["type"] != "boolean" {
			t.Errorf("properties.active.type = %v, want %q", props["active"].(map[string]interface{})["type"], "boolean")
		}

		required := schema["required"].([]interface{})
		if len(required) != 1 || required[0] != "name" {
			t.Errorf("required = %v, want [name]", required)
		}
	})

	t.Run("JSON Schema to Gemini", func(t *testing.T) {
		jsonRaw := json.RawMessage(`{
			"type": "object",
			"description": "A person",
			"properties": {
				"name": {"type": "string", "description": "The name"},
				"age": {"type": "integer"},
				"tags": {"type": "array", "items": {"type": "string"}},
				"active": {"type": "boolean"}
			},
			"required": ["name"]
		}`)

		geminiRaw, err := jsonSchemaToGeminiSchema(jsonRaw)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var schema map[string]interface{}
		if err := json.Unmarshal(geminiRaw, &schema); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		if schema["type"] != "OBJECT" {
			t.Errorf("type = %v, want %q", schema["type"], "OBJECT")
		}

		props := schema["properties"].(map[string]interface{})
		if props["name"].(map[string]interface{})["type"] != "STRING" {
			t.Errorf("properties.name.type = %v, want %q", props["name"].(map[string]interface{})["type"], "STRING")
		}
		if props["age"].(map[string]interface{})["type"] != "INTEGER" {
			t.Errorf("properties.age.type = %v, want %q", props["age"].(map[string]interface{})["type"], "INTEGER")
		}
		if props["tags"].(map[string]interface{})["type"] != "ARRAY" {
			t.Errorf("properties.tags.type = %v, want %q", props["tags"].(map[string]interface{})["type"], "ARRAY")
		}
		tagsItems := props["tags"].(map[string]interface{})["items"].(map[string]interface{})
		if tagsItems["type"] != "STRING" {
			t.Errorf("properties.tags.items.type = %v, want %q", tagsItems["type"], "STRING")
		}
	})

	t.Run("round-trip", func(t *testing.T) {
		// Start with Gemini schema, convert to JSON, convert back
		geminiOriginal := json.RawMessage(`{"type":"OBJECT","properties":{"x":{"type":"STRING"}},"required":["x"]}`)

		jsonSchema, err := geminiSchemaToJSONSchema(geminiOriginal)
		if err != nil {
			t.Fatalf("gemini to json: %v", err)
		}

		geminiBack, err := jsonSchemaToGeminiSchema(jsonSchema)
		if err != nil {
			t.Fatalf("json to gemini: %v", err)
		}

		// Parse both for comparison
		var original, roundTripped map[string]interface{}
		json.Unmarshal(geminiOriginal, &original)
		json.Unmarshal(geminiBack, &roundTripped)

		if roundTripped["type"] != "OBJECT" {
			t.Errorf("round-tripped type = %v, want OBJECT", roundTripped["type"])
		}

		rtProps := roundTripped["properties"].(map[string]interface{})
		if rtProps["x"].(map[string]interface{})["type"] != "STRING" {
			t.Errorf("round-tripped properties.x.type = %v, want STRING", rtProps["x"].(map[string]interface{})["type"])
		}

		rtReq := roundTripped["required"].([]interface{})
		if len(rtReq) != 1 || rtReq[0] != "x" {
			t.Errorf("round-tripped required = %v, want [x]", rtReq)
		}
	})
}

func TestDecodeGeminiRequest_MultipleToolDeclarations(t *testing.T) {
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"tools": [
			{
				"functionDeclarations": [
					{"name": "tool_a", "description": "Tool A"},
					{"name": "tool_b", "description": "Tool B"}
				]
			},
			{
				"functionDeclarations": [
					{"name": "tool_c", "description": "Tool C"}
				]
			}
		]
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(req.Tools) != 3 {
		t.Errorf("Tools len = %d, want 3", len(req.Tools))
	}
}

func TestParseGeminiModelFromURL(t *testing.T) {
	tests := []struct {
		url     string
		want    string
		wantErr bool
	}{
		{"/v1/models/gemini-2.5-pro:generateContent", "gemini-2.5-pro", false},
		{"/v1/models/gemini-2.0-flash:streamGenerateContent", "gemini-2.0-flash", false},
		{"/v1beta/models/gemini-2.5-pro-preview:generateContent", "gemini-2.5-pro-preview", false},
		{"/v1/invalid/path", "", true},
		{"/v1/models/:generateContent", "", true},
		{"/v1/models/gemini-2.5-pro", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			got, err := parseGeminiModelFromURL(tt.url)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestEncodeGeminiRequest_ToolChoiceModes(t *testing.T) {
	tests := []struct {
		irType   string
		wantMode string
	}{
		{"auto", "AUTO"},
		{"none", "NONE"},
		{"required", "ANY"},
	}

	for _, tt := range tests {
		t.Run(tt.irType, func(t *testing.T) {
			req := &Request{
				Model: "gemini-2.5-pro",
				Messages: []Message{
					{Role: RoleUser, Content: []ContentPart{
						{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}},
					}},
				},
				ToolChoice: &ToolChoice{Type: tt.irType},
			}

			_, body, err := EncodeGeminiRequest(req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var raw gemini.Request
			json.Unmarshal(body, &raw)

			if raw.ToolConfig == nil || raw.ToolConfig.FunctionCallingConfig == nil {
				t.Fatal("ToolConfig is nil")
			}
			if raw.ToolConfig.FunctionCallingConfig.Mode != tt.wantMode {
				t.Errorf("Mode = %q, want %q", raw.ToolConfig.FunctionCallingConfig.Mode, tt.wantMode)
			}
		})
	}
}

func TestDecodeGeminiResponse_EmptyCandidates(t *testing.T) {
	body := []byte(`{
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0, "totalTokenCount": 10},
		"modelVersion": "gemini-2.5-pro"
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Model != "gemini-2.5-pro" {
		t.Errorf("Model = %q, want %q", resp.Model, "gemini-2.5-pro")
	}
	if len(resp.Content) != 0 {
		t.Errorf("Content len = %d, want 0", len(resp.Content))
	}
}

func TestEncodeDecodeGeminiRequest_RoundTrip(t *testing.T) {
	temp := 0.5
	topP := 0.8
	topK := 30
	original := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}},
			}},
			{Role: RoleAssistant, Content: []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}},
			}},
		},
		SystemPrompt: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Be helpful"}},
		},
		Temperature: &temp,
		TopP:        &topP,
		TopK:        &topK,
		MaxTokens:   512,
		Thinking: &ThinkingConfig{
			Mode:         "enabled",
			BudgetTokens: 2048,
		},
	}

	model, body, err := EncodeGeminiRequest(original)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	decoded, err := DecodeGeminiRequest("/v1/models/"+model+":generateContent", body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if decoded.Model != original.Model {
		t.Errorf("Model = %q, want %q", decoded.Model, original.Model)
	}
	if len(decoded.Messages) != len(original.Messages) {
		t.Errorf("Messages len = %d, want %d", len(decoded.Messages), len(original.Messages))
	}
	if decoded.Messages[0].Role != RoleUser {
		t.Errorf("Messages[0].Role = %q, want %q", decoded.Messages[0].Role, RoleUser)
	}
	if decoded.Messages[1].Role != RoleAssistant {
		t.Errorf("Messages[1].Role = %q, want %q", decoded.Messages[1].Role, RoleAssistant)
	}
	if decoded.Temperature == nil || *decoded.Temperature != temp {
		t.Errorf("Temperature = %v, want %v", decoded.Temperature, temp)
	}
	if decoded.MaxTokens != original.MaxTokens {
		t.Errorf("MaxTokens = %d, want %d", decoded.MaxTokens, original.MaxTokens)
	}
	if decoded.Thinking == nil || decoded.Thinking.BudgetTokens != 2048 {
		t.Errorf("Thinking = %v, want BudgetTokens=2048", decoded.Thinking)
	}
}

func TestEncodeDecodeGeminiResponse_RoundTrip(t *testing.T) {
	original := &Response{
		Model: "gemini-2.5-pro",
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Hello!"}},
		},
		StopReason: StopReasonEndTurn,
		Usage:      Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	body, err := EncodeGeminiResponse(original)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	decoded, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
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
	if decoded.Content[0].Text.Text != "Hello!" {
		t.Errorf("Content[0].Text = %q, want %q", decoded.Content[0].Text.Text, "Hello!")
	}
	if decoded.Usage.InputTokens != 10 {
		t.Errorf("Usage.InputTokens = %d, want 10", decoded.Usage.InputTokens)
	}
}

// TestDecodeGeminiResponse_ThoughtsTokenCount verifies that the Gemini API
// wire key "thoughtsTokenCount" (with "s" in "thoughts") is decoded correctly
// into Usage.ThinkingTokens. This is a regression test for the prior bug where
// the struct tag was "thinkingTokenCount" (missing the "s").
func TestDecodeGeminiResponse_ThoughtsTokenCount(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {"role": "model", "parts": [{"text": "I thought about it."}]},
			"finishReason": "STOP"
		}],
		"usageMetadata": {
			"promptTokenCount": 20,
			"candidatesTokenCount": 50,
			"totalTokenCount": 70,
			"thoughtsTokenCount": 30
		}
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Usage.ThinkingTokens != 30 {
		t.Errorf("Usage.ThinkingTokens = %d, want 30 (wire key is thoughtsTokenCount)", resp.Usage.ThinkingTokens)
	}
	if resp.Usage.InputTokens != 20 {
		t.Errorf("Usage.InputTokens = %d, want 20", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 50 {
		t.Errorf("Usage.OutputTokens = %d, want 50", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens != 70 {
		t.Errorf("Usage.TotalTokens = %d, want 70", resp.Usage.TotalTokens)
	}
}

// TestEncodeGeminiResponse_ThoughtsTokenCountWireKey verifies that when
// EncodeGeminiResponse serialises a response with ThinkingTokens, the resulting
// JSON uses the key "thoughtsTokenCount" (the correct Gemini API wire key).
func TestEncodeGeminiResponse_ThoughtsTokenCountWireKey(t *testing.T) {
	resp := &Response{
		Model: "gemini-2.5-pro",
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Thought deeply."}},
		},
		StopReason: StopReasonEndTurn,
		Usage: Usage{
			InputTokens:    10,
			OutputTokens:   20,
			TotalTokens:    30,
			ThinkingTokens: 15,
		},
	}

	body, err := EncodeGeminiResponse(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Unmarshal to a raw map to inspect wire keys.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	usageRaw, ok := raw["usageMetadata"]
	if !ok {
		t.Fatal("usageMetadata missing from encoded Gemini response")
	}

	var usageMap map[string]json.RawMessage
	if err := json.Unmarshal(usageRaw, &usageMap); err != nil {
		t.Fatalf("unmarshal usageMetadata: %v", err)
	}

	if _, ok := usageMap["thoughtsTokenCount"]; !ok {
		t.Errorf("expected wire key 'thoughtsTokenCount' in usageMetadata, got keys: %v", func() []string {
			keys := make([]string, 0, len(usageMap))
			for k := range usageMap {
				keys = append(keys, k)
			}
			return keys
		}())
	}
	if _, bad := usageMap["thinkingTokenCount"]; bad {
		t.Error("found incorrect wire key 'thinkingTokenCount' — should be 'thoughtsTokenCount'")
	}
}

// TestDecodeEncodeGeminiResponse_ThinkingTokensRoundTrip verifies that ThinkingTokens
// survives a full Decode → Encode → Decode round-trip via the Gemini codec.
func TestDecodeEncodeGeminiResponse_ThinkingTokensRoundTrip(t *testing.T) {
	original := &Response{
		Model: "gemini-2.5-pro",
		Content: []ContentPart{
			{Type: ContentTypeText, Text: &TextContent{Text: "Round trip!"}},
		},
		StopReason: StopReasonEndTurn,
		Usage: Usage{
			InputTokens:    5,
			OutputTokens:   10,
			TotalTokens:    15,
			ThinkingTokens: 8,
		},
	}

	encoded, err := EncodeGeminiResponse(original)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	decoded, err := DecodeGeminiResponse(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if decoded.Usage.ThinkingTokens != original.Usage.ThinkingTokens {
		t.Errorf("ThinkingTokens round-trip: got %d, want %d", decoded.Usage.ThinkingTokens, original.Usage.ThinkingTokens)
	}
	if decoded.Usage.InputTokens != original.Usage.InputTokens {
		t.Errorf("InputTokens round-trip: got %d, want %d", decoded.Usage.InputTokens, original.Usage.InputTokens)
	}
	if decoded.Usage.OutputTokens != original.Usage.OutputTokens {
		t.Errorf("OutputTokens round-trip: got %d, want %d", decoded.Usage.OutputTokens, original.Usage.OutputTokens)
	}
}

// --- Tests for allowedFunctionNames ---

func TestDecodeGeminiRequest_AllowedFunctionNames(t *testing.T) {
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"toolConfig": {
			"functionCallingConfig": {
				"mode": "ANY",
				"allowedFunctionNames": ["tool_a", "tool_b"]
			}
		}
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
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
	if req.ToolChoice.AllowedToolNames[0] != "tool_a" {
		t.Errorf("AllowedToolNames[0] = %q, want tool_a", req.ToolChoice.AllowedToolNames[0])
	}
	if req.ToolChoice.AllowedToolNames[1] != "tool_b" {
		t.Errorf("AllowedToolNames[1] = %q, want tool_b", req.ToolChoice.AllowedToolNames[1])
	}
}

func TestDecodeGeminiRequest_AllowedFunctionNamesEmpty(t *testing.T) {
	// Empty allowedFunctionNames → AllowedToolNames should be nil
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"toolConfig": {
			"functionCallingConfig": {
				"mode": "AUTO"
			}
		}
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.ToolChoice == nil {
		t.Fatal("ToolChoice is nil")
	}
	if len(req.ToolChoice.AllowedToolNames) != 0 {
		t.Errorf("AllowedToolNames = %v, want empty", req.ToolChoice.AllowedToolNames)
	}
}

func TestEncodeGeminiRequest_AllowedFunctionNames(t *testing.T) {
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type:             "auto",
			AllowedToolNames: []string{"tool_a", "tool_b"},
		},
	}

	model, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model != "gemini-2.5-pro" {
		t.Errorf("model = %q, want gemini-2.5-pro", model)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	tcRaw, ok := m["toolConfig"]
	if !ok {
		t.Fatal("toolConfig missing from encoded request")
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal toolConfig: %v", err)
	}

	fccRaw, ok := tc["functionCallingConfig"]
	if !ok {
		t.Fatal("functionCallingConfig missing from toolConfig")
	}
	var fcc map[string]json.RawMessage
	if err := json.Unmarshal(fccRaw, &fcc); err != nil {
		t.Fatalf("unmarshal functionCallingConfig: %v", err)
	}

	afnRaw, ok := fcc["allowedFunctionNames"]
	if !ok {
		t.Fatal("allowedFunctionNames missing from functionCallingConfig")
	}
	var afn []string
	if err := json.Unmarshal(afnRaw, &afn); err != nil {
		t.Fatalf("unmarshal allowedFunctionNames: %v", err)
	}
	if len(afn) != 2 {
		t.Fatalf("allowedFunctionNames len = %d, want 2", len(afn))
	}
	if afn[0] != "tool_a" || afn[1] != "tool_b" {
		t.Errorf("allowedFunctionNames = %v, want [tool_a tool_b]", afn)
	}
}

func TestEncodeGeminiRequest_NoAllowedFunctionNames(t *testing.T) {
	// No AllowedToolNames → allowedFunctionNames should not appear
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type: "auto",
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if tcRaw, ok := m["toolConfig"]; ok {
		var tc map[string]json.RawMessage
		if err := json.Unmarshal(tcRaw, &tc); err == nil {
			if fccRaw, ok := tc["functionCallingConfig"]; ok {
				var fcc map[string]json.RawMessage
				if err := json.Unmarshal(fccRaw, &fcc); err == nil {
					if _, ok := fcc["allowedFunctionNames"]; ok {
						t.Error("allowedFunctionNames present in functionCallingConfig, want absent")
					}
				}
			}
		}
	}
}

func TestEncodeGeminiRequest_AllowParallelCallsDropped(t *testing.T) {
	// Gemini has no parallel_tool_calls equivalent — silently drop
	allow := true
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}}},
		},
		ToolChoice: &ToolChoice{
			Type:               "auto",
			AllowParallelCalls: &allow,
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// toolConfig should be present (from ToolChoice.Type="auto") but no parallel_tool_calls field
	if tcRaw, ok := m["toolConfig"]; ok {
		var tc map[string]json.RawMessage
		if err := json.Unmarshal(tcRaw, &tc); err == nil {
			if fccRaw, ok := tc["functionCallingConfig"]; ok {
				var fcc map[string]json.RawMessage
				if err := json.Unmarshal(fccRaw, &fcc); err == nil {
					if _, ok := fcc["parallelToolCalls"]; ok {
						t.Error("parallelToolCalls present in functionCallingConfig, want absent")
					}
					if _, ok := fcc["parallel_tool_calls"]; ok {
						t.Error("parallel_tool_calls present in functionCallingConfig, want absent")
					}
				}
			}
		}
	}
}

func TestDecodeEncodeGeminiRequest_AllowedFunctionNamesRoundTrip(t *testing.T) {
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"toolConfig": {
			"functionCallingConfig": {
				"mode": "ANY",
				"allowedFunctionNames": ["tool_a", "tool_b"]
			}
		}
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if req.ToolChoice == nil || len(req.ToolChoice.AllowedToolNames) != 2 {
		t.Fatalf("AllowedToolNames not decoded correctly")
	}

	_, encoded, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	tcRaw, ok := m["toolConfig"]
	if !ok {
		t.Fatal("toolConfig missing after round-trip")
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal toolConfig: %v", err)
	}
	fccRaw, ok := tc["functionCallingConfig"]
	if !ok {
		t.Fatal("functionCallingConfig missing after round-trip")
	}
	var fcc map[string]json.RawMessage
	if err := json.Unmarshal(fccRaw, &fcc); err != nil {
		t.Fatalf("unmarshal functionCallingConfig: %v", err)
	}
	afnRaw, ok := fcc["allowedFunctionNames"]
	if !ok {
		t.Fatal("allowedFunctionNames missing after round-trip")
	}
	var afn []string
	if err := json.Unmarshal(afnRaw, &afn); err != nil {
		t.Fatalf("unmarshal allowedFunctionNames: %v", err)
	}
	if len(afn) != 2 || afn[0] != "tool_a" || afn[1] != "tool_b" {
		t.Errorf("allowedFunctionNames = %v after round-trip, want [tool_a tool_b]", afn)
	}
}

// --- Phase 2 Thinking Output Controls ---

func TestDecodeGeminiRequest_ThinkingConfig_IncludeThoughts(t *testing.T) {
	t.Run("includeThoughts true", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Think"}]}],
			"generationConfig": {"thinkingConfig": {"thinkingBudget": 1024, "includeThoughts": true}}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.Thinking == nil {
			t.Fatal("Thinking is nil")
		}
		if req.Thinking.IncludeThoughts == nil {
			t.Fatal("IncludeThoughts is nil, want non-nil")
		}
		if !*req.Thinking.IncludeThoughts {
			t.Errorf("IncludeThoughts = false, want true")
		}
		if req.Thinking.Mode != "enabled" {
			t.Errorf("Mode = %q, want enabled", req.Thinking.Mode)
		}
		if req.Thinking.BudgetTokens != 1024 {
			t.Errorf("BudgetTokens = %d, want 1024", req.Thinking.BudgetTokens)
		}
	})

	t.Run("includeThoughts false", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Think"}]}],
			"generationConfig": {"thinkingConfig": {"thinkingBudget": 512, "includeThoughts": false}}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.Thinking == nil {
			t.Fatal("Thinking is nil")
		}
		if req.Thinking.IncludeThoughts == nil {
			t.Fatal("IncludeThoughts is nil, want non-nil")
		}
		if *req.Thinking.IncludeThoughts {
			t.Errorf("IncludeThoughts = true, want false")
		}
	})

	t.Run("includeThoughts absent", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Think"}]}],
			"generationConfig": {"thinkingConfig": {"thinkingBudget": 512}}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.Thinking == nil {
			t.Fatal("Thinking is nil")
		}
		if req.Thinking.IncludeThoughts != nil {
			t.Errorf("IncludeThoughts = %v, want nil", *req.Thinking.IncludeThoughts)
		}
	})
}

func TestDecodeGeminiRequest_ThinkingConfig_ThinkingLevel(t *testing.T) {
	t.Run("thinkingLevel set", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Think hard"}]}],
			"generationConfig": {"thinkingConfig": {"thinkingBudget": 2048, "thinkingLevel": "HIGH"}}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.Thinking == nil {
			t.Fatal("Thinking is nil")
		}
		if req.Thinking.Level != "HIGH" {
			t.Errorf("Level = %q, want HIGH", req.Thinking.Level)
		}
	})

	t.Run("thinkingLevel absent", func(t *testing.T) {
		body := []byte(`{
			"contents": [{"role": "user", "parts": [{"text": "Think"}]}],
			"generationConfig": {"thinkingConfig": {"thinkingBudget": 512}}
		}`)

		req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if req.Thinking == nil {
			t.Fatal("Thinking is nil")
		}
		if req.Thinking.Level != "" {
			t.Errorf("Level = %q, want empty", req.Thinking.Level)
		}
	})
}

func TestDecodeGeminiRequest_ThinkingConfig_BothPhase2Fields(t *testing.T) {
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Deep thought"}]}],
		"generationConfig": {"thinkingConfig": {"thinkingBudget": 4096, "includeThoughts": true, "thinkingLevel": "MEDIUM"}}
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if req.Thinking == nil {
		t.Fatal("Thinking is nil")
	}
	if req.Thinking.Mode != "enabled" {
		t.Errorf("Mode = %q, want enabled", req.Thinking.Mode)
	}
	if req.Thinking.BudgetTokens != 4096 {
		t.Errorf("BudgetTokens = %d, want 4096", req.Thinking.BudgetTokens)
	}
	if req.Thinking.IncludeThoughts == nil {
		t.Fatal("IncludeThoughts is nil")
	}
	if !*req.Thinking.IncludeThoughts {
		t.Errorf("IncludeThoughts = false, want true")
	}
	if req.Thinking.Level != "MEDIUM" {
		t.Errorf("Level = %q, want MEDIUM", req.Thinking.Level)
	}
}

func TestEncodeGeminiRequest_ThinkingConfig_IncludeThoughts(t *testing.T) {
	t.Run("include thoughts true", func(t *testing.T) {
		inclTrue := true
		req := &Request{
			Model: "gemini-2.5-pro",
			Messages: []Message{
				{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Think"}}}},
			},
			Thinking: &ThinkingConfig{
				Mode:            "enabled",
				BudgetTokens:    2048,
				IncludeThoughts: &inclTrue,
			},
		}

		_, body, err := EncodeGeminiRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Request
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}

		if raw.GenerationConfig == nil {
			t.Fatal("GenerationConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig == nil {
			t.Fatal("ThinkingConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig.IncludeThoughts == nil {
			t.Fatal("IncludeThoughts is nil")
		}
		if !*raw.GenerationConfig.ThinkingConfig.IncludeThoughts {
			t.Errorf("IncludeThoughts = false, want true")
		}
		if raw.GenerationConfig.ThinkingConfig.ThinkingBudget != 2048 {
			t.Errorf("ThinkingBudget = %d, want 2048", raw.GenerationConfig.ThinkingConfig.ThinkingBudget)
		}
	})

	t.Run("include thoughts false", func(t *testing.T) {
		inclFalse := false
		req := &Request{
			Model: "gemini-2.5-pro",
			Messages: []Message{
				{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Think"}}}},
			},
			Thinking: &ThinkingConfig{
				Mode:            "enabled",
				BudgetTokens:    1024,
				IncludeThoughts: &inclFalse,
			},
		}

		_, body, err := EncodeGeminiRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Request
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}

		if raw.GenerationConfig == nil {
			t.Fatal("GenerationConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig == nil {
			t.Fatal("ThinkingConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig.IncludeThoughts == nil {
			t.Fatal("IncludeThoughts is nil, want *false")
		}
		if *raw.GenerationConfig.ThinkingConfig.IncludeThoughts {
			t.Errorf("IncludeThoughts = true, want false")
		}
	})

	t.Run("include thoughts nil", func(t *testing.T) {
		req := &Request{
			Model: "gemini-2.5-pro",
			Messages: []Message{
				{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Think"}}}},
			},
			Thinking: &ThinkingConfig{
				Mode:         "enabled",
				BudgetTokens: 1024,
			},
		}

		_, body, err := EncodeGeminiRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Request
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}

		if raw.GenerationConfig == nil {
			t.Fatal("GenerationConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig == nil {
			t.Fatal("ThinkingConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig.IncludeThoughts != nil {
			t.Errorf("IncludeThoughts = %v, want nil", *raw.GenerationConfig.ThinkingConfig.IncludeThoughts)
		}
	})
}

func TestEncodeGeminiRequest_ThinkingConfig_ThinkingLevel(t *testing.T) {
	t.Run("level set", func(t *testing.T) {
		req := &Request{
			Model: "gemini-2.5-pro",
			Messages: []Message{
				{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Think hard"}}}},
			},
			Thinking: &ThinkingConfig{
				Mode:         "enabled",
				BudgetTokens: 4096,
				Level:        "HIGH",
			},
		}

		_, body, err := EncodeGeminiRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Request
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}

		if raw.GenerationConfig == nil {
			t.Fatal("GenerationConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig == nil {
			t.Fatal("ThinkingConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig.ThinkingLevel != "HIGH" {
			t.Errorf("ThinkingLevel = %q, want HIGH", raw.GenerationConfig.ThinkingConfig.ThinkingLevel)
		}
	})

	t.Run("level empty", func(t *testing.T) {
		req := &Request{
			Model: "gemini-2.5-pro",
			Messages: []Message{
				{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Think"}}}},
			},
			Thinking: &ThinkingConfig{
				Mode:         "adaptive",
				BudgetTokens: 0,
			},
		}

		_, body, err := EncodeGeminiRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		var raw gemini.Request
		if err := json.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}

		if raw.GenerationConfig == nil {
			t.Fatal("GenerationConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig == nil {
			t.Fatal("ThinkingConfig is nil")
		}
		if raw.GenerationConfig.ThinkingConfig.ThinkingLevel != "" {
			t.Errorf("ThinkingLevel = %q, want empty", raw.GenerationConfig.ThinkingConfig.ThinkingLevel)
		}
	})
}

func TestGeminiThinkingConfig_Phase2_RoundTrip(t *testing.T) {
	// Round-trip: Decode a Gemini request with includeThoughts + thinkingLevel,
	// encode it back, and verify the wire fields are preserved.
	inclTrue := true
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "Deep thought"}]}],
		"generationConfig": {"thinkingConfig": {"thinkingBudget": 2048, "includeThoughts": true, "thinkingLevel": "LOW"}}
	}`)

	req, err := DecodeGeminiRequest("/v1/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("DecodeGeminiRequest: %v", err)
	}

	// Verify decode
	if req.Thinking == nil {
		t.Fatal("Thinking is nil after decode")
	}
	if req.Thinking.IncludeThoughts == nil || !*req.Thinking.IncludeThoughts {
		t.Errorf("IncludeThoughts after decode = %v, want *true", req.Thinking.IncludeThoughts)
	}
	if req.Thinking.Level != "LOW" {
		t.Errorf("Level after decode = %q, want LOW", req.Thinking.Level)
	}

	// Set model to satisfy encode
	req.Model = "gemini-2.5-pro"

	// Encode back
	_, reEncoded, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("EncodeGeminiRequest: %v", err)
	}

	var raw gemini.Request
	if err := json.Unmarshal(reEncoded, &raw); err != nil {
		t.Fatalf("unmarshal re-encoded: %v", err)
	}

	if raw.GenerationConfig == nil {
		t.Fatal("GenerationConfig is nil after round-trip")
	}
	if raw.GenerationConfig.ThinkingConfig == nil {
		t.Fatal("ThinkingConfig is nil after round-trip")
	}
	gtc := raw.GenerationConfig.ThinkingConfig
	if gtc.ThinkingBudget != 2048 {
		t.Errorf("ThinkingBudget after round-trip = %d, want 2048", gtc.ThinkingBudget)
	}
	if gtc.IncludeThoughts == nil || !*gtc.IncludeThoughts {
		t.Errorf("IncludeThoughts after round-trip = %v, want *true", gtc.IncludeThoughts)
	}
	if gtc.ThinkingLevel != "LOW" {
		t.Errorf("ThinkingLevel after round-trip = %q, want LOW", gtc.ThinkingLevel)
	}
	_ = inclTrue // used indirectly via the JSON body
}

func TestEncodeGeminiRequest_ThinkingConfig_OnlyPhase2FieldsNoMode(t *testing.T) {
	// When Thinking.Mode is not set but IncludeThoughts or Level is set,
	// the thinking config should still be written to the wire.
	inclTrue := true
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Think"}}}},
		},
		Thinking: &ThinkingConfig{
			IncludeThoughts: &inclTrue,
			Level:           "MEDIUM",
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	if raw.GenerationConfig == nil {
		t.Fatal("GenerationConfig is nil")
	}
	if raw.GenerationConfig.ThinkingConfig == nil {
		t.Fatal("ThinkingConfig is nil — Phase 2 fields should cause ThinkingConfig to be written")
	}
	if raw.GenerationConfig.ThinkingConfig.IncludeThoughts == nil || !*raw.GenerationConfig.ThinkingConfig.IncludeThoughts {
		t.Errorf("IncludeThoughts = %v, want *true", raw.GenerationConfig.ThinkingConfig.IncludeThoughts)
	}
	if raw.GenerationConfig.ThinkingConfig.ThinkingLevel != "MEDIUM" {
		t.Errorf("ThinkingLevel = %q, want MEDIUM", raw.GenerationConfig.ThinkingConfig.ThinkingLevel)
	}
}
