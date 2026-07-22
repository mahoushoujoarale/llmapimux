package e2e_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	llmapimux "github.com/mahoushoujoarale/llmapimux"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaiopt "github.com/openai/openai-go/v3/option"
)

// ============================================================
// Test 1: Anthropic SDK image (base64) → OpenAI Chat upstream
// ============================================================

func TestE2E_Content_AnthropicSDK_ImageBase64_ToOpenAIChat(t *testing.T) {
	const tinyPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"A tiny pixel"},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":5,"total_tokens":55}}`))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/messages",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"claude-sonnet-4-20250514": "gpt-4o-mini"}).AnthropicHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		option.WithAPIKey("sk-anthropic-inbound"),
		option.WithBaseURL(muxServer.URL),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("describe this image"),
				anthropic.NewImageBlockBase64("image/png", tinyPNG),
			),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify upstream request body.
	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/chat/completions")
	body := decodeCapturedJSONBody(t, got)
	messages, _ := body["messages"].([]any)
	if len(messages) != 1 {
		t.Fatalf("messages len = %d, want 1", len(messages))
	}
	msg, _ := messages[0].(map[string]any)
	content, ok := msg["content"].([]any)
	if !ok {
		t.Fatalf("message.content type = %T, want []any", msg["content"])
	}
	if len(content) != 2 {
		t.Fatalf("message.content len = %d, want 2", len(content))
	}

	// Check that we have one text part and one image_url part.
	var foundText, foundImage bool
	for _, c := range content {
		part, _ := c.(map[string]any)
		switch part["type"] {
		case "text":
			if part["text"] != "describe this image" {
				t.Fatalf("text part text = %v, want 'describe this image'", part["text"])
			}
			foundText = true
		case "image_url":
			imageURL, _ := part["image_url"].(map[string]any)
			url, _ := imageURL["url"].(string)
			if !strings.Contains(url, "data:image/png;base64,") {
				t.Fatalf("image_url.url = %q, want data:image/png;base64,... prefix", url)
			}
			foundImage = true
		}
	}
	if !foundText {
		t.Fatal("expected text part in upstream content")
	}
	if !foundImage {
		t.Fatal("expected image_url part in upstream content")
	}

	// Verify SDK response.
	if len(resp.Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	textBlock, ok := resp.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", resp.Content[0].AsAny())
	}
	if textBlock.Text != "A tiny pixel" {
		t.Fatalf("content[0].text = %q, want 'A tiny pixel'", textBlock.Text)
	}
}

// ============================================================
// Test 2: OpenAI Chat SDK image URL → Anthropic upstream
// ============================================================

func TestE2E_Content_OpenAIChatSDK_ImageURL_ToAnthropic(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"An image from URL"}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":5}}`))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/chat/completions",
		newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, upstream.URL, "sk-ant-upstream", map[string]string{"gpt-4o-mini": "claude-sonnet-4-20250514"}).OpenAIChatHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		openaiopt.WithAPIKey("sk-openai-inbound"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		openaiopt.WithMaxRetries(0),
	)

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
				openai.TextContentPart("describe this image"),
				openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
					URL: "https://example.com/img.png",
				}),
			}),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify upstream request body.
	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/messages")
	body := decodeCapturedJSONBody(t, got)
	messages, _ := body["messages"].([]any)
	if len(messages) != 1 {
		t.Fatalf("messages len = %d, want 1", len(messages))
	}
	msg, _ := messages[0].(map[string]any)
	content, ok := msg["content"].([]any)
	if !ok {
		t.Fatalf("message.content type = %T, want []any", msg["content"])
	}

	// Check that image block has source.type == "url".
	var foundImage bool
	for _, c := range content {
		part, _ := c.(map[string]any)
		if part["type"] == "image" {
			source, _ := part["source"].(map[string]any)
			if source == nil {
				t.Fatal("expected non-nil image source")
			}
			if source["type"] != "url" {
				t.Fatalf("image source.type = %v, want 'url'", source["type"])
			}
			if source["url"] != "https://example.com/img.png" {
				t.Fatalf("image source.url = %v, want 'https://example.com/img.png'", source["url"])
			}
			foundImage = true
		}
	}
	if !foundImage {
		t.Fatal("expected image block in upstream content")
	}

	// Verify SDK response.
	if len(resp.Choices) == 0 {
		t.Fatal("expected non-empty choices")
	}
	if resp.Choices[0].Message.Content != "An image from URL" {
		t.Fatalf("content = %q, want 'An image from URL'", resp.Choices[0].Message.Content)
	}
}

// ============================================================
// Test 3: System prompt — Anthropic → OpenAI developer message
// ============================================================

func TestE2E_Content_SystemPrompt_AnthropicToOpenAIChat(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"I am helpful"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/messages",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"claude-sonnet-4-20250514": "gpt-4o-mini"}).AnthropicHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		option.WithAPIKey("sk-anthropic-inbound"),
		option.WithBaseURL(muxServer.URL),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 128,
		System: []anthropic.TextBlockParam{
			{Text: "You are helpful"},
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify upstream request body has developer message.
	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/chat/completions")
	body := decodeCapturedJSONBody(t, got)
	messages, _ := body["messages"].([]any)

	// Should have at least 2 messages: developer + user.
	if len(messages) < 2 {
		t.Fatalf("messages len = %d, want at least 2", len(messages))
	}

	// Find the developer message and verify it contains the system prompt.
	var foundDeveloper, foundUser bool
	for _, m := range messages {
		msg, _ := m.(map[string]any)
		role, _ := msg["role"].(string)
		switch role {
		case "developer":
			// Content can be a string or array of parts.
			contentStr, strOK := msg["content"].(string)
			contentArr, arrOK := msg["content"].([]any)
			if strOK {
				if !strings.Contains(contentStr, "You are helpful") {
					t.Fatalf("developer content = %q, want to contain 'You are helpful'", contentStr)
				}
			} else if arrOK {
				found := false
				for _, c := range contentArr {
					part, _ := c.(map[string]any)
					if text, ok := part["text"].(string); ok && strings.Contains(text, "You are helpful") {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("developer content parts do not contain 'You are helpful': %v", contentArr)
				}
			} else {
				t.Fatalf("developer content type = %T, expected string or array", msg["content"])
			}
			foundDeveloper = true
		case "user":
			foundUser = true
		}
	}
	if !foundDeveloper {
		t.Fatal("expected developer message in upstream request")
	}
	if !foundUser {
		t.Fatal("expected user message in upstream request")
	}

	// Verify SDK response.
	if len(resp.Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	textBlock, ok := resp.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", resp.Content[0].AsAny())
	}
	if textBlock.Text != "I am helpful" {
		t.Fatalf("content[0].text = %q, want 'I am helpful'", textBlock.Text)
	}
}

// ============================================================
// Test 4: System prompt — OpenAI developer → Anthropic system
// ============================================================

func TestE2E_Content_SystemPrompt_OpenAIChatToAnthropic(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"OK, being concise."}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":5}}`))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/chat/completions",
		newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, upstream.URL, "sk-ant-upstream", map[string]string{"gpt-4o-mini": "claude-sonnet-4-20250514"}).OpenAIChatHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		openaiopt.WithAPIKey("sk-openai-inbound"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		openaiopt.WithMaxRetries(0),
	)

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.DeveloperMessage("Be concise"),
			openai.UserMessage("hello"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify upstream request body has system field (not in messages).
	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/messages")
	body := decodeCapturedJSONBody(t, got)

	// system field should contain "Be concise".
	systemField, ok := body["system"]
	if !ok || systemField == nil {
		t.Fatal("expected 'system' field in upstream request body")
	}

	// system can be a string or array of text blocks.
	switch s := systemField.(type) {
	case string:
		if !strings.Contains(s, "Be concise") {
			t.Fatalf("system = %q, want to contain 'Be concise'", s)
		}
	case []any:
		found := false
		for _, item := range s {
			block, _ := item.(map[string]any)
			if text, ok := block["text"].(string); ok && strings.Contains(text, "Be concise") {
				found = true
				break
			}
		}
		if !found {
			raw, _ := json.Marshal(s)
			t.Fatalf("system blocks do not contain 'Be concise': %s", raw)
		}
	default:
		t.Fatalf("system type = %T, expected string or array", systemField)
	}

	// Verify messages do not contain a developer/system role.
	messages, _ := body["messages"].([]any)
	for _, m := range messages {
		msg, _ := m.(map[string]any)
		role, _ := msg["role"].(string)
		if role == "developer" || role == "system" {
			t.Fatalf("unexpected %q role in messages (should be in system field)", role)
		}
	}

	// Verify SDK response.
	if len(resp.Choices) == 0 {
		t.Fatal("expected non-empty choices")
	}
	if resp.Choices[0].Message.Content != "OK, being concise." {
		t.Fatalf("content = %q, want 'OK, being concise.'", resp.Choices[0].Message.Content)
	}
}

// ============================================================
// Test 5: Multi-turn conversation — Anthropic → OpenAI Chat
// ============================================================

func TestE2E_Content_MultiTurn_AnthropicToOpenAIChat(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Glad to hear it"},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":5,"total_tokens":35}}`))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/messages",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"claude-sonnet-4-20250514": "gpt-4o-mini"}).AnthropicHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		option.WithAPIKey("sk-anthropic-inbound"),
		option.WithBaseURL(muxServer.URL),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Hello")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi there")),
			anthropic.NewUserMessage(anthropic.NewTextBlock("How are you?")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("I'm fine")),
			anthropic.NewUserMessage(anthropic.NewTextBlock("Great")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify upstream request body preserves all 5 turns.
	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/chat/completions")
	body := decodeCapturedJSONBody(t, got)
	messages, _ := body["messages"].([]any)
	if len(messages) != 5 {
		t.Fatalf("messages len = %d, want 5", len(messages))
	}

	expectedRoles := []string{"user", "assistant", "user", "assistant", "user"}
	expectedTexts := []string{"Hello", "Hi there", "How are you?", "I'm fine", "Great"}
	for i, m := range messages {
		msg, _ := m.(map[string]any)
		role, _ := msg["role"].(string)
		if role != expectedRoles[i] {
			t.Fatalf("messages[%d].role = %q, want %q", i, role, expectedRoles[i])
		}

		// Content can be string or array of parts.
		var text string
		switch c := msg["content"].(type) {
		case string:
			text = c
		case []any:
			if len(c) > 0 {
				part, _ := c[0].(map[string]any)
				text, _ = part["text"].(string)
			}
		}
		if text != expectedTexts[i] {
			t.Fatalf("messages[%d].content text = %q, want %q", i, text, expectedTexts[i])
		}
	}

	// Verify SDK response.
	if len(resp.Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	textBlock, ok := resp.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", resp.Content[0].AsAny())
	}
	if textBlock.Text != "Glad to hear it" {
		t.Fatalf("content[0].text = %q, want 'Glad to hear it'", textBlock.Text)
	}
}

// ============================================================
// Test 6: Error format — Anthropic SDK ← OpenAI Chat upstream 400
// ============================================================

func TestE2E_Content_ErrorFormat_AnthropicToOpenAIChat(t *testing.T) {
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid request","type":"invalid_request_error","param":"model","code":"model_not_found"}}`))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/messages",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"claude-sonnet-4-20250514": "gpt-4o-mini"}).AnthropicHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		option.WithAPIKey("sk-anthropic-inbound"),
		option.WithBaseURL(muxServer.URL),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	_, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
		},
	})
	if err == nil {
		t.Fatal("expected error from SDK, got nil")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Fatalf("expected error to contain status 400, got: %v", err)
	}
}

// ============================================================
// Test 7: Error format — OpenAI Chat SDK ← Anthropic upstream 400
// ============================================================

func TestE2E_Content_ErrorFormat_OpenAIChatToAnthropic(t *testing.T) {
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens must be positive"}}`))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/chat/completions",
		newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, upstream.URL, "sk-ant-upstream", map[string]string{"gpt-4o-mini": "claude-sonnet-4-20250514"}).OpenAIChatHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		openaiopt.WithAPIKey("sk-openai-inbound"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		openaiopt.WithMaxRetries(0),
	)

	_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hello"),
		},
	})
	if err == nil {
		t.Fatal("expected error from SDK, got nil")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Fatalf("expected error to contain status 400, got: %v", err)
	}
}

// Ensure imports are used.
var _ = fmt.Sprintf
