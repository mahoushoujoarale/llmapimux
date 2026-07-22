package e2e_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	llmapimux "github.com/mahoushoujoarale/llmapimux"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	anthropicopt "github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaiopt "github.com/openai/openai-go/v3/option"
	"google.golang.org/genai"
)

func TestFallbackE2E_OpenAIChat_NonStreaming(t *testing.T) {
	primary := failingFakeServer(t, llmapimux.ProtocolOpenAIChat, 500)
	fallback := succeedingFakeServer(t, llmapimux.ProtocolOpenAIChat)

	m := newFallbackTestMux(t, llmapimux.ProtocolOpenAIChat, []fallbackTarget{
		{Server: primary, Protocol: llmapimux.ProtocolOpenAIChat, Model: "test-model"},
		{Server: fallback, Protocol: llmapimux.ProtocolOpenAIChat, Model: "test-model"},
	})

	muxServer := newE2EMuxServer(t, "/v1/chat/completions", m.OpenAIChatHandler())
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		openaiopt.WithAPIKey("sk-test-inbound"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, primary.URL, fallback.URL)),
		openaiopt.WithMaxRetries(0),
	)

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: "test-model",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("ping"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected non-empty choices")
	}
	if !strings.Contains(resp.Choices[0].Message.Content, "hello from fallback") {
		t.Fatalf("content = %q, want to contain 'hello from fallback'", resp.Choices[0].Message.Content)
	}
}

func TestFallbackE2E_OpenAIChat_Streaming(t *testing.T) {
	primary := failingFakeServer(t, llmapimux.ProtocolOpenAIChat, 500)
	fallback := succeedingStreamFakeServer(t, llmapimux.ProtocolOpenAIChat)

	m := newFallbackTestMux(t, llmapimux.ProtocolOpenAIChat, []fallbackTarget{
		{Server: primary, Protocol: llmapimux.ProtocolOpenAIChat, Model: "test-model"},
		{Server: fallback, Protocol: llmapimux.ProtocolOpenAIChat, Model: "test-model"},
	})

	muxServer := newE2EMuxServer(t, "/v1/chat/completions", m.OpenAIChatHandler())
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		openaiopt.WithAPIKey("sk-test-inbound"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, primary.URL, fallback.URL)),
		openaiopt.WithMaxRetries(0),
	)

	stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Model: "test-model",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("ping"),
		},
	})

	var fullText string
	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			fullText += chunk.Choices[0].Delta.Content
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(fullText, "hello") {
		t.Fatalf("streamed text = %q, want to contain 'hello'", fullText)
	}
}

func TestFallbackE2E_Anthropic_NonStreaming(t *testing.T) {
	primary := failingFakeServer(t, llmapimux.ProtocolAnthropic, 500)
	fallback := succeedingFakeServer(t, llmapimux.ProtocolAnthropic)

	m := newFallbackTestMux(t, llmapimux.ProtocolAnthropic, []fallbackTarget{
		{Server: primary, Protocol: llmapimux.ProtocolAnthropic, Model: "test-model"},
		{Server: fallback, Protocol: llmapimux.ProtocolAnthropic, Model: "test-model"},
	})

	muxServer := newE2EMuxServer(t, "/v1/messages", m.AnthropicHandler())
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		anthropicopt.WithAPIKey("sk-test-inbound"),
		anthropicopt.WithBaseURL(muxServer.URL),
		anthropicopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, primary.URL, fallback.URL)),
		anthropicopt.WithMaxRetries(0),
	)

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     "test-model",
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("ping")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	textBlock, ok := resp.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", resp.Content[0].AsAny())
	}
	if !strings.Contains(textBlock.Text, "hello from fallback") {
		t.Fatalf("text = %q, want to contain 'hello from fallback'", textBlock.Text)
	}
}

func TestFallbackE2E_Anthropic_Streaming(t *testing.T) {
	primary := failingFakeServer(t, llmapimux.ProtocolAnthropic, 500)
	fallback := succeedingStreamFakeServer(t, llmapimux.ProtocolAnthropic)

	m := newFallbackTestMux(t, llmapimux.ProtocolAnthropic, []fallbackTarget{
		{Server: primary, Protocol: llmapimux.ProtocolAnthropic, Model: "test-model"},
		{Server: fallback, Protocol: llmapimux.ProtocolAnthropic, Model: "test-model"},
	})

	muxServer := newE2EMuxServer(t, "/v1/messages", m.AnthropicHandler())
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		anthropicopt.WithAPIKey("sk-test-inbound"),
		anthropicopt.WithBaseURL(muxServer.URL),
		anthropicopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, primary.URL, fallback.URL)),
		anthropicopt.WithMaxRetries(0),
	)

	stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Model:     "test-model",
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("ping")),
		},
	})

	var eventTypes []string
	message := anthropic.Message{}
	for stream.Next() {
		event := stream.Current()
		eventTypes = append(eventTypes, fmt.Sprintf("%T", event.AsAny()))
		if err := message.Accumulate(event); err != nil {
			t.Fatalf("accumulate %T: %v (seen=%v)", event.AsAny(), err, eventTypes)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if len(message.Content) == 0 {
		t.Fatal("expected non-empty accumulated content")
	}
	textBlock, ok := message.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", message.Content[0].AsAny())
	}
	if !strings.Contains(textBlock.Text, "hello") {
		t.Fatalf("text = %q, want to contain 'hello'", textBlock.Text)
	}
}

func TestFallbackE2E_Gemini_NonStreaming(t *testing.T) {
	primary := failingFakeServer(t, llmapimux.ProtocolGemini, 500)
	fallback := succeedingFakeServer(t, llmapimux.ProtocolGemini)

	m := newFallbackTestMux(t, llmapimux.ProtocolGemini, []fallbackTarget{
		{Server: primary, Protocol: llmapimux.ProtocolGemini, Model: "test-model"},
		{Server: fallback, Protocol: llmapimux.ProtocolGemini, Model: "test-model"},
	})

	muxServer := newE2EMuxServer(t, "/v1/models/", m.GeminiHandler())
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "fake-gemini-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
		HTTPClient: newLocalOnlyHTTPClient(t, muxServer.URL, primary.URL, fallback.URL),
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := newE2EContext(t)
	defer cancel()

	resp, err := client.Models.GenerateContent(ctx, "test-model",
		[]*genai.Content{
			genai.NewContentFromText("ping", genai.RoleUser),
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp == nil || len(resp.Candidates) == 0 {
		t.Fatal("expected non-empty candidates")
	}
	text := resp.Text()
	if !strings.Contains(text, "hello from fallback") {
		t.Fatalf("text = %q, want to contain 'hello from fallback'", text)
	}
}

func TestFallbackE2E_OpenAIChat_CrossProtocolFallback(t *testing.T) {
	primary := failingFakeServer(t, llmapimux.ProtocolOpenAIChat, 500)
	fallback := succeedingFakeServer(t, llmapimux.ProtocolAnthropic)

	m := newFallbackTestMux(t, llmapimux.ProtocolOpenAIChat, []fallbackTarget{
		{Server: primary, Protocol: llmapimux.ProtocolOpenAIChat, Model: "test-model"},
		{Server: fallback, Protocol: llmapimux.ProtocolAnthropic, Model: "test-model"},
	})

	muxServer := newE2EMuxServer(t, "/v1/chat/completions", m.OpenAIChatHandler())
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		openaiopt.WithAPIKey("sk-test-inbound"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, primary.URL, fallback.URL)),
		openaiopt.WithMaxRetries(0),
	)

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: "test-model",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("ping"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected non-empty choices")
	}
	if !strings.Contains(resp.Choices[0].Message.Content, "hello from fallback") {
		t.Fatalf("content = %q, want to contain 'hello from fallback'", resp.Choices[0].Message.Content)
	}
}

func TestFallbackE2E_Anthropic_CrossProtocolFallback(t *testing.T) {
	primary := failingFakeServer(t, llmapimux.ProtocolAnthropic, 500)
	fallback := succeedingFakeServer(t, llmapimux.ProtocolOpenAIChat)

	m := newFallbackTestMux(t, llmapimux.ProtocolAnthropic, []fallbackTarget{
		{Server: primary, Protocol: llmapimux.ProtocolAnthropic, Model: "test-model"},
		{Server: fallback, Protocol: llmapimux.ProtocolOpenAIChat, Model: "test-model"},
	})

	muxServer := newE2EMuxServer(t, "/v1/messages", m.AnthropicHandler())
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		anthropicopt.WithAPIKey("sk-test-inbound"),
		anthropicopt.WithBaseURL(muxServer.URL),
		anthropicopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, primary.URL, fallback.URL)),
		anthropicopt.WithMaxRetries(0),
	)

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     "test-model",
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("ping")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	textBlock, ok := resp.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", resp.Content[0].AsAny())
	}
	if !strings.Contains(textBlock.Text, "hello from fallback") {
		t.Fatalf("text = %q, want to contain 'hello from fallback'", textBlock.Text)
	}
}

func TestFallbackE2E_OpenAIChat_AllFail(t *testing.T) {
	primary := failingFakeServer(t, llmapimux.ProtocolOpenAIChat, 500)
	secondary := failingFakeServer(t, llmapimux.ProtocolOpenAIChat, 500)

	m := newFallbackTestMux(t, llmapimux.ProtocolOpenAIChat, []fallbackTarget{
		{Server: primary, Protocol: llmapimux.ProtocolOpenAIChat, Model: "test-model"},
		{Server: secondary, Protocol: llmapimux.ProtocolOpenAIChat, Model: "test-model"},
	})

	muxServer := newE2EMuxServer(t, "/v1/chat/completions", m.OpenAIChatHandler())
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		openaiopt.WithAPIKey("sk-test-inbound"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, primary.URL, secondary.URL)),
		openaiopt.WithMaxRetries(0),
	)

	_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: "test-model",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("ping"),
		},
	})
	if err == nil {
		t.Fatal("expected error when all targets fail")
	}
	// Verify we got a recognizable error (not a panic or garbled response)
	t.Logf("got expected error: %v", err)
}
