package e2e_test

import (
	"context"
	"net/http"
	"os"
	"testing"

	llmapimux "github.com/mahoushoujoarale/llmapimux"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaiopt "github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

func TestMain(m *testing.M) {
	// Load .env from project root (two levels up from tests/e2e/).
	_ = loadEnvFile("../../.env")
	os.Exit(m.Run())
}

// --- Provider helpers ---

func openAIBaseURL(t *testing.T) string {
	t.Helper()
	skipIfEnvMissing(t, "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")
	return os.Getenv("OPENAI_BASE_URL")
}

func openAIAPIKey() string { return os.Getenv("OPENAI_API_KEY") }
func openAIModel() string  { return os.Getenv("OPENAI_MODEL") }

func geminiBaseURL(t *testing.T) string {
	t.Helper()
	skipIfEnvMissing(t, "GEMINI_BASE_URL", "GEMINI_API_KEY", "GEMINI_MODEL")
	return os.Getenv("GEMINI_BASE_URL")
}

func geminiAPIKey() string { return os.Getenv("GEMINI_API_KEY") }
func geminiModel() string  { return os.Getenv("GEMINI_MODEL") }

func anthropicBaseURL(t *testing.T) string {
	t.Helper()
	skipIfEnvMissing(t, "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL")
	return os.Getenv("ANTHROPIC_BASE_URL")
}

func anthropicAPIKey() string { return os.Getenv("ANTHROPIC_API_KEY") }
func anthropicModel() string  { return os.Getenv("ANTHROPIC_MODEL") }

func openAIResponsesBaseURL(t *testing.T) string {
	t.Helper()
	skipIfEnvMissing(t, "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")
	return os.Getenv("OPENAI_BASE_URL")
}

func collectResponsesOutputText(resp *responses.Response) string {
	if resp == nil {
		return ""
	}
	var text string
	for _, item := range resp.Output {
		if msg := item.AsMessage(); msg.Type == "message" {
			for _, c := range msg.Content {
				if c.Type == "output_text" {
					text += c.Text
				}
			}
		}
	}
	return text
}

// ============================================================
// Passthrough tests (same protocol in and out)
// ============================================================

func TestRealAPI_Passthrough_OpenAIChat(t *testing.T) {
	baseURL := openAIBaseURL(t)
	model := openAIModel()

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/chat/completions": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, baseURL, openAIAPIKey(), map[string]string{model: model}).OpenAIChatHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model: model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Say hello in one word."),
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		text := resp.Choices[0].Message.Content
		if text == "" {
			t.Fatal("expected non-empty response text")
		}
		t.Logf("response: %s", text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Model: model,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Say hello in one word."),
			},
		})

		var chunks int
		var fullText string
		for stream.Next() {
			chunk := stream.Current()
			chunks++
			if len(chunk.Choices) > 0 {
				fullText += chunk.Choices[0].Delta.Content
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		if chunks == 0 {
			t.Fatal("expected at least one streaming chunk")
		}
		if fullText == "" {
			t.Fatal("expected non-empty streamed text")
		}
		t.Logf("streamed %d chunks, text: %s", chunks, fullText)
	})
}

func TestRealAPI_Passthrough_Gemini(t *testing.T) {
	baseURL := geminiBaseURL(t)
	model := geminiModel()

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/models/": newTestMuxWithModelMap(llmapimux.ProtocolGemini, baseURL, geminiAPIKey(), map[string]string{model: model}).GeminiHandler(),
	})
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "unused-inbound-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Models.GenerateContent(ctx, model,
			[]*genai.Content{genai.NewContentFromText("Say hello in one word.", genai.RoleUser)},
			nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil || len(resp.Candidates) == 0 {
			t.Fatal("expected non-empty candidates")
		}
		text := resp.Text()
		if text == "" {
			t.Fatal("expected non-empty response text")
		}
		t.Logf("response: %s", text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		var chunks int
		var fullText string
		for resp, err := range client.Models.GenerateContentStream(ctx, model,
			[]*genai.Content{genai.NewContentFromText("Say hello in one word.", genai.RoleUser)},
			nil,
		) {
			if err != nil {
				t.Fatal(err)
			}
			chunks++
			if resp != nil && len(resp.Candidates) > 0 {
				fullText += resp.Text()
			}
		}
		if chunks == 0 {
			t.Fatal("expected at least one streaming chunk")
		}
		if fullText == "" {
			t.Fatal("expected non-empty streamed text")
		}
		t.Logf("streamed %d chunks, text: %s", chunks, fullText)
	})
}

// ============================================================
// Cross-protocol tests
// ============================================================

func TestRealAPI_Cross_AnthropicToOpenAI(t *testing.T) {
	baseURL := openAIBaseURL(t)
	const inboundModel = "test-model-ant-to-oai"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/messages": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).AnthropicHandler(),
	})
	defer muxServer.Close()

	client := anthropic.NewClient(
		option.WithAPIKey("unused-inbound-key"),
		option.WithBaseURL(muxServer.URL),
		option.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 64,
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("Say hello in one word.")),
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
			t.Fatalf("content[0] type = %T, want TextBlock", resp.Content[0].AsAny())
		}
		if textBlock.Text == "" {
			t.Fatal("expected non-empty text")
		}
		t.Logf("response: %s", textBlock.Text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 64,
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("Say hello in one word.")),
			},
		})

		message := anthropic.Message{}
		var eventCount int
		for stream.Next() {
			event := stream.Current()
			eventCount++
			if err := message.Accumulate(event); err != nil {
				t.Fatalf("accumulate: %v", err)
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		if eventCount == 0 {
			t.Fatal("expected streaming events")
		}
		if len(message.Content) == 0 {
			t.Fatal("expected non-empty accumulated content")
		}
		textBlock, ok := message.Content[0].AsAny().(anthropic.TextBlock)
		if !ok {
			t.Fatalf("content[0] type = %T, want TextBlock", message.Content[0].AsAny())
		}
		if textBlock.Text == "" {
			t.Fatal("expected non-empty text")
		}
		t.Logf("streamed %d events, text: %s", eventCount, textBlock.Text)
	})
}

func TestRealAPI_Cross_AnthropicToGemini(t *testing.T) {
	baseURL := geminiBaseURL(t)
	const inboundModel = "test-model-ant-to-gem"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/messages": newTestMuxWithModelMap(llmapimux.ProtocolGemini, baseURL, geminiAPIKey(), map[string]string{inboundModel: geminiModel()}).AnthropicHandler(),
	})
	defer muxServer.Close()

	client := anthropic.NewClient(
		option.WithAPIKey("unused-inbound-key"),
		option.WithBaseURL(muxServer.URL),
		option.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 64,
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("Say hello in one word.")),
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
			t.Fatalf("content[0] type = %T, want TextBlock", resp.Content[0].AsAny())
		}
		if textBlock.Text == "" {
			t.Fatal("expected non-empty text")
		}
		t.Logf("response: %s", textBlock.Text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 64,
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("Say hello in one word.")),
			},
		})

		message := anthropic.Message{}
		var eventCount int
		for stream.Next() {
			event := stream.Current()
			eventCount++
			if err := message.Accumulate(event); err != nil {
				t.Fatalf("accumulate: %v", err)
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		if eventCount == 0 {
			t.Fatal("expected streaming events")
		}
		if len(message.Content) == 0 {
			t.Fatal("expected non-empty accumulated content")
		}
		textBlock, ok := message.Content[0].AsAny().(anthropic.TextBlock)
		if !ok {
			t.Fatalf("content[0] type = %T, want TextBlock", message.Content[0].AsAny())
		}
		if textBlock.Text == "" {
			t.Fatal("expected non-empty text")
		}
		t.Logf("streamed %d events, text: %s", eventCount, textBlock.Text)
	})
}

func TestRealAPI_Cross_GeminiToOpenAI(t *testing.T) {
	baseURL := openAIBaseURL(t)
	const inboundModel = "test-model-gem-to-oai"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/models/": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).GeminiHandler(),
	})
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "unused-inbound-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Models.GenerateContent(ctx, inboundModel,
			[]*genai.Content{genai.NewContentFromText("Say hello in one word.", genai.RoleUser)},
			nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil || len(resp.Candidates) == 0 {
			t.Fatal("expected non-empty candidates")
		}
		text := resp.Text()
		if text == "" {
			t.Fatal("expected non-empty response text")
		}
		t.Logf("response: %s", text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		var chunks int
		var fullText string
		for resp, err := range client.Models.GenerateContentStream(ctx, inboundModel,
			[]*genai.Content{genai.NewContentFromText("Say hello in one word.", genai.RoleUser)},
			nil,
		) {
			if err != nil {
				t.Fatal(err)
			}
			chunks++
			if resp != nil && len(resp.Candidates) > 0 {
				fullText += resp.Text()
			}
		}
		if chunks == 0 {
			t.Fatal("expected at least one streaming chunk")
		}
		if fullText == "" {
			t.Fatal("expected non-empty streamed text")
		}
		t.Logf("streamed %d chunks, text: %s", chunks, fullText)
	})
}

func TestRealAPI_Passthrough_OpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	model := openAIModel()

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/responses": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, baseURL, openAIAPIKey(), map[string]string{model: model}).OpenAIResponsesHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
			Model: model,
			Input: responses.ResponseNewParamsInputUnion{
				OfString: param.NewOpt("Say hello in one word."),
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		text := collectResponsesOutputText(resp)
		if text == "" {
			t.Fatal("expected non-empty response text")
		}
		t.Logf("response: %s", text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		stream := client.Responses.NewStreaming(ctx, responses.ResponseNewParams{
			Model: model,
			Input: responses.ResponseNewParamsInputUnion{
				OfString: param.NewOpt("Say hello in one word."),
			},
		})

		var chunks int
		var fullText string
		for stream.Next() {
			event := stream.Current()
			chunks++
			if delta := event.AsResponseOutputTextDelta(); delta.Type == "response.output_text.delta" {
				fullText += delta.Delta
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		if chunks == 0 {
			t.Fatal("expected at least one streaming event")
		}
		if fullText == "" {
			t.Fatal("expected non-empty streamed text")
		}
		t.Logf("streamed %d chunks, text: %s", chunks, fullText)
	})
}

func TestRealAPI_Cross_AnthropicToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "test-model-ant-to-oairesp"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/messages": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).AnthropicHandler(),
	})
	defer muxServer.Close()

	client := anthropic.NewClient(
		option.WithAPIKey("unused-inbound-key"),
		option.WithBaseURL(muxServer.URL),
		option.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 64,
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("Say hello in one word.")),
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
			t.Fatalf("content[0] type = %T, want TextBlock", resp.Content[0].AsAny())
		}
		if textBlock.Text == "" {
			t.Fatal("expected non-empty text")
		}
		t.Logf("response: %s", textBlock.Text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 64,
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("Say hello in one word.")),
			},
		})

		message := anthropic.Message{}
		var eventCount int
		for stream.Next() {
			event := stream.Current()
			eventCount++
			if err := message.Accumulate(event); err != nil {
				t.Fatalf("accumulate: %v", err)
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		if eventCount == 0 {
			t.Fatal("expected streaming events")
		}
		if len(message.Content) == 0 {
			t.Fatal("expected non-empty accumulated content")
		}
		textBlock, ok := message.Content[0].AsAny().(anthropic.TextBlock)
		if !ok {
			t.Fatalf("content[0] type = %T, want TextBlock", message.Content[0].AsAny())
		}
		if textBlock.Text == "" {
			t.Fatal("expected non-empty text")
		}
		t.Logf("streamed %d events, text: %s", eventCount, textBlock.Text)
	})
}

func TestRealAPI_Cross_GeminiToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "test-model-gem-to-oairesp"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/models/": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).GeminiHandler(),
	})
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "unused-inbound-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Models.GenerateContent(ctx, inboundModel,
			[]*genai.Content{genai.NewContentFromText("Say hello in one word.", genai.RoleUser)},
			nil,
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil || len(resp.Candidates) == 0 {
			t.Fatal("expected non-empty candidates")
		}
		text := resp.Text()
		if text == "" {
			t.Fatal("expected non-empty response text")
		}
		t.Logf("response: %s", text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		var chunks int
		var fullText string
		for resp, err := range client.Models.GenerateContentStream(ctx, inboundModel,
			[]*genai.Content{genai.NewContentFromText("Say hello in one word.", genai.RoleUser)},
			nil,
		) {
			if err != nil {
				t.Fatal(err)
			}
			chunks++
			if resp != nil && len(resp.Candidates) > 0 {
				fullText += resp.Text()
			}
		}
		if chunks == 0 {
			t.Fatal("expected at least one streaming chunk")
		}
		if fullText == "" {
			t.Fatal("expected non-empty streamed text")
		}
		t.Logf("streamed %d chunks, text: %s", chunks, fullText)
	})
}

func TestRealAPI_Cross_OpenAIChatToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "test-model-oai-to-oairesp"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/chat/completions": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).OpenAIChatHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model: inboundModel,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Say hello in one word."),
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		text := resp.Choices[0].Message.Content
		if text == "" {
			t.Fatal("expected non-empty response text")
		}
		t.Logf("response: %s", text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newRealAPIContext(t)
		defer cancel()

		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Model: inboundModel,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Say hello in one word."),
			},
		})

		var chunks int
		var fullText string
		for stream.Next() {
			chunk := stream.Current()
			chunks++
			if len(chunk.Choices) > 0 {
				fullText += chunk.Choices[0].Delta.Content
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		if chunks == 0 {
			t.Fatal("expected at least one streaming chunk")
		}
		if fullText == "" {
			t.Fatal("expected non-empty streamed text")
		}
		t.Logf("streamed %d chunks, text: %s", chunks, fullText)
	})
}
