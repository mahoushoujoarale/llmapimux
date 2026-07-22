package e2e_test

import (
	"context"
	"encoding/base64"
	_ "embed"
	"net/http"
	"strings"
	"testing"
	"time"

	llmapimux "github.com/mahoushoujoarale/llmapimux"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaiopt "github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

//go:embed cat.jpg
var catJPEG []byte

func newImageContext(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()
	return context.WithTimeout(context.Background(), 60*time.Second)
}

const imagePrompt = "what animal is in this image? answer in one word"

// firstTextBlock finds the first TextBlock in content, skipping ThinkingBlocks.
// Required for models with extended thinking enabled where ThinkingBlock precedes TextBlock.
func firstTextBlock(t *testing.T, content []anthropic.ContentBlockUnion) anthropic.TextBlock {
	t.Helper()
	for _, block := range content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			return tb
		}
	}
	t.Fatalf("no TextBlock found in %d content blocks", len(content))
	return anthropic.TextBlock{}
}

func assertCat(t *testing.T, text string) {
	t.Helper()
	if !strings.Contains(strings.ToLower(text), "cat") {
		t.Fatalf("expected response to contain 'cat', got: %s", text)
	}
}

// --- Anthropic image message builder ---

func anthropicImageMessages() []anthropic.MessageParam {
	imgData := base64.StdEncoding.EncodeToString(catJPEG)
	return []anthropic.MessageParam{
		anthropic.NewUserMessage(
			anthropic.NewImageBlockBase64("image/jpeg", imgData),
			anthropic.NewTextBlock(imagePrompt),
		),
	}
}

// --- OpenAI image message builder ---

func openAIImageMessages() []openai.ChatCompletionMessageParamUnion {
	imgData := base64.StdEncoding.EncodeToString(catJPEG)
	return []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: "data:image/jpeg;base64," + imgData,
			}),
			openai.TextContentPart(imagePrompt),
		}),
	}
}

// --- Gemini image content builder ---

func geminiImageContents() []*genai.Content {
	return []*genai.Content{
		genai.NewContentFromParts([]*genai.Part{
			genai.NewPartFromBytes(catJPEG, "image/jpeg"),
			genai.NewPartFromText(imagePrompt),
		}, genai.RoleUser),
	}
}

// openAIResponsesImageMessages builds a Responses API input containing an embedded
// JPEG image and a text prompt. Returns ResponseNewParamsInputUnion using the
// OfInputItemList variant (a list of input items).
func openAIResponsesImageMessages() responses.ResponseNewParamsInputUnion {
	imgData := base64.StdEncoding.EncodeToString(catJPEG)
	dataURL := "data:image/jpeg;base64," + imgData
	return responses.ResponseNewParamsInputUnion{
		OfInputItemList: responses.ResponseInputParam{
			responses.ResponseInputItemParamOfMessage(
				responses.ResponseInputMessageContentListParam{
					{OfInputImage: &responses.ResponseInputImageParam{
						ImageURL: param.NewOpt(dataURL),
						Detail:   responses.ResponseInputImageDetailAuto,
					}},
					{OfInputText: &responses.ResponseInputTextParam{
						Text: imagePrompt,
					}},
				},
				responses.EasyInputMessageRoleUser,
			),
		},
	}
}

// ============================================================
// Passthrough image tests (same protocol in and out)
// ============================================================

func TestRealAPI_Image_Passthrough_Anthropic(t *testing.T) {
	baseURL := anthropicBaseURL(t)
	model := anthropicModel()

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/messages": newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, baseURL, anthropicAPIKey(), map[string]string{model: model}).AnthropicHandler(),
	})
	defer muxServer.Close()

	client := anthropic.NewClient(
		option.WithAPIKey("unused-inbound-key"),
		option.WithBaseURL(muxServer.URL),
		option.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: 1024,
			Messages:  anthropicImageMessages(),
		})
		if err != nil {
			t.Fatal(err)
		}
		textBlock := firstTextBlock(t, resp.Content)
		t.Logf("response: %s", textBlock.Text)
		assertCat(t, textBlock.Text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: 1024,
			Messages:  anthropicImageMessages(),
		})
		message := anthropic.Message{}
		for stream.Next() {
			if err := message.Accumulate(stream.Current()); err != nil {
				t.Fatalf("accumulate: %v", err)
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatal(err)
		}
		textBlock := firstTextBlock(t, message.Content)
		t.Logf("streamed text: %s", textBlock.Text)
		assertCat(t, textBlock.Text)
	})
}

func TestRealAPI_Image_Passthrough_OpenAI(t *testing.T) {
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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    model,
			Messages: openAIImageMessages(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		text := resp.Choices[0].Message.Content
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Model:    model,
			Messages: openAIImageMessages(),
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
		t.Logf("streamed text: %s", fullText)
		assertCat(t, fullText)
	})
}

func TestRealAPI_Image_Passthrough_Gemini(t *testing.T) {
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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Models.GenerateContent(ctx, model, geminiImageContents(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil || len(resp.Candidates) == 0 {
			t.Fatal("expected non-empty candidates")
		}
		text := resp.Text()
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		var fullText string
		for resp, err := range client.Models.GenerateContentStream(ctx, model, geminiImageContents(), nil) {
			if err != nil {
				t.Fatal(err)
			}
			if resp != nil && len(resp.Candidates) > 0 {
				fullText += resp.Text()
			}
		}
		t.Logf("streamed text: %s", fullText)
		assertCat(t, fullText)
	})
}

// ============================================================
// Cross-protocol image tests
// ============================================================

func TestRealAPI_Image_Cross_AnthropicToOpenAI(t *testing.T) {
	baseURL := openAIBaseURL(t)
	const inboundModel = "img-ant-to-oai"

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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 1024,
			Messages:  anthropicImageMessages(),
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
		t.Logf("response: %s", textBlock.Text)
		assertCat(t, textBlock.Text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 1024,
			Messages:  anthropicImageMessages(),
		})
		message := anthropic.Message{}
		for stream.Next() {
			if err := message.Accumulate(stream.Current()); err != nil {
				t.Fatalf("accumulate: %v", err)
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
			t.Fatalf("content[0] type = %T, want TextBlock", message.Content[0].AsAny())
		}
		t.Logf("streamed text: %s", textBlock.Text)
		assertCat(t, textBlock.Text)
	})
}

func TestRealAPI_Image_Cross_AnthropicToGemini(t *testing.T) {
	baseURL := geminiBaseURL(t)
	const inboundModel = "img-ant-to-gem"

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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 1024,
			Messages:  anthropicImageMessages(),
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
		t.Logf("response: %s", textBlock.Text)
		assertCat(t, textBlock.Text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 1024,
			Messages:  anthropicImageMessages(),
		})
		message := anthropic.Message{}
		for stream.Next() {
			if err := message.Accumulate(stream.Current()); err != nil {
				t.Fatalf("accumulate: %v", err)
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
			t.Fatalf("content[0] type = %T, want TextBlock", message.Content[0].AsAny())
		}
		t.Logf("streamed text: %s", textBlock.Text)
		assertCat(t, textBlock.Text)
	})
}

func TestRealAPI_Image_Cross_OpenAIToAnthropic(t *testing.T) {
	baseURL := anthropicBaseURL(t)
	const inboundModel = "img-oai-to-ant"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/chat/completions": newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, baseURL, anthropicAPIKey(), map[string]string{inboundModel: anthropicModel()}).OpenAIChatHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    inboundModel,
			Messages: openAIImageMessages(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		text := resp.Choices[0].Message.Content
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Model:    inboundModel,
			Messages: openAIImageMessages(),
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
		t.Logf("streamed text: %s", fullText)
		assertCat(t, fullText)
	})
}

func TestRealAPI_Image_Cross_OpenAIToGemini(t *testing.T) {
	baseURL := geminiBaseURL(t)
	const inboundModel = "img-oai-to-gem"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/chat/completions": newTestMuxWithModelMap(llmapimux.ProtocolGemini, baseURL, geminiAPIKey(), map[string]string{inboundModel: geminiModel()}).OpenAIChatHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	t.Run("non-streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    inboundModel,
			Messages: openAIImageMessages(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		text := resp.Choices[0].Message.Content
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Model:    inboundModel,
			Messages: openAIImageMessages(),
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
		t.Logf("streamed text: %s", fullText)
		assertCat(t, fullText)
	})
}

func TestRealAPI_Image_Cross_GeminiToAnthropic(t *testing.T) {
	baseURL := anthropicBaseURL(t)
	const inboundModel = "img-gem-to-ant"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/models/": newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, baseURL, anthropicAPIKey(), map[string]string{inboundModel: anthropicModel()}).GeminiHandler(),
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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Models.GenerateContent(ctx, inboundModel, geminiImageContents(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil || len(resp.Candidates) == 0 {
			t.Fatal("expected non-empty candidates")
		}
		text := resp.Text()
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		var fullText string
		for resp, err := range client.Models.GenerateContentStream(ctx, inboundModel, geminiImageContents(), nil) {
			if err != nil {
				t.Fatal(err)
			}
			if resp != nil && len(resp.Candidates) > 0 {
				fullText += resp.Text()
			}
		}
		t.Logf("streamed text: %s", fullText)
		assertCat(t, fullText)
	})
}

func TestRealAPI_Image_Cross_GeminiToOpenAI(t *testing.T) {
	baseURL := openAIBaseURL(t)
	const inboundModel = "img-gem-to-oai"

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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Models.GenerateContent(ctx, inboundModel, geminiImageContents(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil || len(resp.Candidates) == 0 {
			t.Fatal("expected non-empty candidates")
		}
		text := resp.Text()
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		var fullText string
		for resp, err := range client.Models.GenerateContentStream(ctx, inboundModel, geminiImageContents(), nil) {
			if err != nil {
				t.Fatal(err)
			}
			if resp != nil && len(resp.Candidates) > 0 {
				fullText += resp.Text()
			}
		}
		t.Logf("streamed text: %s", fullText)
		assertCat(t, fullText)
	})
}

func TestRealAPI_Image_Passthrough_OpenAIResponses(t *testing.T) {
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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
			Model: model,
			Input: openAIResponsesImageMessages(),
		})
		if err != nil {
			t.Fatal(err)
		}
		text := collectResponsesOutputText(resp)
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Responses.NewStreaming(ctx, responses.ResponseNewParams{
			Model: model,
			Input: openAIResponsesImageMessages(),
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
		t.Logf("streamed %d chunks, text: %s", chunks, fullText)
		assertCat(t, fullText)
	})
}

func TestRealAPI_Image_Cross_AnthropicToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "img-ant-to-oairesp"

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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 1024,
			Messages:  anthropicImageMessages(),
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
		t.Logf("response: %s", textBlock.Text)
		assertCat(t, textBlock.Text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
			Model:     inboundModel,
			MaxTokens: 1024,
			Messages:  anthropicImageMessages(),
		})
		message := anthropic.Message{}
		for stream.Next() {
			if err := message.Accumulate(stream.Current()); err != nil {
				t.Fatalf("accumulate: %v", err)
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
			t.Fatalf("content[0] type = %T, want TextBlock", message.Content[0].AsAny())
		}
		t.Logf("streamed text: %s", textBlock.Text)
		assertCat(t, textBlock.Text)
	})
}

func TestRealAPI_Image_Cross_GeminiToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "img-gem-to-oairesp"

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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Models.GenerateContent(ctx, inboundModel, geminiImageContents(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil || len(resp.Candidates) == 0 {
			t.Fatal("expected non-empty candidates")
		}
		text := resp.Text()
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		var fullText string
		for resp, err := range client.Models.GenerateContentStream(ctx, inboundModel, geminiImageContents(), nil) {
			if err != nil {
				t.Fatal(err)
			}
			if resp != nil && len(resp.Candidates) > 0 {
				fullText += resp.Text()
			}
		}
		t.Logf("streamed text: %s", fullText)
		assertCat(t, fullText)
	})
}

func TestRealAPI_Image_Cross_OpenAIChatToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "img-oai-to-oairesp"

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
		ctx, cancel := newImageContext(t)
		defer cancel()

		resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    inboundModel,
			Messages: openAIImageMessages(),
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		text := resp.Choices[0].Message.Content
		t.Logf("response: %s", text)
		assertCat(t, text)
	})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := newImageContext(t)
		defer cancel()

		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Model:    inboundModel,
			Messages: openAIImageMessages(),
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
		t.Logf("streamed text: %s", fullText)
		assertCat(t, fullText)
	})
}
