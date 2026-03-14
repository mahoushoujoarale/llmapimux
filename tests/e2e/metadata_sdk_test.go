package e2e_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"testing"

	llmapimux "github.com/llmapimux/llmapimux"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaiopt "github.com/openai/openai-go/v3/option"
	"google.golang.org/genai"
)

// ============================================================
// Test 1: Anthropic SDK → OpenAI Chat — usage/token counts
// ============================================================

func TestE2E_Metadata_AnthropicSDK_UsageFromOpenAIChat(t *testing.T) {
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-abc","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":25,"total_tokens":125}}`))
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
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if resp.Usage.InputTokens != 100 {
		t.Fatalf("input_tokens = %d, want 100", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 25 {
		t.Fatalf("output_tokens = %d, want 25", resp.Usage.OutputTokens)
	}
}

// ============================================================
// Test 2: Anthropic SDK ← OpenAI Chat — stop reason mappings
// ============================================================

func TestE2E_Metadata_AnthropicSDK_StopReasons_FromOpenAIChat(t *testing.T) {
	t.Run("end_turn", func(t *testing.T) {
		upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
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
				anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		if resp.StopReason != anthropic.StopReasonEndTurn {
			t.Fatalf("stop_reason = %q, want %q", resp.StopReason, anthropic.StopReasonEndTurn)
		}
	})

	t.Run("max_tokens", func(t *testing.T) {
		upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
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
				anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		if resp.StopReason != anthropic.StopReasonMaxTokens {
			t.Fatalf("stop_reason = %q, want %q", resp.StopReason, anthropic.StopReasonMaxTokens)
		}
	})

	t.Run("tool_use", func(t *testing.T) {
		upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"test","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
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
				anthropic.NewUserMessage(anthropic.NewTextBlock("use a tool")),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		if resp.StopReason != anthropic.StopReasonToolUse {
			t.Fatalf("stop_reason = %q, want %q", resp.StopReason, anthropic.StopReasonToolUse)
		}
	})
}

// ============================================================
// Test 3: Anthropic SDK ← OpenAI Chat — response ID preserved
// ============================================================

func TestE2E_Metadata_AnthropicSDK_ResponseID_FromOpenAIChat(t *testing.T) {
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-unique123","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
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
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if resp.ID == "" {
		t.Fatal("expected non-empty response ID")
	}
}

// ============================================================
// Test 4: Anthropic SDK ← OpenAI Chat — model name in response
// ============================================================

func TestE2E_Metadata_AnthropicSDK_ModelName_FromOpenAIChat(t *testing.T) {
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
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
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if resp.Model == "" {
		t.Fatal("expected non-empty model in response")
	}
}

// ============================================================
// Test 5: OpenAI Chat SDK ← Anthropic — usage mapping
// ============================================================

func TestE2E_Metadata_OpenAIChatSDK_Usage_FromAnthropic(t *testing.T) {
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg-abc","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":15}}`))
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
			openai.UserMessage("hello"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if resp.Usage.PromptTokens != 50 {
		t.Fatalf("prompt_tokens = %d, want 50", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 15 {
		t.Fatalf("completion_tokens = %d, want 15", resp.Usage.CompletionTokens)
	}
}

// ============================================================
// Test 6: OpenAI Chat SDK ← Anthropic — stop reason mappings
// ============================================================

func TestE2E_Metadata_OpenAIChatSDK_StopReasons_FromAnthropic(t *testing.T) {
	t.Run("stop", func(t *testing.T) {
		upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
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
				openai.UserMessage("hi"),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		if resp.Choices[0].FinishReason != "stop" {
			t.Fatalf("finish_reason = %q, want stop", resp.Choices[0].FinishReason)
		}
	})

	t.Run("length", func(t *testing.T) {
		upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"hi"}],"stop_reason":"max_tokens","usage":{"input_tokens":10,"output_tokens":5}}`))
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
				openai.UserMessage("hi"),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		if resp.Choices[0].FinishReason != "length" {
			t.Fatalf("finish_reason = %q, want length", resp.Choices[0].FinishReason)
		}
	})

	t.Run("tool_calls", func(t *testing.T) {
		upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"tool_use","id":"toolu_1","name":"test","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`))
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
				openai.UserMessage("use a tool"),
			},
		})
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Choices) == 0 {
			t.Fatal("expected non-empty choices")
		}
		choice := resp.Choices[0]
		if choice.FinishReason != "tool_calls" {
			t.Fatalf("finish_reason = %q, want tool_calls", choice.FinishReason)
		}
		if len(choice.Message.ToolCalls) == 0 {
			t.Fatal("expected non-empty tool_calls")
		}
	})
}

// ============================================================
// Test 7: Auth headers — Anthropic inbound → OpenAI Chat outbound
// ============================================================

func TestE2E_Metadata_AuthHeaders_AnthropicToOpenAIChat(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
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
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/chat/completions")
	assertHeaderValue(t, got.Header, "Authorization", "Bearer sk-openai-upstream")
	assertHeaderEmpty(t, got.Header, "x-api-key")
	assertHeaderEmpty(t, got.Header, "anthropic-version")
}

// ============================================================
// Test 8: Auth headers — OpenAI Chat inbound → Anthropic outbound
// ============================================================

func TestE2E_Metadata_AuthHeaders_OpenAIChatToAnthropic(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
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
			openai.UserMessage("hi"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/messages")
	assertHeaderValue(t, got.Header, "x-api-key", "sk-ant-upstream")
	assertHeaderValue(t, got.Header, "anthropic-version", "2023-06-01")
	assertHeaderEmpty(t, got.Header, "Authorization")
}

// ============================================================
// Test 9: Gemini SDK ← OpenAI Chat — usage mapping
// ============================================================

func TestE2E_Metadata_GeminiSDK_Usage_FromOpenAIChat(t *testing.T) {
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":80,"completion_tokens":20,"total_tokens":100}}`))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/models/",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"gemini-2.5-pro": "gpt-4o-mini"}).GeminiHandler(),
	)
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "fake-gemini-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
		HTTPClient: newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL),
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := newE2EContext(t)
	defer cancel()

	resp, err := client.Models.GenerateContent(ctx, "gemini-2.5-pro",
		[]*genai.Content{
			genai.NewContentFromText("hello", genai.RoleUser),
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	if resp.UsageMetadata == nil {
		t.Fatal("expected non-nil UsageMetadata")
	}
	if resp.UsageMetadata.PromptTokenCount != 80 {
		t.Fatalf("prompt_token_count = %d, want 80", resp.UsageMetadata.PromptTokenCount)
	}
	if resp.UsageMetadata.CandidatesTokenCount != 20 {
		t.Fatalf("candidates_token_count = %d, want 20", resp.UsageMetadata.CandidatesTokenCount)
	}
}

// ============================================================
// Test 10: Streaming usage — Anthropic SDK ← OpenAI Chat
// ============================================================

func TestE2E_Metadata_StreamUsage_AnthropicSDK_FromOpenAIChat(t *testing.T) {
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":10,\"total_tokens\":60}}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
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

	stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
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
	if len(eventTypes) == 0 {
		t.Fatal("expected streaming events")
	}

	// Verify streaming completed with correct stop reason and content.
	if message.StopReason != anthropic.StopReasonEndTurn {
		t.Fatalf("stop_reason = %q, want %q", message.StopReason, anthropic.StopReasonEndTurn)
	}
	if len(message.Content) == 0 {
		t.Fatal("expected non-empty accumulated content")
	}
	textBlock, ok := message.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", message.Content[0].AsAny())
	}
	if textBlock.Text != "hi" {
		t.Fatalf("content[0].text = %q, want hi", textBlock.Text)
	}

	// Verify streaming usage propagation:
	// output_tokens should arrive via the message_delta event.
	if message.Usage.OutputTokens != 10 {
		t.Errorf("streaming usage output_tokens = %d, want 10", message.Usage.OutputTokens)
	}
}

// Ensure imports are used.
var (
	_ = json.Marshal
	_ sync.Mutex
)
