package e2e_test

import (
	"fmt"
	"net/http"
	"testing"

	llmapimux "github.com/llmapimux/llmapimux"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func TestE2E_AnthropicSDK_FeasibilityGate(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`))
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
			anthropic.NewUserMessage(anthropic.NewTextBlock("ping")),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if resp == nil {
		t.Fatal("expected non-nil SDK response")
	}
	if resp.ID == "" {
		t.Fatal("expected SDK response id")
	}
	if resp.StopReason != anthropic.StopReasonEndTurn {
		t.Fatalf("stop_reason = %q, want %q", resp.StopReason, anthropic.StopReasonEndTurn)
	}
	if len(resp.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(resp.Content))
	}
	textBlock, ok := resp.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", resp.Content[0].AsAny())
	}
	if textBlock.Text != "pong" {
		t.Fatalf("content[0].text = %q, want pong", textBlock.Text)
	}

	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/chat/completions")
	assertHeaderValue(t, got.Header, "Authorization", "Bearer sk-openai-upstream")
	assertHeaderEmpty(t, got.Header, "x-api-key")
	assertHeaderEmpty(t, got.Header, "anthropic-version")

	body := decodeCapturedJSONBody(t, got)
	if body["model"] != "gpt-4o-mini" {
		t.Fatalf("model = %v, want gpt-4o-mini", body["model"])
	}
	messages, _ := body["messages"].([]any)
	if len(messages) != 1 {
		t.Fatalf("messages len = %d, want 1", len(messages))
	}
	message, _ := messages[0].(map[string]any)
	if message["role"] != "user" {
		t.Fatalf("message.role = %v, want user", message["role"])
	}
	content, ok := message["content"].([]any)
	if !ok {
		t.Fatalf("message.content type = %T, want []any", message["content"])
	}
	if len(content) != 1 {
		t.Fatalf("message.content len = %d, want 1", len(content))
	}
	part, _ := content[0].(map[string]any)
	if part["type"] != "text" {
		t.Fatalf("message.content[0].type = %v, want text", part["type"])
	}
	if part["text"] != "ping" {
		t.Fatalf("message.content[0].text = %v, want ping", part["text"])
	}
}

func TestE2E_AnthropicSDK_Stream_ToOpenAIChat(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"pong\"},\"finish_reason\":null}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
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
			anthropic.NewUserMessage(anthropic.NewTextBlock("ping")),
		},
	})

	var eventTypes []string
	var rawEvents []string
	message := anthropic.Message{}
	for stream.Next() {
		event := stream.Current()
		eventTypes = append(eventTypes, fmt.Sprintf("%T", event.AsAny()))
		rawEvents = append(rawEvents, fmt.Sprintf("%v", event.JSON.Type))
		if err := message.Accumulate(event); err != nil {
			t.Fatalf("accumulate %T: %v (seen=%v raw=%v)", event.AsAny(), err, eventTypes, rawEvents)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	if len(eventTypes) == 0 {
		t.Fatal("expected streaming events")
	}
	if got == nil {
		t.Fatal("expected upstream request to be captured")
	}
	if got.Path != "/v1/chat/completions" {
		t.Fatalf("path = %s, want /v1/chat/completions", got.Path)
	}
	if message.StopReason != anthropic.StopReasonEndTurn {
		t.Fatalf("stop_reason = %q, want %q (seen=%v raw=%v)", message.StopReason, anthropic.StopReasonEndTurn, eventTypes, rawEvents)
	}
	if len(message.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(message.Content))
	}
	textBlock, ok := message.Content[0].AsAny().(anthropic.TextBlock)
	if !ok {
		t.Fatalf("content[0] type = %T, want anthropic.TextBlock", message.Content[0].AsAny())
	}
	if textBlock.Text != "pong" {
		t.Fatalf("content[0].text = %q, want pong (seen=%v raw=%v)", textBlock.Text, eventTypes, rawEvents)
	}
}
