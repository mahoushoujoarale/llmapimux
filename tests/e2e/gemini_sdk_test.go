package e2e_test

import (
	"context"
	"net/http"
	"testing"

	llmapimux "github.com/mahoushoujoarale/llmapimux"

	"google.golang.org/genai"
)

func TestE2E_GeminiSDK_ToOpenAIChat(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`))
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
			genai.NewContentFromText("ping", genai.RoleUser),
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}

	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	if len(resp.Candidates) == 0 {
		t.Fatal("expected non-empty candidates")
	}
	text := resp.Text()
	if text != "pong" {
		t.Fatalf("text = %q, want pong", text)
	}

	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/chat/completions")
	assertHeaderValue(t, got.Header, "Authorization", "Bearer sk-openai-upstream")
	assertHeaderEmpty(t, got.Header, "x-goog-api-key")

	body := decodeCapturedJSONBody(t, got)
	if body["model"] != "gpt-4o-mini" {
		t.Fatalf("model = %v, want gpt-4o-mini", body["model"])
	}
	messages, _ := body["messages"].([]any)
	if len(messages) != 1 {
		t.Fatalf("messages len = %d, want 1", len(messages))
	}
	msg, _ := messages[0].(map[string]any)
	if msg["role"] != "user" {
		t.Fatalf("message.role = %v, want user", msg["role"])
	}
}
