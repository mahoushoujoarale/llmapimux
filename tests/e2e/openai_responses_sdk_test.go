package e2e_test

import (
	"encoding/json"
	"net/http"
	"testing"

	llmapimux "github.com/mahoushoujoarale/llmapimux"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

func anthropicResponseJSON(id, model, text string) []byte {
	resp := map[string]any{
		"id":    id,
		"type":  "message",
		"role":  "assistant",
		"model": model,
		"content": []any{
			map[string]any{"type": "text", "text": text},
		},
		"stop_reason": "end_turn",
		"usage":       map[string]any{"input_tokens": 3, "output_tokens": 1},
	}
	data, _ := json.Marshal(resp)
	return data
}

func TestE2E_OpenAIResponsesSDK_ToAnthropic(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(anthropicResponseJSON("msg-1", "claude-sonnet-4-20250514", "pong"))
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/responses",
		newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, upstream.URL, "sk-ant-outbound", map[string]string{"gpt-4o-mini": "claude-sonnet-4-20250514"}).OpenAIResponsesHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		option.WithAPIKey("sk-openai-inbound"),
		option.WithBaseURL(muxServer.URL+"/v1"),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: "gpt-4o-mini",
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.NewOpt("ping"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if resp == nil {
		t.Fatal("expected non-nil SDK response")
	}
	if resp.ID == "" {
		t.Fatal("expected response id")
	}
	if resp.Status != "completed" {
		t.Fatalf("status = %q, want completed", resp.Status)
	}

	if len(resp.Output) == 0 {
		t.Fatal("expected non-empty output")
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
	if text != "pong" {
		t.Fatalf("output text = %q, want pong", text)
	}

	assertCapturedRequestBasics(t, got, http.MethodPost, "/v1/messages")
	assertHeaderValue(t, got.Header, "x-api-key", "sk-ant-outbound")
	assertHeaderEmpty(t, got.Header, "Authorization")

	body := decodeCapturedJSONBody(t, got)
	if body["model"] != "claude-sonnet-4-20250514" {
		t.Fatalf("model = %v, want claude-sonnet-4-20250514", body["model"])
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
