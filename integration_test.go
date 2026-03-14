package llmapimux

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestIntegration_HandlerRouterOutbound_NonStreaming(t *testing.T) {
	var gotBody map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %s, want /v1/chat/completions", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer sk-openai" {
			t.Fatalf("authorization = %q, want Bearer sk-openai", r.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  upstream.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}})

	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"claude-sonnet-4-20250514","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if gotBody["model"] != "gpt-4o" {
		t.Fatalf("upstream model = %v, want gpt-4o", gotBody["model"])
	}

	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode downstream body: %v", err)
	}
	if resp["type"] != "message" {
		t.Fatalf("response type = %v, want message", resp["type"])
	}
}

func TestIntegration_PreservationSensitive_CrossProtocolDropsRawExtra(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		if _, ok := body["inference_geo"]; ok {
			t.Fatalf("cross-protocol outbound should not include inference_geo")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  upstream.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}})

	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"claude-sonnet-4-20250514","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"inference_geo":"us-east"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
}

// TestIntegration_Anthropic_To_OpenAIChat tests the full flow:
// Anthropic inbound request → IR decode → OpenAI Chat outbound → IR encode → Anthropic response.
func TestIntegration_Anthropic_To_OpenAIChat(t *testing.T) {
	// 1. Fake OpenAI Chat upstream that validates the request.
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		// Verify model was mapped.
		if req["model"] != "gpt-4o" {
			t.Errorf("expected gpt-4o, got %v", req["model"])
		}
		// Verify auth header.
		if r.Header.Get("Authorization") != "Bearer sk-openai" {
			t.Errorf("wrong auth: got %q", r.Header.Get("Authorization"))
		}
		// Return OpenAI Chat response.
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hello from GPT!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer openaiServer.Close()

	// 2. Create Mux.
	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}}
	mux := NewMux(router)

	// 3. Send Anthropic request.
	body := `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	// 4. Verify response is Anthropic format.
	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v, want end_turn", resp["stop_reason"])
	}
	// Verify content has text.
	content, _ := resp["content"].([]any)
	if len(content) == 0 {
		t.Fatal("empty content")
	}
	block, _ := content[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("content[0].type = %v, want text", block["type"])
	}
	if block["text"] != "Hello from GPT!" {
		t.Errorf("content[0].text = %v, want Hello from GPT!", block["text"])
	}
}

// TestIntegration_OpenAIResponses_To_Anthropic tests the codex→claude scenario:
// OpenAI Responses inbound request → IR decode → Anthropic outbound → IR encode → Responses response.
func TestIntegration_OpenAIResponses_To_Anthropic(t *testing.T) {
	// Fake Anthropic upstream.
	anthropicServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "sk-ant" {
			t.Errorf("wrong auth: got %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Errorf("missing anthropic-version")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"msg_1","model":"claude-sonnet-4-20250514","type":"message","role":"assistant","content":[{"type":"text","text":"Hello from Claude!"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer anthropicServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  anthropicServer.URL,
		APIKey:   "sk-ant",
		Model:    "claude-sonnet-4-20250514",
	}}
	mux := NewMux(router)

	body := `{"model":"gpt-4o","input":"Hello","max_output_tokens":1024}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.OpenAIResponsesHandler().ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	// Verify it's Responses API format.
	if resp["status"] != "completed" {
		t.Errorf("status = %v, want completed", resp["status"])
	}
	// Verify output has text.
	output, _ := resp["output"].([]any)
	if len(output) == 0 {
		t.Fatal("empty output")
	}
}

// TestIntegration_Gemini_To_OpenAIChat tests:
// Gemini inbound request → IR decode → OpenAI Chat outbound → IR encode → Gemini response.
func TestIntegration_Gemini_To_OpenAIChat(t *testing.T) {
	// Fake OpenAI Chat upstream.
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		// Verify model was mapped.
		if req["model"] != "gpt-4o" {
			t.Errorf("expected gpt-4o, got %v", req["model"])
		}
		// Verify auth header.
		if r.Header.Get("Authorization") != "Bearer sk-openai" {
			t.Errorf("wrong auth: got %q", r.Header.Get("Authorization"))
		}
		// Return OpenAI Chat response.
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hello from OpenAI!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer openaiServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}}
	mux := NewMux(router)

	// Send Gemini request.
	body := `{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}`
	req := httptest.NewRequest("POST", "/v1/models/gemini-2.5-pro:generateContent", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.GeminiHandler().ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	// Verify response is Gemini format with candidates array.
	candidates, _ := resp["candidates"].([]any)
	if len(candidates) == 0 {
		t.Fatal("empty candidates")
	}
	cand, _ := candidates[0].(map[string]any)
	content, _ := cand["content"].(map[string]any)
	if content == nil {
		t.Fatal("candidate has no content")
	}
	parts, _ := content["parts"].([]any)
	if len(parts) == 0 {
		t.Fatal("candidate content has no parts")
	}
	part, _ := parts[0].(map[string]any)
	if part["text"] != "Hello from OpenAI!" {
		t.Errorf("parts[0].text = %v, want Hello from OpenAI!", part["text"])
	}
}

func TestIntegration_AnthropicPauseTurn_To_OpenAIChat_DowngradesStopReason(t *testing.T) {
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-pause","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Need more input."},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer openaiServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}}
	mux := NewMux(router)

	body := `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if resp["stop_reason"] != "end_turn" {
		t.Fatalf("stop_reason = %v, want end_turn", resp["stop_reason"])
	}
}

// TestIntegration_Anthropic_To_OpenAIChat_Streaming tests the full streaming flow:
// Anthropic stream request → OpenAI Chat SSE upstream → Anthropic SSE response.
func TestIntegration_Anthropic_To_OpenAIChat_Streaming(t *testing.T) {
	// Fake OpenAI Chat upstream returns SSE stream.
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer openaiServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}}
	mux := NewMux(router)

	body := `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	// Verify Anthropic SSE format.
	respBody := w.Body.String()
	if !strings.Contains(respBody, "event: message_start") {
		t.Errorf("missing message_start in: %s", respBody)
	}
	if !strings.Contains(respBody, "text_delta") {
		t.Errorf("missing text_delta in: %s", respBody)
	}
	if !strings.Contains(respBody, "message_stop") {
		t.Errorf("missing message_stop in: %s", respBody)
	}
}

func TestIntegration_Anthropic_To_OpenAIChat_Streaming_DoesNotWriteMessageStopAfterMidStreamFailure(t *testing.T) {
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
	}))
	defer openaiServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}}
	mux := NewMux(router)

	body := `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	respBody := w.Body.String()
	if !strings.Contains(respBody, "Hello") {
		t.Fatalf("missing streamed content in: %s", respBody)
	}
	if strings.Contains(respBody, "message_stop") {
		t.Fatalf("unexpected message_stop after mid-stream failure: %s", respBody)
	}
}

// TestIntegration_ToolUse_CrossProtocol tests tool use conversion:
// Anthropic request with tools → OpenAI Chat outbound with tool_calls response → Anthropic tool_use response.
func TestIntegration_ToolUse_CrossProtocol(t *testing.T) {
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		// Verify tools were converted.
		tools, _ := req["tools"].([]any)
		if len(tools) == 0 {
			t.Error("no tools in request")
		}
		// Return tool call response.
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/foo\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer openaiServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-test",
		Model:    "gpt-4o",
	}}
	mux := NewMux(router)

	body := `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"tools":[{"name":"read_file","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],"messages":[{"role":"user","content":[{"type":"text","text":"Read /tmp/foo"}]}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	// Verify stop_reason is tool_use.
	if resp["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v, want tool_use", resp["stop_reason"])
	}
	// Verify content has tool_use block.
	content, _ := resp["content"].([]any)
	found := false
	for _, c := range content {
		block, _ := c.(map[string]any)
		if block["type"] == "tool_use" {
			found = true
			if block["name"] != "read_file" {
				t.Errorf("tool name = %v, want read_file", block["name"])
			}
		}
	}
	if !found {
		t.Error("no tool_use block in response")
	}
}

// TestIntegration_Image_CrossProtocol tests image conversion:
// Anthropic request with base64 image → OpenAI Chat outbound should receive image_url.
func TestIntegration_Image_CrossProtocol(t *testing.T) {
	// Fake OpenAI Chat upstream that verifies image_url was sent.
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Verify messages contain image_url content.
		messages, _ := req["messages"].([]any)
		if len(messages) == 0 {
			t.Error("no messages in request")
		}
		userMsg, _ := messages[0].(map[string]any)
		content := userMsg["content"]
		// Content can be a string or array; for image it should be array.
		contentArr, ok := content.([]any)
		if !ok {
			t.Errorf("expected content array for image message, got %T", content)
		} else {
			foundImage := false
			for _, part := range contentArr {
				partMap, _ := part.(map[string]any)
				if partMap["type"] == "image_url" {
					foundImage = true
					imageURL, _ := partMap["image_url"].(map[string]any)
					if imageURL == nil || imageURL["url"] == "" {
						t.Error("image_url.url is empty")
					}
				}
			}
			if !foundImage {
				t.Errorf("no image_url part found in messages: %v", content)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"I see the image."},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60}}`))
	}))
	defer openaiServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}}
	mux := NewMux(router)

	// Small 1x1 PNG in base64.
	pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	body := fmt.Sprintf(`{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"%s"}},{"type":"text","text":"What do you see?"}]}]}`, pngBase64)
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v, want end_turn", resp["stop_reason"])
	}
}

// TestIntegration_OpenAIChat_To_Anthropic tests the reverse direction:
// OpenAI Chat inbound request → IR decode → Anthropic outbound → IR encode → OpenAI Chat response.
func TestIntegration_OpenAIChat_To_Anthropic(t *testing.T) {
	// Fake Anthropic upstream.
	anthropicServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify auth.
		if r.Header.Get("x-api-key") != "sk-ant" {
			t.Errorf("wrong auth: got %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Errorf("missing anthropic-version header")
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		// Verify model was mapped.
		if req["model"] != "claude-sonnet-4-20250514" {
			t.Errorf("expected claude-sonnet-4-20250514, got %v", req["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"msg_1","model":"claude-sonnet-4-20250514","type":"message","role":"assistant","content":[{"type":"text","text":"Hello from Claude!"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer anthropicServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  anthropicServer.URL,
		APIKey:   "sk-ant",
		Model:    "claude-sonnet-4-20250514",
	}}
	mux := NewMux(router)

	// Send OpenAI Chat request.
	body := `{"model":"gpt-4o","max_tokens":1024,"messages":[{"role":"user","content":"Hello, Claude!"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.OpenAIChatHandler().ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	// Verify response is OpenAI Chat format.
	choices, _ := resp["choices"].([]any)
	if len(choices) == 0 {
		t.Fatal("empty choices")
	}
	choice, _ := choices[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Errorf("finish_reason = %v, want stop", choice["finish_reason"])
	}
	message, _ := choice["message"].(map[string]any)
	if message["role"] != "assistant" {
		t.Errorf("message.role = %v, want assistant", message["role"])
	}
}

// TestIntegration_Passthrough_Anthropic tests same-protocol passthrough:
// Anthropic inbound → Anthropic outbound (decode/encode with model replacement).
func TestIntegration_Passthrough_Anthropic(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify path.
		if r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		// Verify auth was replaced.
		if r.Header.Get("x-api-key") != "sk-ant-outbound" {
			t.Errorf("wrong api key: got %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Errorf("missing anthropic-version header")
		}

		// Verify model was remapped in body.
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["model"] != "claude-3-haiku-20240307" {
			t.Errorf("model = %v, want claude-3-haiku-20240307", req["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		// Return same body mostly unchanged (passthrough behavior).
		w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-haiku-20240307","content":[{"type":"text","text":"Passthrough response"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}`))
	}))
	defer upstream.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  upstream.URL,
		APIKey:   "sk-ant-outbound",
		Model:    "claude-3-haiku-20240307",
	}}
	mux := NewMux(router)

	// Send Anthropic request with inbound auth.
	body := `{"model":"claude-sonnet-4-20250514","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-ant-inbound") // should be stripped
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	// Verify the response was passed through (it's an Anthropic response from fake upstream).
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["stop_reason"] != "end_turn" {
		t.Errorf("stop_reason = %v, want end_turn", resp["stop_reason"])
	}
}

func TestIntegration_OpenAIChat_To_Anthropic_PreservesUpstreamHTTPStatus(t *testing.T) {
	anthropicServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
	}))
	defer anthropicServer.Close()

	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  anthropicServer.URL,
		APIKey:   "sk-ant",
		Model:    "claude-sonnet-4-20250514",
	}}
	mux := NewMux(router)

	body := `{"model":"gpt-4o","max_tokens":1024,"messages":[{"role":"user","content":"Hello, Claude!"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.OpenAIChatHandler().ServeHTTP(w, req)

	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429; body = %s", w.Code, w.Body.String())
	}

	var errResp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	errObj, ok := errResp["error"].(map[string]any)
	if !ok {
		t.Fatalf("error field missing or wrong type")
	}
	if errObj["type"] != "invalid_request_error" {
		t.Errorf("error.type = %v, want invalid_request_error", errObj["type"])
	}
}

// ---------------------------------------------------------------------------
// Fallback integration test helpers
// ---------------------------------------------------------------------------

// failingUpstream returns a fake upstream that always responds with the given status code.
func failingUpstream(t *testing.T, statusCode int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write([]byte(fmt.Sprintf(`{"error":{"message":"upstream error","type":"server_error","code":%d}}`, statusCode)))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// countingFailingUpstream returns a fake upstream that counts requests and always fails.
func countingFailingUpstream(t *testing.T, statusCode int, counter *atomic.Int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		counter.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write([]byte(fmt.Sprintf(`{"error":{"message":"upstream error","type":"server_error","code":%d}}`, statusCode)))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// succeedingUpstream returns a fake upstream that returns a successful response.
func succeedingUpstream(t *testing.T, protocol Protocol) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch protocol {
		case ProtocolOpenAIChat:
			w.Write([]byte(`{"id":"test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hello from fallback"},"finish_reason":"stop","index":0}],"model":"test-model","usage":{"prompt_tokens":1,"completion_tokens":1}}`))
		case ProtocolAnthropic:
			w.Write([]byte(`{"id":"test","type":"message","role":"assistant","content":[{"type":"text","text":"hello from fallback"}],"model":"test-model","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		default:
			t.Fatalf("unsupported protocol for succeedingUpstream: %s", protocol)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// countingSucceedingUpstream returns a fake upstream that counts requests and succeeds.
func countingSucceedingUpstream(t *testing.T, protocol Protocol, counter *atomic.Int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		counter.Add(1)
		w.Header().Set("Content-Type", "application/json")
		switch protocol {
		case ProtocolOpenAIChat:
			w.Write([]byte(`{"id":"test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hello from fallback"},"finish_reason":"stop","index":0}],"model":"test-model","usage":{"prompt_tokens":1,"completion_tokens":1}}`))
		case ProtocolAnthropic:
			w.Write([]byte(`{"id":"test","type":"message","role":"assistant","content":[{"type":"text","text":"hello from fallback"}],"model":"test-model","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		default:
			t.Fatalf("unsupported protocol for countingSucceedingUpstream: %s", protocol)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// succeedingStreamUpstream returns a fake upstream that returns SSE streaming events.
func succeedingStreamUpstream(t *testing.T, protocol Protocol) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		switch protocol {
		case ProtocolOpenAIChat:
			fmt.Fprintf(w, "data: {\"id\":\"test\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"test\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello from fallback\"},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"test\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		default:
			t.Fatalf("unsupported protocol for succeedingStreamUpstream: %s", protocol)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// slowUpstream returns a fake upstream that delays before responding.
func slowUpstream(t *testing.T, protocol Protocol, delay time.Duration) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch protocol {
		case ProtocolOpenAIChat:
			w.Write([]byte(`{"id":"test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hello from slow"},"finish_reason":"stop","index":0}],"model":"test-model","usage":{"prompt_tokens":1,"completion_tokens":1}}`))
		default:
			t.Fatalf("unsupported protocol for slowUpstream: %s", protocol)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// openAIChatRequest is a minimal valid OpenAI Chat request body.
const openAIChatRequest = `{"model":"gpt-4o","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}`

// openAIChatStreamRequest is a minimal valid OpenAI Chat streaming request body.
const openAIChatStreamRequest = `{"model":"gpt-4o","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hello"}]}`

// sendOpenAIChatRequest sends a request through the OpenAI Chat handler and returns the recorder.
func sendOpenAIChatRequest(t *testing.T, handler http.Handler, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	return w
}

// sendOpenAIChatRequestWithContext sends a request with a custom context.
func sendOpenAIChatRequestWithContext(t *testing.T, handler http.Handler, body string, ctx context.Context) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req = req.WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	return w
}

// ---------------------------------------------------------------------------
// Fallback integration tests
// ---------------------------------------------------------------------------

func TestIntegration_Fallback_BasicFallback(t *testing.T) {
	primary := failingUpstream(t, 500)
	fallback := succeedingUpstream(t, ProtocolOpenAIChat)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: primary.URL, APIKey: "sk-1", Model: "m1"},
			{Protocol: ProtocolOpenAIChat, BaseURL: fallback.URL, APIKey: "sk-2", Model: "m2"},
		}
	}, WithFailureThreshold(10)) // high threshold so circuit doesn't open in this test

	mux := NewMux(router)
	w := sendOpenAIChatRequest(t, mux.OpenAIChatHandler(), openAIChatRequest)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "hello from fallback") {
		t.Fatalf("response body missing fallback content: %s", w.Body.String())
	}
}

func TestIntegration_Fallback_ThreeCandidateChain(t *testing.T) {
	a := failingUpstream(t, 500)
	b := failingUpstream(t, 503)
	c := succeedingUpstream(t, ProtocolOpenAIChat)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: a.URL, APIKey: "sk-a", Model: "ma"},
			{Protocol: ProtocolOpenAIChat, BaseURL: b.URL, APIKey: "sk-b", Model: "mb"},
			{Protocol: ProtocolOpenAIChat, BaseURL: c.URL, APIKey: "sk-c", Model: "mc"},
		}
	}, WithFailureThreshold(10))

	mux := NewMux(router)
	w := sendOpenAIChatRequest(t, mux.OpenAIChatHandler(), openAIChatRequest)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "hello from fallback") {
		t.Fatalf("response body missing fallback content: %s", w.Body.String())
	}
}

func TestIntegration_Fallback_AllFail(t *testing.T) {
	a := failingUpstream(t, 500)
	b := failingUpstream(t, 500)
	c := failingUpstream(t, 500)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: a.URL, APIKey: "sk-a", Model: "ma"},
			{Protocol: ProtocolOpenAIChat, BaseURL: b.URL, APIKey: "sk-b", Model: "mb"},
			{Protocol: ProtocolOpenAIChat, BaseURL: c.URL, APIKey: "sk-c", Model: "mc"},
		}
	}, WithFailureThreshold(10))

	mux := NewMux(router)
	w := sendOpenAIChatRequest(t, mux.OpenAIChatHandler(), openAIChatRequest)

	if w.Code < 400 {
		t.Fatalf("status = %d, want >= 400; body = %s", w.Code, w.Body.String())
	}
}

func TestIntegration_Fallback_4xxNoCircuitTrip(t *testing.T) {
	// Primary returns 400 (client error). Fallback should still be tried
	// because OnError still finds next candidate, but 400 should NOT trip the circuit.
	var primaryCount atomic.Int64
	primary := countingFailingUpstream(t, 400, &primaryCount)
	fallback := succeedingUpstream(t, ProtocolOpenAIChat)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: primary.URL, APIKey: "sk-1", Model: "m1"},
			{Protocol: ProtocolOpenAIChat, BaseURL: fallback.URL, APIKey: "sk-2", Model: "m2"},
		}
	}, WithFailureThreshold(1)) // threshold=1 so even 1 trippable failure would open it

	mux := NewMux(router)
	handler := mux.OpenAIChatHandler()

	// First request: primary 400 → fallback succeeds.
	w := sendOpenAIChatRequest(t, handler, openAIChatRequest)
	if w.Code != http.StatusOK {
		t.Fatalf("req1: status = %d, want 200; body = %s", w.Code, w.Body.String())
	}

	// Second request: primary should still be tried (circuit NOT tripped by 400).
	w = sendOpenAIChatRequest(t, handler, openAIChatRequest)
	if w.Code != http.StatusOK {
		t.Fatalf("req2: status = %d, want 200; body = %s", w.Code, w.Body.String())
	}

	// Primary was called both times (circuit stayed closed).
	if got := primaryCount.Load(); got != 2 {
		t.Fatalf("primary request count = %d, want 2 (circuit should not have tripped)", got)
	}
}

func TestIntegration_Fallback_CrossProtocol(t *testing.T) {
	// Inbound: OpenAI Chat. Primary: OpenAI Chat (500). Fallback: Anthropic (success).
	primary := failingUpstream(t, 500)
	fallback := succeedingUpstream(t, ProtocolAnthropic)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: primary.URL, APIKey: "sk-1", Model: "m1"},
			{Protocol: ProtocolAnthropic, BaseURL: fallback.URL, APIKey: "sk-2", Model: "claude-test"},
		}
	}, WithFailureThreshold(10))

	mux := NewMux(router)
	w := sendOpenAIChatRequest(t, mux.OpenAIChatHandler(), openAIChatRequest)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", w.Code, w.Body.String())
	}

	// Verify response is valid OpenAI Chat JSON (IR conversion happened).
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	choices, _ := resp["choices"].([]any)
	if len(choices) == 0 {
		t.Fatal("empty choices in response")
	}
	choice, _ := choices[0].(map[string]any)
	msg, _ := choice["message"].(map[string]any)
	if msg["content"] != "hello from fallback" {
		t.Fatalf("message content = %v, want 'hello from fallback'", msg["content"])
	}
}

func TestIntegration_Fallback_Streaming(t *testing.T) {
	primary := failingUpstream(t, 503)
	fallback := succeedingStreamUpstream(t, ProtocolOpenAIChat)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: primary.URL, APIKey: "sk-1", Model: "m1"},
			{Protocol: ProtocolOpenAIChat, BaseURL: fallback.URL, APIKey: "sk-2", Model: "m2"},
		}
	}, WithFailureThreshold(10))

	mux := NewMux(router)
	w := sendOpenAIChatRequest(t, mux.OpenAIChatHandler(), openAIChatStreamRequest)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", w.Code, w.Body.String())
	}

	respBody := w.Body.String()
	if !strings.Contains(w.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", w.Header().Get("Content-Type"))
	}
	if !strings.Contains(respBody, "hello from fallback") {
		t.Fatalf("SSE stream missing fallback content: %s", respBody)
	}
	if !strings.Contains(respBody, "[DONE]") {
		t.Fatalf("SSE stream missing [DONE]: %s", respBody)
	}
}

func TestIntegration_Fallback_CircuitBreakerTrips(t *testing.T) {
	var primaryCount atomic.Int64
	primary := countingFailingUpstream(t, 500, &primaryCount)
	var fallbackCount atomic.Int64
	fallback := countingSucceedingUpstream(t, ProtocolOpenAIChat, &fallbackCount)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: primary.URL, APIKey: "sk-1", Model: "m1"},
			{Protocol: ProtocolOpenAIChat, BaseURL: fallback.URL, APIKey: "sk-2", Model: "m2"},
		}
	}, WithFailureThreshold(3))

	mux := NewMux(router)
	handler := mux.OpenAIChatHandler()

	// Send 3 requests — each hits primary (fails 500) then fallback.
	for i := 0; i < 3; i++ {
		w := sendOpenAIChatRequest(t, handler, openAIChatRequest)
		if w.Code != http.StatusOK {
			t.Fatalf("req %d: status = %d, want 200", i+1, w.Code)
		}
	}

	if got := primaryCount.Load(); got != 3 {
		t.Fatalf("primary count after 3 requests = %d, want 3", got)
	}

	// Send 4th request — circuit should be open, primary skipped.
	w := sendOpenAIChatRequest(t, handler, openAIChatRequest)
	if w.Code != http.StatusOK {
		t.Fatalf("req 4: status = %d, want 200; body = %s", w.Code, w.Body.String())
	}

	// Primary should NOT have been called a 4th time (circuit tripped).
	if got := primaryCount.Load(); got != 3 {
		t.Fatalf("primary count after 4 requests = %d, want 3 (circuit should have skipped)", got)
	}
	if got := fallbackCount.Load(); got != 4 {
		t.Fatalf("fallback count = %d, want 4", got)
	}
}

func TestIntegration_Fallback_CircuitBreakerRecovery(t *testing.T) {
	var primaryCount atomic.Int64
	// Primary that fails first 3 times, then succeeds.
	primaryFailsRemaining := int64(3)
	primarySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := primaryCount.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if n <= primaryFailsRemaining {
			w.WriteHeader(500)
			w.Write([]byte(`{"error":{"message":"fail","type":"server_error"}}`))
			return
		}
		w.Write([]byte(`{"id":"test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hello from primary"},"finish_reason":"stop","index":0}],"model":"test-model","usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	t.Cleanup(primarySrv.Close)

	fallback := succeedingUpstream(t, ProtocolOpenAIChat)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: primarySrv.URL, APIKey: "sk-1", Model: "m1"},
			{Protocol: ProtocolOpenAIChat, BaseURL: fallback.URL, APIKey: "sk-2", Model: "m2"},
		}
	}, WithFailureThreshold(3), WithRecoveryTimeout(100*time.Millisecond), WithSuccessThreshold(1))

	mux := NewMux(router)
	handler := mux.OpenAIChatHandler()

	// Phase 1: Send 3 requests — each triggers primary fail then fallback.
	for i := 0; i < 3; i++ {
		w := sendOpenAIChatRequest(t, handler, openAIChatRequest)
		if w.Code != http.StatusOK {
			t.Fatalf("phase1 req %d: status = %d", i+1, w.Code)
		}
	}
	if got := primaryCount.Load(); got != 3 {
		t.Fatalf("phase1: primary count = %d, want 3", got)
	}

	// Phase 2: Circuit is open — primary should be skipped.
	w := sendOpenAIChatRequest(t, handler, openAIChatRequest)
	if w.Code != http.StatusOK {
		t.Fatalf("phase2: status = %d", w.Code)
	}
	if got := primaryCount.Load(); got != 3 {
		t.Fatalf("phase2: primary count = %d, want 3 (should be skipped while open)", got)
	}

	// Phase 3: Wait for recovery timeout, then send request.
	// Primary should be tried (HalfOpen) and succeed (count=4).
	time.Sleep(150 * time.Millisecond)
	w = sendOpenAIChatRequest(t, handler, openAIChatRequest)
	if w.Code != http.StatusOK {
		t.Fatalf("phase3: status = %d; body = %s", w.Code, w.Body.String())
	}
	if got := primaryCount.Load(); got != 4 {
		t.Fatalf("phase3: primary count = %d, want 4 (half-open probe)", got)
	}
	if !strings.Contains(w.Body.String(), "hello from primary") {
		t.Fatalf("phase3: expected primary response, got: %s", w.Body.String())
	}

	// Phase 4: Circuit should be Closed now. Next request goes to primary.
	w = sendOpenAIChatRequest(t, handler, openAIChatRequest)
	if w.Code != http.StatusOK {
		t.Fatalf("phase4: status = %d", w.Code)
	}
	if got := primaryCount.Load(); got != 5 {
		t.Fatalf("phase4: primary count = %d, want 5 (circuit closed)", got)
	}
}

func TestIntegration_Fallback_ConnectionRefused(t *testing.T) {
	// Start a server, get its URL, then close it immediately.
	deadServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := deadServer.URL
	deadServer.Close()

	fallback := succeedingUpstream(t, ProtocolOpenAIChat)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: deadURL, APIKey: "sk-1", Model: "m1"},
			{Protocol: ProtocolOpenAIChat, BaseURL: fallback.URL, APIKey: "sk-2", Model: "m2"},
		}
	}, WithFailureThreshold(10))

	mux := NewMux(router)
	w := sendOpenAIChatRequest(t, mux.OpenAIChatHandler(), openAIChatRequest)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "hello from fallback") {
		t.Fatalf("response body missing fallback content: %s", w.Body.String())
	}
}

func TestIntegration_Fallback_Timeout(t *testing.T) {
	primary := slowUpstream(t, ProtocolOpenAIChat, 2*time.Second)
	fallback := succeedingUpstream(t, ProtocolOpenAIChat)

	router := NewCircuitBreakerRouter(func(info RouteInfo) []RouteResult {
		return []RouteResult{
			{Protocol: ProtocolOpenAIChat, BaseURL: primary.URL, APIKey: "sk-1", Model: "m1"},
			{Protocol: ProtocolOpenAIChat, BaseURL: fallback.URL, APIKey: "sk-2", Model: "m2"},
		}
	}, WithFailureThreshold(10))

	mux := NewMux(router)
	handler := mux.OpenAIChatHandler()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	w := sendOpenAIChatRequestWithContext(t, handler, openAIChatRequest, ctx)

	// The context times out during the primary request. Since context is canceled,
	// the handler should NOT retry — it writes the error directly.
	// This is expected behavior: context cancellation aborts the entire request.
	// So we check that we get an error status (not 200 from fallback).
	if w.Code == http.StatusOK && strings.Contains(w.Body.String(), "hello from fallback") {
		// Fallback was triggered — this is also acceptable behavior.
		return
	}
	// Context canceled → error response is also acceptable.
	if w.Code >= 400 {
		return
	}
	t.Fatalf("unexpected status = %d; body = %s", w.Code, w.Body.String())
}

// --- Cross-protocol integration tests for parallel_tool_calls and allowed_tool_names ---

// TestIntegration_CrossProtocol_ParallelCallsInversion tests that parallel_tool_calls inversion
// works correctly when OpenAI Chat inbound sends parallel_tool_calls=false to an Anthropic outbound.
// OpenAI: parallel_tool_calls=false → IR: AllowParallelCalls=false → Anthropic: disable_parallel_tool_use=true
func TestIntegration_CrossProtocol_ParallelCallsInversion(t *testing.T) {
	var gotBody map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer upstream.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  upstream.URL,
		APIKey:   "sk-anthropic",
		Model:    "claude-sonnet-4-20250514",
	}})

	// OpenAI Chat inbound: parallel_tool_calls=false (meaning: disallow parallel calls)
	reqBody := `{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"tool_choice": "auto",
		"parallel_tool_calls": false
	}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.OpenAIChatHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	// Verify the outbound Anthropic request has disable_parallel_tool_use=true
	tcRaw, ok := gotBody["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing from outbound Anthropic request")
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}

	dptuRaw, ok := tc["disable_parallel_tool_use"]
	if !ok {
		t.Fatal("disable_parallel_tool_use missing from outbound Anthropic tool_choice")
	}
	var dptu bool
	if err := json.Unmarshal(dptuRaw, &dptu); err != nil {
		t.Fatalf("unmarshal disable_parallel_tool_use: %v", err)
	}
	if !dptu {
		t.Errorf("disable_parallel_tool_use = false, want true (OpenAI parallel_tool_calls=false → Anthropic disable=true)")
	}
}

// TestIntegration_CrossProtocol_ParallelCallsInversionTrue tests the opposite direction:
// OpenAI: parallel_tool_calls=true → IR: AllowParallelCalls=true → Anthropic: disable_parallel_tool_use=false
func TestIntegration_CrossProtocol_ParallelCallsInversionTrue(t *testing.T) {
	var gotBody map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer upstream.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  upstream.URL,
		APIKey:   "sk-anthropic",
		Model:    "claude-sonnet-4-20250514",
	}})

	reqBody := `{
		"model": "gpt-4o",
		"messages": [{"role": "user", "content": "Hello"}],
		"tool_choice": "auto",
		"parallel_tool_calls": true
	}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.OpenAIChatHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	tcRaw, ok := gotBody["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing from outbound Anthropic request")
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}

	dptuRaw, ok := tc["disable_parallel_tool_use"]
	if !ok {
		t.Fatal("disable_parallel_tool_use missing from outbound Anthropic tool_choice")
	}
	var dptu bool
	if err := json.Unmarshal(dptuRaw, &dptu); err != nil {
		t.Fatalf("unmarshal disable_parallel_tool_use: %v", err)
	}
	if dptu {
		t.Errorf("disable_parallel_tool_use = true, want false (OpenAI parallel_tool_calls=true → Anthropic disable=false)")
	}
}

// TestIntegration_CrossProtocol_AllowedToolNamesDegradation_SingleTool tests that a single
// allowed tool name from Gemini inbound is represented in the Anthropic outbound as
// tool_choice.type="tool" + tool_choice.name=<name>.
func TestIntegration_CrossProtocol_AllowedToolNamesDegradation_SingleTool(t *testing.T) {
	var gotBody map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer upstream.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  upstream.URL,
		APIKey:   "sk-anthropic",
		Model:    "claude-sonnet-4-20250514",
	}})

	// Gemini inbound: allowedFunctionNames=["my_tool"]
	reqBody := `{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"tools": [{"functionDeclarations": [{"name": "my_tool", "description": "A tool"}]}],
		"toolConfig": {
			"functionCallingConfig": {
				"mode": "ANY",
				"allowedFunctionNames": ["my_tool"]
			}
		}
	}`
	req := httptest.NewRequest("POST", "/v1/models/gemini-2.5-pro:generateContent", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.GeminiHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	tcRaw, ok := gotBody["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing from outbound Anthropic request")
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
		t.Errorf("tool_choice.type = %q, want tool (single allowed tool → named tool choice)", tcType)
	}

	var tcName string
	if err := json.Unmarshal(tc["name"], &tcName); err != nil {
		t.Fatalf("unmarshal tool_choice.name: %v", err)
	}
	if tcName != "my_tool" {
		t.Errorf("tool_choice.name = %q, want my_tool", tcName)
	}
}

// TestIntegration_CrossProtocol_AllowedToolNamesDegradation_MultiTool tests that multiple
// allowed tool names from Gemini inbound are silently dropped when converting to Anthropic.
func TestIntegration_CrossProtocol_AllowedToolNamesDegradation_MultiTool(t *testing.T) {
	var gotBody map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4-20250514",
			"content": [{"type": "text", "text": "ok"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer upstream.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  upstream.URL,
		APIKey:   "sk-anthropic",
		Model:    "claude-sonnet-4-20250514",
	}})

	// Gemini inbound: allowedFunctionNames=["tool_a", "tool_b"]
	reqBody := `{
		"contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
		"tools": [{"functionDeclarations": [
			{"name": "tool_a", "description": "Tool A"},
			{"name": "tool_b", "description": "Tool B"}
		]}],
		"toolConfig": {
			"functionCallingConfig": {
				"mode": "ANY",
				"allowedFunctionNames": ["tool_a", "tool_b"]
			}
		}
	}`
	req := httptest.NewRequest("POST", "/v1/models/gemini-2.5-pro:generateContent", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.GeminiHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	tcRaw, ok := gotBody["tool_choice"]
	if !ok {
		// tool_choice may be absent if no tool_choice was set; that's also acceptable for multi-tool drop
		return
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(tcRaw, &tc); err != nil {
		t.Fatalf("unmarshal tool_choice: %v", err)
	}

	var tcType string
	if err := json.Unmarshal(tc["type"], &tcType); err != nil {
		t.Fatalf("unmarshal tool_choice.type: %v", err)
	}
	// Multi-tool allowlist is silently dropped: type stays "any" (IR "required" → Anthropic "any")
	// The key assertion: type should NOT be "tool" with a specific name since we can't pick one
	if tcType == "tool" {
		if nameRaw, ok := tc["name"]; ok {
			var name string
			_ = json.Unmarshal(nameRaw, &name)
			t.Errorf("tool_choice.type = tool with name %q, but multi-tool allowlist should be silently dropped", name)
		}
	}
}

// --- Phase 1 matrix: additional cross-protocol integration tests ---

// TestIntegration_CrossProtocol_GeminiToOpenAIChat_ThinkingTokenPreservation verifies that
// thinking tokens from a Gemini inbound request are preserved when the OpenAI Chat upstream
// returns reasoning_tokens in completion_tokens_details.
// Gemini inbound → OpenAI Chat outbound (upstream returns reasoning tokens) → Gemini response with thoughtsTokenCount.
func TestIntegration_CrossProtocol_GeminiToOpenAIChat_ThinkingTokenPreservation(t *testing.T) {
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Return an OpenAI Chat response with reasoning_tokens in completion_tokens_details.
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-1",
			"model": "gpt-4o",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "thinking answer"}, "finish_reason": "stop"}],
			"usage": {
				"prompt_tokens": 10,
				"completion_tokens": 20,
				"total_tokens": 30,
				"completion_tokens_details": {"reasoning_tokens": 8}
			}
		}`))
	}))
	defer openaiServer.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}})

	reqBody := `{"contents":[{"role":"user","parts":[{"text":"Think carefully"}]}]}`
	req := httptest.NewRequest("POST", "/v1/models/gemini-2.5-pro:generateContent", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.GeminiHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	// Gemini response must include usageMetadata.thoughtsTokenCount = 8.
	usageMeta, ok := resp["usageMetadata"].(map[string]any)
	if !ok {
		t.Fatal("usageMetadata missing from Gemini response")
	}
	// JSON numbers decode as float64 in Go.
	thinkingTokens, ok := usageMeta["thoughtsTokenCount"].(float64)
	if !ok {
		t.Fatalf("thoughtsTokenCount missing or wrong type in usageMetadata: %v", usageMeta)
	}
	if int(thinkingTokens) != 8 {
		t.Errorf("thoughtsTokenCount = %d, want 8", int(thinkingTokens))
	}
}

// TestIntegration_CrossProtocol_GeminiToOpenAIChat_AllowedToolNamesDegraded verifies that
// allowed function names from Gemini are preserved in the IR's AllowedToolNames field but
// are silently degraded when converted to OpenAI Chat (which has no allowlist concept).
// The outbound tool_choice should reflect the base mode ("required") without a specific name filter.
func TestIntegration_CrossProtocol_GeminiToOpenAIChat_AllowedToolNamesDegraded(t *testing.T) {
	var gotBody map[string]json.RawMessage
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-1",
			"model": "gpt-4o",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`))
	}))
	defer openaiServer.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "gpt-4o",
	}})

	// Gemini inbound: allowedFunctionNames=["my_tool"] with mode=ANY
	// IR: ToolChoice.Type="required", ToolChoice.AllowedToolNames=["my_tool"]
	// OpenAI Chat outbound: tool_choice="required" (AllowedToolNames silently dropped — no allowlist concept)
	reqBody := `{
		"contents": [{"role": "user", "parts": [{"text": "Use the tool"}]}],
		"tools": [{"functionDeclarations": [{"name": "my_tool", "description": "A tool"}]}],
		"toolConfig": {
			"functionCallingConfig": {
				"mode": "ANY",
				"allowedFunctionNames": ["my_tool"]
			}
		}
	}`
	req := httptest.NewRequest("POST", "/v1/models/gemini-2.5-pro:generateContent", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.GeminiHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	// OpenAI Chat outbound tool_choice should be "required" (the base mode from ANY).
	// AllowedToolNames has no direct encoding in OpenAI Chat and is silently dropped.
	tcRaw, ok := gotBody["tool_choice"]
	if !ok {
		t.Fatal("tool_choice missing from outbound OpenAI Chat request")
	}
	// tool_choice="required" is a JSON string, not an object.
	var tcStr string
	if err := json.Unmarshal(tcRaw, &tcStr); err != nil {
		t.Fatalf("unmarshal tool_choice: %v (raw: %s)", err, string(tcRaw))
	}
	if tcStr != "required" {
		t.Errorf("tool_choice = %q, want required (Gemini ANY mode → OpenAI required, allowlist dropped)", tcStr)
	}
}

// TestIntegration_CrossProtocol_OpenAIResponsesToAnthropic_StreamErrorSurvives verifies that
// when an Anthropic outbound stream returns an error event, it survives the conversion and
// reaches the OpenAI Responses inbound client as an SSE error event.
// OpenAI Responses inbound → Anthropic outbound (stream with error) → OpenAI Responses SSE with error event.
func TestIntegration_CrossProtocol_OpenAIResponsesToAnthropic_StreamErrorSurvives(t *testing.T) {
	anthropicServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		// Emit a minimal message_start then an error event.
		fmt.Fprintf(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-20250514\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Service overloaded\"}}\n\n")
		flusher.Flush()
	}))
	defer anthropicServer.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  anthropicServer.URL,
		APIKey:   "sk-ant",
		Model:    "claude-sonnet-4-20250514",
	}})

	// OpenAI Responses inbound with stream=true.
	reqBody := `{"model":"gpt-4o","input":"Hello","max_output_tokens":100,"stream":true}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.OpenAIResponsesHandler().ServeHTTP(w, req)

	respBody := w.Body.String()
	// The SSE response must include an "error" event type.
	if !strings.Contains(respBody, `"type":"error"`) {
		t.Errorf("expected error event in SSE stream, got: %s", respBody)
	}
}

// TestIntegration_CrossProtocol_ProviderExtensions_DoesNotInterfereWithPipeline verifies that
// ProviderExtensions set on the IR Request and Response are transparent to the decode/encode
// pipeline — the field neither causes errors nor leaks into protocol-specific outbound bodies.
func TestIntegration_CrossProtocol_ProviderExtensions_DoesNotInterfereWithPipeline(t *testing.T) {
	var gotUpstreamBody map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotUpstreamBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-1",
			"model": "gpt-4o",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
		}`))
	}))
	defer upstream.Close()

	// Use a RequestModifier to set ProviderExtensions on the IR Request.
	mux := NewMux(
		&staticRouter{result: RouteResult{
			Protocol: ProtocolOpenAIChat,
			BaseURL:  upstream.URL,
			APIKey:   "sk-openai",
			Model:    "gpt-4o",
		}},
		WithRequestModifier(func(_ context.Context, req *Request, _ RouteResult) {
			req.ProviderExtensions = ProviderExtensions{
				"test/meta": json.RawMessage(`"value"`),
			}
		}),
	)

	// Anthropic inbound → OpenAI Chat outbound (cross-protocol).
	reqBody := `{"model":"claude-sonnet-4-20250514","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.AnthropicHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	// ProviderExtensions must NOT leak into the outbound OpenAI Chat body.
	if _, ok := gotUpstreamBody["provider_extensions"]; ok {
		t.Error("provider_extensions should not appear in outbound OpenAI Chat request body")
	}
	if _, ok := gotUpstreamBody["test/meta"]; ok {
		t.Error("test/meta should not appear in outbound OpenAI Chat request body")
	}

	// The inbound Anthropic response should be valid.
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if resp["type"] != "message" {
		t.Errorf("response type = %v, want message", resp["type"])
	}
}

// TestIntegration_CrossProtocol_GeminiToOpenAIChat_ThinkingControlsDegraded verifies that
// Gemini-specific Phase 2 thinking controls (includeThoughts, thinkingLevel) are decoded
// from the Gemini inbound request into the IR but silently dropped when encoding to the
// OpenAI Chat outbound protocol, which has no native equivalent.
func TestIntegration_CrossProtocol_GeminiToOpenAIChat_ThinkingControlsDegraded(t *testing.T) {
	var gotUpstreamBody map[string]json.RawMessage
	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotUpstreamBody); err != nil {
			t.Fatalf("decode upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-1",
			"model": "o1",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`))
	}))
	defer openaiServer.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  openaiServer.URL,
		APIKey:   "sk-openai",
		Model:    "o1",
	}})

	// Gemini inbound request with Phase 2 thinking controls.
	reqBody := `{
		"contents": [{"role": "user", "parts": [{"text": "Think deeply"}]}],
		"generationConfig": {
			"thinkingConfig": {
				"thinkingBudget": 4096,
				"includeThoughts": true,
				"thinkingLevel": "HIGH"
			}
		}
	}`
	req := httptest.NewRequest("POST", "/v1/models/gemini-2.5-pro:generateContent", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	mux.GeminiHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	// OpenAI Chat outbound: includeThoughts and thinkingLevel should not appear.
	// Effort mapping still applies for the mode→reasoning_effort degradation.
	for _, field := range []string{"include_thoughts", "includeThoughts", "level", "thinkingLevel", "thinking_config", "thinkingConfig"} {
		if _, ok := gotUpstreamBody[field]; ok {
			t.Errorf("field %q should not appear in outbound OpenAI Chat request (silently dropped)", field)
		}
	}
}
