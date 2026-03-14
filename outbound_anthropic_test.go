package llmapimux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestAnthropicClient_Send(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify headers
		if r.Header.Get("x-api-key") != "sk-test" {
			t.Errorf("missing api key")
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Errorf("missing anthropic-version")
		}
		// Verify body has correct model
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		json.Unmarshal(body, &parsed)
		if parsed["model"] != "claude-sonnet-4-20250514" {
			t.Errorf("model = %v", parsed["model"])
		}
		// Return response
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":          "msg_123",
			"model":       "claude-sonnet-4-20250514",
			"content":     []map[string]any{{"type": "text", "text": "Hello!"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 10, "output_tokens": 5},
		})
	}))
	defer server.Close()

	client := &AnthropicClient{}
	cfg := OutboundConfig{
		BaseURL: server.URL,
		APIKey:  "sk-test",
	}
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
	}

	resp, err := client.Send(context.Background(), req, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "msg_123" {
		t.Errorf("id = %q", resp.ID)
	}
	if resp.StopReason != StopReasonEndTurn {
		t.Errorf("stop_reason = %q", resp.StopReason)
	}
}

func TestAnthropicClient_SendStream_StopsOnContextCancel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := 0; ; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			fmt.Fprintf(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"chunk-%d\"}}\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer server.Close()

	client := &AnthropicClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := client.SendStream(ctx, req, cfg)
	if err != nil {
		t.Fatal(err)
	}

	select {
	case result, ok := <-ch:
		if !ok {
			t.Fatal("stream channel closed before first item")
		}
		if result.Err != nil {
			t.Fatalf("first stream result err = %v", result.Err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("timeout waiting first stream item")
	}

	cancel()
	closed := make(chan struct{})
	go func() {
		for range ch {
		}
		close(closed)
	}()
	select {
	case <-closed:
	case <-time.After(1 * time.Second):
		t.Fatal("stream channel did not close after context cancellation")
	}
}

func TestAnthropicClient_SendStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-20250514\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := &AnthropicClient{}
	cfg := OutboundConfig{
		BaseURL: server.URL,
		APIKey:  "sk-test",
	}
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
	}

	ch, err := client.SendStream(context.Background(), req, cfg)
	if err != nil {
		t.Fatal(err)
	}

	var events []*StreamEvent
	for result := range ch {
		if result.Err != nil {
			t.Fatal(result.Err)
		}
		events = append(events, result.Event)
	}

	// Verify we got all event types
	if len(events) < 5 {
		t.Fatalf("expected at least 5 events, got %d", len(events))
	}
	if events[0].Type != StreamEventStart {
		t.Errorf("first event = %q", events[0].Type)
	}
}

func TestAnthropicClient_SendStream_TruncatedMessageStopDoesNotEmitStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-20250514\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: message_stop\ndata: {\"type\":\"message_stop\"")
		flusher.Flush()
	}))
	defer server.Close()

	client := &AnthropicClient{}
	cfg := OutboundConfig{
		BaseURL: server.URL,
		APIKey:  "sk-test",
	}
	req := &Request{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
	}

	ch, err := client.SendStream(context.Background(), req, cfg)
	if err != nil {
		t.Fatal(err)
	}

	var events []*StreamEvent
	var streamErr error
	for result := range ch {
		if result.Err != nil {
			streamErr = result.Err
			continue
		}
		events = append(events, result.Event)
	}

	if len(events) == 0 {
		t.Fatal("expected at least one event before truncation")
	}
	last := events[len(events)-1]
	if last.Type == StreamEventStop {
		t.Fatalf("unexpected stop event before truncated stream end")
	}
	if streamErr != nil && !errors.Is(streamErr, io.EOF) {
		t.Fatalf("stream error = %v, want nil or EOF truncation", streamErr)
	}
}

func TestAnthropicClient_Send_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		w.Write([]byte(`{"error":{"message":"bad request"}}`))
	}))
	defer server.Close()

	client := &AnthropicClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{Model: "claude-sonnet-4-20250514", MaxTokens: 1024, Messages: []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}}}

	_, err := client.Send(context.Background(), req, cfg)
	if err == nil {
		t.Fatal("expected error")
	}

	var upstreamErr *UpstreamHTTPError
	if !errors.As(err, &upstreamErr) {
		t.Fatalf("expected UpstreamHTTPError, got %T (%v)", err, err)
	}
	if upstreamErr.StatusCode != 400 {
		t.Errorf("status code = %d, want 400", upstreamErr.StatusCode)
	}
	if string(upstreamErr.Body) != `{"error":{"message":"bad request"}}` {
		t.Errorf("body = %q", string(upstreamErr.Body))
	}
}

func TestAnthropicClient_SendStream_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(429)
		w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer server.Close()

	client := &AnthropicClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{Model: "claude-sonnet-4-20250514", MaxTokens: 1024, Messages: []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}}}

	_, err := client.SendStream(context.Background(), req, cfg)
	if err == nil {
		t.Fatal("expected error")
	}

	var upstreamErr *UpstreamHTTPError
	if !errors.As(err, &upstreamErr) {
		t.Fatalf("expected UpstreamHTTPError, got %T (%v)", err, err)
	}
	if upstreamErr.StatusCode != 429 {
		t.Errorf("status code = %d, want 429", upstreamErr.StatusCode)
	}
	if string(upstreamErr.Body) != `{"error":{"message":"rate limited"}}` {
		t.Errorf("body = %q", string(upstreamErr.Body))
	}
}

func TestAnthropicClient_Send_TransportError(t *testing.T) {
	client := &AnthropicClient{}
	cfg := OutboundConfig{BaseURL: "://bad-url", APIKey: "sk-test"}
	req := &Request{Model: "claude-sonnet-4-20250514", MaxTokens: 1024, Messages: []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}}}

	_, err := client.Send(context.Background(), req, cfg)
	if err == nil {
		t.Fatal("expected error")
	}

	var upstreamErr *UpstreamHTTPError
	if errors.As(err, &upstreamErr) {
		t.Fatalf("expected non-UpstreamHTTPError transport failure, got %T (%v)", err, err)
	}
}

func TestAnthropicClient_SendStreamThenSend_DoesNotMutateRequest(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++

		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal request: %v", err)
		}

		switch callCount {
		case 1:
			if parsed["stream"] != true {
				t.Fatalf("first request stream = %v, want true", parsed["stream"])
			}
			if parsed["model"] != "claude-sonnet-4-20250514" {
				t.Fatalf("first request model = %v, want claude-sonnet-4-20250514", parsed["model"])
			}
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			fmt.Fprintf(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-20250514\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			flusher.Flush()
		case 2:
			if parsed["stream"] == true {
				t.Fatalf("second request stream = true, want false/omitted")
			}
			if parsed["model"] != "claude-sonnet-4-20250514" {
				t.Fatalf("second request model = %v, want claude-sonnet-4-20250514", parsed["model"])
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"id":"msg_2","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		default:
			t.Fatalf("unexpected call count %d", callCount)
		}
	}))
	defer server.Close()

	client := &AnthropicClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{Model: "claude-sonnet-4-20250514", MaxTokens: 1024, Messages: []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}}}

	ch, err := client.SendStream(context.Background(), req, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for r := range ch {
		if r.Err != nil {
			t.Fatal(r.Err)
		}
	}

	if req.Stream {
		t.Fatal("request.Stream mutated to true")
	}
	if req.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("request.Model mutated to %q, want claude-sonnet-4-20250514", req.Model)
	}

	if _, err := client.Send(context.Background(), req, cfg); err != nil {
		t.Fatal(err)
	}

	if req.Stream {
		t.Fatal("request.Stream mutated after Send")
	}
	if req.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("request.Model mutated after Send to %q, want claude-sonnet-4-20250514", req.Model)
	}
}
