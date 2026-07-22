package llmapimux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestGeminiClient_Send(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify URL contains the model
		if !strings.Contains(r.URL.Path, "gemini-2.5-pro") {
			t.Errorf("URL path = %s, expected to contain gemini-2.5-pro", r.URL.Path)
		}
		// Verify URL path uses generateContent action
		if !strings.Contains(r.URL.Path, "generateContent") {
			t.Errorf("URL path = %s, expected to contain generateContent", r.URL.Path)
		}
		if r.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("missing api key header, got %q", r.Header.Get("x-goog-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`))
	}))
	defer server.Close()

	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "test-key"}
	req := &Request{
		Model:     "gemini-2.5-pro",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
	}

	resp, err := client.Send(context.Background(), req, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	if resp.StopReason != StopReasonEndTurn {
		t.Errorf("stop_reason = %q, want end_turn", resp.StopReason)
	}
	if resp.Usage.PromptTokens != 10 {
		t.Errorf("input_tokens = %d, want 10", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("output_tokens = %d, want 5", resp.Usage.CompletionTokens)
	}
	if len(resp.Content) == 0 {
		t.Fatal("expected non-empty content")
	}
	if resp.Content[0].Text == nil || resp.Content[0].Text.Text != "Hello!" {
		t.Errorf("content[0].text = %v, want Hello!", resp.Content[0].Text)
	}
}

func TestGeminiClient_SendStream_StopsOnContextCancel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := 0; ; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"chunk-%d\"}]}}]}\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer server.Close()

	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "test-key"}
	req := &Request{
		Model:     "gemini-2.5-pro",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
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

func TestGeminiClient_SendStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "streamGenerateContent") {
			t.Errorf("expected stream URL, got %s", r.URL.Path)
		}
		if !strings.Contains(r.URL.RawQuery, "alt=sse") {
			t.Errorf("expected alt=sse query param, got %s", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\" world\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"totalTokenCount\":15}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "test-key"}
	req := &Request{
		Model:     "gemini-2.5-pro",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
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

	if len(events) == 0 {
		t.Fatal("expected at least one event")
	}

	// First event should be a delta with "Hello"
	if events[0].Type != StreamEventDelta {
		t.Errorf("events[0].type = %q, want delta", events[0].Type)
	}
	if events[0].Delta == nil || events[0].Delta.Text == nil || events[0].Delta.Text.Text != "Hello" {
		t.Errorf("events[0].delta text = %v, want Hello", events[0].Delta)
	}

	// Last event should have a stop reason or be a stop event
	last := events[len(events)-1]
	if last.StopReason == nil && last.Type != StreamEventStop {
		t.Errorf("last event has no stop reason, type = %q", last.Type)
	}
}

func TestGeminiClient_SendStream_TruncatedStopChunkDoesNotEmitStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10")
		flusher.Flush()
	}))
	defer server.Close()

	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "test-key"}
	req := &Request{
		Model:     "gemini-2.5-pro",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
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

func TestGeminiClient_Send_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		w.Write([]byte(`{"error":{"code":400,"message":"bad request","status":"INVALID_ARGUMENT"}}`))
	}))
	defer server.Close()

	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "test-key"}
	req := &Request{
		Model:     "gemini-2.5-pro",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
	}

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
	if string(upstreamErr.Body) != `{"error":{"code":400,"message":"bad request","status":"INVALID_ARGUMENT"}}` {
		t.Errorf("body = %q", string(upstreamErr.Body))
	}
}

func TestGeminiClient_SendStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		io.WriteString(w, `{"error":{"code":500,"message":"internal error","status":"INTERNAL"}}`)
	}))
	defer server.Close()

	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "test-key"}
	req := &Request{
		Model:     "gemini-2.5-pro",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
	}

	_, err := client.SendStream(context.Background(), req, cfg)
	if err == nil {
		t.Fatal("expected error")
	}

	var upstreamErr *UpstreamHTTPError
	if !errors.As(err, &upstreamErr) {
		t.Fatalf("expected UpstreamHTTPError, got %T (%v)", err, err)
	}
	if upstreamErr.StatusCode != 500 {
		t.Errorf("status code = %d, want 500", upstreamErr.StatusCode)
	}
	if string(upstreamErr.Body) != `{"error":{"code":500,"message":"internal error","status":"INTERNAL"}}` {
		t.Errorf("body = %q", string(upstreamErr.Body))
	}
}

func TestGeminiClient_Send_TransportError(t *testing.T) {
	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: "://bad-url", APIKey: "test-key"}
	req := &Request{
		Model:     "gemini-2.5-pro",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
	}

	_, err := client.Send(context.Background(), req, cfg)
	if err == nil {
		t.Fatal("expected error")
	}

	var upstreamErr *UpstreamHTTPError
	if errors.As(err, &upstreamErr) {
		t.Fatalf("expected non-UpstreamHTTPError transport failure, got %T (%v)", err, err)
	}
}

func TestGeminiClient_SendStreamThenSend_DoesNotMutateRequest(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++

		if !strings.Contains(r.URL.Path, "gemini-2.5-pro") {
			t.Fatalf("URL path = %s, expected model gemini-2.5-pro", r.URL.Path)
		}

		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal request: %v", err)
		}

		switch callCount {
		case 1:
			if _, ok := parsed["generationConfig"]; ok {
				// ignore; stream flag is path-level for Gemini, not body-level
			}
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\" world\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}\n\n")
			flusher.Flush()
		case 2:
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`))
		default:
			t.Fatalf("unexpected call count %d", callCount)
		}
	}))
	defer server.Close()

	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "test-key"}
	req := &Request{
		Model:     "gemini-2.5-pro",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}},
		},
	}

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
	if req.Model != "gemini-2.5-pro" {
		t.Fatalf("request.Model mutated to %q, want gemini-2.5-pro", req.Model)
	}

	if _, err := client.Send(context.Background(), req, cfg); err != nil {
		t.Fatal(err)
	}

	if req.Stream {
		t.Fatal("request.Stream mutated after Send")
	}
	if req.Model != "gemini-2.5-pro" {
		t.Fatalf("request.Model mutated after Send to %q, want gemini-2.5-pro", req.Model)
	}
}

func TestGeminiStreamSplit_NoDuplicateDelta(t *testing.T) {
	// Gemini chunk with text + finishReason in same response → should yield
	// exactly ONE delta event (text content) + ONE stop event.
	chunk := `{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", chunk)
	}))
	defer server.Close()

	client := &GeminiClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "test-key"}
	req := &Request{
		Model:    "gemini-2.5-pro",
		Messages: []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}}},
		Stream:   true,
	}

	ch, err := client.SendStream(context.Background(), req, cfg)
	if err != nil {
		t.Fatal(err)
	}

	var events []*StreamEvent
	for result := range ch {
		if result.Err != nil {
			t.Fatalf("unexpected error: %v", result.Err)
		}
		events = append(events, result.Event)
	}

	// Expect: start(optional), exactly 1 delta, exactly 1 stop
	var deltaCount, stopCount int
	for _, e := range events {
		switch e.Type {
		case StreamEventDelta:
			deltaCount++
		case StreamEventStop:
			stopCount++
		}
	}
	if deltaCount != 1 {
		t.Errorf("got %d delta events, want exactly 1 (duplicate detected)", deltaCount)
	}
	if stopCount != 1 {
		t.Errorf("got %d stop events, want exactly 1", stopCount)
	}
}
