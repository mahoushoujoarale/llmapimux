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

func TestOpenAIChatClient_Send(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer sk-test" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		json.Unmarshal(body, &parsed)
		if parsed["model"] != "gpt-4o" {
			t.Errorf("model = %v", parsed["model"])
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer server.Close()

	client := &OpenAIChatClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
	}

	resp, err := client.Send(context.Background(), req, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "chatcmpl-1" {
		t.Errorf("id = %q", resp.ID)
	}
	if resp.StopReason != StopReasonEndTurn {
		t.Errorf("stop_reason = %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("input_tokens = %d", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("output_tokens = %d", resp.Usage.OutputTokens)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("total_tokens = %d", resp.Usage.TotalTokens)
	}
	if len(resp.Content) == 0 {
		t.Fatal("expected content")
	}
	if resp.Content[0].Type != ContentTypeText {
		t.Errorf("content type = %q", resp.Content[0].Type)
	}
	if resp.Content[0].Text == nil || resp.Content[0].Text.Text != "Hello!" {
		t.Errorf("content text = %v", resp.Content[0].Text)
	}
}

func TestOpenAIChatClient_SendStream_StopsOnContextCancel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := 0; ; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"chunk-%d\"},\"finish_reason\":null}]}\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer server.Close()

	client := &OpenAIChatClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
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

func TestOpenAIChatClient_SendStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	defer server.Close()

	client := &OpenAIChatClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
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

	if len(events) < 3 {
		t.Fatalf("expected at least 3 events, got %d", len(events))
	}

	// First event should be start
	if events[0].Type != StreamEventStart {
		t.Errorf("first event type = %q, want StreamEventStart", events[0].Type)
	}

	// Find a delta event with text content
	found := false
	for _, e := range events {
		if e.Type == StreamEventDelta && e.Delta != nil && e.Delta.Type == ContentTypeText {
			if e.Delta.Text != nil && e.Delta.Text.Text == "Hello" {
				found = true
			}
		}
	}
	if !found {
		t.Error("expected delta event with text 'Hello'")
	}

	// Last event should be stop
	last := events[len(events)-1]
	if last.Type != StreamEventStop {
		t.Errorf("last event type = %q, want StreamEventStop", last.Type)
	}
	if last.StopReason == nil || *last.StopReason != StopReasonEndTurn {
		t.Errorf("last event stop_reason = %v", last.StopReason)
	}
}

func TestOpenAIChatClient_SendStream_TruncatedStopChunkDoesNotEmitStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]")
		flusher.Flush()
	}))
	defer server.Close()

	client := &OpenAIChatClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
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

func TestOpenAIChatClient_Send_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		w.Write([]byte(`{"error":{"message":"bad request"}}`))
	}))
	defer server.Close()

	client := &OpenAIChatClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
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
	if string(upstreamErr.Body) != `{"error":{"message":"bad request"}}` {
		t.Errorf("body = %q", string(upstreamErr.Body))
	}
}

func TestOpenAIChatClient_SendStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(502)
		w.Write([]byte(`{"error":{"message":"bad gateway"}}`))
	}))
	defer server.Close()

	client := &OpenAIChatClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
	}

	_, err := client.SendStream(context.Background(), req, cfg)
	if err == nil {
		t.Fatal("expected error")
	}

	var upstreamErr *UpstreamHTTPError
	if !errors.As(err, &upstreamErr) {
		t.Fatalf("expected UpstreamHTTPError, got %T (%v)", err, err)
	}
	if upstreamErr.StatusCode != 502 {
		t.Errorf("status code = %d, want 502", upstreamErr.StatusCode)
	}
	if string(upstreamErr.Body) != `{"error":{"message":"bad gateway"}}` {
		t.Errorf("body = %q", string(upstreamErr.Body))
	}
}

func TestOpenAIChatClient_Send_TransportError(t *testing.T) {
	client := &OpenAIChatClient{}
	cfg := OutboundConfig{BaseURL: "://bad-url", APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
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

func TestOpenAIChatClient_SendStreamThenSend_DoesNotMutateRequest(t *testing.T) {
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
			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		case 2:
			if parsed["stream"] == true {
				t.Fatalf("second request stream = true, want false/omitted")
			}
			if parsed["model"] != "gpt-4o" {
				t.Fatalf("second request model = %v, want gpt-4o", parsed["model"])
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		default:
			t.Fatalf("unexpected call count %d", callCount)
		}
	}))
	defer server.Close()

	client := &OpenAIChatClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
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
	if req.Model != "gpt-4o" {
		t.Fatalf("request.Model mutated to %q, want gpt-4o", req.Model)
	}

	if _, err := client.Send(context.Background(), req, cfg); err != nil {
		t.Fatal(err)
	}

	if req.Stream {
		t.Fatal("request.Stream mutated after Send")
	}
	if req.Model != "gpt-4o" {
		t.Fatalf("request.Model mutated after Send to %q, want gpt-4o", req.Model)
	}
}
