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

func TestOpenAIResponsesClient_Send(t *testing.T) {
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
		w.Write([]byte(`{"id":"resp_1","object":"response","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello!"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`))
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
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
	if resp.ID != "resp_1" {
		t.Errorf("id = %q", resp.ID)
	}
	if resp.StopReason != StopReasonEndTurn {
		t.Errorf("stop_reason = %q", resp.StopReason)
	}
	if resp.Usage.PromptTokens != 10 {
		t.Errorf("input_tokens = %d", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("output_tokens = %d", resp.Usage.CompletionTokens)
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

func TestOpenAIResponsesClient_Send_WebSearchAddsInclude(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]json.RawMessage
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal request: %v", err)
		}

		includeRaw, ok := parsed["include"]
		if !ok {
			t.Fatal("include missing from request")
		}
		var include []string
		if err := json.Unmarshal(includeRaw, &include); err != nil {
			t.Fatalf("unmarshal include: %v", err)
		}
		found := false
		for _, item := range include {
			if item == "web_search_call.action.sources" {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("include = %v, want web_search_call.action.sources", include)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello!"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`))
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: 1024,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}},
		Tools: []Tool{
			{Type: "web_search", Name: "web_search"},
		},
	}

	if _, err := client.Send(context.Background(), req, cfg); err != nil {
		t.Fatal(err)
	}
}

func TestOpenAIResponsesClient_SendStream_StopsOnContextCancel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := 0; ; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			fmt.Fprintf(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"chunk-%d\"}\n\n", i)
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
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

func TestOpenAIResponsesClient_SendStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"status\":\"in_progress\",\"output\":[]}}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}}\n\n")
		flusher.Flush()

		fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
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

func TestOpenAIResponsesClient_SendStream_TruncatedCompletedDoesNotEmitStop(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"status\":\"in_progress\",\"output\":[]}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"status\":\"completed\"")
		flusher.Flush()
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
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
	if streamErr != nil && !errors.Is(streamErr, io.EOF) && !strings.Contains(streamErr.Error(), "unexpected EOF") {
		t.Fatalf("stream error = %v, want nil or EOF-style truncation", streamErr)
	}
}

func TestOpenAIResponsesClient_SendStream_LifecycleSequence(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"status\":\"in_progress\",\"output\":[]}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"Hello\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{Model: "gpt-4o", Messages: []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}}}

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

	// response.content_part.added and response.content_part.done are now skipped;
	// only the canonical response.output_item.added / response.output_item.done events
	// produce ContentBlockStart / ContentBlockStop in the IR stream.
	if len(events) != 5 {
		t.Fatalf("events len = %d, want 5", len(events))
	}
	want := []StreamEventType{
		StreamEventStart,
		StreamEventContentBlockStart,
		StreamEventDelta,
		StreamEventContentBlockStop,
		StreamEventStop,
	}
	for i, wantType := range want {
		if events[i].Type != wantType {
			t.Fatalf("events[%d].Type = %q, want %q", i, events[i].Type, wantType)
		}
	}
	if events[2].Delta == nil || events[2].Delta.Text == nil || events[2].Delta.Text.Text != "Hello" {
		t.Fatalf("delta event = %+v, want text Hello", events[2].Delta)
	}
}

func TestOpenAIResponsesClient_SendStream_ReasoningModelIndexRemap(t *testing.T) {
	// Simulates a reasoning model (e.g. o-series) that produces:
	// - A reasoning item at output_index=0 (should be suppressed)
	// - A message item at output_index=1 (should be remapped to index=0)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		fmt.Fprintf(w, "event: response.created\ndata: {\"response\":{\"id\":\"resp_1\",\"model\":\"o4-mini\",\"status\":\"in_progress\"}}\n\n")
		flusher.Flush()

		// Reasoning item (type="reasoning") — should be suppressed entirely
		fmt.Fprintf(w, "event: response.output_item.added\ndata: {\"output_index\":0,\"item\":{\"type\":\"reasoning\"}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.output_item.done\ndata: {\"output_index\":0}\n\n")
		flusher.Flush()

		// Message item at index=1 — should be remapped to index=0
		fmt.Fprintf(w, "event: response.output_item.added\ndata: {\"output_index\":1,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.output_text.delta\ndata: {\"output_index\":1,\"delta\":\"Hello\"}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.output_item.done\ndata: {\"output_index\":1}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "event: response.completed\ndata: {\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3,\"total_tokens\":8}}}\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
	cfg := OutboundConfig{BaseURL: server.URL, APIKey: "sk-test"}
	req := &Request{Model: "o4-mini", Messages: []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}}}}}}

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

	// Expect: Start, ContentBlockStart(0), Delta(0), ContentBlockStop(0), Stop
	if len(events) != 5 {
		types := make([]StreamEventType, len(events))
		for i, e := range events {
			types[i] = e.Type
		}
		t.Fatalf("events len = %d, want 5; types = %v", len(events), types)
	}
	want := []StreamEventType{
		StreamEventStart,
		StreamEventContentBlockStart,
		StreamEventDelta,
		StreamEventContentBlockStop,
		StreamEventStop,
	}
	for i, wantType := range want {
		if events[i].Type != wantType {
			t.Errorf("events[%d].Type = %q, want %q", i, events[i].Type, wantType)
		}
	}
	// All indexed events should have been remapped to index 0
	if events[1].Index != 0 {
		t.Errorf("ContentBlockStart.Index = %d, want 0 (remapped from 1)", events[1].Index)
	}
	if events[2].Index != 0 {
		t.Errorf("Delta.Index = %d, want 0 (remapped from 1)", events[2].Index)
	}
	if events[3].Index != 0 {
		t.Errorf("ContentBlockStop.Index = %d, want 0 (remapped from 1)", events[3].Index)
	}
}

func TestOpenAIResponsesClient_Send_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		w.Write([]byte(`{"error":{"message":"bad request"}}`))
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
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

func TestOpenAIResponsesClient_SendStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(503)
		w.Write([]byte(`{"error":{"message":"service unavailable"}}`))
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
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
	if upstreamErr.StatusCode != 503 {
		t.Errorf("status code = %d, want 503", upstreamErr.StatusCode)
	}
	if string(upstreamErr.Body) != `{"error":{"message":"service unavailable"}}` {
		t.Errorf("body = %q", string(upstreamErr.Body))
	}
}

func TestOpenAIResponsesClient_Send_TransportError(t *testing.T) {
	client := &OpenAIResponsesClient{}
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

func TestOpenAIResponsesClient_SendStreamThenSend_DoesNotMutateRequest(t *testing.T) {
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
			fmt.Fprintf(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"status\":\"in_progress\",\"output\":[]}}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4o\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n")
			flusher.Flush()
		case 2:
			if parsed["stream"] == true {
				t.Fatalf("second request stream = true, want false/omitted")
			}
			if parsed["model"] != "gpt-4o" {
				t.Fatalf("second request model = %v, want gpt-4o", parsed["model"])
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"id":"resp_2","object":"response","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
		default:
			t.Fatalf("unexpected call count %d", callCount)
		}
	}))
	defer server.Close()

	client := &OpenAIResponsesClient{}
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
