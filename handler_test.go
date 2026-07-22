package llmapimux

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// staticRouter always routes to a fixed target.
type staticRouter struct {
	result RouteResult
	err    error
}

func (r *staticRouter) Route(_ context.Context, _ RouteInfo) (RouteResult, error) {
	return r.result, r.err
}

func (r *staticRouter) OnError(_ context.Context, _ RouteInfo, _ RouteResult, sendErr SendError) (RouteResult, error) {
	return RouteResult{}, sendErr.Err
}

func (r *staticRouter) OnSuccess(_ context.Context, _ RouteInfo, _ RouteResult) {}

// rejectAuth always rejects.
type rejectAuth struct{}

func (a *rejectAuth) Authenticate(_ context.Context, _ string) error {
	return fmt.Errorf("unauthorized")
}

type recordingStatsReporter struct {
	mu            sync.Mutex
	order         []string
	starts        []RequestStartEvent
	firstBytes    []FirstByteEvent
	chunks        []StreamChunkEvent
	completions   []CompleteEvent
	attemptErrors []AttemptErrorEvent
	onChunk       func(StreamChunkEvent)
	onComplete    func(CompleteEvent)
}

type testAttemptController struct {
	acquire func(ctx context.Context, info RouteInfo, target RouteResult, routeAttempt int, retryAttempt int) (AttemptAdmission, error)
	retry   func(ctx context.Context, info RouteInfo, target RouteResult, sendErr SendError, routeAttempt int, retryAttempt int) (time.Duration, bool)
}

func (c *testAttemptController) Acquire(ctx context.Context, info RouteInfo, target RouteResult, routeAttempt int, retryAttempt int) (AttemptAdmission, error) {
	if c.acquire != nil {
		return c.acquire(ctx, info, target, routeAttempt, retryAttempt)
	}
	return AttemptAdmission{}, nil
}

func (c *testAttemptController) RetryDelay(ctx context.Context, info RouteInfo, target RouteResult, sendErr SendError, routeAttempt int, retryAttempt int) (time.Duration, bool) {
	if c.retry != nil {
		return c.retry(ctx, info, target, sendErr, routeAttempt, retryAttempt)
	}
	return 0, false
}

type testPermit struct {
	release func()
}

func (p testPermit) Release() {
	if p.release != nil {
		p.release()
	}
}

func (r *recordingStatsReporter) OnRequestStart(_ context.Context, e RequestStartEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.order = append(r.order, "start")
	r.starts = append(r.starts, e)
}

func (r *recordingStatsReporter) OnFirstByte(_ context.Context, e FirstByteEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.order = append(r.order, "first_byte")
	r.firstBytes = append(r.firstBytes, e)
}

func (r *recordingStatsReporter) OnStreamChunk(_ context.Context, e StreamChunkEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.order = append(r.order, "chunk")
	r.chunks = append(r.chunks, e)
	if r.onChunk != nil {
		r.onChunk(e)
	}
}

func runStreamingRequestWithOpenAIChatUpstream(t *testing.T, events []*StreamEvent) (*httptest.ResponseRecorder, *recordingStatsReporter) {
	t.Helper()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		time.Sleep(2 * time.Millisecond)
		for _, event := range events {
			data, err := EncodeOpenAIChatStreamChunk(event)
			if err != nil {
				t.Fatalf("encode stream chunk: %v", err)
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			if flusher != nil {
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stream hi"}],"stream":true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)
	return w, reporter
}

func (r *recordingStatsReporter) OnComplete(_ context.Context, e CompleteEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.order = append(r.order, "complete")
	r.completions = append(r.completions, e)
	if r.onComplete != nil {
		r.onComplete(e)
	}
}

func (r *recordingStatsReporter) OnAttemptError(_ context.Context, e AttemptErrorEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.order = append(r.order, "attempt_error")
	r.attemptErrors = append(r.attemptErrors, e)
}

func TestHandler_AttemptControllerLimitsStreamingUntilStreamEnds(t *testing.T) {
	var (
		mu            sync.Mutex
		activeStreams int
		maxActive     int
		acquires      int
		releases      int
	)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		activeStreams++
		if activeStreams > maxActive {
			maxActive = activeStreams
		}
		mu.Unlock()
		defer func() {
			mu.Lock()
			activeStreams--
			mu.Unlock()
		}()

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		time.Sleep(50 * time.Millisecond)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	gate := make(chan struct{}, 1)
	controller := &testAttemptController{
		acquire: func(ctx context.Context, info RouteInfo, target RouteResult, routeAttempt int, retryAttempt int) (AttemptAdmission, error) {
			select {
			case gate <- struct{}{}:
				mu.Lock()
				acquires++
				mu.Unlock()
				return AttemptAdmission{
					Permit: testPermit{release: func() {
						<-gate
						mu.Lock()
						releases++
						mu.Unlock()
					}},
				}, nil
			case <-ctx.Done():
				return AttemptAdmission{}, ctx.Err()
			}
		},
	}

	h := &Handler{
		codec:             &openaiChatCodec{},
		router:            &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		attemptController: controller,
	}

	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}`
			r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != http.StatusOK {
				t.Errorf("status = %d, body = %s", w.Code, w.Body.String())
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if maxActive != 1 {
		t.Fatalf("max active upstream streams = %d, want 1", maxActive)
	}
	if acquires != 2 || releases != 2 {
		t.Fatalf("acquires/releases = %d/%d, want 2/2", acquires, releases)
	}
}

func TestHandler_RetriesStreamingSetupErrorBeforeFallback(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"service unavailable"}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	controller := &testAttemptController{
		retry: func(ctx context.Context, info RouteInfo, target RouteResult, sendErr SendError, routeAttempt int, retryAttempt int) (time.Duration, bool) {
			if sendErr.StatusCode == http.StatusServiceUnavailable && retryAttempt == 0 {
				return time.Millisecond, true
			}
			return 0, false
		},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:             &openaiChatCodec{},
		router:            &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:             reporter,
		attemptController: controller,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if callCount != 2 {
		t.Fatalf("upstream calls = %d, want 2", callCount)
	}
	if len(reporter.attemptErrors) != 1 {
		t.Fatalf("attempt errors = %d, want 1", len(reporter.attemptErrors))
	}
	if !reporter.attemptErrors[0].WillRetry || reporter.attemptErrors[0].RetryAttempt != 0 {
		t.Fatalf("attempt error retry fields = willRetry:%v retryAttempt:%d, want true/0", reporter.attemptErrors[0].WillRetry, reporter.attemptErrors[0].RetryAttempt)
	}
	if got := reporter.completions[0].RetryAttempts; got != 1 {
		t.Fatalf("complete RetryAttempts = %d, want 1", got)
	}
}

func TestHandler_DoesNotRetryAfterStreamingStarts(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {not-json}\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	controller := &testAttemptController{
		retry: func(ctx context.Context, info RouteInfo, target RouteResult, sendErr SendError, routeAttempt int, retryAttempt int) (time.Duration, bool) {
			return time.Millisecond, true
		},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:             &openaiChatCodec{},
		router:            &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:             reporter,
		attemptController: controller,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if callCount != 1 {
		t.Fatalf("upstream calls = %d, want 1", callCount)
	}
	if len(reporter.completions) != 1 || reporter.completions[0].Status != CompletionStatusError {
		t.Fatalf("completion = %+v, want one error completion", reporter.completions)
	}
	if reporter.completions[0].RetryAttempts != 0 {
		t.Fatalf("RetryAttempts = %d, want 0", reporter.completions[0].RetryAttempts)
	}
}

func TestHandler_OnRequestStart_TracksOriginalAndMappedModel(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
	}))
	defer upstream.Close()

	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if len(reporter.starts) != 1 {
		t.Fatalf("start events = %d, want 1", len(reporter.starts))
	}
	start := reporter.starts[0]
	if start.IRRequest == nil {
		t.Fatal("start.IRRequest is nil")
	}
	if start.IRRequest.OriginalModel != "gpt-4o" {
		t.Fatalf("OriginalModel = %q, want %q", start.IRRequest.OriginalModel, "gpt-4o")
	}
	if start.IRRequest.Model != "gpt-4o-mini" {
		t.Fatalf("Model = %q, want %q", start.IRRequest.Model, "gpt-4o-mini")
	}
}

func TestHandler_NonStreaming_EmitsStatsLifecycle(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	}))
	defer upstream.Close()

	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if got, want := reporter.order, []string{"start", "first_byte", "complete"}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("order = %v, want %v", got, want)
	}
	if len(reporter.starts) != 1 || len(reporter.firstBytes) != 1 || len(reporter.completions) != 1 {
		t.Fatalf("events count start/first/complete = %d/%d/%d, want 1/1/1", len(reporter.starts), len(reporter.firstBytes), len(reporter.completions))
	}

	start := reporter.starts[0]
	first := reporter.firstBytes[0]
	complete := reporter.completions[0]

	if start.RequestID == "" {
		t.Fatal("start.RequestID is empty")
	}
	if first.RequestID != start.RequestID || complete.RequestID != start.RequestID {
		t.Fatalf("request IDs mismatch: start=%q first=%q complete=%q", start.RequestID, first.RequestID, complete.RequestID)
	}
	if start.InboundProtocol != ProtocolOpenAIChat {
		t.Fatalf("start inbound protocol = %q, want %q", start.InboundProtocol, ProtocolOpenAIChat)
	}
	if start.OutboundProtocol != ProtocolOpenAIChat {
		t.Fatalf("start outbound protocol = %q, want %q", start.OutboundProtocol, ProtocolOpenAIChat)
	}
	if complete.InboundProtocol != ProtocolOpenAIChat || complete.OutboundProtocol != ProtocolOpenAIChat {
		t.Fatalf("complete protocols inbound/outbound = %q/%q, want %q/%q", complete.InboundProtocol, complete.OutboundProtocol, ProtocolOpenAIChat, ProtocolOpenAIChat)
	}
	if complete.IRResponse == nil {
		t.Fatal("complete.IRResponse is nil")
	}
	if complete.Usage.TotalTokens != 8 || complete.Usage.PromptTokens != 5 || complete.Usage.CompletionTokens != 3 {
		t.Fatalf("usage = %+v, want input=5 output=3 total=8", complete.Usage)
	}
	if complete.StopReason != StopReasonEndTurn {
		t.Fatalf("stop reason = %q, want %q", complete.StopReason, StopReasonEndTurn)
	}
	if complete.ActualModel != "gpt-4o-mini" {
		t.Fatalf("actual model = %q, want %q", complete.ActualModel, "gpt-4o-mini")
	}
	if first.TTFB <= 0 {
		t.Fatalf("TTFB = %v, want > 0", first.TTFB)
	}
	if complete.TTFB <= 0 {
		t.Fatalf("complete.TTFB = %v, want > 0", complete.TTFB)
	}
	if complete.TotalLatency < complete.TTFB {
		t.Fatalf("TotalLatency = %v, want >= TTFB %v", complete.TotalLatency, complete.TTFB)
	}
	if complete.OutputThroughput <= 0 {
		t.Fatalf("OutputThroughput = %v, want > 0", complete.OutputThroughput)
	}
	if complete.Status != CompletionStatusSuccess {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusSuccess)
	}
	if complete.Error != nil {
		t.Fatalf("error = %v, want nil", complete.Error)
	}
	if first.Time.Before(start.Time) {
		t.Fatalf("first byte time %v before start time %v", first.Time, start.Time)
	}
	if complete.Time.Before(first.Time) {
		t.Fatalf("complete time %v before first byte time %v", complete.Time, first.Time)
	}
	if complete.Time.Sub(start.Time) < 0 {
		t.Fatalf("complete occurs before start")
	}
	if complete.TotalLatency <= 0 {
		t.Fatalf("total latency = %v, want > 0", complete.TotalLatency)
	}
	if complete.TotalLatency > 0 && complete.Usage.CompletionTokens > 0 {
		expectedMin := float64(complete.Usage.CompletionTokens) / complete.TotalLatency.Seconds()
		if complete.OutputThroughput <= 0 || complete.OutputThroughput > expectedMin*1.5+1 {
			t.Fatalf("output throughput = %v looks invalid, expected roughly %v", complete.OutputThroughput, expectedMin)
		}
	}
	if time.Until(complete.Time) > time.Second {
		t.Fatalf("complete time %v is unexpectedly in the future", complete.Time)
	}
}

func TestHandler_UnsupportedOutboundProtocol_EmitsCompleteAfterStart(t *testing.T) {
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec: &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{
			Protocol: Protocol("unsupported-protocol"),
			Model:    "routed-model",
		}},
		stats: reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 502 {
		t.Fatalf("status = %d, want 502", w.Code)
	}
	if got, want := reporter.order, []string{"start", "attempt_error", "complete"}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("order = %v, want %v", got, want)
	}
	if len(reporter.starts) != 1 || len(reporter.firstBytes) != 0 || len(reporter.completions) != 1 || len(reporter.attemptErrors) != 1 {
		t.Fatalf("events count start/first/complete/attempt_error = %d/%d/%d/%d, want 1/0/1/1", len(reporter.starts), len(reporter.firstBytes), len(reporter.completions), len(reporter.attemptErrors))
	}

	start := reporter.starts[0]
	complete := reporter.completions[0]

	if start.RequestID == "" {
		t.Fatal("start.RequestID is empty")
	}
	if complete.RequestID != start.RequestID {
		t.Fatalf("request IDs mismatch: start=%q complete=%q", start.RequestID, complete.RequestID)
	}
	if complete.Status != CompletionStatusError {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusError)
	}
	if complete.Error == nil || complete.Error.Error() != "unsupported outbound protocol" {
		t.Fatalf("error = %v, want unsupported outbound protocol", complete.Error)
	}
	if complete.IRResponse != nil {
		t.Fatal("complete.IRResponse should be nil")
	}
	if complete.InboundProtocol != ProtocolOpenAIChat {
		t.Fatalf("inbound protocol = %q, want %q", complete.InboundProtocol, ProtocolOpenAIChat)
	}
	if complete.OutboundProtocol != Protocol("unsupported-protocol") {
		t.Fatalf("outbound protocol = %q, want %q", complete.OutboundProtocol, Protocol("unsupported-protocol"))
	}
	if complete.Time.Before(start.Time) {
		t.Fatalf("complete time %v before start time %v", complete.Time, start.Time)
	}
}

func TestHandler_Streaming_EmitsStatsLifecycle(t *testing.T) {
	stopReason := StopReasonEndTurn
	events := []*StreamEvent{
		{Type: StreamEventStart, Response: &Response{ID: "chatcmpl-1", Model: "gpt-4o-mini"}},
		{Type: StreamEventDelta, Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: "Hello"}}},
		{Type: StreamEventDelta, Usage: &Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}},
		{Type: StreamEventStop, StopReason: &stopReason, Usage: &Usage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8}},
	}

	w, reporter := runStreamingRequestWithOpenAIChatUpstream(t, events)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if len(reporter.starts) != 1 {
		t.Fatalf("start events = %d, want 1", len(reporter.starts))
	}
	if len(reporter.firstBytes) != 1 {
		t.Fatalf("first byte events = %d, want 1", len(reporter.firstBytes))
	}
	if len(reporter.chunks) != len(events) {
		t.Fatalf("chunk events = %d, want %d", len(reporter.chunks), len(events))
	}
	if len(reporter.completions) != 1 {
		t.Fatalf("complete events = %d, want 1", len(reporter.completions))
	}

	if got, want := reporter.order, []string{"start", "first_byte", "chunk", "chunk", "chunk", "chunk", "complete"}; len(got) != len(want) {
		t.Fatalf("order len = %d, want %d, order=%v", len(got), len(want), got)
	} else {
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("order[%d] = %q, want %q (full=%v)", i, got[i], want[i], got)
			}
		}
	}

	start := reporter.starts[0]
	first := reporter.firstBytes[0]
	complete := reporter.completions[0]

	if start.RequestID == "" {
		t.Fatal("start.RequestID is empty")
	}
	if first.RequestID != start.RequestID || complete.RequestID != start.RequestID {
		t.Fatalf("request IDs mismatch: start=%q first=%q complete=%q", start.RequestID, first.RequestID, complete.RequestID)
	}
	for i, chunk := range reporter.chunks {
		if chunk.RequestID != start.RequestID {
			t.Fatalf("chunk[%d] requestID=%q, want %q", i, chunk.RequestID, start.RequestID)
		}
		if chunk.SequenceNum != i+1 {
			t.Fatalf("chunk[%d] sequence=%d, want %d", i, chunk.SequenceNum, i+1)
		}
		if chunk.IREvent == nil {
			t.Fatalf("chunk[%d] IREvent is nil", i)
		}
		if i == 0 && chunk.InterChunkDelay != 0 {
			t.Fatalf("chunk[0] InterChunkDelay = %v, want 0", chunk.InterChunkDelay)
		}
		if i > 0 && chunk.InterChunkDelay <= 0 {
			t.Fatalf("chunk[%d] InterChunkDelay = %v, want > 0", i, chunk.InterChunkDelay)
		}
	}

	if complete.IRResponse != nil {
		t.Fatal("complete.IRResponse should be nil for streaming")
	}
	if complete.Usage.PromptTokens != 5 || complete.Usage.CompletionTokens != 3 || complete.Usage.TotalTokens != 8 {
		t.Fatalf("complete usage = %+v, want input=5 output=3 total=8", complete.Usage)
	}
	if complete.StopReason != StopReasonEndTurn {
		t.Fatalf("complete stop reason = %q, want %q", complete.StopReason, StopReasonEndTurn)
	}
	if complete.ActualModel != "gpt-4o-mini" {
		t.Fatalf("complete actual model = %q, want %q", complete.ActualModel, "gpt-4o-mini")
	}
	if complete.Status != CompletionStatusSuccess {
		t.Fatalf("complete status = %q, want %q", complete.Status, CompletionStatusSuccess)
	}
	if complete.TTFB <= 0 {
		t.Fatalf("complete TTFB = %v, want > 0", complete.TTFB)
	}
	if first.TTFB <= 0 {
		t.Fatalf("first TTFB = %v, want > 0", first.TTFB)
	}
	if complete.TotalLatency < complete.TTFB {
		t.Fatalf("total latency = %v, want >= TTFB %v", complete.TotalLatency, complete.TTFB)
	}
	if complete.OutputThroughput < 0 {
		t.Fatalf("output throughput = %v, want >= 0", complete.OutputThroughput)
	}
	if first.Time.Before(start.Time) {
		t.Fatalf("first time %v before start time %v", first.Time, start.Time)
	}
	if complete.Time.Before(first.Time) {
		t.Fatalf("complete time %v before first-byte time %v", complete.Time, first.Time)
	}
}

type earlyExitStreamingCodec struct {
	inner *openaiChatCodec
}

func (c *earlyExitStreamingCodec) Protocol() Protocol {
	return c.inner.Protocol()
}

func (c *earlyExitStreamingCodec) KnownFields() map[string]bool {
	return c.inner.KnownFields()
}

func (c *earlyExitStreamingCodec) ExtractAPIKey(r *http.Request) string {
	return c.inner.ExtractAPIKey(r)
}

func (c *earlyExitStreamingCodec) DecodeRequest(r *http.Request, body []byte) (*Request, error) {
	return c.inner.DecodeRequest(r, body)
}

func (c *earlyExitStreamingCodec) WriteError(w http.ResponseWriter, statusCode int, msg string) {
	c.inner.WriteError(w, statusCode, msg)
}

func (c *earlyExitStreamingCodec) EncodeResponse(resp *Response) ([]byte, error) {
	return c.inner.EncodeResponse(resp)
}

func (c *earlyExitStreamingCodec) WriteStreamingResponse(_ *SSEWriter, ch <-chan StreamResult) {
	<-ch
}

func TestHandler_Streaming_StreamError_DoesNotEmitChunk(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"broken\"\n\n")
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stream hi"}],"stream":true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if len(reporter.starts) != 1 {
		t.Fatalf("start events = %d, want 1", len(reporter.starts))
	}
	if len(reporter.firstBytes) != 0 {
		t.Fatalf("first byte events = %d, want 0", len(reporter.firstBytes))
	}
	if len(reporter.chunks) != 0 {
		t.Fatalf("chunk events = %d, want 0", len(reporter.chunks))
	}
	if len(reporter.completions) != 1 {
		t.Fatalf("complete events = %d, want 1", len(reporter.completions))
	}
	if reporter.completions[0].Status != CompletionStatusError {
		t.Fatalf("status = %q, want %q", reporter.completions[0].Status, CompletionStatusError)
	}
}

func TestHandler_Streaming_DownstreamEarlyExit_DoesNotDeadlock(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := 0; ; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			event := &StreamEvent{Type: StreamEventDelta, Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: fmt.Sprintf("chunk-%d", i)}}}
			data, err := EncodeOpenAIChatStreamChunk(event)
			if err != nil {
				return
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstream.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reporter := &recordingStatsReporter{}
	reporter.onComplete = func(CompleteEvent) { cancel() }
	h := &Handler{
		codec:  &earlyExitStreamingCodec{inner: &openaiChatCodec{}},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stream hi"}],"stream":true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)).WithContext(ctx)
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		h.ServeHTTP(w, req)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handler did not return; possible deadlock")
	}

	if len(reporter.completions) != 1 {
		t.Fatalf("complete events = %d, want 1", len(reporter.completions))
	}
	before := len(reporter.order)
	time.Sleep(20 * time.Millisecond)
	after := len(reporter.order)
	if before != after {
		t.Fatalf("events changed after complete: before=%d after=%d", before, after)
	}
	if reporter.order[len(reporter.order)-1] != "complete" {
		t.Fatalf("last event = %q, want complete", reporter.order[len(reporter.order)-1])
	}
}

func TestHandler_Streaming_CancellationWins(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := 0; ; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			event := &StreamEvent{Type: StreamEventDelta, Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: fmt.Sprintf("chunk-%d", i)}}}
			data, err := EncodeOpenAIChatStreamChunk(event)
			if err != nil {
				return
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(5 * time.Millisecond)
		}
	}))
	defer upstream.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reporter := &recordingStatsReporter{}
	reporter.onChunk = func(StreamChunkEvent) { cancel() }

	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"stream hi"}],"stream":true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)).WithContext(ctx)
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if len(reporter.completions) != 1 {
		t.Fatalf("complete events = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.Status != CompletionStatusCanceled {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusCanceled)
	}
	if !errors.Is(complete.Error, context.Canceled) && !errors.Is(complete.Error, context.DeadlineExceeded) {
		t.Fatalf("complete error = %v, want canceled/deadline exceeded", complete.Error)
	}
	before := len(reporter.order)
	time.Sleep(20 * time.Millisecond)
	after := len(reporter.order)
	if before != after {
		t.Fatalf("events changed after complete: before=%d after=%d", before, after)
	}
	if reporter.order[len(reporter.order)-1] != "complete" {
		t.Fatalf("last event = %q, want complete", reporter.order[len(reporter.order)-1])
	}
}

type trackingRouter struct {
	called bool
}

func (r *trackingRouter) Route(_ context.Context, _ RouteInfo) (RouteResult, error) {
	r.called = true
	return RouteResult{}, nil
}

func (r *trackingRouter) OnError(_ context.Context, _ RouteInfo, _ RouteResult, sendErr SendError) (RouteResult, error) {
	return RouteResult{}, sendErr.Err
}

func (r *trackingRouter) OnSuccess(_ context.Context, _ RouteInfo, _ RouteResult) {}

type trackingCodec struct {
	decodeCalled bool
}

func (c *trackingCodec) Protocol() Protocol { return ProtocolOpenAIChat }

func (c *trackingCodec) KnownFields() map[string]bool { return nil }

func (c *trackingCodec) ExtractAPIKey(_ *http.Request) string { return "test-key" }

func (c *trackingCodec) DecodeRequest(_ *http.Request, _ []byte) (*Request, error) {
	c.decodeCalled = true
	return &Request{}, nil
}

func (c *trackingCodec) WriteError(w http.ResponseWriter, statusCode int, msg string) {
	http.Error(w, msg, statusCode)
}

func (c *trackingCodec) EncodeResponse(_ *Response) ([]byte, error) {
	return nil, fmt.Errorf("unexpected call")
}

func (c *trackingCodec) WriteStreamingResponse(_ *SSEWriter, _ <-chan StreamResult) {
	panic("unexpected call")
}

func TestHandler_NonStreaming_UpstreamError_EmitsCompleteWithError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"internal server error"}`))
	}))
	defer upstream.Close()

	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 500 {
		t.Fatalf("status = %d, want 500", w.Code)
	}
	if got, want := reporter.order, []string{"start", "attempt_error", "complete"}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("order = %v, want %v", got, want)
	}
	if len(reporter.firstBytes) != 0 {
		t.Fatalf("first byte events = %d, want 0", len(reporter.firstBytes))
	}
	complete := reporter.completions[0]
	if complete.Status != CompletionStatusError {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusError)
	}
	if complete.Error == nil {
		t.Fatal("complete.Error should be non-nil")
	}
	if complete.IRResponse != nil {
		t.Fatal("complete.IRResponse should be nil on error")
	}
	if complete.TotalLatency <= 0 {
		t.Fatalf("TotalLatency = %v, want > 0", complete.TotalLatency)
	}
}

func TestHandler_Streaming_SendStreamError_EmitsCompleteWithError(t *testing.T) {
	// Upstream that closes immediately without SSE headers, causing SendStream to fail.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":"service unavailable"}`))
	}))
	defer upstream.Close()

	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"}},
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if len(reporter.starts) != 1 {
		t.Fatalf("start events = %d, want 1", len(reporter.starts))
	}
	if len(reporter.completions) != 1 {
		t.Fatalf("complete events = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.Status != CompletionStatusError {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusError)
	}
	if complete.Error == nil {
		t.Fatal("complete.Error should be non-nil")
	}
	if complete.RequestID != reporter.starts[0].RequestID {
		t.Fatalf("request ID mismatch: start=%q complete=%q", reporter.starts[0].RequestID, complete.RequestID)
	}
}

func TestNoopStatsReporter_NoPanic(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
	}))
	defer upstream.Close()

	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o"}},
		// stats is nil — will use NoopStatsReporter
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

func TestHandler_NonStreaming_CrossProtocol(t *testing.T) {
	// Fake OpenAI Chat upstream.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	}))
	defer upstream.Close()

	h := &Handler{
		codec:  &anthropicCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o"}},
	}

	body := `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}`
	r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp["type"] != "message" {
		t.Errorf("type = %v, want message", resp["type"])
	}
	if resp["model"] != "gpt-4o" {
		t.Errorf("model = %v, want gpt-4o", resp["model"])
	}
}

func TestHandler_AuthReject_BeforeDecodeAndRoute(t *testing.T) {
	codec := &trackingCodec{}
	router := &trackingRouter{}
	h := &Handler{
		codec:  codec,
		router: router,
		auth:   &rejectAuth{},
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 401 {
		t.Fatalf("status = %d, want 401", w.Code)
	}
	if codec.decodeCalled {
		t.Fatal("DecodeRequest should not be called when auth fails")
	}
	if router.called {
		t.Fatal("Route should not be called when auth fails")
	}
}

func TestHandler_RouteError(t *testing.T) {
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{err: fmt.Errorf("unknown model")},
	}

	body := `{"model":"unknown","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 400 {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestHandler_RawExtra_SameProtocol_Preserved(t *testing.T) {
	var upstreamReq map[string]json.RawMessage
	// Fake OpenAI Chat upstream.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&upstreamReq); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
	}))
	defer upstream.Close()

	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o"}},
	}

	// Request includes protocol-specific fields service_tier and seed.
	// These are now known fields in ChatRequest (promoted from RawExtra), so they
	// are decoded into the struct but have no IR mapping — they are not forwarded
	// upstream and are not round-tripped to the response.
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"service_tier":"priority","seed":42}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	// service_tier and seed are known struct fields with no IR mapping, so they
	// are not forwarded upstream (EncodeOpenAIChatRequest only sets IR-mapped fields).
	if _, ok := upstreamReq["service_tier"]; ok {
		t.Fatal("service_tier should not be forwarded by IR encode")
	}
	if _, ok := upstreamReq["seed"]; ok {
		t.Fatal("seed should not be forwarded by IR encode")
	}

	// Since service_tier and seed are now known fields, they are no longer captured
	// in RawExtra and are not merged back into the response.
	var resp map[string]json.RawMessage
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode downstream response: %v", err)
	}
	if _, ok := resp["service_tier"]; ok {
		t.Fatal("service_tier should not appear in response (now a known field, not in RawExtra)")
	}
	if _, ok := resp["seed"]; ok {
		t.Fatal("seed should not appear in response (now a known field, not in RawExtra)")
	}
}

// fallbackRouter returns a primary target from Route(), and on OnError returns
// the fallback target. On a second OnError it gives up.
type fallbackRouter struct {
	mu             sync.Mutex
	primary        RouteResult
	fallback       RouteResult
	onErrorCalls   int
	onSuccessCalls int
	successTarget  RouteResult
}

func (r *fallbackRouter) Route(_ context.Context, _ RouteInfo) (RouteResult, error) {
	return r.primary, nil
}

func (r *fallbackRouter) OnError(_ context.Context, _ RouteInfo, _ RouteResult, sendErr SendError) (RouteResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.onErrorCalls++
	if r.onErrorCalls == 1 {
		return r.fallback, nil
	}
	return RouteResult{}, sendErr.Err
}

func (r *fallbackRouter) OnSuccess(_ context.Context, _ RouteInfo, target RouteResult) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.onSuccessCalls++
	r.successTarget = target
}

func TestHandler_NonStreaming_SuccessWithoutFallback(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	}))
	defer upstream.Close()

	router := &fallbackRouter{
		primary:  RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"},
		fallback: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test2", Model: "gpt-4o"},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: router,
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if len(reporter.attemptErrors) != 0 {
		t.Fatalf("attempt errors = %d, want 0", len(reporter.attemptErrors))
	}
	if len(reporter.completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.AttemptNum != 1 {
		t.Fatalf("AttemptNum = %d, want 1", complete.AttemptNum)
	}
	if complete.Status != CompletionStatusSuccess {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusSuccess)
	}
	router.mu.Lock()
	defer router.mu.Unlock()
	if router.onSuccessCalls != 1 {
		t.Fatalf("OnSuccess calls = %d, want 1", router.onSuccessCalls)
	}
	if !reflect.DeepEqual(router.successTarget, router.primary) {
		t.Fatalf("OnSuccess called with wrong target")
	}
}

func TestHandler_NonStreaming_FallbackOnError(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			// First call fails.
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error":"internal server error"}`))
			return
		}
		// Second call succeeds.
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Fallback!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	}))
	defer upstream.Close()

	router := &fallbackRouter{
		primary:  RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"},
		fallback: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test2", Model: "gpt-4o"},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: router,
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	// Verify attempt error was fired.
	if len(reporter.attemptErrors) != 1 {
		t.Fatalf("attempt errors = %d, want 1", len(reporter.attemptErrors))
	}
	ae := reporter.attemptErrors[0]
	if ae.AttemptNum != 1 {
		t.Fatalf("attempt error AttemptNum = %d, want 1", ae.AttemptNum)
	}
	if !reflect.DeepEqual(ae.Target, router.primary) {
		t.Fatalf("attempt error target mismatch")
	}
	if ae.SendErr.StatusCode != 500 {
		t.Fatalf("attempt error StatusCode = %d, want 500", ae.SendErr.StatusCode)
	}

	// Verify completion.
	if len(reporter.completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.AttemptNum != 2 {
		t.Fatalf("AttemptNum = %d, want 2", complete.AttemptNum)
	}
	if complete.Status != CompletionStatusSuccess {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusSuccess)
	}
	if complete.OutboundProtocol != ProtocolOpenAIChat {
		t.Fatalf("outbound protocol = %q, want %q", complete.OutboundProtocol, ProtocolOpenAIChat)
	}

	// Verify OnSuccess was called with fallback target.
	router.mu.Lock()
	defer router.mu.Unlock()
	if router.onSuccessCalls != 1 {
		t.Fatalf("OnSuccess calls = %d, want 1", router.onSuccessCalls)
	}
	if !reflect.DeepEqual(router.successTarget, router.fallback) {
		t.Fatalf("OnSuccess called with wrong target, got %+v want %+v", router.successTarget, router.fallback)
	}
}

func TestHandler_NonStreaming_RetryExhaustedBeforeFallback(t *testing.T) {
	primaryCalls := 0
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":"service unavailable"}`))
	}))
	defer primary.Close()

	fallbackCalls := 0
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalls++
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-2","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Fallback!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	}))
	defer fallback.Close()

	controller := &testAttemptController{
		retry: func(ctx context.Context, info RouteInfo, target RouteResult, sendErr SendError, routeAttempt int, retryAttempt int) (time.Duration, bool) {
			if routeAttempt == 1 && retryAttempt == 0 && sendErr.StatusCode == http.StatusServiceUnavailable {
				return time.Millisecond, true
			}
			return 0, false
		},
	}
	router := &fallbackRouter{
		primary:  RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: primary.URL, APIKey: "sk-test", Model: "gpt-4o-mini"},
		fallback: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: fallback.URL, APIKey: "sk-test2", Model: "gpt-4o"},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:             &openaiChatCodec{},
		router:            router,
		stats:             reporter,
		attemptController: controller,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if primaryCalls != 2 {
		t.Fatalf("primary calls = %d, want 2", primaryCalls)
	}
	if fallbackCalls != 1 {
		t.Fatalf("fallback calls = %d, want 1", fallbackCalls)
	}
	if len(reporter.attemptErrors) != 2 {
		t.Fatalf("attempt errors = %d, want 2", len(reporter.attemptErrors))
	}
	if !reporter.attemptErrors[0].WillRetry || reporter.attemptErrors[0].RetryAttempt != 0 {
		t.Fatalf("first attempt error = %+v, want willRetry retryAttempt=0", reporter.attemptErrors[0])
	}
	if reporter.attemptErrors[1].WillRetry || reporter.attemptErrors[1].RetryAttempt != 1 {
		t.Fatalf("second attempt error = %+v, want terminal retryAttempt=1", reporter.attemptErrors[1])
	}
	if len(reporter.completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.AttemptNum != 2 {
		t.Fatalf("AttemptNum = %d, want 2", complete.AttemptNum)
	}
	if complete.RetryAttempts != 1 {
		t.Fatalf("RetryAttempts = %d, want 1", complete.RetryAttempts)
	}
}

func TestHandler_NonStreaming_FallbackGivesUp(t *testing.T) {
	// Both primary and fallback fail.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"internal server error"}`))
	}))
	defer upstream.Close()

	router := &fallbackRouter{
		primary:  RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"},
		fallback: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test2", Model: "gpt-4o"},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: router,
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 500 {
		t.Fatalf("status = %d, want 500", w.Code)
	}
	// Two attempt errors: primary and fallback.
	if len(reporter.attemptErrors) != 2 {
		t.Fatalf("attempt errors = %d, want 2", len(reporter.attemptErrors))
	}
	if reporter.attemptErrors[0].AttemptNum != 1 {
		t.Fatalf("first attempt error AttemptNum = %d, want 1", reporter.attemptErrors[0].AttemptNum)
	}
	if reporter.attemptErrors[1].AttemptNum != 2 {
		t.Fatalf("second attempt error AttemptNum = %d, want 2", reporter.attemptErrors[1].AttemptNum)
	}

	// Complete event should be error.
	if len(reporter.completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.Status != CompletionStatusError {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusError)
	}
	if complete.AttemptNum != 2 {
		t.Fatalf("AttemptNum = %d, want 2", complete.AttemptNum)
	}

	// OnSuccess should not be called.
	router.mu.Lock()
	defer router.mu.Unlock()
	if router.onSuccessCalls != 0 {
		t.Fatalf("OnSuccess calls = %d, want 0", router.onSuccessCalls)
	}
}

func TestHandler_NonStreaming_ContextCancellation_NoRetry(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		// Cancel context on first call.
		cancel()
		// Return error.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"internal server error"}`))
	}))
	defer upstream.Close()

	router := &fallbackRouter{
		primary:  RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"},
		fallback: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test2", Model: "gpt-4o"},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: router,
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)).WithContext(ctx)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	// Should not retry — only 1 call to upstream.
	if callCount != 1 {
		t.Fatalf("upstream calls = %d, want 1", callCount)
	}
	// No attempt errors fired (context cancellation skips retry).
	if len(reporter.attemptErrors) != 0 {
		t.Fatalf("attempt errors = %d, want 0", len(reporter.attemptErrors))
	}
	if len(reporter.completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.AttemptNum != 1 {
		t.Fatalf("AttemptNum = %d, want 1", complete.AttemptNum)
	}
}

func TestHandler_NonStreaming_CrossProtocolFallback(t *testing.T) {
	// Primary: Anthropic (fails), Fallback: OpenAI Chat (succeeds).
	anthropicServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"internal error"}}`))
	}))
	defer anthropicServer.Close()

	openaiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Fallback from OpenAI!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))
	}))
	defer openaiServer.Close()

	router := &fallbackRouter{
		primary:  RouteResult{Protocol: ProtocolAnthropic, BaseURL: anthropicServer.URL, APIKey: "sk-ant", Model: "claude-sonnet-4-20250514"},
		fallback: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: openaiServer.URL, APIKey: "sk-openai", Model: "gpt-4o"},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: router,
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	// Verify attempt error was fired for the Anthropic failure.
	if len(reporter.attemptErrors) != 1 {
		t.Fatalf("attempt errors = %d, want 1", len(reporter.attemptErrors))
	}
	ae := reporter.attemptErrors[0]
	if ae.Target.Protocol != ProtocolAnthropic {
		t.Fatalf("attempt error target protocol = %q, want %q", ae.Target.Protocol, ProtocolAnthropic)
	}

	// Verify completion reflects the final target.
	if len(reporter.completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.AttemptNum != 2 {
		t.Fatalf("AttemptNum = %d, want 2", complete.AttemptNum)
	}
	if complete.OutboundProtocol != ProtocolOpenAIChat {
		t.Fatalf("outbound protocol = %q, want %q", complete.OutboundProtocol, ProtocolOpenAIChat)
	}
	if complete.Status != CompletionStatusSuccess {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusSuccess)
	}

	// OnRequestStart should have primary's outbound protocol.
	if len(reporter.starts) != 1 {
		t.Fatalf("starts = %d, want 1", len(reporter.starts))
	}
	if reporter.starts[0].OutboundProtocol != ProtocolAnthropic {
		t.Fatalf("start outbound protocol = %q, want %q", reporter.starts[0].OutboundProtocol, ProtocolAnthropic)
	}
}

func TestHandler_Streaming_FallbackOnSendStreamError(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			// First call fails.
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte(`{"error":"service unavailable"}`))
			return
		}
		// Second call succeeds with streaming.
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	router := &fallbackRouter{
		primary:  RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o-mini"},
		fallback: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test2", Model: "gpt-4o"},
	}
	reporter := &recordingStatsReporter{}
	h := &Handler{
		codec:  &openaiChatCodec{},
		router: router,
		stats:  reporter,
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	// Verify attempt error was fired.
	if len(reporter.attemptErrors) != 1 {
		t.Fatalf("attempt errors = %d, want 1", len(reporter.attemptErrors))
	}
	ae := reporter.attemptErrors[0]
	if ae.AttemptNum != 1 {
		t.Fatalf("attempt error AttemptNum = %d, want 1", ae.AttemptNum)
	}

	// Verify completion.
	if len(reporter.completions) != 1 {
		t.Fatalf("completions = %d, want 1", len(reporter.completions))
	}
	complete := reporter.completions[0]
	if complete.AttemptNum != 2 {
		t.Fatalf("AttemptNum = %d, want 2", complete.AttemptNum)
	}
	if complete.Status != CompletionStatusSuccess {
		t.Fatalf("status = %q, want %q", complete.Status, CompletionStatusSuccess)
	}

	// Verify OnSuccess was called with fallback target.
	router.mu.Lock()
	defer router.mu.Unlock()
	if router.onSuccessCalls != 1 {
		t.Fatalf("OnSuccess calls = %d, want 1", router.onSuccessCalls)
	}
	if !reflect.DeepEqual(router.successTarget, router.fallback) {
		t.Fatalf("OnSuccess called with wrong target")
	}

	// Verify stream content is in response.
	if !strings.Contains(w.Body.String(), "Hello") {
		t.Fatalf("response body missing streamed content: %s", w.Body.String())
	}
}

func TestHandler_BuildSendError_UpstreamHTTPError(t *testing.T) {
	origErr := &UpstreamHTTPError{
		StatusCode: 429,
		Header:     http.Header{"Retry-After": []string{"30"}},
		Body:       []byte(`{"error":"rate limited"}`),
	}
	se := buildSendError(fmt.Errorf("send: %w", origErr), 1)
	if se.StatusCode != 429 {
		t.Fatalf("StatusCode = %d, want 429", se.StatusCode)
	}
	if se.Header.Get("Retry-After") != "30" {
		t.Fatalf("Header Retry-After = %q, want 30", se.Header.Get("Retry-After"))
	}
	if se.IsTimeout {
		t.Fatal("IsTimeout should be false")
	}
	if se.IsConnError {
		t.Fatal("IsConnError should be false")
	}
	if se.AttemptNum != 1 {
		t.Fatalf("AttemptNum = %d, want 1", se.AttemptNum)
	}
}

func TestHandler_BuildSendError_Timeout(t *testing.T) {
	se := buildSendError(fmt.Errorf("send: %w", context.DeadlineExceeded), 2)
	if !se.IsTimeout {
		t.Fatal("IsTimeout should be true")
	}
	if se.AttemptNum != 2 {
		t.Fatalf("AttemptNum = %d, want 2", se.AttemptNum)
	}
}

func TestHandler_RawExtra_CrossProtocol_Dropped(t *testing.T) {
	// Anthropic inbound -> OpenAI Chat outbound.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the outbound request does NOT contain Anthropic-specific fields.
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		if _, ok := body["inference_geo"]; ok {
			t.Error("Anthropic-specific field 'inference_geo' should not appear in OpenAI outbound")
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
	}))
	defer upstream.Close()

	h := &Handler{
		codec:  &anthropicCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o"}},
	}

	body := `{"model":"claude-sonnet-4-20250514","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}],"inference_geo":"us-east"}`
	r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()

	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
}

func TestHandler_OutboundExtra_Injected(t *testing.T) {
	var upstreamBody map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&upstreamBody)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
	}))
	defer upstream.Close()

	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o"}},
		reqMod: func(ctx context.Context, req *Request, target RouteResult) {
			req.OutboundExtra = map[string]json.RawMessage{
				"service_tier": json.RawMessage(`"priority"`),
			}
		},
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if string(upstreamBody["service_tier"]) != `"priority"` {
		t.Fatalf("service_tier = %s, want \"priority\"", string(upstreamBody["service_tier"]))
	}
}

func TestHandler_OutboundExtra_CrossProtocol(t *testing.T) {
	var upstreamBody map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&upstreamBody)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
	}))
	defer upstream.Close()

	h := &Handler{
		codec:  &anthropicCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o"}},
		reqMod: func(ctx context.Context, req *Request, target RouteResult) {
			if target.Protocol == ProtocolOpenAIChat {
				req.OutboundExtra = map[string]json.RawMessage{
					"service_tier": json.RawMessage(`"priority"`),
				}
			}
		},
	}

	body := `{"model":"claude-3","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	r.Header.Set("x-api-key", "sk-test")
	r.Header.Set("anthropic-version", "2023-06-01")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if string(upstreamBody["service_tier"]) != `"priority"` {
		t.Fatalf("service_tier = %s, want \"priority\"", string(upstreamBody["service_tier"]))
	}
}

func TestHandler_OutboundExtra_ResetOnFallback(t *testing.T) {
	var attempt1Body, attempt2Body map[string]json.RawMessage
	callCount := 0

	upstreamFail := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&attempt1Body)
		callCount++
		w.WriteHeader(500)
		w.Write([]byte(`{"error":"fail"}`))
	}))
	defer upstreamFail.Close()

	upstreamOK := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&attempt2Body)
		callCount++
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","content":[{"type":"text","text":"Hi"}],"model":"claude-3","stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":1}}`))
	}))
	defer upstreamOK.Close()

	router := &fallbackRouter{
		primary:  RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstreamFail.URL, APIKey: "sk-test", Model: "gpt-4o"},
		fallback: RouteResult{Protocol: ProtocolAnthropic, BaseURL: upstreamOK.URL, APIKey: "sk-test", Model: "claude-3"},
	}

	h := &Handler{
		codec:  &openaiChatCodec{},
		router: router,
		reqMod: func(ctx context.Context, req *Request, target RouteResult) {
			if target.Protocol == ProtocolOpenAIChat {
				req.OutboundExtra = map[string]json.RawMessage{
					"service_tier": json.RawMessage(`"priority"`),
				}
			}
		},
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if string(attempt1Body["service_tier"]) != `"priority"` {
		t.Fatalf("attempt 1 service_tier = %s, want \"priority\"", string(attempt1Body["service_tier"]))
	}
	if _, ok := attempt2Body["service_tier"]; ok {
		t.Fatal("attempt 2 should not have service_tier")
	}
}

func TestHandler_OutboundExtra_Streaming(t *testing.T) {
	var upstreamBody map[string]json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&upstreamBody)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}\n\ndata: [DONE]\n\n")
	}))
	defer upstream.Close()

	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o"}},
		reqMod: func(ctx context.Context, req *Request, target RouteResult) {
			req.OutboundExtra = map[string]json.RawMessage{
				"service_tier": json.RawMessage(`"priority"`),
			}
		},
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if string(upstreamBody["service_tier"]) != `"priority"` {
		t.Fatalf("service_tier = %s, want \"priority\"", string(upstreamBody["service_tier"]))
	}
}

func TestHandler_OutboundExtra_NilModifier(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`))
	}))
	defer upstream.Close()

	h := &Handler{
		codec:  &openaiChatCodec{},
		router: &staticRouter{result: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: upstream.URL, APIKey: "sk-test", Model: "gpt-4o"}},
	}

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != 200 {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
}
