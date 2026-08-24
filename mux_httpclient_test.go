package llmapimux

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// TestWithHTTPClient_InjectedTransportUsedWithProxy verifies that when a
// caller injects an *http.Transport and a RouteResult carries a ProxyURL,
// the derived proxy transport is a clone of the injected one — i.e. the
// injected dialer (and thus keepalive settings) is what actually dials.
func TestWithHTTPClient_InjectedTransportUsedWithProxy(t *testing.T) {
	var dialed atomic.Bool
	injected := &http.Transport{
		// Recording dialer: marks usage and refuses to connect (the proxy URL
		// is unreachable anyway; we only assert the dial path goes through
		// the injected dialer).
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			dialed.Store(true)
			return nil, errors.New("blocked in test")
		},
	}
	mux := NewMux(
		&staticRouter{result: RouteResult{
			Protocol: ProtocolOpenAIChat,
			BaseURL:  "http://upstream.invalid",
			APIKey:   "sk-openai",
			Model:    "gpt-4o",
			ProxyURL: "http://proxy.invalid:1",
		}},
		WithHTTPClient(&http.Client{Transport: injected}),
	)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.OpenAIChatHandler().ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		t.Fatalf("expected failure via unreachable proxy, got status %d", w.Code)
	}
	if !dialed.Load() {
		t.Fatal("outbound dial did not go through the injected transport's dialer — injected transport was ignored")
	}
	// The injected transport itself must not be mutated (Proxy still unset).
	if injected.Proxy != nil {
		t.Fatal("injected transport Proxy should not be mutated")
	}
}

// TestWithHTTPClient_DirectNoProxy verifies injection works without a ProxyURL.
func TestWithHTTPClient_DirectNoProxy(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	var used atomic.Bool
	injected := &http.Transport{
		DialContext: (&net.Dialer{}).DialContext,
	}
	injected.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		used.Store(true)
		return (&net.Dialer{}).DialContext(ctx, network, addr)
	}
	mux := NewMux(
		&staticRouter{result: RouteResult{
			Protocol: ProtocolOpenAIChat,
			BaseURL:  upstream.URL,
			APIKey:   "sk-openai",
			Model:    "gpt-4o",
		}},
		WithHTTPClient(&http.Client{Transport: injected}),
	)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.OpenAIChatHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if !used.Load() {
		t.Fatal("injected transport was not used for the outbound request")
	}
}
