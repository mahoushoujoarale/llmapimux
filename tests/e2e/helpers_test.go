package e2e_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	llmapimux "github.com/llmapimux/llmapimux"
)

type e2eCapturedRequest struct {
	Method string
	Path   string
	Header http.Header
	Body   []byte
}

func assertCapturedRequestBasics(t *testing.T, got *e2eCapturedRequest, method, path string) {
	t.Helper()
	if got == nil {
		t.Fatal("expected upstream request to be captured")
	}
	if got.Method != method {
		t.Fatalf("method = %s, want %s", got.Method, method)
	}
	if got.Path != path {
		t.Fatalf("path = %s, want %s", got.Path, path)
	}
}

func assertHeaderValue(t *testing.T, h http.Header, key, want string) {
	t.Helper()
	if got := h.Get(key); got != want {
		t.Fatalf("%s = %q, want %q", key, got, want)
	}
}

func assertHeaderEmpty(t *testing.T, h http.Header, key string) {
	t.Helper()
	if got := h.Get(key); got != "" {
		t.Fatalf("%s = %q, want empty", key, got)
	}
}

func decodeCapturedJSONBody(t *testing.T, got *e2eCapturedRequest) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(got.Body, &body); err != nil {
		t.Fatalf("unmarshal upstream body: %v", err)
	}
	return body
}

func newE2EContext(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()
	return context.WithTimeout(context.Background(), 5*time.Second)
}

func newLocalOnlyHTTPClient(t *testing.T, allowedBaseURLs ...string) *http.Client {
	t.Helper()

	allowedHosts := make(map[string]struct{}, len(allowedBaseURLs))
	for _, baseURL := range allowedBaseURLs {
		u, err := url.Parse(baseURL)
		if err != nil {
			t.Fatalf("parse allowed base url %q: %v", baseURL, err)
		}
		allowedHosts[u.Host] = struct{}{}
	}

	transport := &http.Transport{
		Proxy: nil,
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			if _, ok := allowedHosts[addr]; !ok {
				return nil, fmt.Errorf("e2e network guard blocked host %q", addr)
			}
			var d net.Dialer
			return d.DialContext(ctx, network, addr)
		},
	}

	return &http.Client{
		Transport: transport,
		Timeout:   5 * time.Second,
	}
}

func newE2EUpstreamServer(t *testing.T, handler func(http.ResponseWriter, *http.Request, *e2eCapturedRequest)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read upstream request body: %v", err)
		}
		captured := &e2eCapturedRequest{
			Method: r.Method,
			Path:   r.URL.Path,
			Header: r.Header.Clone(),
			Body:   bytes.Clone(body),
		}
		handler(w, r, captured)
	}))
}

func newE2EMuxServer(t *testing.T, inboundPath string, handler http.Handler) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle(inboundPath, handler)
	return httptest.NewServer(mux)
}

func TestE2EHelpers_MuxServerRoutesToUpstream(t *testing.T) {
	var got *e2eCapturedRequest
	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		got = captured
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	})
	defer upstream.Close()

	server := newE2EMuxServer(t,
		"/v1/chat/completions",
		newTestMux(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-test").OpenAIChatHandler(),
	)
	defer server.Close()

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if got == nil {
		t.Fatal("expected upstream request to be captured")
	}
	if got.Method != http.MethodPost {
		t.Fatalf("method = %s, want POST", got.Method)
	}
	if got.Path != "/v1/chat/completions" {
		t.Fatalf("path = %s, want /v1/chat/completions", got.Path)
	}
}

func TestE2E_NetworkGuardBlocksExternalHosts(t *testing.T) {
	client := newLocalOnlyHTTPClient(t, "http://127.0.0.1:1")

	_, err := client.Get("http://example.com")
	if err == nil {
		t.Fatal("expected external host to be blocked")
	}
	if !strings.Contains(err.Error(), "e2e network guard blocked host") {
		t.Fatalf("err = %v, want network guard block", err)
	}
}

// loadEnvFile parses a .env file and sets environment variables.
// Lines starting with # or empty lines are skipped.
func loadEnvFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		os.Setenv(strings.TrimSpace(key), strings.TrimSpace(value))
	}
	return scanner.Err()
}

// skipIfEnvMissing skips the test if any of the given env vars is empty.
func skipIfEnvMissing(t *testing.T, keys ...string) {
	t.Helper()
	for _, k := range keys {
		if os.Getenv(k) == "" {
			t.Skipf("skipping: env var %s not set", k)
		}
	}
}

// newRealAPIContext returns a 30-second timeout context for real API tests.
func newRealAPIContext(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()
	return context.WithTimeout(context.Background(), 30*time.Second)
}

// newRealMuxServer creates a mux test server with multiple handlers registered.
// paths is a map of path → http.Handler pairs.
func newRealMuxServer(t *testing.T, paths map[string]http.Handler) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	for p, h := range paths {
		mux.Handle(p, h)
	}
	return httptest.NewServer(mux)
}

// staticRouter is a simple Router that always returns a fixed RouteResult.
type staticRouter struct {
	result llmapimux.RouteResult
}

func (r *staticRouter) Route(_ context.Context, info llmapimux.RouteInfo) (llmapimux.RouteResult, error) {
	res := r.result
	// If model is not explicitly set in result, pass through the requested model
	if res.Model == "" {
		res.Model = info.Model
	}
	return res, nil
}

func (r *staticRouter) OnError(_ context.Context, _ llmapimux.RouteInfo, _ llmapimux.RouteResult, sendErr llmapimux.SendError) (llmapimux.RouteResult, error) {
	return llmapimux.RouteResult{}, sendErr.Err
}

func (r *staticRouter) OnSuccess(_ context.Context, _ llmapimux.RouteInfo, _ llmapimux.RouteResult) {}

// newTestMux creates a Mux routing to the given target (model pass-through).
func newTestMux(protocol llmapimux.Protocol, baseURL, apiKey string, opts ...llmapimux.MuxOption) *llmapimux.Mux {
	return llmapimux.NewMux(&staticRouter{
		result: llmapimux.RouteResult{
			Protocol: protocol,
			BaseURL:  baseURL,
			APIKey:   apiKey,
		},
	}, opts...)
}

// mappingRouter routes to a fixed target with model name translation.
type mappingRouter struct {
	protocol llmapimux.Protocol
	baseURL  string
	apiKey   string
	modelMap map[string]string
}

func (r *mappingRouter) Route(_ context.Context, info llmapimux.RouteInfo) (llmapimux.RouteResult, error) {
	model := info.Model
	if mapped, ok := r.modelMap[model]; ok {
		model = mapped
	}
	return llmapimux.RouteResult{Protocol: r.protocol, BaseURL: r.baseURL, APIKey: r.apiKey, Model: model}, nil
}

func (r *mappingRouter) OnError(_ context.Context, _ llmapimux.RouteInfo, _ llmapimux.RouteResult, sendErr llmapimux.SendError) (llmapimux.RouteResult, error) {
	return llmapimux.RouteResult{}, sendErr.Err
}

func (r *mappingRouter) OnSuccess(_ context.Context, _ llmapimux.RouteInfo, _ llmapimux.RouteResult) {}

// newTestMuxWithModelMap creates a Mux with model name translation.
func newTestMuxWithModelMap(protocol llmapimux.Protocol, baseURL, apiKey string, modelMap map[string]string, opts ...llmapimux.MuxOption) *llmapimux.Mux {
	return llmapimux.NewMux(&mappingRouter{
		protocol: protocol,
		baseURL:  baseURL,
		apiKey:   apiKey,
		modelMap: modelMap,
	}, opts...)
}

// --- Fallback test helpers ---

// failingFakeServer returns a httptest.Server that responds with the given status code.
func failingFakeServer(t *testing.T, protocol llmapimux.Protocol, statusCode int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		switch protocol {
		case llmapimux.ProtocolOpenAIChat:
			w.Write([]byte(fmt.Sprintf(`{"error":{"message":"upstream error","type":"server_error","code":"%d"}}`, statusCode)))
		case llmapimux.ProtocolAnthropic:
			w.Write([]byte(fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":"upstream error %d"}}`, statusCode)))
		case llmapimux.ProtocolGemini:
			w.Write([]byte(fmt.Sprintf(`{"error":{"code":%d,"message":"upstream error","status":"INTERNAL"}}`, statusCode)))
		default:
			w.Write([]byte(fmt.Sprintf(`{"error":{"message":"upstream error","code":%d}}`, statusCode)))
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// succeedingFakeServer returns a httptest.Server with a valid success response for the protocol.
func succeedingFakeServer(t *testing.T, protocol llmapimux.Protocol) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch protocol {
		case llmapimux.ProtocolOpenAIChat:
			w.Write([]byte(`{"id":"test","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"hello from fallback"},"finish_reason":"stop","index":0}],"model":"test-model","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		case llmapimux.ProtocolAnthropic:
			w.Write([]byte(`{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"hello from fallback"}],"model":"test-model","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
		case llmapimux.ProtocolGemini:
			w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"hello from fallback"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}`))
		default:
			t.Fatalf("unsupported protocol for succeedingFakeServer: %s", protocol)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// succeedingStreamFakeServer returns a httptest.Server that streams SSE events.
func succeedingStreamFakeServer(t *testing.T, protocol llmapimux.Protocol) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		switch protocol {
		case llmapimux.ProtocolOpenAIChat:
			fmt.Fprintf(w, "data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"index\":0}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"test\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		case llmapimux.ProtocolAnthropic:
			fmt.Fprintf(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"test-model\",\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			flusher.Flush()
		default:
			t.Fatalf("unsupported protocol for succeedingStreamFakeServer: %s", protocol)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// fallbackTarget describes a fake upstream target for fallback tests.
type fallbackTarget struct {
	Server   *httptest.Server
	Protocol llmapimux.Protocol
	Model    string
}

// newFallbackTestMux creates a Mux with CircuitBreakerRouter pointing to multiple fake upstreams.
func newFallbackTestMux(t *testing.T, inboundProtocol llmapimux.Protocol, targets []fallbackTarget) *llmapimux.Mux {
	t.Helper()
	candidates := make([]llmapimux.RouteResult, len(targets))
	for i, tgt := range targets {
		candidates[i] = llmapimux.RouteResult{
			Protocol: tgt.Protocol,
			BaseURL:  tgt.Server.URL,
			APIKey:   "sk-test-" + fmt.Sprintf("%d", i),
			Model:    tgt.Model,
		}
	}
	router := llmapimux.NewCircuitBreakerRouter(func(info llmapimux.RouteInfo) []llmapimux.RouteResult {
		return candidates
	})
	return llmapimux.NewMux(router)
}
