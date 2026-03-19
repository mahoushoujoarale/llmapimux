package llmapimux

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// trimBaseURL removes any trailing slashes from a base URL to prevent
// double-slash issues when concatenating paths.
func trimBaseURL(u string) string {
	return strings.TrimRight(u, "/")
}

// StreamResult carries either a StreamEvent or an error from mid-stream failures.
type StreamResult struct {
	Event *StreamEvent
	Err   error
}

// UpstreamHTTPError represents an HTTP 4xx/5xx error returned by the upstream provider.
type UpstreamHTTPError struct {
	StatusCode int
	Header     http.Header
	Body       []byte
}

func (e *UpstreamHTTPError) Error() string {
	return fmt.Sprintf("upstream HTTP status %d: %s", e.StatusCode, string(e.Body))
}

func newUpstreamHTTPError(op string, statusCode int, header http.Header, body []byte) error {
	return fmt.Errorf("%s: %w", op, &UpstreamHTTPError{StatusCode: statusCode, Header: header.Clone(), Body: body})
}

// resolveUpstreamStatusCode returns the HTTP status code for an upstream error.
// If err wraps an UpstreamHTTPError, its StatusCode is used; otherwise http.StatusBadGateway.
func resolveUpstreamStatusCode(err error) int {
	var upstreamErr *UpstreamHTTPError
	if errors.As(err, &upstreamErr) {
		return upstreamErr.StatusCode
	}
	return http.StatusBadGateway
}

// proxyClients caches http.Client instances per proxy URL to reuse connections.
var (
	proxyClientsMu sync.RWMutex
	proxyClients   = make(map[string]*http.Client)
)

// httpClientForProxy returns an *http.Client configured for the given proxy URL.
// If proxyURL is empty, returns base (or http.DefaultClient if base is nil).
// Proxy clients are cached by URL to reuse underlying transports.
func httpClientForProxy(base *http.Client, proxyURL string) *http.Client {
	if proxyURL == "" {
		if base != nil {
			return base
		}
		return http.DefaultClient
	}

	proxyClientsMu.RLock()
	c, ok := proxyClients[proxyURL]
	proxyClientsMu.RUnlock()
	if ok {
		return c
	}

	proxyClientsMu.Lock()
	defer proxyClientsMu.Unlock()
	// Double-check after acquiring write lock.
	if c, ok = proxyClients[proxyURL]; ok {
		return c
	}
	u, err := url.Parse(proxyURL)
	if err != nil {
		// Invalid proxy URL — fall back to direct connection rather than
		// silently caching a misconfigured client.
		if base != nil {
			return base
		}
		return http.DefaultClient
	}
	c = &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyURL(u),
		},
	}
	proxyClients[proxyURL] = c
	return c
}

// applyExtraHeaders copies extra headers from OutboundConfig onto the HTTP request.
// Protocol-specific headers (Authorization, x-api-key, Content-Type, etc.) set by the
// client are NOT overwritten — extra headers are applied first, then protocol headers.
func applyExtraHeaders(httpReq *http.Request, cfg OutboundConfig) {
	for k, vs := range cfg.Header {
		for _, v := range vs {
			httpReq.Header.Add(k, v)
		}
	}
}

// Client sends an IR Request to a provider and returns an IR Response.
type Client interface {
	Send(ctx context.Context, req *Request, cfg OutboundConfig) (*Response, error)
	SendStream(ctx context.Context, req *Request, cfg OutboundConfig) (<-chan StreamResult, error)
}

// NewClient returns the appropriate outbound client for the given protocol.
func NewClient(protocol Protocol) Client {
	switch protocol {
	case ProtocolAnthropic:
		return &AnthropicClient{}
	case ProtocolOpenAIChat:
		return &OpenAIChatClient{}
	case ProtocolOpenAIResponses:
		return &OpenAIResponsesClient{}
	case ProtocolGemini:
		return &GeminiClient{}
	default:
		return nil
	}
}
