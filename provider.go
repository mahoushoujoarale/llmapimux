package llmapimux

import (
	"context"
	"net/http"
)

type Protocol string

const (
	ProtocolOpenAIChat      Protocol = "openai_chat"
	ProtocolOpenAIResponses Protocol = "openai_responses"
	ProtocolAnthropic       Protocol = "anthropic"
	ProtocolGemini          Protocol = "gemini"
)

// SendError carries structured information about a failed send attempt.
type SendError struct {
	AttemptNum  int
	StatusCode  int
	Header      http.Header
	IsTimeout   bool
	IsConnError bool
	Err         error
}

// Router determines the outbound target for each request.
type Router interface {
	Route(ctx context.Context, info RouteInfo) (RouteResult, error)
	OnError(ctx context.Context, info RouteInfo, target RouteResult, sendErr SendError) (RouteResult, error)
	OnSuccess(ctx context.Context, info RouteInfo, target RouteResult)
}

type routerFunc struct {
	fn func(ctx context.Context, info RouteInfo) (RouteResult, error)
}

func (r *routerFunc) Route(ctx context.Context, info RouteInfo) (RouteResult, error) {
	return r.fn(ctx, info)
}

func (r *routerFunc) OnError(_ context.Context, _ RouteInfo, _ RouteResult, sendErr SendError) (RouteResult, error) {
	return RouteResult{}, sendErr.Err
}

func (r *routerFunc) OnSuccess(_ context.Context, _ RouteInfo, _ RouteResult) {}

// RouterFunc wraps a Route function into a full Router implementation.
// OnError returns the error directly (no fallback). OnSuccess is a no-op.
func RouterFunc(fn func(ctx context.Context, info RouteInfo) (RouteResult, error)) Router {
	return &routerFunc{fn: fn}
}

// RouteInfo carries explicit, immutable routing inputs.
type RouteInfo struct {
	RequestID       string
	Model           string
	InboundProtocol Protocol
	Stream          bool
	HasTools        bool
	HasMedia        bool
	APIKey          string
}

// RouteResult is the outbound target decided by the Router.
type RouteResult struct {
	Protocol Protocol
	BaseURL  string
	APIKey   string
	Model    string
	ProxyURL string      // optional HTTP/HTTPS proxy URL (e.g. "http://proxy:8080")
	Header   http.Header // optional extra headers to send to upstream
}

// Authenticator validates inbound API keys.
type Authenticator interface {
	Authenticate(ctx context.Context, apiKey string) error
}

// OutboundConfig holds the base URL and API key for outbound calls.
// Protocol is determined by the Client implementation.
type OutboundConfig struct {
	BaseURL  string
	APIKey   string
	ProxyURL string      // optional HTTP/HTTPS proxy URL
	Header   http.Header // optional extra headers to send to upstream
}

// RequestModifier is called before each outbound send attempt.
// It may inspect the IR Request and current RouteResult, and set
// req.OutboundExtra to inject additional fields into the outbound request body.
type RequestModifier func(ctx context.Context, req *Request, target RouteResult)
