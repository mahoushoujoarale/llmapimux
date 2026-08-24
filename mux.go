package llmapimux

import (
	"net/http"
	"reflect"
)

// Mux is the core entry point that creates inbound handlers for a given router.
type Mux struct {
	router                Router
	auth                  Authenticator
	stats                 StatsReporter
	reqMod                RequestModifier
	attemptController     AttemptController
	httpClient            *http.Client // injected into every outbound client
	preserveOriginalModel bool
	mapDeveloperToSystem  bool
}

// OpenAIChatHandler returns an http.Handler for OpenAI Chat Completions inbound requests.
func (m *Mux) OpenAIChatHandler() http.Handler {
	return &Handler{codec: &openaiChatCodec{}, router: m.router, auth: m.auth, stats: m.stats, reqMod: m.reqMod, attemptController: m.attemptController, httpClient: m.httpClient, preserveOriginalModel: m.preserveOriginalModel, mapDeveloperToSystem: m.mapDeveloperToSystem}
}

// OpenAIResponsesHandler returns an http.Handler for OpenAI Responses API inbound requests.
func (m *Mux) OpenAIResponsesHandler() http.Handler {
	return &Handler{codec: &openaiResponsesCodec{}, router: m.router, auth: m.auth, stats: m.stats, reqMod: m.reqMod, attemptController: m.attemptController, httpClient: m.httpClient, preserveOriginalModel: m.preserveOriginalModel, mapDeveloperToSystem: m.mapDeveloperToSystem}
}

// AnthropicHandler returns an http.Handler for Anthropic Messages inbound requests.
func (m *Mux) AnthropicHandler() http.Handler {
	return &Handler{codec: &anthropicCodec{}, router: m.router, auth: m.auth, stats: m.stats, reqMod: m.reqMod, attemptController: m.attemptController, httpClient: m.httpClient, preserveOriginalModel: m.preserveOriginalModel, mapDeveloperToSystem: m.mapDeveloperToSystem}
}

// GeminiHandler returns an http.Handler for Gemini GenerateContent inbound requests.
func (m *Mux) GeminiHandler() http.Handler {
	return &Handler{codec: &geminiCodec{}, router: m.router, auth: m.auth, stats: m.stats, reqMod: m.reqMod, attemptController: m.attemptController, httpClient: m.httpClient, preserveOriginalModel: m.preserveOriginalModel, mapDeveloperToSystem: m.mapDeveloperToSystem}
}

// MuxOption configures a Mux.
type MuxOption func(*Mux)

// WithAuthenticator sets an Authenticator on the Mux.
func WithAuthenticator(auth Authenticator) MuxOption {
	return func(m *Mux) { m.auth = auth }
}

// WithStatsReporter sets a StatsReporter on the Mux.
func WithStatsReporter(r StatsReporter) MuxOption {
	return func(m *Mux) {
		if r == nil {
			m.stats = NoopStatsReporter{}
			return
		}
		rv := reflect.ValueOf(r)
		if rv.Kind() == reflect.Pointer && rv.IsNil() {
			m.stats = NoopStatsReporter{}
			return
		}
		m.stats = r
	}
}

// WithRequestModifier sets a RequestModifier that is called before each
// outbound send attempt, allowing callers to set Request.OutboundExtra.
func WithRequestModifier(fn RequestModifier) MuxOption {
	return func(m *Mux) { m.reqMod = fn }
}

// WithAttemptController sets a controller that can gate and retry physical
// outbound send attempts. Nil keeps the default no-controller behavior.
func WithAttemptController(controller AttemptController) MuxOption {
	return func(m *Mux) { m.attemptController = controller }
}

func WithPreserveOriginalModel(enabled bool) MuxOption {
	return func(m *Mux) { m.preserveOriginalModel = enabled }
}

// WithHTTPClient sets a shared *http.Client that is injected into every
// outbound client created by the Mux. This lets callers control connection
// behavior (proxy, TCP keepalive, timeouts, connection pooling) in one place.
// When a RouteResult carries a ProxyURL, the injected transport is cloned and
// only its Proxy field is overridden, so caller dial/keepalive settings are
// preserved. Nil keeps the default client behavior.
func WithHTTPClient(c *http.Client) MuxOption {
	return func(m *Mux) { m.httpClient = c }
}

// WithMapDeveloperToSystem configures the gateway to emit SystemPrompt as
// "system" role (instead of "developer") in outbound OpenAI Chat Completions
// requests. Many downstream OpenAI-compatible providers (e.g. vLLM, some
// OpenAI-compatible servers) don't support the "developer" message role and
// will reject requests containing it with a 400 error.
//
// When enabled, the IR's SystemPrompt — which already consolidates all system
// and developer content (equivalent to vLLM's _consolidate_system_messages) —
// is emitted as a single "system" role message at position 0 instead of a
// "developer" role message. This matches the behavior of vLLM PR #43590
// ("Fold developer-role input messages into system instructions") adapted to
// the gateway's IR-based architecture.
func WithMapDeveloperToSystem(enabled bool) MuxOption {
	return func(m *Mux) { m.mapDeveloperToSystem = enabled }
}

// NewMux creates a new Mux with a Router and optional configuration.
func NewMux(router Router, opts ...MuxOption) *Mux {
	m := &Mux{router: router, stats: NoopStatsReporter{}}
	for _, opt := range opts {
		opt(m)
	}
	return m
}
