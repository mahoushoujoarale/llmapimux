package llmapimux

import (
	"net/http"
	"reflect"
)

// Mux is the core entry point that creates inbound handlers for a given router.
type Mux struct {
	router            Router
	auth              Authenticator
	stats             StatsReporter
	reqMod            RequestModifier
	attemptController AttemptController
}

// OpenAIChatHandler returns an http.Handler for OpenAI Chat Completions inbound requests.
func (m *Mux) OpenAIChatHandler() http.Handler {
	return &Handler{codec: &openaiChatCodec{}, router: m.router, auth: m.auth, stats: m.stats, reqMod: m.reqMod, attemptController: m.attemptController}
}

// OpenAIResponsesHandler returns an http.Handler for OpenAI Responses API inbound requests.
func (m *Mux) OpenAIResponsesHandler() http.Handler {
	return &Handler{codec: &openaiResponsesCodec{}, router: m.router, auth: m.auth, stats: m.stats, reqMod: m.reqMod, attemptController: m.attemptController}
}

// AnthropicHandler returns an http.Handler for Anthropic Messages inbound requests.
func (m *Mux) AnthropicHandler() http.Handler {
	return &Handler{codec: &anthropicCodec{}, router: m.router, auth: m.auth, stats: m.stats, reqMod: m.reqMod, attemptController: m.attemptController}
}

// GeminiHandler returns an http.Handler for Gemini GenerateContent inbound requests.
func (m *Mux) GeminiHandler() http.Handler {
	return &Handler{codec: &geminiCodec{}, router: m.router, auth: m.auth, stats: m.stats, reqMod: m.reqMod, attemptController: m.attemptController}
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

// NewMux creates a new Mux with a Router and optional configuration.
func NewMux(router Router, opts ...MuxOption) *Mux {
	m := &Mux{router: router, stats: NoopStatsReporter{}}
	for _, opt := range opts {
		opt(m)
	}
	return m
}
