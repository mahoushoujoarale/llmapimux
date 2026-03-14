# llmapimux

[中文文档](README_zh.md)

A Go SDK providing `http.Handler` implementations for proxying and multiplexing LLM API requests across multiple protocols. Translate between OpenAI, Anthropic, and Google Gemini APIs transparently.

## Supported Protocols

| Protocol | Inbound (receive) | Outbound (send) |
|---|---|---|
| OpenAI Chat Completions | Yes | Yes |
| OpenAI Responses API | Yes | Yes |
| Anthropic Messages | Yes | Yes |
| Gemini GenerateContent | Yes | Yes |

Any inbound protocol can be routed to any outbound protocol — llmapimux handles the conversion automatically via a unified intermediate representation (IR).

## Installation

```bash
go get github.com/llmapimux/llmapimux
```

Requires Go 1.21+.

## Quick Start

```go
package main

import (
	"context"
	"net/http"

	"github.com/llmapimux/llmapimux"
)

// SimpleRouter routes all requests to a single OpenAI-compatible backend.
type SimpleRouter struct{}

func (r *SimpleRouter) Route(ctx context.Context, info llmapimux.RouteInfo) (llmapimux.RouteResult, error) {
	return llmapimux.RouteResult{
		Protocol: llmapimux.ProtocolOpenAIChat,
		BaseURL:  "https://api.openai.com",
		APIKey:   "sk-your-api-key",
		Model:    info.Model,
	}, nil
}

func main() {
	mux := llmapimux.NewMux(&SimpleRouter{})

	http.Handle("/v1/chat/completions", mux.OpenAIChatHandler())
	http.Handle("/v1/responses", mux.OpenAIResponsesHandler())
	http.Handle("/v1/messages", mux.AnthropicHandler())
	http.Handle("/v1/models/", mux.GeminiHandler()) // trailing slash for prefix matching

	http.ListenAndServe(":8080", nil)
}
```

This creates a proxy server that accepts requests in any of the 4 protocols and forwards them to an OpenAI backend, converting the protocol as needed.

## Architecture

```
Inbound Request
    │
    ▼
┌─────────────────┐
│  Protocol Handler│ (OpenAI Chat / Responses / Anthropic / Gemini)
│  Decode → IR     │
└────────┬────────┘
         │
    ┌────▼────┐
    │  Router  │ (your routing logic)
    └────┬────┘
         │
┌────────▼────────┐
│  Outbound Client │ (OpenAI Chat / Responses / Anthropic / Gemini)
│  IR → Encode     │
└─────────────────┘
         │
         ▼
   Target LLM API
```

**Key design choices:**
- **Unified IR with 2N adapters** covers N² protocol paths (4 protocols = 8 adapters for 16 combinations)
- **No same-protocol passthrough** — all requests go through decode → IR → encode, ensuring consistent behavior
- **RawExtra side channel** preserves protocol-specific fields for same-protocol roundtrips
- **No retry logic** — errors are forwarded as-is
- **Context propagation** — client disconnect cancels the upstream request
- **Zero external dependencies** — only the Go standard library

## Core Concepts

### Router

The `Router` interface is the only required component. It decides where each request goes:

```go
type Router interface {
    Route(ctx context.Context, info RouteInfo) (RouteResult, error)
}
```

`RouteInfo` provides: `RequestID`, `Model`, `InboundProtocol`, `Stream`, `HasTools`, `HasMedia`, `APIKey`.

`RouteResult` specifies: `Protocol`, `BaseURL`, `APIKey`, `Model`.

### Authenticator

Optional inbound authentication:

```go
type Authenticator interface {
    Authenticate(ctx context.Context, apiKey string) error
}

mux := llmapimux.NewMux(router, llmapimux.WithAuthenticator(myAuth))
```

### StatsReporter

Optional observability hook for request lifecycle events:

```go
type StatsReporter interface {
    OnRequestStart(ctx context.Context, e RequestStartEvent)
    OnFirstByte(ctx context.Context, e FirstByteEvent)
    OnStreamChunk(ctx context.Context, e StreamChunkEvent)
    OnComplete(ctx context.Context, e CompleteEvent)
}

mux := llmapimux.NewMux(router, llmapimux.WithStatsReporter(myReporter))
```

Embed `NoopStatsReporter` to only implement the methods you need.

## Protocol Notes

- **Gemini**: model is extracted from URL path (not request body); inbound handler must be registered with trailing-slash path (e.g. `/v1/models/`) for prefix matching
- **Anthropic**: `redacted_thinking` content round-trips exactly; inbound auth accepts both `x-api-key` header and `Authorization: Bearer` token (x-api-key takes precedence)
- **OpenAI Chat**: both `system` and `developer` roles map to IR system prompt; outbound emits `developer` role
- **OpenAI Responses**: stateless proxy (no `previous_response_id` support); built-in tools are silently dropped
- Cross-protocol fields not representable in the target protocol are silently dropped

## Streaming

Both streaming and non-streaming modes are supported for all protocol combinations. Streaming responses use Server-Sent Events (SSE). The `Stream` field in `RouteInfo` lets your router make protocol-aware decisions.

## Testing

```bash
go test ./...                          # Unit + integration tests
cd tests/e2e && go test ./...          # E2E tests with real SDK clients
```

E2E tests use real SDK clients (`anthropic-sdk-go`, `openai-go/v3`, `google.golang.org/genai`) against local fake servers. Real API tests require a `.env` file and skip automatically when credentials are missing.

## License

[MIT](LICENSE)
