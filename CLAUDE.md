# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`github.com/llmapimux/llmapimux` is a Go SDK providing `http.Handler` implementations for proxying LLM API requests across 4 protocols: OpenAI Chat Completions, OpenAI Responses API, Anthropic Messages, and Gemini GenerateContent. It uses a unified intermediate representation (IR) pipeline for request/response normalization and protocol conversion across all routes.

## Build & Test Commands

```bash
go build ./...           # Build all packages
go test ./...            # Run all tests (unit + integration, no SDK deps)
go test -run TestName ./...  # Run a single test by name
go vet ./...             # Static analysis

# E2E SDK tests (in separate submodule):
cd tests/e2e && go test ./...              # Run all E2E tests (fake-server tests always run; real API tests skip if .env missing)
cd tests/e2e && go test -run TestRealAPI ./...  # Run only real API tests
```

## Architecture

**Data flow**: Inbound request → decode to IR (plus protocol-specific RawExtra side channel) → `RequestModifier` hook (sets `OutboundExtra`) → encode to outbound protocol (+ merge `OutboundExtra`) → target API.

**File layout** (single root package, no subdirectories):
- `mux.go`, `message.go`, `provider.go` — `Mux` entry point, IR types (Request/Response/Message/ContentPart/StreamEvent), routing interfaces
- `handler.go`, `handler_helpers.go`, `codec_*.go` — 4 inbound `http.Handler` entrypoints plus shared decode/encode helpers and protocol codec adapters
- `outbound_*.go` — 4 HTTP clients, encode IR and send to target API
- `convert_*.go` — 4 bidirectional converters (protocol ↔ IR), 8 total conversion directions covering 16 protocol combinations
- `outbound.go`, `sse.go`, `rawextra.go` — shared outbound HTTP utilities, SSE encode/decode utilities, and RawExtra side-channel helpers
- `stats.go` — StatsReporter interface and lifecycle event types
- `circuit_breaker.go` — Built-in `CircuitBreakerRouter` implementation with configurable circuit breaker logic
- `tests/e2e/` — E2E submodule (`go.mod` with `replace ../..`): SDK fake-server tests + real API integration tests

**Key design choices**:
- Unified IR with 2N adapters covers N² protocol paths
- All protocol routes go through decode → IR (+ RawExtra) → encode; no same-protocol raw-body passthrough path
- Cross-protocol fields not representable in target are silently dropped
- Fallback retry: Handler retries via `Router.OnError`/`OnSuccess` when upstream fails (IR decoded once, re-encoded per attempt); streaming fallback only before HTTP 200 committed
- Built-in `CircuitBreakerRouter`: per-target circuit breaker (Closed→Open→HalfOpen→Closed) with configurable thresholds, `CandidateFunc` for target selection, and lazy attempt tracking cleanup
- Context propagation: client disconnect cancels upstream request and retry loop
- `RequestModifier` hook: called per-attempt in the retry loop (after model assignment, before send), allows callers to set `Request.OutboundExtra` which gets merged into the outbound JSON body. Reset to `nil` before each attempt to prevent cross-target leakage. Registered via `WithRequestModifier` MuxOption.

## Testing Notes

- E2E SDK tests live in `tests/e2e/` submodule with its own `go.mod` (uses `replace` to import parent module); run with `cd tests/e2e && go test ./...`
- Root `go test ./...` runs unit + integration tests only — no SDK dependencies
- Fake-server E2E tests use real SDK clients (`anthropic-sdk-go`, `openai-go/v3`, `google.golang.org/genai`) against local `httptest.Server` fakes
- Real API tests (`TestRealAPI_*`) load credentials from `.env` at project root; they skip automatically when env vars are missing
- Required `.env` vars for real API tests: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `GEMINI_BASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`
- `newLocalOnlyHTTPClient` blocks all non-localhost outbound calls to prevent accidental external requests in fake-server tests
- Gemini inbound handler must be registered with trailing-slash path (e.g. `/v1/models/`) for prefix matching
- Fallback integration tests (`TestIntegration_Fallback_*`) verify CircuitBreakerRouter + Handler full pipeline with fake upstreams
- Fallback E2E tests (`TestFallbackE2E_*`) verify fallback behavior through real SDK clients (openai-go, anthropic-sdk-go, genai)

## Fallback & Circuit Breaker

- **Router interface**: `Route` (primary target), `OnError` (fallback on failure), `OnSuccess` (success callback). `RouterFunc` adapter wraps a simple Route function (no fallback).
- **SendError**: carries `StatusCode`, `Header`, `IsTimeout`, `IsConnError`, `Err` from failed upstream attempts. Built from `UpstreamHTTPError` (which includes response `Header`).
- **Handler retry loop** (`handler.go`): decodes IR once, retries by re-encoding per attempt. `retryLoopState.handleSendError` is the shared error-handling helper for both streaming and non-streaming paths. Context cancellation breaks the loop without retry.
- **Streaming boundary**: fallback only possible when `SendStream()` returns error (pre-HTTP-200). Once channel returned, headers written to client, no fallback.
- **RawExtra on retry**: evaluated once after first `Route()`. Cross-protocol fallback clears it; same-protocol fallback re-populates.
- **OutboundExtra on retry**: reset to `nil` and `RequestModifier` re-invoked before each attempt, so the modifier can condition extras on the current target (e.g., add `service_tier` only for OpenAI targets).
- **StatsReporter**: `OnAttemptError` fires per failed attempt; `CompleteEvent.AttemptNum` reflects final attempt number; `RequestStartEvent.OutboundProtocol` is the initial target (not final).
- **CircuitBreakerRouter** (`circuit_breaker.go`): takes `CandidateFunc` + `CBOption`s. State machine: Closed (pass) → Open (after `failureThreshold` consecutive trippable failures) → HalfOpen (after `recoveryTimeout`) → Closed (after `successThreshold` successes). HalfOpen failure immediately reopens. Per-circuit `sync.Mutex`, `sync.RWMutex` for circuit map, lazy cleanup of attempt records every 100 calls (5 min TTL). `defaultShouldTrip`: 5xx + timeout + conn error (4xx does NOT trip but still triggers fallback via `OnError`).

## Protocol-Specific Gotchas

- **Gemini**: model is in URL path (not body); no native "tool" role; no explicit `tool_use` finish reason (infer from FunctionCall parts); streaming uses `streamGenerateContent?alt=sse`; ResponseFormat requires JSON Schema → Gemini Schema conversion
- **Anthropic**: `redacted_thinking` must round-trip exactly; `pause_turn` stop reason has no equivalent in other protocols (maps to `end_turn`)
- **OpenAI Chat**: both `system` and `developer` roles → IR SystemPrompt; outbound emits `developer` role; reads both `max_tokens` and `max_completion_tokens`
- **OpenAI Responses**: stateless proxy (no `previous_response_id` support); built-in tools silently dropped
- **Auth headers**: OpenAI `Authorization: Bearer`, Anthropic `x-api-key` or `Authorization: Bearer` (inbound, x-api-key preferred) / `x-api-key` + `anthropic-version` (outbound), Gemini `x-goog-api-key` or `?key=`
- **Anthropic streaming**: cross-protocol IR streams omit `content_block_start`/`content_block_stop` lifecycle events; `inbound_anthropic.go:handleStreaming` injects them synthetically — required for the Anthropic SDK accumulator's `AsAny()` to return populated text
