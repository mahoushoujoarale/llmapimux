package llmapimux

import (
	"context"
	"time"
)

// StatsReporter receives observability events from the unified Handler.
// Callbacks are serialized per request. Non-streaming callbacks run on the request goroutine;
// streaming callbacks may run on a dedicated wrapper goroutine.
// Implementations should be non-blocking or use buffered channels internally.
// Reporter panics are treated as caller bugs and are NOT recovered by llmapimux.
type StatsReporter interface {
	OnRequestStart(ctx context.Context, e RequestStartEvent)
	OnFirstByte(ctx context.Context, e FirstByteEvent)
	OnStreamChunk(ctx context.Context, e StreamChunkEvent)
	OnComplete(ctx context.Context, e CompleteEvent)
	OnAttemptError(ctx context.Context, e AttemptErrorEvent)
}

// NoopStatsReporter provides empty implementations of all StatsReporter methods.
// Embed this in your implementation to override only the methods you care about.
// Also used as the default when no StatsReporter is configured (avoids nil checks).
type NoopStatsReporter struct{}

func (NoopStatsReporter) OnRequestStart(context.Context, RequestStartEvent) {}
func (NoopStatsReporter) OnFirstByte(context.Context, FirstByteEvent)       {}
func (NoopStatsReporter) OnStreamChunk(context.Context, StreamChunkEvent)   {}
func (NoopStatsReporter) OnComplete(context.Context, CompleteEvent)         {}
func (NoopStatsReporter) OnAttemptError(context.Context, AttemptErrorEvent) {}

type RequestStartEvent struct {
	RequestID        string
	Time             time.Time
	InboundProtocol  Protocol
	OutboundProtocol Protocol
	Streaming        bool
	IRRequest        *Request
}

type FirstByteEvent struct {
	RequestID string
	Time      time.Time
	TTFB      time.Duration
}

type StreamChunkEvent struct {
	RequestID       string
	Time            time.Time
	SequenceNum     int
	ElapsedTime     time.Duration
	InterChunkDelay time.Duration
	IREvent         *StreamEvent
}

type CompletionStatus string

const (
	CompletionStatusSuccess  CompletionStatus = "success"
	CompletionStatusError    CompletionStatus = "error"
	CompletionStatusCanceled CompletionStatus = "canceled"
)

type CompleteEvent struct {
	RequestID        string
	Time             time.Time
	Status           CompletionStatus
	Error            error
	InboundProtocol  Protocol
	OutboundProtocol Protocol

	TTFB         time.Duration
	TotalLatency time.Duration

	Usage Usage

	OutputThroughput float64
	TPOT             time.Duration // Time Per Output Token = (TotalLatency - TTFB) / CompletionTokens (0 if non-streaming or no output tokens)
	Chunks           int           // Total streaming chunks received (0 if non-streaming)

	StopReason  StopReason
	ActualModel string

	IRResponse *Response

	AttemptNum    int // which route/fallback attempt succeeded (1 = no fallback)
	RetryAttempts int // physical retries across this request
	QueueWait     time.Duration
}

// AttemptErrorEvent is fired each time a send attempt fails and is retried.
type AttemptErrorEvent struct {
	RequestID    string
	AttemptNum   int         // which route/fallback attempt failed (1 = primary from Route())
	RetryAttempt int         // physical retry attempt within this route target (0 = first try)
	Target       RouteResult // the target that failed
	SendErr      SendError   // error details
	WillRetry    bool
	RetryDelay   time.Duration
}
