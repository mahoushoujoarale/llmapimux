package llmapimux

import (
	"context"
	"time"
)

// AttemptPermit is held for the lifetime of one outbound send attempt.
// For streaming attempts, Release is called only after the stream is fully
// consumed or the client request is canceled.
type AttemptPermit interface {
	Release()
}

// AttemptAdmission describes the result of admitting an outbound attempt.
type AttemptAdmission struct {
	Permit       AttemptPermit
	WaitDuration time.Duration
	LimitKey     string
	Limit        int
	Active       int
}

// AttemptController can delay, reject, or retry outbound send attempts.
// routeAttempt is the fallback-chain attempt number, starting at 1.
// retryAttempt is the retry number for the current target, starting at 0.
type AttemptController interface {
	Acquire(ctx context.Context, info RouteInfo, target RouteResult, routeAttempt int, retryAttempt int) (AttemptAdmission, error)
	RetryDelay(ctx context.Context, info RouteInfo, target RouteResult, sendErr SendError, routeAttempt int, retryAttempt int) (time.Duration, bool)
}
