package llmapimux

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// CircuitState represents the state of a circuit breaker.
type CircuitState int

const (
	CircuitClosed   CircuitState = iota // Normal — requests pass through
	CircuitOpen                         // Tripped — requests rejected
	CircuitHalfOpen                     // Probing — limited requests allowed
)

// CBOption configures a CircuitBreakerRouter.
type CBOption func(*cbConfig)

// WithFailureThreshold sets the number of consecutive failures before the
// circuit opens. Default is 5.
func WithFailureThreshold(n int) CBOption {
	return func(c *cbConfig) { c.failureThreshold = n }
}

// WithRecoveryTimeout sets the duration a circuit stays open before
// transitioning to half-open. Default is 30s.
func WithRecoveryTimeout(d time.Duration) CBOption {
	return func(c *cbConfig) { c.recoveryTimeout = d }
}

// WithSuccessThreshold sets the number of consecutive successes in half-open
// state required to close the circuit. Default is 2.
func WithSuccessThreshold(n int) CBOption {
	return func(c *cbConfig) { c.successThreshold = n }
}

// WithShouldTrip sets the function that decides whether a SendError should
// count as a circuit-breaker failure. By default, 5xx status codes, timeouts,
// and connection errors trip the circuit; 4xx errors do not.
func WithShouldTrip(fn func(SendError) bool) CBOption {
	return func(c *cbConfig) { c.shouldTrip = fn }
}

// WithHalfOpenMax sets the maximum number of concurrent probing requests
// allowed while the circuit is half-open. Default is 1.
func WithHalfOpenMax(n int) CBOption {
	return func(c *cbConfig) { c.halfOpenMax = n }
}

// WithOnStateChange registers a callback that fires on circuit state
// transitions. The callback is called with the circuit's mutex held — it must
// not block or call back into the CircuitBreakerRouter.
func WithOnStateChange(fn func(key string, from, to CircuitState)) CBOption {
	return func(c *cbConfig) { c.onStateChange = fn }
}

// WithCircuitKeyFunc sets the function used to derive the circuit-breaker map
// key from a RouteResult. Default is rr.BaseURL.
func WithCircuitKeyFunc(fn func(RouteResult) string) CBOption {
	return func(c *cbConfig) { c.keyFunc = fn }
}

// CandidateFunc returns an ordered list of candidate targets for a request.
type CandidateFunc func(info RouteInfo) []RouteResult

// NewCircuitBreakerRouter creates a Router with circuit breaker logic.
// fn provides the ordered candidate list for each request. Options configure
// thresholds, timeouts, and callbacks.
func NewCircuitBreakerRouter(fn CandidateFunc, opts ...CBOption) Router {
	cfg := cbConfig{
		failureThreshold: 5,
		recoveryTimeout:  30 * time.Second,
		successThreshold: 2,
		shouldTrip:       defaultShouldTrip,
		halfOpenMax:      1,
		onStateChange:    func(string, CircuitState, CircuitState) {},
		keyFunc:          func(rr RouteResult) string { return rr.BaseURL },
	}
	for _, o := range opts {
		o(&cfg)
	}
	return &CircuitBreakerRouter{
		fn:       fn,
		cfg:      cfg,
		circuits: make(map[string]*circuitBreaker),
		attempts: make(map[string]*attemptRecord),
	}
}

func defaultShouldTrip(se SendError) bool {
	if se.IsTimeout || se.IsConnError {
		return true
	}
	return se.StatusCode >= 500
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

type cbConfig struct {
	failureThreshold int
	recoveryTimeout  time.Duration
	successThreshold int
	shouldTrip       func(SendError) bool
	halfOpenMax      int
	onStateChange    func(key string, from, to CircuitState)
	keyFunc          func(RouteResult) string
}

type circuitBreaker struct {
	mu               sync.Mutex
	state            CircuitState
	consecutiveFails int
	consecutiveSuccs int
	lastFailTime     time.Time
	halfOpenInFlight int
}

// CircuitBreakerRouter implements the Router interface with circuit breaker
// logic wrapping a CandidateFunc.
type CircuitBreakerRouter struct {
	fn  CandidateFunc
	cfg cbConfig

	circuitMu sync.RWMutex
	circuits  map[string]*circuitBreaker

	attemptMu    sync.Mutex
	attempts     map[string]*attemptRecord
	cleanupCount int64 // atomic; used for lazy cleanup
}

type attemptRecord struct {
	targets   map[string]struct{} // circuit keys already attempted
	createdAt time.Time
}

// ---------------------------------------------------------------------------
// Router interface
// ---------------------------------------------------------------------------

var errAllCircuitsBroken = errors.New("all targets are circuit-broken")

// Route picks the first healthy candidate from the candidate list.
func (r *CircuitBreakerRouter) Route(_ context.Context, info RouteInfo) (RouteResult, error) {
	r.lazyCleanup()

	candidates := r.fn(info)
	for _, c := range candidates {
		key := r.cfg.keyFunc(c)
		cb := r.getOrCreateCircuit(key)

		if rr, ok := r.tryCandidate(cb, key, c); ok {
			r.recordAttempt(info.RequestID, key)
			return rr, nil
		}
	}
	return RouteResult{}, errAllCircuitsBroken
}

// OnError handles a failed send attempt: updates circuit state, then tries to
// find the next healthy candidate.
func (r *CircuitBreakerRouter) OnError(_ context.Context, info RouteInfo, target RouteResult, sendErr SendError) (RouteResult, error) {
	r.lazyCleanup()

	key := r.cfg.keyFunc(target)
	cb := r.getOrCreateCircuit(key)

	cb.mu.Lock()
	if cb.state == CircuitHalfOpen {
		cb.halfOpenInFlight--
	}
	if r.cfg.shouldTrip(sendErr) {
		cb.consecutiveFails++
		cb.consecutiveSuccs = 0
		if cb.state == CircuitHalfOpen {
			// Any trippable failure in HalfOpen → back to Open immediately.
			old := cb.state
			cb.state = CircuitOpen
			cb.lastFailTime = time.Now()
			cb.halfOpenInFlight = 0
			r.cfg.onStateChange(key, old, CircuitOpen)
		} else if cb.consecutiveFails >= r.cfg.failureThreshold && cb.state != CircuitOpen {
			old := cb.state
			cb.state = CircuitOpen
			cb.lastFailTime = time.Now()
			r.cfg.onStateChange(key, old, CircuitOpen)
		}
	}
	cb.mu.Unlock()

	// Record this target as attempted.
	r.recordAttempt(info.RequestID, key)

	// Try to find the next candidate, skipping already-attempted ones.
	attempted := r.getAttempted(info.RequestID)
	candidates := r.fn(info)
	for _, c := range candidates {
		cKey := r.cfg.keyFunc(c)
		if _, skip := attempted[cKey]; skip {
			continue
		}
		cCB := r.getOrCreateCircuit(cKey)
		if rr, ok := r.tryCandidate(cCB, cKey, c); ok {
			r.recordAttempt(info.RequestID, cKey)
			return rr, nil
		}
	}

	// No more candidates — clean up attempt record.
	r.deleteAttempt(info.RequestID)
	return RouteResult{}, errAllCircuitsBroken
}

// OnSuccess handles a successful send: resets failure counters and may close a
// half-open circuit.
func (r *CircuitBreakerRouter) OnSuccess(_ context.Context, info RouteInfo, target RouteResult) {
	key := r.cfg.keyFunc(target)
	cb := r.getOrCreateCircuit(key)

	cb.mu.Lock()
	if cb.state == CircuitHalfOpen {
		cb.halfOpenInFlight--
		cb.consecutiveSuccs++
		if cb.consecutiveSuccs >= r.cfg.successThreshold {
			old := cb.state
			cb.state = CircuitClosed
			cb.halfOpenInFlight = 0
			cb.consecutiveFails = 0
			cb.consecutiveSuccs = 0
			r.cfg.onStateChange(key, old, CircuitClosed)
		}
	}
	cb.consecutiveFails = 0
	cb.mu.Unlock()

	r.deleteAttempt(info.RequestID)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// getOrCreateCircuit returns the circuit for key, creating one if needed.
func (r *CircuitBreakerRouter) getOrCreateCircuit(key string) *circuitBreaker {
	r.circuitMu.RLock()
	cb, ok := r.circuits[key]
	r.circuitMu.RUnlock()
	if ok {
		return cb
	}

	r.circuitMu.Lock()
	defer r.circuitMu.Unlock()
	// Double-check after acquiring write lock.
	if cb, ok = r.circuits[key]; ok {
		return cb
	}
	cb = &circuitBreaker{}
	r.circuits[key] = cb
	return cb
}

// tryCandidate checks whether the given circuit allows a request through.
// It returns true if the request is allowed, and may transition Open→HalfOpen.
// The caller must NOT hold cb.mu.
func (r *CircuitBreakerRouter) tryCandidate(cb *circuitBreaker, key string, c RouteResult) (RouteResult, bool) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case CircuitClosed:
		return c, true

	case CircuitOpen:
		if time.Since(cb.lastFailTime) >= r.cfg.recoveryTimeout {
			old := cb.state
			cb.state = CircuitHalfOpen
			cb.consecutiveSuccs = 0
			cb.halfOpenInFlight = 0
			r.cfg.onStateChange(key, old, CircuitHalfOpen)
			// Fall through to HalfOpen handling below.
		} else {
			return RouteResult{}, false
		}
		fallthrough

	case CircuitHalfOpen:
		if cb.halfOpenInFlight < r.cfg.halfOpenMax {
			cb.halfOpenInFlight++
			return c, true
		}
		return RouteResult{}, false
	}

	return RouteResult{}, false
}

// recordAttempt adds key to the attempt record for the given request ID.
func (r *CircuitBreakerRouter) recordAttempt(requestID, key string) {
	r.attemptMu.Lock()
	defer r.attemptMu.Unlock()
	rec, ok := r.attempts[requestID]
	if !ok {
		rec = &attemptRecord{
			targets:   make(map[string]struct{}),
			createdAt: time.Now(),
		}
		r.attempts[requestID] = rec
	}
	rec.targets[key] = struct{}{}
}

// getAttempted returns a copy of the set of circuit keys attempted for requestID.
func (r *CircuitBreakerRouter) getAttempted(requestID string) map[string]struct{} {
	r.attemptMu.Lock()
	defer r.attemptMu.Unlock()
	rec, ok := r.attempts[requestID]
	if !ok {
		return nil
	}
	// Return a copy so callers don't need to hold the lock.
	out := make(map[string]struct{}, len(rec.targets))
	for k := range rec.targets {
		out[k] = struct{}{}
	}
	return out
}

// deleteAttempt removes the attempt record for the given request ID.
func (r *CircuitBreakerRouter) deleteAttempt(requestID string) {
	r.attemptMu.Lock()
	defer r.attemptMu.Unlock()
	delete(r.attempts, requestID)
}

const cleanupInterval = 100

// lazyCleanup periodically scans and removes stale attempt records.
func (r *CircuitBreakerRouter) lazyCleanup() {
	count := atomic.AddInt64(&r.cleanupCount, 1)
	if count%cleanupInterval != 0 {
		return
	}

	cutoff := time.Now().Add(-5 * time.Minute)
	r.attemptMu.Lock()
	defer r.attemptMu.Unlock()
	for id, rec := range r.attempts {
		if rec.createdAt.Before(cutoff) {
			delete(r.attempts, id)
		}
	}
}
