package llmapimux

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// helpers -------------------------------------------------------------------

func makeCandidates(baseURLs ...string) CandidateFunc {
	return func(_ RouteInfo) []RouteResult {
		out := make([]RouteResult, len(baseURLs))
		for i, u := range baseURLs {
			out[i] = RouteResult{BaseURL: u, Protocol: ProtocolOpenAIChat}
		}
		return out
	}
}

func infoWith(requestID string) RouteInfo {
	return RouteInfo{RequestID: requestID, Model: "m"}
}

var ctx = context.Background()

func tripN(t *testing.T, r Router, info RouteInfo, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		target, err := r.Route(ctx, info)
		if err != nil {
			t.Fatalf("Route failed on trip iteration %d: %v", i, err)
		}
		_, _ = r.OnError(ctx, info, target, SendError{
			StatusCode: 500, Err: errors.New("fail"),
		})
		// Use a new requestID each time so attempt tracking doesn't interfere.
		info = infoWith(info.RequestID + "_")
	}
}

// tests ---------------------------------------------------------------------

func TestCB_BasicRouting(t *testing.T) {
	r := NewCircuitBreakerRouter(makeCandidates("a", "b", "c"))
	rr, err := r.Route(ctx, infoWith("r1"))
	if err != nil {
		t.Fatal(err)
	}
	if rr.BaseURL != "a" {
		t.Fatalf("expected first candidate 'a', got %q", rr.BaseURL)
	}
}

func TestCB_CircuitOpensAfterFailures(t *testing.T) {
	threshold := 3
	r := NewCircuitBreakerRouter(
		makeCandidates("a", "b"),
		WithFailureThreshold(threshold),
	)

	// Trip circuit "a" with threshold failures.
	tripN(t, r, infoWith("r1"), threshold)

	// Next route should skip "a" and return "b".
	rr, err := r.Route(ctx, infoWith("r2"))
	if err != nil {
		t.Fatal(err)
	}
	if rr.BaseURL != "b" {
		t.Fatalf("expected 'b' after 'a' opened, got %q", rr.BaseURL)
	}
}

func TestCB_CircuitRecovery(t *testing.T) {
	var transitions []struct{ key string; from, to CircuitState }
	var mu sync.Mutex

	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(1),
		WithRecoveryTimeout(10*time.Millisecond),
		WithSuccessThreshold(1),
		WithOnStateChange(func(key string, from, to CircuitState) {
			mu.Lock()
			transitions = append(transitions, struct{ key string; from, to CircuitState }{key, from, to})
			mu.Unlock()
		}),
	)

	// Trip the circuit.
	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	// Circuit should be Open — route should fail.
	_, err := r.Route(ctx, infoWith("r2"))
	if err == nil {
		t.Fatal("expected error when circuit is open")
	}

	// Wait for recovery timeout.
	time.Sleep(15 * time.Millisecond)

	// Should transition to HalfOpen and allow a probe.
	target, err = r.Route(ctx, infoWith("r3"))
	if err != nil {
		t.Fatalf("expected route to succeed in half-open: %v", err)
	}

	// Success should close the circuit.
	r.OnSuccess(ctx, infoWith("r3"), target)

	// Verify transitions: Closed→Open, Open→HalfOpen, HalfOpen→Closed.
	mu.Lock()
	defer mu.Unlock()
	if len(transitions) != 3 {
		t.Fatalf("expected 3 transitions, got %d: %+v", len(transitions), transitions)
	}
	expected := []struct{ from, to CircuitState }{
		{CircuitClosed, CircuitOpen},
		{CircuitOpen, CircuitHalfOpen},
		{CircuitHalfOpen, CircuitClosed},
	}
	for i, e := range expected {
		if transitions[i].from != e.from || transitions[i].to != e.to {
			t.Errorf("transition %d: expected %d→%d, got %d→%d",
				i, e.from, e.to, transitions[i].from, transitions[i].to)
		}
	}
}

func TestCB_HalfOpenMaxConcurrency(t *testing.T) {
	maxProbes := 2
	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(1),
		WithRecoveryTimeout(1*time.Millisecond),
		WithHalfOpenMax(maxProbes),
	)

	// Trip the circuit.
	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	time.Sleep(5 * time.Millisecond)

	// Should allow up to maxProbes concurrent probes.
	var targets []RouteResult
	for i := 0; i < maxProbes; i++ {
		rr, err := r.Route(ctx, infoWith("probe_"+string(rune('a'+i))))
		if err != nil {
			t.Fatalf("probe %d should succeed: %v", i, err)
		}
		targets = append(targets, rr)
	}

	// Next probe should be rejected (all half-open slots full).
	_, err := r.Route(ctx, infoWith("probe_extra"))
	if err == nil {
		t.Fatal("expected error when half-open slots are full")
	}

	// Complete one probe successfully to free a slot.
	r.OnSuccess(ctx, infoWith("probe_a"), targets[0])

	// Now another probe should be allowed.
	_, err = r.Route(ctx, infoWith("probe_retry"))
	if err != nil {
		t.Fatalf("expected probe to succeed after slot freed: %v", err)
	}
}

func TestCB_ShouldTripFiltering(t *testing.T) {
	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(1),
	)

	// 4xx errors should NOT trip the circuit.
	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 400, Err: errors.New("bad request")})

	// Circuit should still be closed.
	_, err := r.Route(ctx, infoWith("r2"))
	if err != nil {
		t.Fatalf("circuit should still be closed after 4xx: %v", err)
	}

	// 5xx error should trip.
	target, _ = r.Route(ctx, infoWith("r3"))
	r.OnError(ctx, infoWith("r3"), target, SendError{StatusCode: 500, Err: errors.New("server error")})

	// Circuit should be open.
	_, err = r.Route(ctx, infoWith("r4"))
	if err == nil {
		t.Fatal("expected error after 5xx tripped circuit")
	}
}

func TestCB_ShouldTripTimeout(t *testing.T) {
	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(1),
	)

	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{IsTimeout: true, Err: errors.New("timeout")})

	_, err := r.Route(ctx, infoWith("r2"))
	if err == nil {
		t.Fatal("expected error after timeout tripped circuit")
	}
}

func TestCB_ShouldTripConnError(t *testing.T) {
	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(1),
	)

	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{IsConnError: true, Err: errors.New("conn refused")})

	_, err := r.Route(ctx, infoWith("r2"))
	if err == nil {
		t.Fatal("expected error after conn error tripped circuit")
	}
}

func TestCB_OnSuccessResetsFailures(t *testing.T) {
	threshold := 3
	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(threshold),
	)

	// Accumulate threshold-1 failures.
	for i := 0; i < threshold-1; i++ {
		target, _ := r.Route(ctx, infoWith("f"+string(rune('0'+i))))
		r.OnError(ctx, infoWith("f"+string(rune('0'+i))), target, SendError{StatusCode: 500, Err: errors.New("fail")})
	}

	// One success should reset the counter.
	target, _ := r.Route(ctx, infoWith("s1"))
	r.OnSuccess(ctx, infoWith("s1"), target)

	// Now threshold-1 more failures should not trip.
	for i := 0; i < threshold-1; i++ {
		target, _ = r.Route(ctx, infoWith("g"+string(rune('0'+i))))
		r.OnError(ctx, infoWith("g"+string(rune('0'+i))), target, SendError{StatusCode: 500, Err: errors.New("fail")})
	}

	// Should still be routable.
	_, err := r.Route(ctx, infoWith("final"))
	if err != nil {
		t.Fatalf("circuit should not have opened: %v", err)
	}
}

func TestCB_AllCircuitsOpen(t *testing.T) {
	r := NewCircuitBreakerRouter(
		makeCandidates("a", "b"),
		WithFailureThreshold(1),
	)

	// Trip both circuits.
	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	target, _ = r.Route(ctx, infoWith("r2"))
	if target.BaseURL != "b" {
		t.Fatalf("expected 'b', got %q", target.BaseURL)
	}
	r.OnError(ctx, infoWith("r2"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	_, err := r.Route(ctx, infoWith("r3"))
	if err == nil {
		t.Fatal("expected error when all circuits are open")
	}
}

func TestCB_FallbackToNextCandidate(t *testing.T) {
	r := NewCircuitBreakerRouter(makeCandidates("a", "b", "c"))

	target, err := r.Route(ctx, infoWith("r1"))
	if err != nil {
		t.Fatal(err)
	}
	if target.BaseURL != "a" {
		t.Fatalf("expected 'a', got %q", target.BaseURL)
	}

	// Error on "a" should fall back to "b".
	next, err := r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 500, Err: errors.New("fail")})
	if err != nil {
		t.Fatal(err)
	}
	if next.BaseURL != "b" {
		t.Fatalf("expected fallback to 'b', got %q", next.BaseURL)
	}

	// Error on "b" should fall back to "c".
	next, err = r.OnError(ctx, infoWith("r1"), next, SendError{StatusCode: 500, Err: errors.New("fail")})
	if err != nil {
		t.Fatal(err)
	}
	if next.BaseURL != "c" {
		t.Fatalf("expected fallback to 'c', got %q", next.BaseURL)
	}

	// Error on "c" should fail — all attempted.
	_, err = r.OnError(ctx, infoWith("r1"), next, SendError{StatusCode: 500, Err: errors.New("fail")})
	if err == nil {
		t.Fatal("expected error when all candidates exhausted")
	}
}

func TestCB_CrossRequestIsolation(t *testing.T) {
	r := NewCircuitBreakerRouter(makeCandidates("a", "b"))

	// Request 1 routes to "a".
	t1, _ := r.Route(ctx, infoWith("req1"))
	if t1.BaseURL != "a" {
		t.Fatalf("req1 expected 'a', got %q", t1.BaseURL)
	}

	// Request 2 should independently route to "a" as well.
	t2, _ := r.Route(ctx, infoWith("req2"))
	if t2.BaseURL != "a" {
		t.Fatalf("req2 expected 'a', got %q", t2.BaseURL)
	}

	// Error on req1 falls back to "b".
	next, err := r.OnError(ctx, infoWith("req1"), t1, SendError{StatusCode: 500, Err: errors.New("fail")})
	if err != nil {
		t.Fatal(err)
	}
	if next.BaseURL != "b" {
		t.Fatalf("req1 fallback expected 'b', got %q", next.BaseURL)
	}

	// req2 is still on "a", success should work fine.
	r.OnSuccess(ctx, infoWith("req2"), t2)
}

func TestCB_CustomKeyFunc(t *testing.T) {
	// Key by Model instead of BaseURL.
	r := NewCircuitBreakerRouter(
		func(_ RouteInfo) []RouteResult {
			return []RouteResult{
				{BaseURL: "url1", Model: "m1"},
				{BaseURL: "url2", Model: "m2"},
			}
		},
		WithCircuitKeyFunc(func(rr RouteResult) string { return rr.Model }),
		WithFailureThreshold(1),
	)

	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	// "m1" circuit should be open; should route to "m2".
	rr, err := r.Route(ctx, infoWith("r2"))
	if err != nil {
		t.Fatal(err)
	}
	if rr.Model != "m2" {
		t.Fatalf("expected model 'm2', got %q", rr.Model)
	}
}

func TestCB_OnStateChangeCallback(t *testing.T) {
	var calls []CircuitState
	var mu sync.Mutex

	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(1),
		WithRecoveryTimeout(1*time.Millisecond),
		WithSuccessThreshold(1),
		WithOnStateChange(func(_ string, _, to CircuitState) {
			mu.Lock()
			calls = append(calls, to)
			mu.Unlock()
		}),
	)

	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	time.Sleep(5 * time.Millisecond)

	target, _ = r.Route(ctx, infoWith("r2"))
	r.OnSuccess(ctx, infoWith("r2"), target)

	mu.Lock()
	defer mu.Unlock()
	expected := []CircuitState{CircuitOpen, CircuitHalfOpen, CircuitClosed}
	if len(calls) != len(expected) {
		t.Fatalf("expected %d callbacks, got %d: %v", len(expected), len(calls), calls)
	}
	for i, e := range expected {
		if calls[i] != e {
			t.Errorf("callback %d: expected %d, got %d", i, e, calls[i])
		}
	}
}

func TestCB_LazyCleanup(t *testing.T) {
	r := NewCircuitBreakerRouter(makeCandidates("a"))
	cbr := r.(*CircuitBreakerRouter)

	// Manually insert a stale attempt record.
	cbr.attemptMu.Lock()
	cbr.attempts["stale"] = &attemptRecord{
		targets:   map[string]struct{}{"x": {}},
		createdAt: time.Now().Add(-10 * time.Minute),
	}
	cbr.attemptMu.Unlock()

	// Force the cleanup counter to trigger on next call.
	atomic.StoreInt64(&cbr.cleanupCount, cleanupInterval-1)

	// This Route call should trigger cleanup.
	_, _ = r.Route(ctx, infoWith("trigger"))

	cbr.attemptMu.Lock()
	_, exists := cbr.attempts["stale"]
	cbr.attemptMu.Unlock()
	if exists {
		t.Fatal("stale attempt record should have been cleaned up")
	}
}

func TestCB_DefaultOptions(t *testing.T) {
	r := NewCircuitBreakerRouter(makeCandidates("a"))
	cbr := r.(*CircuitBreakerRouter)
	if cbr.cfg.failureThreshold != 5 {
		t.Errorf("default failureThreshold: got %d, want 5", cbr.cfg.failureThreshold)
	}
	if cbr.cfg.recoveryTimeout != 30*time.Second {
		t.Errorf("default recoveryTimeout: got %v, want 30s", cbr.cfg.recoveryTimeout)
	}
	if cbr.cfg.successThreshold != 2 {
		t.Errorf("default successThreshold: got %d, want 2", cbr.cfg.successThreshold)
	}
	if cbr.cfg.halfOpenMax != 1 {
		t.Errorf("default halfOpenMax: got %d, want 1", cbr.cfg.halfOpenMax)
	}
}

func TestCB_HalfOpenFailureReopens(t *testing.T) {
	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(1),
		WithRecoveryTimeout(1*time.Millisecond),
		WithSuccessThreshold(3),
	)

	// Trip the circuit.
	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	time.Sleep(5 * time.Millisecond)

	// Circuit transitions to HalfOpen; probe allowed.
	target, err := r.Route(ctx, infoWith("r2"))
	if err != nil {
		t.Fatal(err)
	}

	// Failure in HalfOpen should immediately reopen.
	_, _ = r.OnError(ctx, infoWith("r2"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	// Should be Open again, route should fail.
	_, err = r.Route(ctx, infoWith("r3"))
	if err == nil {
		t.Fatal("expected error after half-open failure reopened circuit")
	}
}

func TestCB_CustomShouldTrip(t *testing.T) {
	// Only trip on 503.
	r := NewCircuitBreakerRouter(
		makeCandidates("a"),
		WithFailureThreshold(1),
		WithShouldTrip(func(se SendError) bool {
			return se.StatusCode == 503
		}),
	)

	target, _ := r.Route(ctx, infoWith("r1"))
	r.OnError(ctx, infoWith("r1"), target, SendError{StatusCode: 500, Err: errors.New("fail")})

	// 500 should not trip with custom shouldTrip.
	_, err := r.Route(ctx, infoWith("r2"))
	if err != nil {
		t.Fatal("500 should not trip with custom shouldTrip")
	}

	// 503 should trip.
	target, _ = r.Route(ctx, infoWith("r3"))
	r.OnError(ctx, infoWith("r3"), target, SendError{StatusCode: 503, Err: errors.New("service unavailable")})

	_, err = r.Route(ctx, infoWith("r4"))
	if err == nil {
		t.Fatal("expected error after 503 tripped circuit")
	}
}

func TestCB_ConcurrentAccess(t *testing.T) {
	r := NewCircuitBreakerRouter(
		makeCandidates("a", "b", "c"),
		WithFailureThreshold(2),
		WithRecoveryTimeout(1*time.Millisecond),
	)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			info := infoWith("conc_" + string(rune('A'+i%26)))
			target, err := r.Route(ctx, info)
			if err != nil {
				return
			}
			if i%3 == 0 {
				r.OnError(ctx, info, target, SendError{StatusCode: 500, Err: errors.New("fail")})
			} else {
				r.OnSuccess(ctx, info, target)
			}
		}(i)
	}
	wg.Wait()
	// If we get here without deadlock or panic, the test passes.
}
