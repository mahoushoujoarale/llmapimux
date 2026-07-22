package llmapimux

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"syscall"
	"time"
)

// Handler is a unified http.Handler that delegates protocol-specific behavior
// to an inboundCodec and routing decisions to a Router.
type Handler struct {
	codec             inboundCodec
	router            Router
	auth              Authenticator // nil = no auth
	stats             StatsReporter
	reqMod            RequestModifier // nil = no modification
	attemptController AttemptController
}

// buildSendError constructs a SendError from the error returned by Send/SendStream.
func buildSendError(err error, attemptNum int) SendError {
	se := SendError{
		AttemptNum: attemptNum,
		Err:        err,
	}

	// Check for upstream HTTP error.
	var upstreamErr *UpstreamHTTPError
	if errors.As(err, &upstreamErr) {
		se.StatusCode = upstreamErr.StatusCode
		se.Header = upstreamErr.Header
	}

	// Check for timeout.
	if errors.Is(err, context.DeadlineExceeded) {
		se.IsTimeout = true
	} else {
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			se.IsTimeout = true
		}
	}

	// Check for connection error.
	if errors.Is(err, syscall.ECONNREFUSED) {
		se.IsConnError = true
	} else {
		var netOpErr *net.OpError
		if errors.As(err, &netOpErr) {
			se.IsConnError = true
		}
	}

	return se
}

// retryLoopState holds all the shared state needed by the retry loop and its
// helper methods (error handling, streaming, non-streaming response writing).
type retryLoopState struct {
	h          *Handler
	w          http.ResponseWriter
	r          *http.Request
	req        *Request
	body       []byte
	info       RouteInfo
	startTime  time.Time
	stats      StatsReporter
	target     RouteResult
	attemptNum int

	totalRetryAttempts int
	totalQueueWait     time.Duration
}

// writeErrorAndComplete writes an error response to the client and fires OnComplete.
// Used by handleSendError for the two terminal error paths (context canceled, no more fallback targets).
func (s *retryLoopState) writeErrorAndComplete(statusCode int, msg string, compErr error) {
	s.h.codec.WriteError(s.w, statusCode, msg)
	now := time.Now()
	s.stats.OnComplete(s.r.Context(), CompleteEvent{
		RequestID:        s.info.RequestID,
		Time:             now,
		Status:           resolveCompletionStatus(compErr),
		Error:            compErr,
		InboundProtocol:  s.info.InboundProtocol,
		OutboundProtocol: s.target.Protocol,
		TTFB:             0,
		TotalLatency:     now.Sub(s.startTime),
		Usage:            Usage{},
		OutputThroughput: 0,
		IRResponse:       nil,
		AttemptNum:       s.attemptNum,
		RetryAttempts:    s.totalRetryAttempts,
		QueueWait:        s.totalQueueWait,
	})
}

func (s *retryLoopState) acquireAttempt(retryAttempt int) (AttemptAdmission, error) {
	if s.h.attemptController == nil {
		return AttemptAdmission{}, nil
	}
	admission, err := s.h.attemptController.Acquire(s.r.Context(), s.info, s.target, s.attemptNum, retryAttempt)
	if admission.WaitDuration > 0 {
		s.totalQueueWait += admission.WaitDuration
	}
	return admission, err
}

func releaseAttempt(admission AttemptAdmission) {
	if admission.Permit != nil {
		admission.Permit.Release()
	}
}

func (s *retryLoopState) recordAttemptError(builtErr SendError, retryAttempt int, willRetry bool, retryDelay time.Duration) {
	s.stats.OnAttemptError(s.r.Context(), AttemptErrorEvent{
		RequestID:    s.info.RequestID,
		AttemptNum:   s.attemptNum,
		RetryAttempt: retryAttempt,
		Target:       s.target,
		SendErr:      builtErr,
		WillRetry:    willRetry,
		RetryDelay:   retryDelay,
	})
}

func (s *retryLoopState) shouldRetry(builtErr SendError, retryAttempt int) (time.Duration, bool) {
	if s.h.attemptController == nil {
		return 0, false
	}
	return s.h.attemptController.RetryDelay(s.r.Context(), s.info, s.target, builtErr, s.attemptNum, retryAttempt)
}

func (s *retryLoopState) sleepBeforeRetry(delay time.Duration) bool {
	if delay <= 0 {
		return true
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()

	select {
	case <-timer.C:
		return true
	case <-s.r.Context().Done():
		s.writeErrorAndComplete(499, "context canceled", s.r.Context().Err())
		return false
	}
}

func (s *retryLoopState) writeAcquireErrorAndComplete(err error) {
	statusCode := http.StatusServiceUnavailable
	msg := "attempt controller error: " + err.Error()
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		statusCode = 499
		msg = "context canceled"
	}
	s.writeErrorAndComplete(statusCode, msg, err)
}

// handleTerminalSendError processes a send error from either the streaming or
// non-streaming path after same-target retries have been exhausted. It calls
// Router.OnError and returns (nextTarget, true) if the
// caller should continue to the next attempt, or (zero, false) if it should return after
// writing the error response. The errPrefix is used for the user-facing error message.
// If sendErr is nil (client == nil case), it creates a synthetic error.
func (s *retryLoopState) handleTerminalSendError(sendErr error, builtErr SendError, errPrefix string) (RouteResult, bool) {
	// Check context cancellation — do not retry.
	if s.r.Context().Err() != nil {
		statusCode := 499
		if sendErr != nil {
			statusCode = resolveUpstreamStatusCode(sendErr)
		}
		msg := "context canceled"
		if sendErr != nil {
			msg = errPrefix + sendErr.Error()
		}
		s.writeErrorAndComplete(statusCode, msg, s.r.Context().Err())
		return RouteResult{}, false
	}

	nextTarget, retryErr := s.h.router.OnError(s.r.Context(), s.info, s.target, builtErr)
	if retryErr != nil {
		statusCode := 502
		if sendErr != nil {
			statusCode = resolveUpstreamStatusCode(sendErr)
		}
		msg := builtErr.Err.Error()
		if sendErr != nil {
			msg = errPrefix + sendErr.Error()
		}
		s.writeErrorAndComplete(statusCode, msg, sendErr)
		return RouteResult{}, false
	}

	// Advance to next target and re-populate RawExtra if protocol matches inbound.
	s.target = nextTarget
	s.attemptNum++
	if s.req.InboundProtocol == s.target.Protocol {
		populateRawExtraIfNeeded(s.req, s.body, s.h.codec.KnownFields())
	} else {
		s.req.RawExtra = nil
	}
	return nextTarget, true
}

func (s *retryLoopState) handleSendError(sendErr error, errPrefix string, retryAttempt int) (retrySameTarget bool, continueFallback bool) {
	// Check context cancellation before asking the controller for retry advice.
	if s.r.Context().Err() != nil {
		statusCode := 499
		if sendErr != nil {
			statusCode = resolveUpstreamStatusCode(sendErr)
		}
		msg := "context canceled"
		if sendErr != nil {
			msg = errPrefix + sendErr.Error()
		}
		s.writeErrorAndComplete(statusCode, msg, s.r.Context().Err())
		return false, false
	}

	builtErr := buildSendError(sendErr, s.attemptNum)
	if retryDelay, ok := s.shouldRetry(builtErr, retryAttempt); ok {
		s.recordAttemptError(builtErr, retryAttempt, true, retryDelay)
		if !s.sleepBeforeRetry(retryDelay) {
			return false, false
		}
		s.totalRetryAttempts++
		return true, true
	}

	s.recordAttemptError(builtErr, retryAttempt, false, 0)
	_, ok := s.handleTerminalSendError(sendErr, builtErr, errPrefix)
	return false, ok
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// 1. Extract API key
	apiKey := h.codec.ExtractAPIKey(r)

	// 2. Authenticate (optional)
	if h.auth != nil {
		if err := h.auth.Authenticate(r.Context(), apiKey); err != nil {
			h.codec.WriteError(w, 401, err.Error())
			return
		}
	}

	// 3. Read body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		h.codec.WriteError(w, 502, "failed to read request body")
		return
	}

	// 4. Decode to IR (stream detection is codec-specific)
	req, err := h.codec.DecodeRequest(r, body)
	if err != nil {
		h.codec.WriteError(w, 400, "failed to decode request: "+err.Error())
		return
	}

	stats := h.stats
	if stats == nil {
		stats = NoopStatsReporter{}
	}

	// 5. Build route info and route
	requestID := generateRequestID()
	inboundProtocol := h.codec.Protocol()
	info := RouteInfo{
		RequestID:       requestID,
		Model:           req.Model,
		InboundProtocol: inboundProtocol,
		Stream:          req.Stream,
		HasTools:        len(req.Tools) > 0,
		HasMedia:        hasMediaContent(req),
		APIKey:          apiKey,
	}
	result, err := h.router.Route(r.Context(), info)
	if err != nil {
		h.codec.WriteError(w, 400, err.Error())
		return
	}

	// On-demand RawExtra extraction is only needed for same-protocol preservation.
	if req.InboundProtocol == result.Protocol {
		populateRawExtraIfNeeded(req, body, h.codec.KnownFields())
	}

	// 6. Apply route result (OriginalModel set once, Model updated per attempt)
	req.OriginalModel = req.Model
	req.Model = result.Model

	startTime := time.Now()
	// OnRequestStart fires ONCE before the retry loop with the primary target.
	stats.OnRequestStart(r.Context(), RequestStartEvent{
		RequestID:        requestID,
		Time:             startTime,
		InboundProtocol:  inboundProtocol,
		OutboundProtocol: result.Protocol,
		Streaming:        req.Stream,
		IRRequest:        req,
	})

	// 7. Retry loop
	loop := &retryLoopState{
		h:          h,
		w:          w,
		r:          r,
		req:        req,
		body:       body,
		info:       info,
		startTime:  startTime,
		stats:      stats,
		target:     result,
		attemptNum: 1,
	}

	for {
		target := loop.target
		client := NewClient(target.Protocol)
		if client == nil {
			// Treat unsupported protocol as a send error so the router can fallback.
			syntheticErr := errors.New("unsupported outbound protocol")
			builtErr := buildSendError(syntheticErr, loop.attemptNum)
			loop.recordAttemptError(builtErr, 0, false, 0)
			if _, ok := loop.handleTerminalSendError(syntheticErr, builtErr, ""); !ok {
				return
			}
			continue
		}

		for retryAttempt := 0; ; retryAttempt++ {
			req.Model = loop.target.Model

			// Call RequestModifier to allow caller to set OutboundExtra per attempt.
			req.OutboundExtra = nil
			if h.reqMod != nil {
				h.reqMod(r.Context(), req, loop.target)
			}

			admission, err := loop.acquireAttempt(retryAttempt)
			if err != nil {
				releaseAttempt(admission)
				loop.writeAcquireErrorAndComplete(err)
				return
			}

			outCfg := OutboundConfig{BaseURL: loop.target.BaseURL, APIKey: loop.target.APIKey, ProxyURL: loop.target.ProxyURL, Header: loop.target.Header}

			if req.Stream {
				ch, sendErr := client.SendStream(r.Context(), req, outCfg)
				if sendErr != nil {
					releaseAttempt(admission)
					retrySameTarget, ok := loop.handleSendError(sendErr, "upstream stream error: ", retryAttempt)
					if !ok {
						return
					}
					if retrySameTarget {
						continue
					}
					break
				}

				// SendStream succeeded — commit to streaming. No more fallback.
				h.router.OnSuccess(r.Context(), info, loop.target)
				loop.handleStreaming(ch)
				releaseAttempt(admission)
				return
			}

			// Non-streaming path.
			resp, sendErr := client.Send(r.Context(), req, outCfg)
			releaseAttempt(admission)
			if sendErr != nil {
				retrySameTarget, ok := loop.handleSendError(sendErr, "upstream error: ", retryAttempt)
				if !ok {
					return
				}
				if retrySameTarget {
					continue
				}
				break
			}

			// Non-streaming success — record TTFB immediately after Send returns.
			firstByteTime := time.Now()
			h.router.OnSuccess(r.Context(), info, loop.target)
			loop.handleNonStreaming(resp, firstByteTime)
			return
		}
	}
}

func (s *retryLoopState) handleNonStreaming(resp *Response, firstByteTime time.Time) {
	var (
		err  error
		ttfb = firstByteTime.Sub(s.startTime)
	)

	defer func() {
		now := time.Now()
		totalLatency := now.Sub(s.startTime)
		usage := Usage{}
		stopReason := StopReason("")
		actualModel := ""
		if resp != nil {
			usage = resp.Usage
			stopReason = resp.StopReason
			actualModel = resp.Model
		}
		throughput := 0.0
		if totalLatency > 0 {
			throughput = float64(usage.CompletionTokens) / totalLatency.Seconds()
		}
		tpot := time.Duration(0)
		if usage.CompletionTokens > 0 && totalLatency > ttfb {
			tpot = (totalLatency - ttfb) / time.Duration(usage.CompletionTokens)
		}
		s.stats.OnComplete(s.r.Context(), CompleteEvent{
			RequestID:        s.info.RequestID,
			Time:             now,
			Status:           resolveCompletionStatus(err),
			Error:            err,
			InboundProtocol:  s.info.InboundProtocol,
			OutboundProtocol: s.target.Protocol,
			TTFB:             ttfb,
			TotalLatency:     totalLatency,
			Usage:            usage,
			OutputThroughput: throughput,
			TPOT:             tpot,
			StopReason:       stopReason,
			ActualModel:      actualModel,
			IRResponse:       resp,
			AttemptNum:       s.attemptNum,
			RetryAttempts:    s.totalRetryAttempts,
			QueueWait:        s.totalQueueWait,
		})
	}()

	s.stats.OnFirstByte(s.r.Context(), FirstByteEvent{
		RequestID: s.info.RequestID,
		Time:      firstByteTime,
		TTFB:      ttfb,
	})

	data, err := s.h.codec.EncodeResponse(resp)
	if err != nil {
		s.h.codec.WriteError(s.w, 502, "failed to encode response: "+err.Error())
		return
	}
	// Merge RawExtra only for same-protocol roundtrip
	if s.req.InboundProtocol == s.target.Protocol {
		data, _ = mergeRawExtra(data, s.req.RawExtra, s.h.codec.KnownFields())
	}
	s.w.Header().Set("Content-Type", "application/json")
	s.w.WriteHeader(http.StatusOK)
	_, err = s.w.Write(data)
}

func resolveCompletionStatus(err error) CompletionStatus {
	switch {
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return CompletionStatusCanceled
	case err != nil:
		return CompletionStatusError
	default:
		return CompletionStatusSuccess
	}
}

type streamSummary struct {
	ttfb        time.Duration
	usage       Usage
	stopReason  StopReason
	actualModel string
	streamErr   error
	chunks      int
}

func (s *retryLoopState) wrapStreamForStats(ch <-chan StreamResult) (<-chan StreamResult, <-chan streamSummary, chan struct{}) {
	ctx := s.r.Context()
	wrappedCh := make(chan StreamResult)
	summaryDone := make(chan streamSummary, 1)
	stopForwarding := make(chan struct{})

	go func() {
		defer close(wrappedCh)
		var summary streamSummary
		firstByteSent := false
		lastChunkTime := s.startTime
		seq := 0

		for {
			select {
			case <-stopForwarding:
				summaryDone <- summary
				close(summaryDone)
				return
			case <-ctx.Done():
				if summary.streamErr == nil {
					summary.streamErr = ctx.Err()
				}
				summaryDone <- summary
				close(summaryDone)
				return
			case result, ok := <-ch:
				if !ok {
					summaryDone <- summary
					close(summaryDone)
					return
				}

				if result.Err != nil {
					summary.streamErr = result.Err
					select {
					case wrappedCh <- result:
					case <-stopForwarding:
					}
					summaryDone <- summary
					close(summaryDone)
					return
				}
				if result.Event == nil {
					select {
					case wrappedCh <- result:
					case <-stopForwarding:
					}
					continue
				}

				now := time.Now()
				if !firstByteSent {
					firstByteSent = true
					summary.ttfb = now.Sub(s.startTime)
					s.stats.OnFirstByte(ctx, FirstByteEvent{
						RequestID: s.info.RequestID,
						Time:      now,
						TTFB:      summary.ttfb,
					})
				}

				seq++
				summary.chunks++
				interChunkDelay := time.Duration(0)
				if seq > 1 {
					interChunkDelay = now.Sub(lastChunkTime)
				}
				s.stats.OnStreamChunk(ctx, StreamChunkEvent{
					RequestID:       s.info.RequestID,
					Time:            now,
					SequenceNum:     seq,
					ElapsedTime:     now.Sub(s.startTime),
					InterChunkDelay: interChunkDelay,
					IREvent:         result.Event,
				})
				lastChunkTime = now

				if result.Event.Usage != nil {
					mergeStreamUsage(&summary.usage, result.Event.Usage)
				}
				if result.Event.Response != nil {
					// Some protocols (e.g. Anthropic message_start) carry usage
					// inside Response.Usage rather than the top-level Usage field.
					mergeStreamUsage(&summary.usage, &result.Event.Response.Usage)
					if result.Event.Response.Model != "" {
						summary.actualModel = result.Event.Response.Model
					}
				}
				if result.Event.StopReason != nil {
					summary.stopReason = *result.Event.StopReason
				}

				select {
				case wrappedCh <- result:
				case <-stopForwarding:
					summaryDone <- summary
					close(summaryDone)
					return
				}
			}
		}
	}()

	return wrappedCh, summaryDone, stopForwarding
}

func (s *retryLoopState) handleStreaming(ch <-chan StreamResult) {
	wrappedCh, summaryDone, stopForwarding := s.wrapStreamForStats(ch)

	s.w.Header().Set("Content-Type", "text/event-stream")
	s.w.Header().Set("Cache-Control", "no-cache")
	s.w.WriteHeader(http.StatusOK)
	s.h.codec.WriteStreamingResponse(NewSSEWriter(s.w), wrappedCh)

	close(stopForwarding)
	summary := <-summaryDone
	err := summary.streamErr
	if ctxErr := s.r.Context().Err(); errors.Is(ctxErr, context.Canceled) || errors.Is(ctxErr, context.DeadlineExceeded) {
		err = ctxErr
	}

	now := time.Now()
	totalLatency := now.Sub(s.startTime)
	throughput := 0.0
	if totalLatency > 0 {
		throughput = float64(summary.usage.CompletionTokens) / totalLatency.Seconds()
	}
	tpot := time.Duration(0)
	if summary.usage.CompletionTokens > 0 && totalLatency > summary.ttfb {
		tpot = (totalLatency - summary.ttfb) / time.Duration(summary.usage.CompletionTokens)
	}

	s.stats.OnComplete(s.r.Context(), CompleteEvent{
		RequestID:        s.info.RequestID,
		Time:             now,
		Status:           resolveCompletionStatus(err),
		Error:            err,
		InboundProtocol:  s.info.InboundProtocol,
		OutboundProtocol: s.target.Protocol,
		TTFB:             summary.ttfb,
		TotalLatency:     totalLatency,
		Usage:            summary.usage,
		OutputThroughput: throughput,
		TPOT:             tpot,
		Chunks:           summary.chunks,
		StopReason:       summary.stopReason,
		ActualModel:      summary.actualModel,
		IRResponse:       nil,
		AttemptNum:       s.attemptNum,
		RetryAttempts:    s.totalRetryAttempts,
		QueueWait:        s.totalQueueWait,
	})
}

// mergeStreamUsage merges non-zero fields from src into dst.
// This handles protocols like Anthropic where usage is split across
// multiple streaming events (message_start has input tokens, message_delta has output tokens).
func mergeStreamUsage(dst *Usage, src *Usage) {
	if src.PromptTokens != 0 {
		dst.PromptTokens = src.PromptTokens
	}
	if src.PromptCacheHitTokens != 0 {
		dst.PromptCacheHitTokens = src.PromptCacheHitTokens
	}
	if src.PromptCacheWriteTokens != 0 {
		dst.PromptCacheWriteTokens = src.PromptCacheWriteTokens
	}
	if src.PromptAudioTokens != 0 {
		dst.PromptAudioTokens = src.PromptAudioTokens
	}
	if src.CompletionTokens != 0 {
		dst.CompletionTokens = src.CompletionTokens
	}
	if src.CompletionReasoningTokens != 0 {
		dst.CompletionReasoningTokens = src.CompletionReasoningTokens
	}
	if src.CompletionAudioTokens != 0 {
		dst.CompletionAudioTokens = src.CompletionAudioTokens
	}
	if src.CompletionAcceptedPrediction != 0 {
		dst.CompletionAcceptedPrediction = src.CompletionAcceptedPrediction
	}
	if src.CompletionRejectedPrediction != 0 {
		dst.CompletionRejectedPrediction = src.CompletionRejectedPrediction
	}
	if src.ServerToolUseTokens != 0 {
		dst.ServerToolUseTokens = src.ServerToolUseTokens
	}
	if src.TotalTokens != 0 {
		dst.TotalTokens = src.TotalTokens
	}
}
