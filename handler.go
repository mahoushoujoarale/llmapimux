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
	codec  inboundCodec
	router Router
	auth   Authenticator  // nil = no auth
	stats  StatsReporter
	reqMod RequestModifier // nil = no modification
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

// retryLoopState holds all the shared state needed by the retry loop error-handling helper.
type retryLoopState struct {
	h               *Handler
	w               http.ResponseWriter
	r               *http.Request
	req             *Request
	body            []byte
	info            RouteInfo
	requestID       string
	inboundProtocol Protocol
	startTime       time.Time
	stats           StatsReporter
	target          RouteResult
	attemptNum      int
}

// handleSendError processes a send error from either the streaming or non-streaming path.
// It fires OnAttemptError, calls Router.OnError, and returns (nextTarget, true) if the
// caller should continue to the next attempt, or (zero, false) if it should return after
// writing the error response. The errPrefix is used for the user-facing error message.
// If sendErr is nil (client == nil case), it creates a synthetic error.
func (s *retryLoopState) handleSendError(sendErr error, errPrefix string) (RouteResult, bool) {
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
		s.h.codec.WriteError(s.w, statusCode, msg)
		now := time.Now()
		s.stats.OnComplete(s.r.Context(), CompleteEvent{
			RequestID:        s.requestID,
			Time:             now,
			Status:           resolveCompletionStatus(s.r.Context().Err()),
			Error:            s.r.Context().Err(),
			InboundProtocol:  s.inboundProtocol,
			OutboundProtocol: s.target.Protocol,
			TTFB:             0,
			TotalLatency:     now.Sub(s.startTime),
			Usage:            Usage{},
			OutputThroughput: 0,
			IRResponse:       nil,
			AttemptNum:       s.attemptNum,
		})
		return RouteResult{}, false
	}

	builtErr := buildSendError(sendErr, s.attemptNum)
	s.stats.OnAttemptError(s.r.Context(), AttemptErrorEvent{
		RequestID:  s.requestID,
		AttemptNum: s.attemptNum,
		Target:     s.target,
		SendErr:    builtErr,
	})

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
		s.h.codec.WriteError(s.w, statusCode, msg)
		now := time.Now()
		s.stats.OnComplete(s.r.Context(), CompleteEvent{
			RequestID:        s.requestID,
			Time:             now,
			Status:           resolveCompletionStatus(sendErr),
			Error:            sendErr,
			InboundProtocol:  s.inboundProtocol,
			OutboundProtocol: s.target.Protocol,
			TTFB:             0,
			TotalLatency:     now.Sub(s.startTime),
			Usage:            Usage{},
			OutputThroughput: 0,
			IRResponse:       nil,
			AttemptNum:       s.attemptNum,
		})
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
		h:               h,
		w:               w,
		r:               r,
		req:             req,
		body:            body,
		info:            info,
		requestID:       requestID,
		inboundProtocol: inboundProtocol,
		startTime:       startTime,
		stats:           stats,
		target:          result,
		attemptNum:      1,
	}

	for {
		target := loop.target
		client := NewClient(target.Protocol)
		if client == nil {
			// Treat unsupported protocol as a send error so the router can fallback.
			syntheticErr := errors.New("unsupported outbound protocol")
			if _, ok := loop.handleSendError(syntheticErr, ""); !ok {
				return
			}
			continue
		}

		req.Model = loop.target.Model

		// Call RequestModifier to allow caller to set OutboundExtra per attempt.
		req.OutboundExtra = nil
		if h.reqMod != nil {
			h.reqMod(r.Context(), req, loop.target)
		}

		outCfg := OutboundConfig{BaseURL: loop.target.BaseURL, APIKey: loop.target.APIKey, ProxyURL: loop.target.ProxyURL, Header: loop.target.Header}

		if req.Stream {
			ch, sendErr := client.SendStream(r.Context(), req, outCfg)
			if sendErr != nil {
				if _, ok := loop.handleSendError(sendErr, "upstream stream error: "); !ok {
					return
				}
				continue
			}

			// SendStream succeeded — commit to streaming. No more fallback.
			h.router.OnSuccess(r.Context(), info, loop.target)
			h.handleStreaming(w, r, req, ch, loop.target.Protocol, inboundProtocol, requestID, startTime, stats, loop.attemptNum)
			return
		}

		// Non-streaming path.
		resp, sendErr := client.Send(r.Context(), req, outCfg)
		if sendErr != nil {
			if _, ok := loop.handleSendError(sendErr, "upstream error: "); !ok {
				return
			}
			continue
		}

		// Non-streaming success — record TTFB immediately after Send returns.
		firstByteTime := time.Now()
		h.router.OnSuccess(r.Context(), info, loop.target)
		h.handleNonStreaming(w, r, req, resp, loop.target.Protocol, inboundProtocol, requestID, startTime, firstByteTime, stats, loop.attemptNum)
		return
	}
}

func (h *Handler) handleNonStreaming(w http.ResponseWriter, r *http.Request, req *Request, resp *Response, outboundProtocol Protocol, inboundProtocol Protocol, requestID string, startTime time.Time, firstByteTime time.Time, stats StatsReporter, attemptNum int) {
	var (
		err  error
		ttfb = firstByteTime.Sub(startTime)
	)

	defer func() {
		now := time.Now()
		totalLatency := now.Sub(startTime)
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
			throughput = float64(usage.OutputTokens) / totalLatency.Seconds()
		}
		stats.OnComplete(r.Context(), CompleteEvent{
			RequestID:        requestID,
			Time:             now,
			Status:           resolveCompletionStatus(err),
			Error:            err,
			InboundProtocol:  inboundProtocol,
			OutboundProtocol: outboundProtocol,
			TTFB:             ttfb,
			TotalLatency:     totalLatency,
			Usage:            usage,
			OutputThroughput: throughput,
			StopReason:       stopReason,
			ActualModel:      actualModel,
			IRResponse:       resp,
			AttemptNum:       attemptNum,
		})
	}()

	stats.OnFirstByte(r.Context(), FirstByteEvent{
		RequestID: requestID,
		Time:      firstByteTime,
		TTFB:      ttfb,
	})

	data, err := h.codec.EncodeResponse(resp)
	if err != nil {
		h.codec.WriteError(w, 502, "failed to encode response: "+err.Error())
		return
	}
	// Merge RawExtra only for same-protocol roundtrip
	if req.InboundProtocol == outboundProtocol {
		data, _ = mergeRawExtra(data, req.RawExtra, h.codec.KnownFields())
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, err = w.Write(data)
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
}

func (h *Handler) wrapStreamForStats(ctx context.Context, ch <-chan StreamResult, requestID string, startTime time.Time, stats StatsReporter) (<-chan StreamResult, <-chan streamSummary, chan struct{}) {
	wrappedCh := make(chan StreamResult)
	summaryDone := make(chan streamSummary, 1)
	stopForwarding := make(chan struct{})

	go func() {
		defer close(wrappedCh)
		var summary streamSummary
		firstByteSent := false
		lastChunkTime := startTime
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
					summary.ttfb = now.Sub(startTime)
					stats.OnFirstByte(ctx, FirstByteEvent{
						RequestID: requestID,
						Time:      now,
						TTFB:      summary.ttfb,
					})
				}

				seq++
				interChunkDelay := time.Duration(0)
				if seq > 1 {
					interChunkDelay = now.Sub(lastChunkTime)
				}
				stats.OnStreamChunk(ctx, StreamChunkEvent{
					RequestID:       requestID,
					Time:            now,
					SequenceNum:     seq,
					ElapsedTime:     now.Sub(startTime),
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

func (h *Handler) handleStreaming(w http.ResponseWriter, r *http.Request, _ *Request, ch <-chan StreamResult, outboundProtocol Protocol, inboundProtocol Protocol, requestID string, startTime time.Time, stats StatsReporter, attemptNum int) {
	wrappedCh, summaryDone, stopForwarding := h.wrapStreamForStats(r.Context(), ch, requestID, startTime, stats)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	h.codec.WriteStreamingResponse(NewSSEWriter(w), wrappedCh)

	close(stopForwarding)
	summary := <-summaryDone
	err := summary.streamErr
	if ctxErr := r.Context().Err(); errors.Is(ctxErr, context.Canceled) || errors.Is(ctxErr, context.DeadlineExceeded) {
		err = ctxErr
	}

	now := time.Now()
	totalLatency := now.Sub(startTime)
	throughput := 0.0
	if totalLatency > 0 {
		throughput = float64(summary.usage.OutputTokens) / totalLatency.Seconds()
	}

	stats.OnComplete(r.Context(), CompleteEvent{
		RequestID:        requestID,
		Time:             now,
		Status:           resolveCompletionStatus(err),
		Error:            err,
		InboundProtocol:  inboundProtocol,
		OutboundProtocol: outboundProtocol,
		TTFB:             summary.ttfb,
		TotalLatency:     totalLatency,
		Usage:            summary.usage,
		OutputThroughput: throughput,
		StopReason:       summary.stopReason,
		ActualModel:      summary.actualModel,
		IRResponse:       nil,
		AttemptNum:       attemptNum,
	})
}

// mergeStreamUsage merges non-zero fields from src into dst.
// This handles protocols like Anthropic where usage is split across
// multiple streaming events (message_start has InputTokens, message_delta has OutputTokens).
func mergeStreamUsage(dst *Usage, src *Usage) {
	if src.InputTokens != 0 {
		dst.InputTokens = src.InputTokens
	}
	if src.OutputTokens != 0 {
		dst.OutputTokens = src.OutputTokens
	}
	if src.TotalTokens != 0 {
		dst.TotalTokens = src.TotalTokens
	}
	if src.CacheReadTokens != 0 {
		dst.CacheReadTokens = src.CacheReadTokens
	}
	if src.CacheCreationTokens != 0 {
		dst.CacheCreationTokens = src.CacheCreationTokens
	}
	if src.ThinkingTokens != 0 {
		dst.ThinkingTokens = src.ThinkingTokens
	}
}
