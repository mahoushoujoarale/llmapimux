package llmapimux

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Helper: build an Anthropic-style split-usage stream (the hardest case for
// codecs that normally receive all usage in a single event).
//
// Anthropic streaming sends:
//   message_start  → StreamEventStart  { Response.Usage: {PromptTokens, ...} }
//   content deltas → StreamEventDelta  { Delta: text/tool }
//   message_delta  → StreamEventDelta  { StopReason, Usage: {CompletionTokens} }
//   message_stop   → StreamEventStop   { StopReason: nil }
//
// This is the canonical test case for usage/stop_reason deferral.
// ---------------------------------------------------------------------------

// anthropicStyleStreamEvents returns a sequence of StreamResults simulating
// an Anthropic streaming response with full usage and an end_turn stop reason.
func anthropicStyleStreamEvents() []StreamResult {
	endTurn := StopReasonEndTurn
	return []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "msg_test",
				Model: "claude-sonnet-4-20250514",
				Usage: Usage{
					PromptTokens:          50,
					PromptCacheHitTokens:  10,
					PromptCacheWriteTokens: 5,
				},
			},
		}},
		{Event: &StreamEvent{
			Type:  StreamEventDelta,
			Index: 0,
			Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: "Hello world"}},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventDelta,
			StopReason: &endTurn,
			Usage:      &Usage{CompletionTokens: 7},
		}},
		{Event: &StreamEvent{
			Type: StreamEventStop,
		}},
	}
}

// anthropicStyleToolUseStreamEvents returns a stream where the stop reason is
// tool_use (the most critical case — must NOT be reported as "stop").
func anthropicStyleToolUseStreamEvents() []StreamResult {
	toolUse := StopReasonToolUse
	return []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "msg_tool",
				Model: "claude-sonnet-4-20250514",
				Usage: Usage{PromptTokens: 80},
			},
		}},
		{Event: &StreamEvent{
			Type:  StreamEventDelta,
			Index: 0,
			Delta: &ContentPart{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{ID: "tu_1", Name: "get_weather"}},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventDelta,
			StopReason: &toolUse,
			Usage:      &Usage{CompletionTokens: 15},
		}},
		{Event: &StreamEvent{
			Type: StreamEventStop,
		}},
	}
}

// nativeOpenAIChatStreamEvents returns a stream where usage already arrives
// on the stop event (native OpenAI Chat format). The codec should NOT double-
// count this usage.
func nativeOpenAIChatStreamEvents() []StreamResult {
	stop := StopReasonEndTurn
	return []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "chatcmpl-1",
				Model: "gpt-4o",
			},
		}},
		{Event: &StreamEvent{
			Type:  StreamEventDelta,
			Index: 0,
			Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: "Hi"}},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventStop,
			StopReason: &stop,
			Usage: &Usage{
				PromptTokens:     30,
				CompletionTokens: 5,
			},
		}},
	}
}

// noUsageStreamEvents returns a stream with no usage at all.
func noUsageStreamEvents() []StreamResult {
	stop := StopReasonEndTurn
	return []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "msg_nousage",
				Model: "claude-sonnet-4-20250514",
			},
		}},
		{Event: &StreamEvent{
			Type:  StreamEventDelta,
			Index: 0,
			Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: "No usage"}},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventStop,
			StopReason: &stop,
		}},
	}
}

func sendStreamToCodec(codec inboundCodec, events []StreamResult) string {
	ch := make(chan StreamResult, len(events))
	for _, e := range events {
		ch <- e
	}
	close(ch)

	w := httptest.NewRecorder()
	sseWriter := NewSSEWriter(w)
	codec.WriteStreamingResponse(sseWriter, ch)
	return w.Body.String()
}

// ---------------------------------------------------------------------------
// OpenAI Chat codec tests
// ---------------------------------------------------------------------------

func TestOpenAIChatCodec_Stream_AnthropicStyleUsage(t *testing.T) {
	body := sendStreamToCodec(&openaiChatCodec{}, anthropicStyleStreamEvents())

	// prompt_tokens must be deferred from message_start
	if !strings.Contains(body, `"prompt_tokens":50`) {
		t.Errorf("missing prompt_tokens=50 (deferred from message_start):\n%s", body)
	}
	// cached_tokens from prompt_tokens_details
	if !strings.Contains(body, `"cached_tokens":10`) {
		t.Errorf("missing cached_tokens=10:\n%s", body)
	}
	// completion_tokens from message_delta
	if !strings.Contains(body, `"completion_tokens":7`) {
		t.Errorf("missing completion_tokens=7:\n%s", body)
	}
	// finish_reason must be "stop" (end_turn → stop)
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish_reason 'stop':\n%s", body)
	}
	// [DONE] sentinel
	if !strings.Contains(body, "[DONE]") {
		t.Errorf("missing in SSE output:\n%s", body)
	}
}

func TestOpenAIChatCodec_Stream_ToolUseStopReason(t *testing.T) {
	body := sendStreamToCodec(&openaiChatCodec{}, anthropicStyleToolUseStreamEvents())

	// finish_reason must be "tool_calls", NOT "stop"
	if !strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Errorf("finish_reason should be 'tool_calls', got something else:\n%s", body)
	}
	// prompt_tokens must be deferred from message_start
	if !strings.Contains(body, `"prompt_tokens":80`) {
		t.Errorf("missing prompt_tokens=80:\n%s", body)
	}
}

func TestOpenAIChatCodec_Stream_NativeOpenAIChat(t *testing.T) {
	body := sendStreamToCodec(&openaiChatCodec{}, nativeOpenAIChatStreamEvents())

	// usage is already on the stop event — should appear exactly once
	count := strings.Count(body, `"prompt_tokens":30`)
	if count != 1 {
		t.Errorf("prompt_tokens=30 should appear exactly once, got %d:\n%s", count, body)
	}
	if !strings.Contains(body, `"completion_tokens":5`) {
		t.Errorf("missing completion_tokens=5:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish_reason 'stop':\n%s", body)
	}
}

func TestOpenAIChatCodec_Stream_NoUsage(t *testing.T) {
	body := sendStreamToCodec(&openaiChatCodec{}, noUsageStreamEvents())

	// Should still have finish_reason
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish_reason 'stop':\n%s", body)
	}
	// Should NOT have usage section at all
	if strings.Contains(body, `"prompt_tokens"`) {
		t.Errorf("should not have prompt_tokens when no usage:\n%s", body)
	}
}

func TestOpenAIChatCodec_Stream_ContentFilterStopReason(t *testing.T) {
	contentFilter := StopReasonContentFilter
	events := []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "msg_cf",
				Model: "claude-sonnet-4-20250514",
				Usage: Usage{PromptTokens: 20},
			},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventDelta,
			StopReason: &contentFilter,
			Usage:      &Usage{CompletionTokens: 0},
		}},
		{Event: &StreamEvent{
			Type: StreamEventStop,
		}},
	}

	body := sendStreamToCodec(&openaiChatCodec{}, events)

	if !strings.Contains(body, `"finish_reason":"content_filter"`) {
		t.Errorf("finish_reason should be 'content_filter', got:\n%s", body)
	}
}

// ---------------------------------------------------------------------------
// OpenAI Responses codec tests
// ---------------------------------------------------------------------------

func TestOpenAIResponsesCodec_Stream_AnthropicStyleUsage(t *testing.T) {
	body := sendStreamToCodec(&openaiResponsesCodec{}, anthropicStyleStreamEvents())

	// response.completed should carry the deferred usage
	if !strings.Contains(body, `"input_tokens":50`) {
		t.Errorf("missing input_tokens=50 (deferred from message_start):\n%s", body)
	}
	if !strings.Contains(body, `"output_tokens":7`) {
		t.Errorf("missing output_tokens=7:\n%s", body)
	}
	// status should be "completed" (end_turn → completed)
	if !strings.Contains(body, `"status":"completed"`) {
		t.Errorf("missing status 'completed':\n%s", body)
	}
}

func TestOpenAIResponsesCodec_Stream_ToolUseStopReason(t *testing.T) {
	body := sendStreamToCodec(&openaiResponsesCodec{}, anthropicStyleToolUseStreamEvents())

	// tool_use → status "completed" (not "incomplete" or "failed")
	if !strings.Contains(body, `"status":"completed"`) {
		t.Errorf("status should be 'completed' for tool_use:\n%s", body)
	}
	if !strings.Contains(body, `"input_tokens":80`) {
		t.Errorf("missing input_tokens=80:\n%s", body)
	}
}

func TestOpenAIResponsesCodec_Stream_ContentFilterStopReason(t *testing.T) {
	contentFilter := StopReasonContentFilter
	events := []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "msg_cf_resp",
				Model: "claude-sonnet-4-20250514",
				Usage: Usage{PromptTokens: 20},
			},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventDelta,
			StopReason: &contentFilter,
		}},
		{Event: &StreamEvent{
			Type: StreamEventStop,
		}},
	}

	body := sendStreamToCodec(&openaiResponsesCodec{}, events)

	// content_filter → status "failed"
	if !strings.Contains(body, `"status":"failed"`) {
		t.Errorf("status should be 'failed' for content_filter:\n%s", body)
	}
}

func TestOpenAIResponsesCodec_Stream_MaxTokensStopReason(t *testing.T) {
	maxTokens := StopReasonMaxTokens
	events := []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "msg_max",
				Model: "claude-sonnet-4-20250514",
				Usage: Usage{PromptTokens: 30},
			},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventDelta,
			StopReason: &maxTokens,
			Usage:      &Usage{CompletionTokens: 100},
		}},
		{Event: &StreamEvent{
			Type: StreamEventStop,
		}},
	}

	body := sendStreamToCodec(&openaiResponsesCodec{}, events)

	// max_tokens → status "incomplete"
	if !strings.Contains(body, `"status":"incomplete"`) {
		t.Errorf("status should be 'incomplete' for max_tokens:\n%s", body)
	}
}

// ---------------------------------------------------------------------------
// Gemini codec tests
// ---------------------------------------------------------------------------

func TestGeminiCodec_Stream_AnthropicStyleUsage(t *testing.T) {
	body := sendStreamToCodec(&geminiCodec{}, anthropicStyleStreamEvents())

	// Find the last SSE chunk with usageMetadata
	type geminiChunk struct {
		UsageMetadata *struct {
			PromptTokenCount        int `json:"promptTokenCount,omitempty"`
			CandidatesTokenCount   int `json:"candidatesTokenCount,omitempty"`
			CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
		} `json:"usageMetadata,omitempty"`
		Candidates []struct {
			FinishReason string `json:"finishReason,omitempty"`
		} `json:"candidates,omitempty"`
	}

	// Parse all SSE data lines
	var lastChunk geminiChunk
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var chunk geminiChunk
		if err := json.Unmarshal([]byte(line[6:]), &chunk); err != nil {
			continue
		}
		if chunk.UsageMetadata != nil || len(chunk.Candidates) > 0 {
			lastChunk = chunk
		}
	}

	if lastChunk.UsageMetadata == nil {
		t.Fatalf("no usageMetadata found in Gemini SSE output:\n%s", body)
	}
	if lastChunk.UsageMetadata.PromptTokenCount != 50 {
		t.Errorf("promptTokenCount = %d, want 50 (deferred from message_start)", lastChunk.UsageMetadata.PromptTokenCount)
	}
	if lastChunk.UsageMetadata.CandidatesTokenCount != 7 {
		t.Errorf("candidatesTokenCount = %d, want 7", lastChunk.UsageMetadata.CandidatesTokenCount)
	}
	if lastChunk.UsageMetadata.CachedContentTokenCount != 10 {
		t.Errorf("cachedContentTokenCount = %d, want 10", lastChunk.UsageMetadata.CachedContentTokenCount)
	}
	if lastChunk.Candidates[0].FinishReason != "STOP" {
		t.Errorf("finishReason = %q, want STOP (end_turn → STOP)", lastChunk.Candidates[0].FinishReason)
	}
}

func TestGeminiCodec_Stream_ToolUseStopReason(t *testing.T) {
	body := sendStreamToCodec(&geminiCodec{}, anthropicStyleToolUseStreamEvents())

	type geminiChunk struct {
		Candidates []struct {
			FinishReason string `json:"finishReason,omitempty"`
		} `json:"candidates,omitempty"`
	}

	var lastChunk geminiChunk
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var chunk geminiChunk
		if err := json.Unmarshal([]byte(line[6:]), &chunk); err != nil {
			continue
		}
		if len(chunk.Candidates) > 0 {
			lastChunk = chunk
		}
	}

	if len(lastChunk.Candidates) == 0 {
		t.Fatalf("no candidates found in Gemini SSE output:\n%s", body)
	}
	if lastChunk.Candidates[0].FinishReason != "STOP" {
		// Gemini maps tool_use → STOP (not a separate finish reason)
		t.Errorf("finishReason = %q, want STOP for tool_use", lastChunk.Candidates[0].FinishReason)
	}
}

// ---------------------------------------------------------------------------
// Anthropic codec tests
// ---------------------------------------------------------------------------

func TestAnthropicCodec_Stream_AnthropicStyleUsage(t *testing.T) {
	body := sendStreamToCodec(&anthropicCodec{}, anthropicStyleStreamEvents())

	// message_start should carry input tokens
	if !strings.Contains(body, "event: message_start") {
		t.Errorf("missing event: message_start:\n%s", body)
	}
	// input_tokens = PromptTokens(50) - CacheWrite(5) - CacheRead(10) = 35
	if !strings.Contains(body, `"input_tokens":35`) {
		t.Errorf("missing input_tokens=35:\n%s", body)
	}
	// cache_creation_input_tokens = 5
	if !strings.Contains(body, `"cache_creation_input_tokens":5`) {
		t.Errorf("missing cache_creation_input_tokens=5:\n%s", body)
	}
	// cache_read_input_tokens = 10
	if !strings.Contains(body, `"cache_read_input_tokens":10`) {
		t.Errorf("missing cache_read_input_tokens=10:\n%s", body)
	}
	// message_delta should carry output_tokens and stop_reason
	if !strings.Contains(body, "event: message_delta") {
		t.Errorf("missing event: message_delta:\n%s", body)
	}
	if !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Errorf("missing stop_reason 'end_turn':\n%s", body)
	}
	// message_stop
	if !strings.Contains(body, "event: message_stop") {
		t.Errorf("missing event: message_stop:\n%s", body)
	}
}

func TestAnthropicCodec_Stream_OpenAIChatStyleUsage(t *testing.T) {
	// When upstream is OpenAI Chat, StreamEventStop carries both usage and stop_reason.
	// Anthropic codec should emit a synthetic message_delta with the usage/stop_reason
	// before the message_stop.
	body := sendStreamToCodec(&anthropicCodec{}, nativeOpenAIChatStreamEvents())

	if !strings.Contains(body, "event: message_start") {
		t.Errorf("missing event: message_start:\n%s", body)
	}
	if !strings.Contains(body, "event: message_delta") {
		t.Errorf("missing event: message_delta (should be synthesized from StreamEventStop):\n%s", body)
	}
	if !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Errorf("missing stop_reason in message_delta:\n%s", body)
	}
	if !strings.Contains(body, "event: message_stop") {
		t.Errorf("missing event: message_stop:\n%s", body)
	}
}

// ---------------------------------------------------------------------------
// Cross-protocol edge case: stop event already has stop_reason
// (should not be overwritten by a delta's stop_reason)
// ---------------------------------------------------------------------------

func TestOpenAIChatCodec_Stream_StopEventHasPriorityOverDelta(t *testing.T) {
	// If both StreamEventDelta and StreamEventStop have StopReason,
	// the stop event's StopReason should take priority (not overwritten).
	toolUse := StopReasonToolUse
	endTurn := StopReasonEndTurn
	events := []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "msg_priority",
				Model: "gpt-4o",
				Usage: Usage{PromptTokens: 40},
			},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventDelta,
			StopReason: &toolUse, // delta says tool_use
			Usage:      &Usage{CompletionTokens: 5},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventStop,
			StopReason: &endTurn, // stop says end_turn — should win
		}},
	}

	body := sendStreamToCodec(&openaiChatCodec{}, events)

	// Stop event's end_turn should take priority
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("finish_reason should be 'stop' (from StreamEventStop), not 'tool_calls':\n%s", body)
	}
}

// ---------------------------------------------------------------------------
// Edge case: multiple usage deltas before stop
// ---------------------------------------------------------------------------

func TestOpenAIChatCodec_Stream_MultipleUsageDeltas(t *testing.T) {
	// Anthropic can send multiple deltas with usage, e.g.
	// thinking tokens + completion tokens in separate deltas.
	events := []StreamResult{
		{Event: &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				ID:    "msg_multi",
				Model: "claude-sonnet-4-20250514",
				Usage: Usage{PromptTokens: 60, PromptCacheHitTokens: 15},
			},
		}},
		{Event: &StreamEvent{
			Type:  StreamEventDelta,
			Index: 0,
			Delta: &ContentPart{Type: ContentTypeThinking, Thinking: &ThinkingContent{Thinking: "hmm"}},
		}},
		{Event: &StreamEvent{
			Type:  StreamEventDelta,
			Usage: &Usage{CompletionReasoningTokens: 20},
		}},
		{Event: &StreamEvent{
			Type:  StreamEventDelta,
			Index: 0,
			Delta: &ContentPart{Type: ContentTypeText, Text: &TextContent{Text: "Answer"}},
		}},
		{Event: &StreamEvent{
			Type:       StreamEventDelta,
			StopReason: func() *StopReason { r := StopReasonEndTurn; return &r }(),
			Usage:      &Usage{CompletionTokens: 8},
		}},
		{Event: &StreamEvent{
			Type: StreamEventStop,
		}},
	}

	body := sendStreamToCodec(&openaiChatCodec{}, events)

	// All usage should be merged
	if !strings.Contains(body, `"prompt_tokens":60`) {
		t.Errorf("missing prompt_tokens=60:\n%s", body)
	}
	if !strings.Contains(body, `"cached_tokens":15`) {
		t.Errorf("missing cached_tokens=15:\n%s", body)
	}
	if !strings.Contains(body, `"completion_tokens":8`) {
		t.Errorf("missing completion_tokens=8:\n%s", body)
	}
	if !strings.Contains(body, `"reasoning_tokens":20`) {
		t.Errorf("missing reasoning_tokens=20:\n%s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("missing finish_reason 'stop':\n%s", body)
	}
}
