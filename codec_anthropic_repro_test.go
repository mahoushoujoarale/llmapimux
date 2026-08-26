package llmapimux

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestAnthropicCodecSSESequence_NormalTurn verifies the exact SSE event
// sequence a Claude Code client receives for a typical wanqing OpenAI stream
// (reasoning + text + tool_calls + trailing usage chunk).
func TestAnthropicCodecSSESequence_NormalTurn(t *testing.T) {
	chunks := []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"think."}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"answer."}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Read","arguments":"{\"a\":1}"}}]}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,"prompt_tokens_details":{"cached_tokens":80}}}`,
		" data: [DONE]",
	}

	// Feed chunks through the outbound decoder into a channel, mimicking
	// OpenAIChatClient.SendStream (including drainTrailingUsage semantics).
	eventsCh := make(chan StreamResult, 64)
	var stopEvent *StreamEvent
	for _, c := range chunks {
		raw := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(c), "data:"))
		if raw == "[DONE]" || raw == "" {
			continue
		}
		evs, err := DecodeOpenAIChatStreamChunks([]byte(raw))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		for _, ev := range evs {
			if ev.Type == StreamEventStop {
				// drainTrailingUsage: subsequent usage chunks merged into stop.
				stopEvent = ev
				continue
			}
			if stopEvent != nil && ev.Usage != nil {
				if stopEvent.Usage == nil {
					u := *ev.Usage
					stopEvent.Usage = &u
				} else {
					mergeStreamUsage(stopEvent.Usage, ev.Usage)
				}
				continue
			}
			eventsCh <- StreamResult{Event: ev}
		}
	}
	if stopEvent != nil {
		eventsCh <- StreamResult{Event: stopEvent}
	}
	close(eventsCh)

	var sb strings.Builder
	w := NewSSEWriter(&nopFlusher{&sb})
	(&anthropicCodec{}).WriteStreamingResponse(w, eventsCh)

	out := sb.String()
	t.Logf("SSE stream sent to Claude Code:\n%s", out)

	// Required lifecycle checks.
	for _, want := range []string{
		"event: message_start",
		"event: content_block_start", // thinking
		"thinking_delta",
		"event: content_block_stop",
		"event: content_block_start", // text
		"text_delta",
		"event: content_block_stop",
		"event: content_block_start", // tool_use
		"input_json_delta",
		"event: content_block_stop",
		"event: message_delta",
		`"stop_reason":"tool_use"`,
		`"cache_read_input_tokens":80`,
		"event: message_stop",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("SSE stream missing %q", want)
		}
	}

	// message_stop must be the LAST event (Claude Code treats a stream without
	// it as a failed turn).
	if !strings.HasSuffix(strings.TrimSpace(out), "event: message_stop\ndata: {\"type\":\"message_stop\"}") {
		t.Errorf("message_stop is not the final event")
	}
}

// TestAnthropicCodecSSESequence_InterleavedToolCalls pins the behavior when a
// provider interleaves deltas of parallel tool calls (tc0 and tc1 alternating),
// which some OpenAI-compatible backends do.
func TestAnthropicCodecSSESequence_InterleavedToolCalls(t *testing.T) {
	chunks := []string{
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_A","type":"function","function":{"name":"Read","arguments":"{\"a\":"}}]}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_B","type":"function","function":{"name":"Write","arguments":"{\"b\":"}}]}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]}}]}`,
		`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	}

	eventsCh := make(chan StreamResult, 64)
	var stopEvent *StreamEvent
	for _, c := range chunks {
		evs, err := DecodeOpenAIChatStreamChunks([]byte(c))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		for _, ev := range evs {
			if ev.Type == StreamEventStop {
				stopEvent = ev
				continue
			}
			eventsCh <- StreamResult{Event: ev}
		}
	}
	if stopEvent != nil {
		eventsCh <- StreamResult{Event: stopEvent}
	}
	close(eventsCh)

	var sb strings.Builder
	w := NewSSEWriter(&nopFlusher{&sb})
	(&anthropicCodec{}).WriteStreamingResponse(w, eventsCh)

	out := sb.String()
	t.Logf("SSE stream (interleaved):\n%s", out)

	// Each logical tool call must end up in exactly one content block with its
	// full argument JSON — no id-less fragments.
	starts := strings.Count(out, `"type":"tool_use"`)
	if starts != 2 {
		t.Errorf("tool_use block starts = %d, want 2 (interleaving split a tool call into multiple blocks)", starts)
	}
	frags := collectPartialJSON(t, out)
	if len(frags) != 2 || frags[0] != `{"a":1}` || frags[1] != `{"b":2}` {
		t.Errorf("interleaved arguments must reassemble into one delta per call, got %v", frags)
	}
	if strings.Count(out, `"id":"call_A"`) != 1 || strings.Count(out, `"id":"call_B"`) != 1 {
		t.Errorf("each tool call must appear exactly once: %s", out)
	}
	// The flushed tool blocks must precede message_delta / message_stop.
	if strings.Index(out, `"id":"call_B"`) > strings.Index(out, "event: message_delta") {
		t.Errorf("buffered tool blocks must flush before message_delta")
	}
}

// TestAnthropicCodecSSESequence_NativeToolPassthrough verifies that tool blocks
// opened by a real upstream content_block_start (native Anthropic upstream or
// OpenAI Responses function_call items) still stream live — the buffering added
// for interleaved OpenAI Chat tool_call deltas must not change their behaviour.
func TestAnthropicCodecSSESequence_NativeToolPassthrough(t *testing.T) {
	toolStart := &StreamEvent{
		Type:  StreamEventContentBlockStart,
		Index: 0,
		Delta: &ContentPart{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{ID: "tu_1", Name: "Read"}},
	}
	toolDelta := func(args string) *StreamEvent {
		return &StreamEvent{
			Type:  StreamEventDelta,
			Index: 0,
			Delta: &ContentPart{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{Arguments: json.RawMessage(args)}},
		}
	}
	eventsCh := make(chan StreamResult, 8)
	eventsCh <- StreamResult{Event: toolStart}
	eventsCh <- StreamResult{Event: toolDelta(`{"pa`)}
	eventsCh <- StreamResult{Event: toolDelta(`th":"x"}`)}
	eventsCh <- StreamResult{Event: &StreamEvent{Type: StreamEventContentBlockStop, Index: 0}}
	eventsCh <- StreamResult{Event: &StreamEvent{Type: StreamEventStop, StopReason: stopReasonPtr(StopReasonToolUse)}}
	close(eventsCh)

	var sb strings.Builder
	w := NewSSEWriter(&nopFlusher{&sb})
	(&anthropicCodec{}).WriteStreamingResponse(w, eventsCh)

	out := sb.String()
	t.Logf("SSE stream (native passthrough):\n%s", out)

	// Deltas must stream live: both argument fragments arrive as separate
	// input_json_delta events on the same block, not deferred to the stop flush.
	if strings.Count(out, "event: content_block_delta") != 2 {
		t.Errorf("live tool deltas were buffered/merged: %s", out)
	}
	fragments := collectPartialJSON(t, out)
	if strings.Join(fragments, "") != `{"path":"x"}` {
		t.Errorf("live tool delta fragments = %v, want [{\"pa\" + \"th\":\"x\"}] reassembled", fragments)
	}
	if !strings.Contains(out, `"stop_reason":"tool_use"`) {
		t.Errorf("missing stop_reason: %s", out)
	}
}

// collectPartialJSON extracts the partial_json value of every input_json_delta
// event from an Anthropic SSE stream, in order.
func collectPartialJSON(t *testing.T, sse string) []string {
	t.Helper()
	var fragments []string
	for _, line := range strings.Split(sse, "\n") {
		if !strings.HasPrefix(line, "data: ") || !strings.Contains(line, "input_json_delta") {
			continue
		}
		var ev struct {
			Delta struct {
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev); err != nil {
			t.Fatalf("unmarshal delta event: %v", err)
		}
		fragments = append(fragments, ev.Delta.PartialJSON)
	}
	return fragments
}

type nopFlusher struct{ w *strings.Builder }

func (n *nopFlusher) Write(p []byte) (int, error) { return n.w.Write(p) }
func (n *nopFlusher) Flush()                      {}
