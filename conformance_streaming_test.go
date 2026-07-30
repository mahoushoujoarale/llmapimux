package llmapimux

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestStreamingConformanceMatrix drives streaming responses through all 16
// inbound × outbound protocol combinations and asserts the SSE stream the client
// receives is well-formed for its protocol and preserves all content.
//
// Streaming was the least-covered area: before this test only 3 cross-protocol
// streaming paths were exercised, all of them Anthropic-inbound, and none used
// Gemini as an outbound target. Multi-block streams (thinking + text + parallel
// tool calls) — where the block-lifecycle bugs lived — were untested entirely.
func TestStreamingConformanceMatrix(t *testing.T) {
	for _, sc := range streamingScenarios() {
		for _, inbound := range allProtocols {
			for _, outbound := range allProtocols {
				name := sc.name + "/" + protocolName(inbound) + "_to_" + protocolName(outbound)
				t.Run(name, func(t *testing.T) {
					runStreamingCase(t, sc, inbound, outbound)
				})
			}
		}
	}
}

// streamingScenario is one logical streamed reply, expressed as raw SSE per
// outbound protocol, plus the expectations for the client-facing stream.
type streamingScenario struct {
	name string
	// request bodies keyed by inbound protocol
	inbound map[Protocol]inboundFixture
	// raw upstream SSE payloads keyed by outbound protocol
	upstreamSSE map[Protocol]string
	// wantText is the assistant text the client must end up with.
	wantText string
	// wantThinking is the reasoning text the client must receive, when the target
	// protocol can express it.
	wantThinking string
	// wantToolCalls lists tool names the client must observe, in order.
	wantToolCalls []string
}

func runStreamingCase(t *testing.T, sc streamingScenario, inbound, outbound Protocol) {
	fixture, ok := sc.inbound[inbound]
	if !ok {
		t.Skipf("no fixture for inbound %s", protocolName(inbound))
	}
	sse, ok := sc.upstreamSSE[outbound]
	if !ok {
		t.Skipf("no upstream SSE for outbound %s", protocolName(outbound))
	}

	var captured []byte
	var requestURI string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		requestURI = r.URL.RequestURI()
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sse))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer server.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: outbound,
		BaseURL:  server.URL,
		APIKey:   "test-key",
		Model:    outboundModelFor(outbound),
	}})

	req := httptest.NewRequest("POST", fixture.path, strings.NewReader(fixture.body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	inboundHandler(mux, inbound).ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	// codeflicker-fix: LOGIC-Issue-002/tb3m3jp0fdxv42afyew5
	// Streaming must use the target protocol's transport and valid request body.
	assertStreamingOutboundRequest(t, outbound, requestURI, captured)

	out := w.Body.String()
	assembled, problems := assembleClientStream(inbound, out)
	if len(problems) > 0 {
		t.Errorf("malformed %s SSE stream:\n  - %s\nstream:\n%s",
			protocolName(inbound), strings.Join(problems, "\n  - "), out)
	}

	if assembled.text != sc.wantText {
		t.Errorf("assembled text = %q, want %q\nstream:\n%s", assembled.text, sc.wantText, out)
	}
	// Thinking only survives to protocols that can express reasoning; Gemini
	// inbound renders thoughts as parts, which assembleClientStream folds into
	// thinking as well.
	if sc.wantThinking != "" && assembled.thinking != sc.wantThinking {
		t.Errorf("assembled thinking = %q, want %q\nstream:\n%s", assembled.thinking, sc.wantThinking, out)
	}
	if len(sc.wantToolCalls) > 0 {
		if len(assembled.toolNames) != len(sc.wantToolCalls) {
			t.Errorf("tool calls = %v, want %v\nstream:\n%s", assembled.toolNames, sc.wantToolCalls, out)
		} else {
			for i, name := range sc.wantToolCalls {
				if assembled.toolNames[i] != name {
					t.Errorf("tool call[%d] = %q, want %q", i, assembled.toolNames[i], name)
				}
			}
		}
	}
}

// assertStreamingOutboundRequest checks both the protocol body and the transport
// switch required to receive SSE. A valid body without stream mode is still a
// broken streaming conversion.
func assertStreamingOutboundRequest(t *testing.T, outbound Protocol, requestURI string, body []byte) {
	t.Helper()
	if len(body) == 0 {
		t.Fatal("upstream received no streaming request body")
	}
	if problems := validateOutboundBody(outbound, body); len(problems) > 0 {
		t.Errorf("invalid streaming outbound %s request:\n  - %s\nbody: %s", protocolName(outbound), strings.Join(problems, "\n  - "), body)
	}
	if outbound == ProtocolGemini {
		if !strings.Contains(requestURI, ":streamGenerateContent") || !strings.Contains(requestURI, "alt=sse") {
			t.Errorf("Gemini streaming URI = %q, want :streamGenerateContent?alt=sse", requestURI)
		}
		return
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("parse streaming %s request: %v", protocolName(outbound), err)
	}
	var stream bool
	if err := json.Unmarshal(raw["stream"], &stream); err != nil || !stream {
		t.Errorf("%s streaming request has stream=%v (err=%v), want true; body: %s", protocolName(outbound), stream, err, body)
	}
}

// assembledStream is the client-visible result of consuming an SSE stream.
type assembledStream struct {
	text      string
	thinking  string
	toolNames []string
	toolArgs  []string
}

// assembleClientStream parses the SSE stream the gateway produced for the inbound
// protocol, reassembling content the way a real SDK would and reporting any
// protocol violations it finds.
func assembleClientStream(p Protocol, raw string) (assembledStream, []string) {
	switch p {
	case ProtocolAnthropic:
		return assembleAnthropicStream(raw)
	case ProtocolOpenAIChat:
		return assembleOpenAIChatStream(raw)
	case ProtocolOpenAIResponses:
		return assembleOpenAIResponsesStream(raw)
	case ProtocolGemini:
		return assembleGeminiStream(raw)
	}
	return assembledStream{}, []string{"unknown protocol " + string(p)}
}

// sseFrame is one parsed SSE event.
type sseFrame struct {
	event string
	data  string
}

// parseSSEFrames splits a raw SSE body into frames.
func parseSSEFrames(raw string) []sseFrame {
	var frames []sseFrame
	for _, block := range strings.Split(raw, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var f sseFrame
		var dataLines []string
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				f.event = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
			}
		}
		f.data = strings.Join(dataLines, "\n")
		frames = append(frames, f)
	}
	return frames
}

// assembleAnthropicStream validates the strict Anthropic event grammar:
// message_start, then strictly sequential content blocks (start → delta* → stop),
// then an optional message_delta, then message_stop.
func assembleAnthropicStream(raw string) (assembledStream, []string) {
	var out assembledStream
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	sawStart := false
	sawStop := false
	openIndex := -1
	openType := ""
	var seenIndices []int
	var text, thinking strings.Builder
	var currentArgs strings.Builder

	for _, f := range parseSSEFrames(raw) {
		switch f.event {
		case "message_start":
			if sawStart {
				add("duplicate message_start")
			}
			sawStart = true

		case "content_block_start":
			if !sawStart {
				add("content_block_start before message_start")
			}
			if openIndex != -1 {
				add("content_block_start(index=?) while block %d is still open", openIndex)
			}
			var ev struct {
				Index        int `json:"index"`
				ContentBlock struct {
					Type string `json:"type"`
					Name string `json:"name"`
					ID   string `json:"id"`
				} `json:"content_block"`
			}
			if err := json.Unmarshal([]byte(f.data), &ev); err != nil {
				add("content_block_start not JSON: %v", err)
				continue
			}
			if ev.ContentBlock.Type == "" {
				add("content_block_start(index=%d) has no content_block.type", ev.Index)
			}
			for _, seen := range seenIndices {
				if seen == ev.Index {
					add("content block index %d reused", ev.Index)
				}
			}
			seenIndices = append(seenIndices, ev.Index)
			openIndex = ev.Index
			openType = ev.ContentBlock.Type
			if openType == "tool_use" || openType == "server_tool_use" {
				out.toolNames = append(out.toolNames, ev.ContentBlock.Name)
				currentArgs.Reset()
			}

		case "content_block_delta":
			var ev struct {
				Index int `json:"index"`
				Delta struct {
					Type        string  `json:"type"`
					Text        *string `json:"text"`
					Thinking    *string `json:"thinking"`
					Signature   *string `json:"signature"`
					PartialJSON *string `json:"partial_json"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(f.data), &ev); err != nil {
				add("content_block_delta not JSON: %v", err)
				continue
			}
			if openIndex == -1 {
				add("content_block_delta(index=%d) with no open block", ev.Index)
			} else if ev.Index != openIndex {
				add("content_block_delta(index=%d) does not match open block %d", ev.Index, openIndex)
			}
			switch ev.Delta.Type {
			case "text_delta":
				if ev.Delta.Text == nil {
					add("text_delta has no text field")
				} else {
					text.WriteString(*ev.Delta.Text)
				}
			case "thinking_delta":
				if ev.Delta.Thinking == nil {
					add("thinking_delta has no thinking field")
				} else {
					thinking.WriteString(*ev.Delta.Thinking)
				}
			case "signature_delta":
				// Signature is metadata; nothing to assemble.
			case "input_json_delta":
				if ev.Delta.PartialJSON != nil {
					currentArgs.WriteString(*ev.Delta.PartialJSON)
				}
			default:
				add("unknown content_block_delta type %q", ev.Delta.Type)
			}

		case "content_block_stop":
			var ev struct {
				Index int `json:"index"`
			}
			_ = json.Unmarshal([]byte(f.data), &ev)
			if openIndex == -1 {
				add("content_block_stop(index=%d) with no open block", ev.Index)
			} else if ev.Index != openIndex {
				add("content_block_stop(index=%d) does not close open block %d", ev.Index, openIndex)
			}
			if openType == "tool_use" || openType == "server_tool_use" {
				out.toolArgs = append(out.toolArgs, currentArgs.String())
			}
			openIndex = -1
			openType = ""

		case "message_delta":
			var ev struct {
				Delta struct {
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(f.data), &ev); err != nil {
				add("message_delta not JSON: %v", err)
				continue
			}
			if openIndex != -1 {
				add("message_delta while block %d is still open", openIndex)
			}
			if ev.Delta.StopReason == "" {
				add("message_delta has an empty stop_reason")
			} else if !validAnthropicStopReason(ev.Delta.StopReason) {
				add("message_delta stop_reason %q is not a valid enum value", ev.Delta.StopReason)
			}

		case "message_stop":
			if openIndex != -1 {
				add("message_stop while block %d is still open", openIndex)
			}
			sawStop = true

		case "error":
			// Errors terminate the stream; the caller decides whether that is expected.

		case "ping":
			// Ignored.

		default:
			add("unknown Anthropic SSE event %q", f.event)
		}
	}

	if !sawStart {
		add("stream has no message_start")
	}
	if !sawStop {
		add("stream has no message_stop")
	}
	if openIndex != -1 {
		add("content block %d was never closed", openIndex)
	}

	out.text = text.String()
	out.thinking = thinking.String()
	return out, problems
}

func validAnthropicStopReason(s string) bool {
	switch s {
	case "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal":
		return true
	}
	return false
}

// assembleOpenAIChatStream reassembles a Chat Completions delta stream.
func assembleOpenAIChatStream(raw string) (assembledStream, []string) {
	var out assembledStream
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	var text, thinking strings.Builder
	toolByIndex := map[int]*struct {
		name string
		args strings.Builder
	}{}
	var toolOrder []int
	sawDone := false
	sawFinish := false
	sawErr := false

	for _, f := range parseSSEFrames(raw) {
		if f.event != "" {
			add("Chat Completions streams must not use event: lines, got %q", f.event)
		}
		if f.data == "[DONE]" {
			sawDone = true
			continue
		}
		if sawDone {
			add("chunk received after the [DONE] sentinel")
		}
		// An in-band error object terminates the stream.
		var maybeErr struct {
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(f.data), &maybeErr); err == nil && maybeErr.Error != nil {
			sawErr = true
			continue
		}

		var chunk struct {
			Object  string `json:"object"`
			Choices []struct {
				Index int `json:"index"`
				Delta *struct {
					Role             string  `json:"role"`
					Content          *string `json:"content"`
					ReasoningContent *string `json:"reasoning_content"`
					Refusal          *string `json:"refusal"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(f.data), &chunk); err != nil {
			add("chunk is not valid JSON: %v", err)
			continue
		}
		if chunk.Object != "chat.completion.chunk" {
			add("chunk object = %q, want chat.completion.chunk", chunk.Object)
		}
		for _, ch := range chunk.Choices {
			if ch.FinishReason != nil && *ch.FinishReason != "" {
				if !validChatFinishReason(*ch.FinishReason) {
					add("finish_reason %q is not a valid enum value", *ch.FinishReason)
				}
				sawFinish = true
			}
			if ch.Delta == nil {
				continue
			}
			if ch.Delta.Content != nil {
				text.WriteString(*ch.Delta.Content)
			}
			if ch.Delta.ReasoningContent != nil {
				thinking.WriteString(*ch.Delta.ReasoningContent)
			}
			for _, tc := range ch.Delta.ToolCalls {
				entry, ok := toolByIndex[tc.Index]
				if !ok {
					entry = &struct {
						name string
						args strings.Builder
					}{}
					toolByIndex[tc.Index] = entry
					toolOrder = append(toolOrder, tc.Index)
				}
				if tc.Function.Name != "" {
					entry.name = tc.Function.Name
				}
				entry.args.WriteString(tc.Function.Arguments)
			}
		}
	}

	if !sawErr {
		if !sawFinish {
			add("stream has no finish_reason chunk")
		}
		if !sawDone {
			add("stream has no [DONE] sentinel")
		}
	}

	sortInts(toolOrder)
	for _, idx := range toolOrder {
		out.toolNames = append(out.toolNames, toolByIndex[idx].name)
		out.toolArgs = append(out.toolArgs, toolByIndex[idx].args.String())
	}
	out.text = text.String()
	out.thinking = thinking.String()
	return out, problems
}

// assembleOpenAIResponsesStream reassembles a Responses event stream.
func assembleOpenAIResponsesStream(raw string) (assembledStream, []string) {
	var out assembledStream
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	var text, thinking strings.Builder
	toolByIndex := map[int]*struct {
		name string
		args strings.Builder
	}{}
	var toolOrder []int
	sawCreated := false
	sawTerminal := false

	for _, f := range parseSSEFrames(raw) {
		if f.event == "" {
			add("Responses streams require event: lines")
			continue
		}
		var ev struct {
			Type        string `json:"type"`
			OutputIndex *int   `json:"output_index"`
			Delta       string `json:"delta"`
			Item        *struct {
				Type   string `json:"type"`
				Name   string `json:"name"`
				CallID string `json:"call_id"`
			} `json:"item"`
			Response *struct {
				Status string `json:"status"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(f.data), &ev); err != nil {
			add("event %s data is not JSON: %v", f.event, err)
			continue
		}
		if ev.Type != "" && ev.Type != f.event {
			add("event line %q does not match payload type %q", f.event, ev.Type)
		}
		idx := 0
		if ev.OutputIndex != nil {
			idx = *ev.OutputIndex
		}

		switch f.event {
		case "response.created":
			sawCreated = true
		case "response.output_item.added":
			if ev.Item != nil && ev.Item.Type == "function_call" {
				entry, ok := toolByIndex[idx]
				if !ok {
					entry = &struct {
						name string
						args strings.Builder
					}{}
					toolByIndex[idx] = entry
					toolOrder = append(toolOrder, idx)
				}
				entry.name = ev.Item.Name
			}
		case "response.output_text.delta":
			text.WriteString(ev.Delta)
		case "response.reasoning_summary_text.delta":
			thinking.WriteString(ev.Delta)
		case "response.function_call_arguments.delta":
			entry, ok := toolByIndex[idx]
			if !ok {
				entry = &struct {
					name string
					args strings.Builder
				}{}
				toolByIndex[idx] = entry
				toolOrder = append(toolOrder, idx)
			}
			entry.args.WriteString(ev.Delta)
		case "response.output_item.done", "response.content_part.added",
			"response.content_part.done", "response.refusal.delta":
			// Lifecycle / non-text events.
		case "response.completed", "response.incomplete", "response.failed", "error":
			sawTerminal = true
			if f.event == "response.completed" && ev.Response != nil &&
				!isOpenAIResponsesStatus(ev.Response.Status) {
				add("response.completed status %q is not valid", ev.Response.Status)
			}
		default:
			add("unknown Responses SSE event %q", f.event)
		}
	}

	if !sawCreated {
		add("stream has no response.created")
	}
	if !sawTerminal {
		add("stream has no terminal event (response.completed/failed/incomplete/error)")
	}

	sortInts(toolOrder)
	for _, idx := range toolOrder {
		out.toolNames = append(out.toolNames, toolByIndex[idx].name)
		out.toolArgs = append(out.toolArgs, toolByIndex[idx].args.String())
	}
	out.text = text.String()
	out.thinking = thinking.String()
	return out, problems
}

// assembleGeminiStream reassembles a Gemini streamGenerateContent stream.
func assembleGeminiStream(raw string) (assembledStream, []string) {
	var out assembledStream
	var problems []string
	add := func(format string, args ...any) {
		problems = append(problems, fmt.Sprintf(format, args...))
	}

	var text, thinking strings.Builder
	sawFinish := false
	sawErr := false

	for _, f := range parseSSEFrames(raw) {
		if f.event != "" {
			add("Gemini streams must not use event: lines, got %q", f.event)
		}
		if f.data == "[DONE]" {
			add("Gemini streams must not emit a [DONE] sentinel")
			continue
		}
		var chunk struct {
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
			Candidates []struct {
				Content *struct {
					Role  string `json:"role"`
					Parts []struct {
						Text         string `json:"text"`
						Thought      *bool  `json:"thought"`
						FunctionCall *struct {
							Name string          `json:"name"`
							Args json.RawMessage `json:"args"`
						} `json:"functionCall"`
					} `json:"parts"`
				} `json:"content"`
				FinishReason string `json:"finishReason"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal([]byte(f.data), &chunk); err != nil {
			add("chunk is not valid JSON: %v", err)
			continue
		}
		if chunk.Error != nil {
			sawErr = true
			continue
		}
		for _, cand := range chunk.Candidates {
			if cand.FinishReason != "" {
				if !isGeminiFinishReason(cand.FinishReason) {
					add("finishReason %q is not a valid enum value", cand.FinishReason)
				}
				sawFinish = true
			}
			if cand.Content == nil {
				continue
			}
			if cand.Content.Role != "" && cand.Content.Role != "model" {
				add("candidate role = %q, want model", cand.Content.Role)
			}
			for _, part := range cand.Content.Parts {
				switch {
				case part.FunctionCall != nil:
					out.toolNames = append(out.toolNames, part.FunctionCall.Name)
					out.toolArgs = append(out.toolArgs, string(part.FunctionCall.Args))
				case part.Thought != nil && *part.Thought:
					thinking.WriteString(part.Text)
				default:
					text.WriteString(part.Text)
				}
			}
		}
	}

	if !sawErr && !sawFinish {
		add("stream has no finishReason")
	}

	out.text = text.String()
	out.thinking = thinking.String()
	return out, problems
}

func sortInts(v []int) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}

// --- Streaming scenarios ---

func streamingScenarios() []streamingScenario {
	return []streamingScenario{
		streamTextScenario(),
		streamThinkingThenTextScenario(),
		streamToolCallScenario(),
		streamParallelToolCallsScenario(),
		streamContentWithFinishReasonScenario(),
	}
}

// streamingRequestFixtures builds streaming request bodies for all four inbound
// protocols.
func streamingRequestFixtures() map[Protocol]inboundFixture {
	return map[Protocol]inboundFixture{
		ProtocolAnthropic: {
			path: "/v1/messages",
			body: `{"model":"claude-3","max_tokens":1000,"stream":true,"messages":[{"role":"user","content":"Hi"}]}`,
		},
		ProtocolOpenAIChat: {
			path: "/v1/chat/completions",
			body: `{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"Hi"}]}`,
		},
		ProtocolOpenAIResponses: {
			path: "/v1/responses",
			body: `{"model":"gpt-4o","stream":true,"input":"Hi"}`,
		},
		ProtocolGemini: {
			path: "/v1beta/models/gemini-2.0:streamGenerateContent?alt=sse",
			body: `{"contents":[{"role":"user","parts":[{"text":"Hi"}]}]}`,
		},
	}
}

// streamTextScenario: plain text streamed in two chunks.
func streamTextScenario() streamingScenario {
	return streamingScenario{
		name:    "stream_text",
		inbound: streamingRequestFixtures(),
		upstreamSSE: map[Protocol]string{
			ProtocolAnthropic: sseJoin(
				`event: message_start`, `data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-3","usage":{"input_tokens":5,"output_tokens":0}}}`,
				`event: content_block_start`, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				`event: content_block_delta`, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello "}}`,
				`event: content_block_delta`, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}`,
				`event: content_block_stop`, `data: {"type":"content_block_stop","index":0}`,
				`event: message_delta`, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
				`event: message_stop`, `data: {"type":"message_stop"}`,
			),
			ProtocolOpenAIChat: sseJoin(
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello "},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
				`data: [DONE]`,
			),
			ProtocolOpenAIResponses: sseJoin(
				`event: response.created`, `data: {"type":"response.created","response":{"id":"r1","object":"response","model":"gpt-4o","status":"in_progress"}}`,
				`event: response.output_item.added`, `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant","content":[]}}`,
				`event: response.output_text.delta`, `data: {"type":"response.output_text.delta","output_index":0,"delta":"Hello "}`,
				`event: response.output_text.delta`, `data: {"type":"response.output_text.delta","output_index":0,"delta":"world"}`,
				`event: response.output_item.done`, `data: {"type":"response.output_item.done","output_index":0}`,
				`event: response.completed`, `data: {"type":"response.completed","response":{"id":"r1","object":"response","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}`,
			),
			ProtocolGemini: sseJoin(
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello "}]}}],"modelVersion":"gemini-2.5-pro"}`,
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			),
		},
		wantText: "Hello world",
	}
}

// streamThinkingThenTextScenario: a reasoning block followed by a text block. This
// exercises multi-block streaming and the block-lifecycle normalisation that the
// Anthropic outbound codec performs.
func streamThinkingThenTextScenario() streamingScenario {
	return streamingScenario{
		name:    "stream_thinking_then_text",
		inbound: streamingRequestFixtures(),
		upstreamSSE: map[Protocol]string{
			ProtocolAnthropic: sseJoin(
				`event: message_start`, `data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-3","usage":{"input_tokens":5,"output_tokens":0}}}`,
				`event: content_block_start`, `data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
				`event: content_block_delta`, `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}`,
				`event: content_block_stop`, `data: {"type":"content_block_stop","index":0}`,
				`event: content_block_start`, `data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
				`event: content_block_delta`, `data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}`,
				`event: content_block_stop`, `data: {"type":"content_block_stop","index":1}`,
				`event: message_delta`, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
				`event: message_stop`, `data: {"type":"message_stop"}`,
			),
			ProtocolOpenAIChat: sseJoin(
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"reasoning_content":"Let me think"},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}`,
				`data: [DONE]`,
			),
			ProtocolOpenAIResponses: sseJoin(
				`event: response.created`, `data: {"type":"response.created","response":{"id":"r1","object":"response","model":"gpt-4o","status":"in_progress"}}`,
				`event: response.output_item.added`, `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning"}}`,
				`event: response.reasoning_summary_text.delta`, `data: {"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"Let me think"}`,
				`event: response.output_item.done`, `data: {"type":"response.output_item.done","output_index":0}`,
				`event: response.output_item.added`, `data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","role":"assistant","content":[]}}`,
				`event: response.output_text.delta`, `data: {"type":"response.output_text.delta","output_index":1,"delta":"Answer"}`,
				`event: response.output_item.done`, `data: {"type":"response.output_item.done","output_index":0}`,
				`event: response.completed`, `data: {"type":"response.completed","response":{"id":"r1","object":"response","status":"completed","usage":{"input_tokens":5,"output_tokens":4,"total_tokens":9}}}`,
			),
			ProtocolGemini: sseJoin(
				`data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"Let me think"}]}}],"modelVersion":"gemini-2.5-pro"}`,
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Answer"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"totalTokenCount":9}}`,
			),
		},
		wantText: "Answer",
		// codeflicker-fix: LOGIC-Issue-003/tb3m3jp0fdxv42afyew5
		wantThinking: "Let me think",
	}
}

// streamToolCallScenario: a single streamed tool call with incremental arguments.
func streamToolCallScenario() streamingScenario {
	return streamingScenario{
		name:    "stream_tool_call",
		inbound: streamingRequestFixtures(),
		upstreamSSE: map[Protocol]string{
			ProtocolAnthropic: sseJoin(
				`event: message_start`, `data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-3","usage":{"input_tokens":5,"output_tokens":0}}}`,
				`event: content_block_start`, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"get_weather","input":{}}}`,
				`event: content_block_delta`, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
				`event: content_block_delta`, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`,
				`event: content_block_stop`, `data: {"type":"content_block_stop","index":0}`,
				`event: message_delta`, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}`,
				`event: message_stop`, `data: {"type":"message_stop"}`,
			),
			ProtocolOpenAIChat: sseJoin(
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}`,
				`data: [DONE]`,
			),
			ProtocolOpenAIResponses: sseJoin(
				`event: response.created`, `data: {"type":"response.created","response":{"id":"r1","object":"response","model":"gpt-4o","status":"in_progress"}}`,
				`event: response.output_item.added`, `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"t1","name":"get_weather"}}`,
				`event: response.function_call_arguments.delta`, `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"city\":"}`,
				`event: response.function_call_arguments.delta`, `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"Paris\"}"}`,
				`event: response.output_item.done`, `data: {"type":"response.output_item.done","output_index":0}`,
				`event: response.completed`, `data: {"type":"response.completed","response":{"id":"r1","object":"response","status":"completed","usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}}`,
			),
			ProtocolGemini: sseJoin(
				`data: {"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"},"id":"t1"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":7,"totalTokenCount":12},"modelVersion":"gemini-2.5-pro"}`,
			),
		},
		wantText:      "",
		wantToolCalls: []string{"get_weather"},
	}
}

// streamParallelToolCallsScenario: two tool calls streamed in the same response.
// Each must land in its own content block / tool_call index; merging them corrupts
// both argument payloads.
func streamParallelToolCallsScenario() streamingScenario {
	return streamingScenario{
		name:    "stream_parallel_tool_calls",
		inbound: streamingRequestFixtures(),
		upstreamSSE: map[Protocol]string{
			ProtocolAnthropic: sseJoin(
				`event: message_start`, `data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-3","usage":{"input_tokens":5,"output_tokens":0}}}`,
				`event: content_block_start`, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"tool_a","input":{}}}`,
				`event: content_block_delta`, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}`,
				`event: content_block_stop`, `data: {"type":"content_block_stop","index":0}`,
				`event: content_block_start`, `data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t2","name":"tool_b","input":{}}}`,
				`event: content_block_delta`, `data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"b\":2}"}}`,
				`event: content_block_stop`, `data: {"type":"content_block_stop","index":1}`,
				`event: message_delta`, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`,
				`event: message_stop`, `data: {"type":"message_stop"}`,
			),
			// Both tool calls batched into a single chunk — the shape that used to
			// silently drop everything after tool_calls[0].
			ProtocolOpenAIChat: sseJoin(
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[`+
					`{"index":0,"id":"t1","type":"function","function":{"name":"tool_a","arguments":"{\"a\":1}"}},`+
					`{"index":1,"id":"t2","type":"function","function":{"name":"tool_b","arguments":"{\"b\":2}"}}`+
					`]},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}}`,
				`data: [DONE]`,
			),
			ProtocolOpenAIResponses: sseJoin(
				`event: response.created`, `data: {"type":"response.created","response":{"id":"r1","object":"response","model":"gpt-4o","status":"in_progress"}}`,
				`event: response.output_item.added`, `data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"t1","name":"tool_a"}}`,
				`event: response.function_call_arguments.delta`, `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"a\":1}"}`,
				`event: response.output_item.done`, `data: {"type":"response.output_item.done","output_index":0}`,
				`event: response.output_item.added`, `data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"t2","name":"tool_b"}}`,
				`event: response.function_call_arguments.delta`, `data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"b\":2}"}`,
				`event: response.output_item.done`, `data: {"type":"response.output_item.done","output_index":1}`,
				`event: response.completed`, `data: {"type":"response.completed","response":{"id":"r1","object":"response","status":"completed","usage":{"input_tokens":5,"output_tokens":9,"total_tokens":14}}}`,
			),
			ProtocolGemini: sseJoin(
				`data: {"candidates":[{"content":{"role":"model","parts":[` +
					`{"functionCall":{"name":"tool_a","args":{"a":1},"id":"t1"}},` +
					`{"functionCall":{"name":"tool_b","args":{"b":2},"id":"t2"}}` +
					`]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":9,"totalTokenCount":14},"modelVersion":"gemini-2.5-pro"}`,
			),
		},
		wantToolCalls: []string{"tool_a", "tool_b"},
	}
}

// streamContentWithFinishReasonScenario: the last text token arrives in the same
// chunk as finish_reason. vLLM, DeepSeek and Azure all do this; a decoder that
// checks finish_reason first silently drops the final token.
func streamContentWithFinishReasonScenario() streamingScenario {
	return streamingScenario{
		name:    "stream_content_with_finish_reason",
		inbound: streamingRequestFixtures(),
		upstreamSSE: map[Protocol]string{
			ProtocolOpenAIChat: sseJoin(
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello "},"finish_reason":null}]}`,
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
				`data: [DONE]`,
			),
			// Gemini natively puts content and finishReason in one chunk.
			ProtocolGemini: sseJoin(
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello "}]}}],"modelVersion":"gemini-2.5-pro"}`,
				`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}`,
			),
		},
		wantText: "Hello world",
	}
}

// sseJoin builds a raw SSE payload from individual lines, inserting the blank-line
// frame separators. Lines beginning with "event:" are paired with the following
// "data:" line.
func sseJoin(lines ...string) string {
	var b strings.Builder
	for i := 0; i < len(lines); i++ {
		line := lines[i]
		if strings.HasPrefix(line, "event: ") && i+1 < len(lines) && strings.HasPrefix(lines[i+1], "data: ") {
			b.WriteString(line)
			b.WriteString("\n")
			b.WriteString(lines[i+1])
			b.WriteString("\n\n")
			i++
			continue
		}
		b.WriteString(line)
		b.WriteString("\n\n")
	}
	return b.String()
}
