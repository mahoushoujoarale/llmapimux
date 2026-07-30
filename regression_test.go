package llmapimux

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// This file holds one regression test per bug found during the conversion audit.
// Each test is named after the defect it locks down and documents why the old
// behaviour was wrong, so a future refactor that reintroduces the bug fails with
// an explanation rather than a bare assertion mismatch.

// --- Streaming decode ---

// Regression: DecodeOpenAIChatStreamChunk returned a single event, so a chunk that
// carried both content and finish_reason lost the content. vLLM, DeepSeek and
// Azure all emit the final token together with finish_reason.
func TestRegression_ChatChunk_ContentAndFinishReasonBothSurvive(t *testing.T) {
	chunk := []byte(`{"id":"c","object":"chat.completion.chunk","choices":[
		{"index":0,"delta":{"content":"final"},"finish_reason":"stop"}]}`)

	events, err := DecodeOpenAIChatStreamChunks(chunk)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	var sawText, sawStop bool
	for _, ev := range events {
		if ev.Type == StreamEventDelta && ev.Delta != nil && ev.Delta.Text != nil &&
			ev.Delta.Text.Text == "final" {
			sawText = true
		}
		if ev.Type == StreamEventStop {
			sawStop = true
		}
	}
	if !sawText {
		t.Error("content delta was dropped when finish_reason shared the chunk")
	}
	if !sawStop {
		t.Error("stop event missing")
	}
}

// Regression: only tool_calls[0] was decoded, so parallel tool calls batched into
// one chunk lost every call after the first.
func TestRegression_ChatChunk_AllToolCallsDecoded(t *testing.T) {
	chunk := []byte(`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[
		{"index":0,"id":"a","type":"function","function":{"name":"tool_a","arguments":"{}"}},
		{"index":1,"id":"b","type":"function","function":{"name":"tool_b","arguments":"{}"}},
		{"index":2,"id":"c","type":"function","function":{"name":"tool_c","arguments":"{}"}}
	]},"finish_reason":null}]}`)

	events, err := DecodeOpenAIChatStreamChunks(chunk)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	var names []string
	for _, ev := range events {
		if ev.Delta != nil && ev.Delta.ToolUse != nil {
			names = append(names, ev.Delta.ToolUse.Name)
		}
	}
	if len(names) != 3 {
		t.Fatalf("decoded tool calls = %v, want 3", names)
	}
	for i, want := range []string{"tool_a", "tool_b", "tool_c"} {
		if names[i] != want {
			t.Errorf("tool[%d] = %q, want %q", i, names[i], want)
		}
	}
}

// Regression: reasoning, content and tool_calls in one chunk were mutually
// exclusive — the first match won and the rest were discarded.
func TestRegression_ChatChunk_ReasoningContentAndToolCallCoexist(t *testing.T) {
	chunk := []byte(`{"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{
		"reasoning_content":"think","content":"say",
		"tool_calls":[{"index":0,"id":"t","type":"function","function":{"name":"f","arguments":"{}"}}]
	},"finish_reason":null}]}`)

	events, err := DecodeOpenAIChatStreamChunks(chunk)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	var sawThinking, sawText, sawTool bool
	for _, ev := range events {
		if ev.Delta == nil {
			continue
		}
		switch ev.Delta.Type {
		case ContentTypeThinking:
			sawThinking = true
		case ContentTypeText:
			sawText = true
		case ContentTypeToolUse:
			sawTool = true
		}
	}
	if !sawThinking || !sawText || !sawTool {
		t.Errorf("thinking=%v text=%v tool=%v; all three must survive", sawThinking, sawText, sawTool)
	}
}

// Regression: an explicit Gemini finishReason of MAX_TOKENS/SAFETY was overridden
// by tool-use inference whenever the chunk also contained a functionCall, so
// truncated and filtered responses were reported as a clean tool call.
func TestRegression_GeminiStream_ExplicitFinishReasonWins(t *testing.T) {
	cases := []struct {
		finishReason string
		want         StopReason
	}{
		{"MAX_TOKENS", StopReasonMaxTokens},
		{"SAFETY", StopReasonContentFilter},
		{"STOP_SEQUENCE", StopReasonStopSequence},
		{"STOP", StopReasonToolUse}, // tool_use inference only applies to STOP
	}
	for _, tc := range cases {
		t.Run(tc.finishReason, func(t *testing.T) {
			chunk := []byte(`{"candidates":[{"content":{"role":"model","parts":[
				{"functionCall":{"name":"f","args":{}}}]},"finishReason":"` + tc.finishReason + `"}]}`)
			events, err := DecodeGeminiStreamChunk(chunk)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			var got *StopReason
			for _, ev := range events {
				if ev != nil && ev.StopReason != nil {
					got = ev.StopReason
				}
			}
			if got == nil {
				t.Fatalf("no stop reason decoded from %s", tc.finishReason)
			}
			if *got != tc.want {
				t.Errorf("stop reason = %q, want %q", *got, tc.want)
			}
		})
	}
}

// --- Anthropic streaming codec ---

// Regression: EncodeAnthropicStreamEvent dereferenced event.Delta on
// content_block_start without a nil check, panicking on the nil-Delta events that
// DecodeOpenAIResponsesStreamEvent legitimately produces for unmapped item types.
func TestRegression_AnthropicCodec_NilDeltaDoesNotPanic(t *testing.T) {
	ch := make(chan StreamResult, 4)
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStart, Response: &Response{Model: "m"}}}
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventContentBlockStart, Index: 0, Delta: nil}}
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStop, StopReason: stopReasonPtr(StopReasonEndTurn)}}
	close(ch)

	out := runAnthropicStreamCodec(t, ch)
	if !strings.Contains(out, "content_block_start") {
		t.Errorf("nil Delta should still open a block, got:\n%s", out)
	}
	if _, problems := assembleAnthropicStream(out); len(problems) > 0 {
		t.Errorf("stream is malformed: %v\n%s", problems, out)
	}
}

// Regression: the codec tracked several simultaneously-open content blocks, so an
// IR stream that switched content type without an explicit stop produced
// overlapping blocks. The Anthropic SDK accumulator requires strictly sequential
// blocks and mis-assembles overlapping ones.
func TestRegression_AnthropicCodec_NoOverlappingBlocks(t *testing.T) {
	ch := make(chan StreamResult, 8)
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStart, Response: &Response{Model: "m"}}}
	// Thinking then text with no intervening content_block_stop.
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventDelta, Delta: &ContentPart{
		Type: ContentTypeThinking, Thinking: &ThinkingContent{Thinking: "t"}}}}
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventDelta, Delta: &ContentPart{
		Type: ContentTypeText, Text: &TextContent{Text: "x"}}}}
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStop, StopReason: stopReasonPtr(StopReasonEndTurn)}}
	close(ch)

	out := runAnthropicStreamCodec(t, ch)
	assembled, problems := assembleAnthropicStream(out)
	if len(problems) > 0 {
		t.Errorf("overlapping or malformed blocks: %v\n%s", problems, out)
	}
	if assembled.thinking != "t" || assembled.text != "x" {
		t.Errorf("thinking=%q text=%q, want t/x", assembled.thinking, assembled.text)
	}
}

// Regression: message_delta.stop_sequence had no IR field, so the matched stop
// sequence was lost on every Anthropic→Anthropic streaming round-trip.
func TestRegression_AnthropicCodec_StopSequenceSurvives(t *testing.T) {
	ch := make(chan StreamResult, 4)
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStart, Response: &Response{Model: "m"}}}
	ch <- StreamResult{Event: &StreamEvent{
		Type:         StreamEventStop,
		StopReason:   stopReasonPtr(StopReasonStopSequence),
		StopSequence: "END",
	}}
	close(ch)

	out := runAnthropicStreamCodec(t, ch)
	if !strings.Contains(out, `"stop_sequence":"END"`) {
		t.Errorf("stop_sequence was dropped:\n%s", out)
	}
}

// Regression: a usage-only delta (no Delta, no StopReason) made
// EncodeAnthropicStreamEvent return an error, which the codec treated as fatal and
// tore the stream down mid-response. It must be skipped and its usage folded into
// the terminating message_delta instead.
func TestRegression_AnthropicCodec_UsageOnlyDeltaDoesNotKillStream(t *testing.T) {
	ch := make(chan StreamResult, 6)
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStart, Response: &Response{Model: "m"}}}
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventDelta, Delta: &ContentPart{
		Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}}
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventDelta, Usage: &Usage{CompletionTokens: 7}}}
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStop, StopReason: stopReasonPtr(StopReasonEndTurn)}}
	close(ch)

	out := runAnthropicStreamCodec(t, ch)
	assembled, problems := assembleAnthropicStream(out)
	if len(problems) > 0 {
		t.Errorf("stream malformed: %v\n%s", problems, out)
	}
	if assembled.text != "hi" {
		t.Errorf("text = %q, want hi", assembled.text)
	}
	if !strings.Contains(out, `"output_tokens":7`) {
		t.Errorf("usage from the usage-only delta was lost:\n%s", out)
	}
}

// Regression: a mid-stream transport error was swallowed — the codec just stopped
// writing, which a client cannot distinguish from a clean end of stream.
func TestRegression_AnthropicCodec_StreamErrorIsSurfaced(t *testing.T) {
	ch := make(chan StreamResult, 4)
	ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStart, Response: &Response{Model: "m"}}}
	ch <- StreamResult{Err: errf("upstream exploded")}
	close(ch)

	out := runAnthropicStreamCodec(t, ch)
	if !strings.Contains(out, "event: error") {
		t.Errorf("transport error was not surfaced as an SSE error event:\n%s", out)
	}
	if !strings.Contains(out, "upstream exploded") {
		t.Errorf("error message missing:\n%s", out)
	}
}

// Same guarantee for the other three inbound protocols.
func TestRegression_AllCodecs_StreamErrorIsSurfaced(t *testing.T) {
	cases := []struct {
		protocol Protocol
		wantSub  string
	}{
		{ProtocolAnthropic, "event: error"},
		{ProtocolOpenAIChat, `"error"`},
		{ProtocolOpenAIResponses, "event: error"},
		{ProtocolGemini, `"error"`},
	}
	for _, tc := range cases {
		t.Run(protocolName(tc.protocol), func(t *testing.T) {
			ch := make(chan StreamResult, 4)
			ch <- StreamResult{Event: &StreamEvent{Type: StreamEventStart, Response: &Response{Model: "m"}}}
			ch <- StreamResult{Err: errf("boom")}
			close(ch)

			rec := httptest.NewRecorder()
			writer := NewSSEWriter(rec)
			codecFor(tc.protocol).WriteStreamingResponse(writer, ch)
			out := rec.Body.String()
			if !strings.Contains(out, tc.wantSub) || !strings.Contains(out, "boom") {
				t.Errorf("%s stream did not report the error:\n%s", protocolName(tc.protocol), out)
			}
		})
	}
}

// --- Tool results ---

// Regression: encodeOpenAIChatToolMessage had a `break` after the first tool
// result, so an Anthropic turn packing N parallel results produced one OpenAI tool
// message. The remaining tool_call_ids were left unanswered, which providers
// reject with a hard 400.
func TestRegression_ChatEncode_ParallelToolResultsFanOut(t *testing.T) {
	req := &Request{
		Model:     "gpt-4o",
		MaxTokens: 100,
		Tools: []Tool{
			{Name: "a", Parameters: json.RawMessage(`{"type":"object"}`)},
			{Name: "b", Parameters: json.RawMessage(`{"type":"object"}`)},
		},
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "go"}}}},
			{Role: RoleAssistant, Content: []ContentPart{
				{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{ID: "c1", Name: "a", Arguments: json.RawMessage(`{}`)}},
				{Type: ContentTypeToolUse, ToolUse: &ToolUseContent{ID: "c2", Name: "b", Arguments: json.RawMessage(`{}`)}},
			}},
			{Role: RoleUser, Content: []ContentPart{
				{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{ToolUseID: "c1",
					Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "r1"}}}}},
				{Type: ContentTypeToolResult, ToolResult: &ToolResultContent{ToolUseID: "c2",
					Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "r2"}}}}},
			}},
		},
	}

	body, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if problems := validateOpenAIChatOutbound(body); len(problems) > 0 {
		t.Errorf("invalid outbound request: %v\n%s", problems, body)
	}

	var out struct {
		Messages []struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var ids []string
	for _, m := range out.Messages {
		if m.Role == "tool" {
			ids = append(ids, m.ToolCallID)
		}
	}
	if len(ids) != 2 || ids[0] != "c1" || ids[1] != "c2" {
		t.Errorf("tool messages = %v, want [c1 c2]", ids)
	}
}

// Regression: the "tool" role decoder assumed a string content, so the array form
// (emitted by Responses-style SDKs) was silently dropped, producing a tool result
// with no content.
func TestRegression_ChatDecode_ArrayFormToolContent(t *testing.T) {
	body := []byte(`{"model":"gpt-4o","messages":[
		{"role":"user","content":"go"},
		{"role":"assistant","tool_calls":[{"index":0,"id":"t","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"t","content":[{"type":"text","text":"result body"}]}
	]}`)

	req, err := DecodeOpenAIChatRequest(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	last := req.Messages[len(req.Messages)-1]
	if len(last.Content) != 1 || last.Content[0].ToolResult == nil {
		t.Fatalf("last message = %+v, want one tool_result part", last)
	}
	got := toolResultText(last.Content[0].ToolResult)
	if got != "result body" {
		t.Errorf("tool result text = %q, want %q", got, "result body")
	}
}

// Regression: functionResponse.response was double-encoded — an existing JSON
// object was re-marshalled into a string and wrapped, so a Gemini→Gemini
// round-trip destroyed the structure the model had produced.
func TestRegression_GeminiEncode_FunctionResponseKeepsStructure(t *testing.T) {
	req := &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeToolResult,
				ToolResult: &ToolResultContent{
					ToolUseID: "t1",
					Name:      "get_weather",
					Content: []ContentPart{{Type: ContentTypeText,
						Text: &TextContent{Text: `{"temp":18,"unit":"C"}`}}},
				}}}},
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if problems := validateGeminiOutbound(body); len(problems) > 0 {
		t.Errorf("invalid outbound request: %v\n%s", problems, body)
	}

	var out struct {
		Contents []struct {
			Parts []struct {
				FunctionResponse *struct {
					Response map[string]any `json:"response"`
				} `json:"functionResponse"`
			} `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	fr := out.Contents[0].Parts[0].FunctionResponse
	if fr == nil {
		t.Fatalf("no functionResponse in %s", body)
	}
	if fr.Response["temp"] != float64(18) || fr.Response["unit"] != "C" {
		t.Errorf("response = %v; the JSON object must pass through verbatim", fr.Response)
	}
}

// Regression: ToolResultContent.IsError was dropped for every protocol without a
// structured error flag, so a failed tool looked successful to the model.
func TestRegression_ToolResultIsErrorMarkedInline(t *testing.T) {
	result := &ToolResultContent{
		ToolUseID: "t",
		IsError:   true,
		Content:   []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "file not found"}}},
	}
	got := toolResultTextWithError(result)
	if !strings.Contains(got, "file not found") {
		t.Errorf("text = %q, must retain the message", got)
	}
	if !strings.HasPrefix(got, toolErrorPrefix) {
		t.Errorf("text = %q, must be marked as an error", got)
	}
}

// --- Request validity ---

// Regression: when every declared tool was filtered out during encoding (e.g. an
// Anthropic server-side tool with no counterpart), tool_choice was left behind,
// producing a request the provider rejects.
func TestRegression_ToolChoiceDroppedWhenAllToolsFiltered(t *testing.T) {
	req := &Request{
		Model:     "m",
		MaxTokens: 100,
		Tools: []Tool{
			{Type: "web_search_20250305", Name: "web_search"},
		},
		ToolChoice: &ToolChoice{Type: "tool", ToolName: "web_search"},
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "search"}}}},
		},
	}

	for _, tc := range []struct {
		name     string
		protocol Protocol
		encode   func(*Request) ([]byte, error)
	}{
		{"openai_chat", ProtocolOpenAIChat, EncodeOpenAIChatRequest},
		{"openai_responses", ProtocolOpenAIResponses, EncodeOpenAIResponsesRequest},
		{"gemini", ProtocolGemini, func(r *Request) ([]byte, error) { _, b, e := EncodeGeminiRequest(r); return b, e }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body, err := tc.encode(req)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if problems := validateOutboundBody(tc.protocol, body); len(problems) > 0 {
				t.Errorf("invalid outbound request: %v\n%s", problems, body)
			}
		})
	}
}

// Regression: a message whose every part had no target representation encoded to
// an empty content array, which OpenAI and Gemini both reject.
func TestRegression_NoEmptyContentArrays(t *testing.T) {
	// A document part has no OpenAI Chat representation at all.
	req := &Request{
		Model:     "m",
		MaxTokens: 100,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeDocument,
				Document: &DocumentContent{MediaType: "application/pdf", Data: []byte("%PDF-"), Title: "report.pdf"}}}},
		},
	}

	chatBody, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("encode chat: %v", err)
	}
	if problems := validateOpenAIChatOutbound(chatBody); len(problems) > 0 {
		t.Errorf("invalid chat request: %v\n%s", problems, chatBody)
	}
	// The placeholder must mention the document so the model is not left guessing.
	if !strings.Contains(string(chatBody), "report.pdf") {
		t.Errorf("document was dropped without a placeholder:\n%s", chatBody)
	}

	_, geminiBody, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("encode gemini: %v", err)
	}
	if problems := validateGeminiOutbound(geminiBody); len(problems) > 0 {
		t.Errorf("invalid gemini request: %v\n%s", problems, geminiBody)
	}
}

// Regression: response_format.json_schema.name is required by OpenAI but has no
// equivalent in Anthropic or Gemini, so a cross-protocol request arrived without
// one and was rejected. A name must be synthesised.
func TestRegression_JSONSchemaNameAlwaysPresent(t *testing.T) {
	req := &Request{
		Model:     "m",
		MaxTokens: 100,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "json"}}}}},
		ResponseFormat: &ResponseFormat{
			Type:       "json_schema",
			JSONSchema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}}}`),
		},
	}

	chatBody, err := EncodeOpenAIChatRequest(req)
	if err != nil {
		t.Fatalf("encode chat: %v", err)
	}
	if problems := validateOpenAIChatOutbound(chatBody); len(problems) > 0 {
		t.Errorf("invalid chat request: %v\n%s", problems, chatBody)
	}

	respBody, err := EncodeOpenAIResponsesRequest(req)
	if err != nil {
		t.Fatalf("encode responses: %v", err)
	}
	if problems := validateOpenAIResponsesOutbound(respBody); len(problems) > 0 {
		t.Errorf("invalid responses request: %v\n%s", problems, respBody)
	}
}

// Regression: a data: URI in a Responses input_image was stored as ImageContent.URL
// verbatim, so cross-protocol encoders emitted source.type=url with a data: URI —
// which Anthropic and Gemini both reject.
func TestRegression_ResponsesDataURIDecodedToBytes(t *testing.T) {
	body := []byte(`{"model":"gpt-4o","input":[{"type":"message","role":"user","content":[
		{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8="}]}]}`)

	req, err := DecodeOpenAIResponsesRequest(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	img := req.Messages[0].Content[0].Image
	if img == nil {
		t.Fatal("no image part decoded")
	}
	if img.URL != "" {
		t.Errorf("URL = %q, a data: URI must not be kept as a URL", img.URL)
	}
	if img.MediaType != "image/png" || string(img.Data) != "hello" {
		t.Errorf("MediaType=%q Data=%q, want image/png / hello", img.MediaType, img.Data)
	}

	// It must now encode into every protocol's native inline-image form.
	anthropicBody, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("encode anthropic: %v", err)
	}
	if !strings.Contains(string(anthropicBody), `"type":"base64"`) {
		t.Errorf("anthropic image is not base64-sourced:\n%s", anthropicBody)
	}
}

// Regression: optional Anthropic request fields were emitted as explicit nulls
// (no omitempty), which strict proxies in front of the API reject.
func TestRegression_AnthropicRequestOmitsNulls(t *testing.T) {
	req := &Request{
		Model:     "claude-3",
		MaxTokens: 100,
		Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}}},
	}
	body, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	for _, field := range []string{"tool_choice", "thinking", "system", "tools",
		"temperature", "top_p", "top_k", "stop_sequences"} {
		if strings.Contains(string(body), `"`+field+`":null`) {
			t.Errorf("field %q emitted as explicit null:\n%s", field, body)
		}
	}
}

// Regression: reasoning_effort → Anthropic thinking produced
// {"type":"enabled"} with no budget_tokens, which the API rejects.
func TestRegression_AnthropicThinkingAlwaysHasBudget(t *testing.T) {
	for _, effort := range []string{"minimal", "low", "medium", "high", ""} {
		t.Run("effort_"+effort, func(t *testing.T) {
			req := &Request{
				Model:     "claude-3",
				MaxTokens: 20000,
				Messages:  []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}}},
				Thinking:  &ThinkingConfig{Mode: "enabled", Effort: effort},
			}
			body, err := EncodeAnthropicRequest(req)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if problems := validateAnthropicOutbound(body); len(problems) > 0 {
				t.Errorf("invalid outbound request: %v\n%s", problems, body)
			}
		})
	}
}

// Regression: an explicit Gemini thinkingBudget of 0 (thinking off) decoded to
// "adaptive" and then re-encoded to nothing, silently turning thinking back on.
func TestRegression_GeminiThinkingDisabledRoundTrips(t *testing.T) {
	body := []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"generationConfig":{"thinkingConfig":{"thinkingBudget":0}}}`)

	req, err := DecodeGeminiRequest("/v1beta/models/gemini-2.5-pro:generateContent", body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.Thinking == nil || req.Thinking.Mode != "disabled" {
		t.Fatalf("Thinking = %+v, want mode=disabled", req.Thinking)
	}

	_, out, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !strings.Contains(string(out), `"thinkingBudget":0`) {
		t.Errorf("re-encoded request lost the explicit zero budget:\n%s", out)
	}
}

// Regression: non-string enum values were silently dropped by
// convertJSONSchemaToGemini (`case string` only), so an integer enum lost every
// constraint.
func TestRegression_GeminiSchemaNumericEnumPreserved(t *testing.T) {
	req := &Request{
		Model:    "gemini-2.5-pro",
		Messages: []Message{{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}}},
		ResponseFormat: &ResponseFormat{
			Type: "json_schema",
			JSONSchema: json.RawMessage(`{"type":"object","properties":{
				"n":{"type":"integer","enum":[1,2,3]},
				"b":{"type":"boolean","enum":[true,false]},
				"s":{"type":"string","enum":["x"],"minLength":1,"maxLength":4,"pattern":"^x$"}
			}}`),
		},
	}

	_, body, err := EncodeGeminiRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	s := string(body)
	for _, want := range []string{`"1"`, `"2"`, `"3"`, `"true"`, `"false"`,
		`"minLength":1`, `"maxLength":4`, `"pattern":"^x$"`} {
		if !strings.Contains(s, want) {
			t.Errorf("encoded schema is missing %s:\n%s", want, body)
		}
	}
	if problems := validateGeminiOutbound(body); len(problems) > 0 {
		t.Errorf("invalid gemini request: %v", problems)
	}
}

// Regression: an unmapped stop reason leaked through as a finish_reason /
// status / finishReason value outside each protocol's closed enum, breaking SDK
// parsers.
func TestRegression_StopReasonsStayInEnum(t *testing.T) {
	// "refusal" and "model_context_window_exceeded" reach the IR verbatim from
	// Anthropic and have no OpenAI/Gemini equivalent.
	for _, reason := range []StopReason{"refusal", "model_context_window_exceeded", "totally_unknown"} {
		t.Run(string(reason), func(t *testing.T) {
			resp := &Response{
				Model:      "m",
				StopReason: reason,
				Content:    []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "x"}}},
			}

			chatBody, err := EncodeOpenAIChatResponse(resp)
			if err != nil {
				t.Fatalf("encode chat: %v", err)
			}
			if _, err := extractResponseText(ProtocolOpenAIChat, chatBody); err != nil {
				t.Errorf("chat response invalid: %v\n%s", err, chatBody)
			}

			respBody, err := EncodeOpenAIResponsesResponse(resp)
			if err != nil {
				t.Fatalf("encode responses: %v", err)
			}
			if _, err := extractResponseText(ProtocolOpenAIResponses, respBody); err != nil {
				t.Errorf("responses response invalid: %v\n%s", err, respBody)
			}

			geminiBody, err := EncodeGeminiResponse(resp)
			if err != nil {
				t.Fatalf("encode gemini: %v", err)
			}
			if _, err := extractResponseText(ProtocolGemini, geminiBody); err != nil {
				t.Errorf("gemini response invalid: %v\n%s", err, geminiBody)
			}
		})
	}
}

// --- RawExtra direction ---

// Regression: RawExtra (the inbound request's unknown fields) was merged into the
// *response* body, so request-only options such as service_tier and safetySettings
// were echoed back to the client as if the model had produced them. RawExtra is an
// outbound-request mechanism; it must reach the upstream, not the client.
func TestRegression_RawExtraGoesUpstreamNotToClient(t *testing.T) {
	var captured []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = readAllBody(r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer server.Close()

	mux := NewMux(&staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  server.URL,
		APIKey:   "k",
		Model:    "gpt-4o",
	}})

	// service_tier is not an IR field, so it lands in RawExtra.
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"service_tier":"flex"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.OpenAIChatHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(string(captured), `"service_tier":"flex"`) {
		t.Errorf("service_tier did not reach the upstream request:\n%s", captured)
	}
	if strings.Contains(w.Body.String(), "service_tier") {
		t.Errorf("request-only field leaked into the client response:\n%s", w.Body.String())
	}
}

// --- Thinking / reasoning preservation ---

// Regression: thinking signatures and redacted_thinking blocks were dropped when
// an assistant turn passed through the Chat or Responses request history. Anthropic
// rejects a replayed thinking block whose signature is missing, so a fallback from
// an OpenAI target back to Anthropic would fail.
func TestRegression_ThinkingSignatureSurvivesRequestHistory(t *testing.T) {
	original := &Request{
		Model:     "m",
		MaxTokens: 1000,
		Messages: []Message{
			{Role: RoleUser, Content: []ContentPart{{Type: ContentTypeText, Text: &TextContent{Text: "hi"}}}},
			{Role: RoleAssistant, Content: []ContentPart{
				{Type: ContentTypeThinking, Thinking: &ThinkingContent{Thinking: "reasoning", Signature: "sig-abc"}},
				{Type: ContentTypeRedactedThinking, RedactedThinking: &RedactedThinkingContent{Data: "enc-xyz"}},
				{Type: ContentTypeText, Text: &TextContent{Text: "answer"}},
			}},
		},
	}

	t.Run("openai_chat", func(t *testing.T) {
		body, err := EncodeOpenAIChatRequest(original)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		back, err := DecodeOpenAIChatRequest(body)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		assertThinkingPreserved(t, back)
	})

	t.Run("openai_responses", func(t *testing.T) {
		body, err := EncodeOpenAIResponsesRequest(original)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		back, err := DecodeOpenAIResponsesRequest(body)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		assertThinkingPreserved(t, back)
	})
}

func assertThinkingPreserved(t *testing.T, req *Request) {
	t.Helper()
	var sig, redacted string
	for _, m := range req.Messages {
		for _, p := range m.Content {
			if p.Type == ContentTypeThinking && p.Thinking != nil && p.Thinking.Signature != "" {
				sig = p.Thinking.Signature
			}
			if p.Type == ContentTypeRedactedThinking && p.RedactedThinking != nil {
				redacted = p.RedactedThinking.Data
			}
		}
	}
	if sig != "sig-abc" {
		t.Errorf("thinking signature = %q, want sig-abc", sig)
	}
	if redacted != "enc-xyz" {
		t.Errorf("redacted_thinking data = %q, want enc-xyz", redacted)
	}
}

// --- Anthropic system field dual form ---

// Regression: the Anthropic decoder typed `system` as []ContentBlock, so the
// plain-string form the API also accepts failed with a JSON unmarshal error.
func TestRegression_AnthropicSystemAcceptsString(t *testing.T) {
	body := []byte(`{"model":"claude-3","max_tokens":100,"system":"Be brief.",
		"messages":[{"role":"user","content":"hi"}]}`)

	req, err := DecodeAnthropicRequest(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(req.SystemPrompt) != 1 || req.SystemPrompt[0].Text == nil ||
		req.SystemPrompt[0].Text.Text != "Be brief." {
		t.Fatalf("SystemPrompt = %+v, want one text part", req.SystemPrompt)
	}

	// It must re-encode into the block form the API also accepts.
	out, err := EncodeAnthropicRequest(req)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if problems := validateAnthropicOutbound(out); len(problems) > 0 {
		t.Errorf("invalid outbound request: %v\n%s", problems, out)
	}
	if !strings.Contains(string(out), "Be brief.") {
		t.Errorf("system prompt lost:\n%s", out)
	}
}

// --- Helpers ---

// runAnthropicStreamCodec runs the Anthropic inbound codec over ch and returns the
// raw SSE it produced.
func runAnthropicStreamCodec(t *testing.T, ch <-chan StreamResult) string {
	t.Helper()
	rec := httptest.NewRecorder()
	writer := NewSSEWriter(rec)
	codecFor(ProtocolAnthropic).WriteStreamingResponse(writer, ch)
	return rec.Body.String()
}

// codecFor returns the inbound codec for a protocol.
func codecFor(p Protocol) inboundCodec {
	switch p {
	case ProtocolAnthropic:
		return &anthropicCodec{}
	case ProtocolOpenAIChat:
		return &openaiChatCodec{}
	case ProtocolOpenAIResponses:
		return &openaiResponsesCodec{}
	case ProtocolGemini:
		return &geminiCodec{}
	}
	panic("unknown protocol " + string(p))
}

// readAllBody drains an http.Request body.
func readAllBody(r *http.Request) ([]byte, error) {
	defer r.Body.Close() //nolint:errcheck
	var sb strings.Builder
	buf := make([]byte, 4096)
	for {
		n, err := r.Body.Read(buf)
		sb.Write(buf[:n])
		if err != nil {
			if err.Error() == "EOF" {
				return []byte(sb.String()), nil
			}
			return []byte(sb.String()), nil
		}
	}
}
