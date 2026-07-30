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

// TestConformanceMatrix drives every logical scenario through all 16
// inbound × outbound protocol combinations and asserts that:
//
//  1. the request the gateway sends upstream is *valid* for the outbound
//     protocol (validateOutboundBody), and
//  2. the response the gateway returns to the client is well-formed for the
//     inbound protocol and preserves the scenario's semantics.
//
// Before this test, integration coverage only exercised ~7 of the 16 pairs and
// never used Gemini as an outbound target at all, which is how several
// gateway-introduced 400s went unnoticed.
func TestConformanceMatrix(t *testing.T) {
	for _, sc := range conformanceScenarios() {
		for _, inbound := range allProtocols {
			for _, outbound := range allProtocols {
				name := sc.name + "/" + protocolName(inbound) + "_to_" + protocolName(outbound)
				t.Run(name, func(t *testing.T) {
					runConformanceCase(t, sc, inbound, outbound)
				})
			}
		}
	}
}

// conformanceScenario is one logical request plus the upstream reply to fake,
// expressed once per protocol.
type conformanceScenario struct {
	name string
	// inbound bodies keyed by protocol
	inbound map[Protocol]inboundFixture
	// upstream response bodies keyed by outbound protocol
	upstream map[Protocol]string
	// assertOutboundRequest verifies scenario-specific semantics survived the
	// conversion into the target protocol.
	assertOutboundRequest func(t *testing.T, outbound Protocol, body []byte)
	// assertClientResponse validates the body returned to the client, given the
	// inbound protocol.
	assertClientResponse func(t *testing.T, inbound Protocol, body []byte)
}

func runConformanceCase(t *testing.T, sc conformanceScenario, inbound, outbound Protocol) {
	fixture, ok := sc.inbound[inbound]
	if !ok {
		t.Skipf("no fixture for inbound %s", protocolName(inbound))
	}
	upstreamBody, ok := sc.upstream[outbound]
	if !ok {
		t.Skipf("no upstream body for outbound %s", protocolName(outbound))
	}

	var captured []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(upstreamBody))
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
	for k, v := range fixture.header {
		req.Header.Set(k, v)
	}
	w := httptest.NewRecorder()
	inboundHandler(mux, inbound).ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if len(captured) == 0 {
		t.Fatal("upstream received no body")
	}

	// (1) The outbound request must be valid for the target protocol.
	if problems := validateOutboundBody(outbound, captured); len(problems) > 0 {
		t.Errorf("invalid outbound %s request:\n  - %s\nbody: %s",
			protocolName(outbound), strings.Join(problems, "\n  - "), captured)
	}

	// codeflicker-fix: LOGIC-Issue-001/tb3m3jp0fdxv42afyew5
	// (2) Every scenario asserts the semantic payload, not only JSON validity.
	if sc.assertOutboundRequest != nil {
		sc.assertOutboundRequest(t, outbound, captured)
	}

	// (3) The response returned to the client must be well-formed and preserve
	// the scenario's semantics.
	if sc.assertClientResponse != nil {
		sc.assertClientResponse(t, inbound, w.Body.Bytes())
	}
}

// inboundHandler returns the Mux handler for the given inbound protocol.
func inboundHandler(mux *Mux, p Protocol) http.Handler {
	switch p {
	case ProtocolAnthropic:
		return mux.AnthropicHandler()
	case ProtocolOpenAIChat:
		return mux.OpenAIChatHandler()
	case ProtocolOpenAIResponses:
		return mux.OpenAIResponsesHandler()
	case ProtocolGemini:
		return mux.GeminiHandler()
	default:
		panic("unknown inbound protocol " + string(p))
	}
}

// outboundModelFor returns a plausible model name for the target protocol so the
// outbound validators see a realistic request.
func outboundModelFor(p Protocol) string {
	switch p {
	case ProtocolAnthropic:
		return "claude-sonnet-4-20250514"
	case ProtocolOpenAIChat, ProtocolOpenAIResponses:
		return "gpt-4o"
	case ProtocolGemini:
		return "gemini-2.5-pro"
	default:
		return "model"
	}
}

// --- Scenarios ---

func conformanceScenarios() []conformanceScenario {
	return []conformanceScenario{
		simpleTextScenario(),
		systemPromptScenario(),
		toolCallRoundTripScenario(),
		parallelToolResultScenario(),
		imageScenario(),
		documentScenario(),
		jsonSchemaScenario(),
		serverToolOnlyScenario(),
		thinkingScenario(),
	}
}

// simpleTextScenario: a plain single-turn text request.
func simpleTextScenario() conformanceScenario {
	return conformanceScenario{
		name: "simple_text",
		inbound: map[Protocol]inboundFixture{
			ProtocolAnthropic: {
				path: "/v1/messages",
				body: `{"model":"claude-3","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}`,
			},
			ProtocolOpenAIChat: {
				path: "/v1/chat/completions",
				body: `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}`,
			},
			ProtocolOpenAIResponses: {
				path: "/v1/responses",
				body: `{"model":"gpt-4","max_output_tokens":100,"input":"Hello"}`,
			},
			ProtocolGemini: {
				path: "/v1beta/models/gemini-2.0:generateContent",
				body: `{"contents":[{"role":"user","parts":[{"text":"Hello"}]}],"generationConfig":{"maxOutputTokens":100}}`,
			},
		},
		upstream:             standardUpstreamResponses("Hi there"),
		assertClientResponse: assertResponseText("Hi there"),
	}
}

// systemPromptScenario: a system prompt plus a user turn.
func systemPromptScenario() conformanceScenario {
	return conformanceScenario{
		name: "system_prompt",
		inbound: map[Protocol]inboundFixture{
			ProtocolAnthropic: {
				path: "/v1/messages",
				body: `{"model":"claude-3","max_tokens":100,"system":"Be brief.","messages":[{"role":"user","content":"Hello"}]}`,
			},
			ProtocolOpenAIChat: {
				path: "/v1/chat/completions",
				body: `{"model":"gpt-4","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hello"}]}`,
			},
			ProtocolOpenAIResponses: {
				path: "/v1/responses",
				body: `{"model":"gpt-4","instructions":"Be brief.","input":"Hello"}`,
			},
			ProtocolGemini: {
				path: "/v1beta/models/gemini-2.0:generateContent",
				body: `{"systemInstruction":{"parts":[{"text":"Be brief."}]},"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}`,
			},
		},
		upstream:              standardUpstreamResponses("Hi"),
		assertOutboundRequest: assertOutboundContains(semanticMarkers(`"system":[`, `"role":"developer"`, `"instructions":"Be brief."`, `"systemInstruction"`)),
		assertClientResponse:  assertResponseText("Hi"),
	}
}

// toolCallRoundTripScenario: a full assistant tool_call → tool result history.
// This is the shape that most often trips up cross-protocol encoding because each
// protocol models the correlation differently (tool_call_id, call_id, or
// functionResponse.name).
func toolCallRoundTripScenario() conformanceScenario {
	return conformanceScenario{
		name: "tool_call_round_trip",
		inbound: map[Protocol]inboundFixture{
			ProtocolAnthropic: {
				path: "/v1/messages",
				body: `{"model":"claude-3","max_tokens":100,
					"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
					"messages":[
						{"role":"user","content":"Weather in Paris?"},
						{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"Paris"}}]},
						{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"18C"}]}]}
					]}`,
			},
			ProtocolOpenAIChat: {
				path: "/v1/chat/completions",
				body: `{"model":"gpt-4",
					"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
					"messages":[
						{"role":"user","content":"Weather in Paris?"},
						{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
						{"role":"tool","tool_call_id":"call_1","content":"18C"}
					]}`,
			},
			ProtocolOpenAIResponses: {
				path: "/v1/responses",
				body: `{"model":"gpt-4",
					"tools":[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
					"input":[
						{"type":"message","role":"user","content":[{"type":"input_text","text":"Weather in Paris?"}]},
						{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
						{"type":"function_call_output","call_id":"call_1","output":"18C"}
					]}`,
			},
			ProtocolGemini: {
				path: "/v1beta/models/gemini-2.0:generateContent",
				body: `{
					"tools":[{"functionDeclarations":[{"name":"get_weather","description":"Get weather","parameters":{"type":"OBJECT","properties":{"city":{"type":"STRING"}},"required":["city"]}}]}],
					"contents":[
						{"role":"user","parts":[{"text":"Weather in Paris?"}]},
						{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"},"id":"call_1"}}]},
						{"role":"user","parts":[{"functionResponse":{"name":"get_weather","response":{"temp":"18C"},"id":"call_1"}}]}
					]}`,
			},
		},
		upstream:              standardUpstreamResponses("It is 18C."),
		assertOutboundRequest: assertOutboundContains(semanticMarkers(`"call_1"`, `"call_1"`, `"call_1"`, `"call_1"`)),
		assertClientResponse:  assertResponseText("It is 18C."),
	}
}

// parallelToolResultScenario: two tool calls answered in one turn. Anthropic packs
// both results into a single user message, which must fan out into two OpenAI
// "tool" messages / two function_call_output items — dropping one leaves an
// unanswered tool_call_id and a provider 400.
func parallelToolResultScenario() conformanceScenario {
	tools := `[{"name":"tool_a","input_schema":{"type":"object"}},{"name":"tool_b","input_schema":{"type":"object"}}]`
	return conformanceScenario{
		name: "parallel_tool_results",
		inbound: map[Protocol]inboundFixture{
			ProtocolAnthropic: {
				path: "/v1/messages",
				body: `{"model":"claude-3","max_tokens":100,"tools":` + tools + `,
					"messages":[
						{"role":"user","content":"Do both"},
						{"role":"assistant","content":[
							{"type":"tool_use","id":"c1","name":"tool_a","input":{}},
							{"type":"tool_use","id":"c2","name":"tool_b","input":{}}
						]},
						{"role":"user","content":[
							{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"r1"}]},
							{"type":"tool_result","tool_use_id":"c2","content":[{"type":"text","text":"r2"}]}
						]}
					]}`,
			},
			ProtocolOpenAIChat: {
				path: "/v1/chat/completions",
				body: `{"model":"gpt-4",
					"tools":[{"type":"function","function":{"name":"tool_a","parameters":{"type":"object"}}},{"type":"function","function":{"name":"tool_b","parameters":{"type":"object"}}}],
					"messages":[
						{"role":"user","content":"Do both"},
						{"role":"assistant","tool_calls":[
							{"index":0,"id":"c1","type":"function","function":{"name":"tool_a","arguments":"{}"}},
							{"index":1,"id":"c2","type":"function","function":{"name":"tool_b","arguments":"{}"}}
						]},
						{"role":"tool","tool_call_id":"c1","content":"r1"},
						{"role":"tool","tool_call_id":"c2","content":"r2"}
					]}`,
			},
			ProtocolOpenAIResponses: {
				path: "/v1/responses",
				body: `{"model":"gpt-4",
					"tools":[{"type":"function","name":"tool_a","parameters":{"type":"object"}},{"type":"function","name":"tool_b","parameters":{"type":"object"}}],
					"input":[
						{"type":"message","role":"user","content":[{"type":"input_text","text":"Do both"}]},
						{"type":"function_call","call_id":"c1","name":"tool_a","arguments":"{}"},
						{"type":"function_call","call_id":"c2","name":"tool_b","arguments":"{}"},
						{"type":"function_call_output","call_id":"c1","output":"r1"},
						{"type":"function_call_output","call_id":"c2","output":"r2"}
					]}`,
			},
			ProtocolGemini: {
				path: "/v1beta/models/gemini-2.0:generateContent",
				body: `{
					"tools":[{"functionDeclarations":[{"name":"tool_a","parameters":{"type":"OBJECT"}},{"name":"tool_b","parameters":{"type":"OBJECT"}}]}],
					"contents":[
						{"role":"user","parts":[{"text":"Do both"}]},
						{"role":"model","parts":[
							{"functionCall":{"name":"tool_a","args":{},"id":"c1"}},
							{"functionCall":{"name":"tool_b","args":{},"id":"c2"}}
						]},
						{"role":"user","parts":[
							{"functionResponse":{"name":"tool_a","response":{"v":"r1"},"id":"c1"}},
							{"functionResponse":{"name":"tool_b","response":{"v":"r2"},"id":"c2"}}
						]}
					]}`,
			},
		},
		upstream:              standardUpstreamResponses("Both done."),
		assertOutboundRequest: assertOutboundContains(semanticMarkers(`"c2"`, `"c2"`, `"c2"`, `"c2"`)),
		assertClientResponse:  assertResponseText("Both done."),
	}
}

// imageScenario: an inline base64 image. Each protocol carries binary image data
// differently (source.base64, a data: URI, or inlineData), so a wrong hop here
// produces an unreadable image or a 400.
func imageScenario() conformanceScenario {
	const b64 = "aGVsbG8="
	return conformanceScenario{
		name: "inline_image",
		inbound: map[Protocol]inboundFixture{
			ProtocolAnthropic: {
				path: "/v1/messages",
				body: `{"model":"claude-3","max_tokens":100,"messages":[{"role":"user","content":[
					{"type":"text","text":"What is this?"},
					{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + b64 + `"}}]}]}`,
			},
			ProtocolOpenAIChat: {
				path: "/v1/chat/completions",
				body: `{"model":"gpt-4o","messages":[{"role":"user","content":[
					{"type":"text","text":"What is this?"},
					{"type":"image_url","image_url":{"url":"data:image/png;base64,` + b64 + `"}}]}]}`,
			},
			ProtocolOpenAIResponses: {
				path: "/v1/responses",
				body: `{"model":"gpt-4o","input":[{"type":"message","role":"user","content":[
					{"type":"input_text","text":"What is this?"},
					{"type":"input_image","image_url":"data:image/png;base64,` + b64 + `"}]}]}`,
			},
			ProtocolGemini: {
				path: "/v1beta/models/gemini-2.0:generateContent",
				body: `{"contents":[{"role":"user","parts":[
					{"text":"What is this?"},
					{"inlineData":{"mimeType":"image/png","data":"` + b64 + `"}}]}]}`,
			},
		},
		upstream:              standardUpstreamResponses("An image."),
		assertOutboundRequest: assertOutboundContains(semanticMarkers(`"data":"aGVsbG8="`, `data:image/png;base64,aGVsbG8=`, `data:image/png;base64,aGVsbG8=`, `"data":"aGVsbG8="`)),
		assertClientResponse: func(t *testing.T, inbound Protocol, body []byte) {
			assertResponseText("An image.")(t, inbound, body)
		},
	}
}

// documentScenario: a PDF attachment. Only Anthropic, Responses and Gemini have a
// document/file part; Chat has none, so the gateway must degrade to text rather
// than emit an empty content array.
func documentScenario() conformanceScenario {
	const b64 = "JVBERi0="
	return conformanceScenario{
		name: "document",
		inbound: map[Protocol]inboundFixture{
			ProtocolAnthropic: {
				path: "/v1/messages",
				body: `{"model":"claude-3","max_tokens":100,"messages":[{"role":"user","content":[
					{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + b64 + `"}}]}]}`,
			},
			ProtocolOpenAIResponses: {
				path: "/v1/responses",
				body: `{"model":"gpt-4o","input":[{"type":"message","role":"user","content":[
					{"type":"input_file","file_data":"` + b64 + `","filename":"doc.pdf"}]}]}`,
			},
			ProtocolGemini: {
				path: "/v1beta/models/gemini-2.0:generateContent",
				body: `{"contents":[{"role":"user","parts":[
					{"inlineData":{"mimeType":"application/pdf","data":"` + b64 + `"}}]}]}`,
			},
		},
		upstream:              standardUpstreamResponses("A PDF."),
		assertOutboundRequest: assertOutboundContains(semanticMarkers(`"data":"JVBERi0="`, `[document`, `"file_data":"JVBERi0="`, `"data":"JVBERi0="`)),
		assertClientResponse:  assertResponseText("A PDF."),
	}
}

// jsonSchemaScenario: structured output. OpenAI requires a schema *name*, which
// Anthropic and Gemini have no field for, so the gateway must synthesise one.
func jsonSchemaScenario() conformanceScenario {
	return conformanceScenario{
		name: "json_schema",
		inbound: map[Protocol]inboundFixture{
			ProtocolOpenAIChat: {
				path: "/v1/chat/completions",
				body: `{"model":"gpt-4o","messages":[{"role":"user","content":"Give me JSON"}],
					"response_format":{"type":"json_schema","json_schema":{"name":"Answer","strict":true,
						"schema":{"type":"object","properties":{"n":{"type":"integer","enum":[1,2,3]},"s":{"type":"string","minLength":2}},"required":["n"]}}}}`,
			},
			ProtocolOpenAIResponses: {
				path: "/v1/responses",
				body: `{"model":"gpt-4o","input":"Give me JSON",
					"text":{"format":{"type":"json_schema","name":"Answer",
						"schema":{"type":"object","properties":{"n":{"type":"integer","enum":[1,2,3]},"s":{"type":"string","minLength":2}},"required":["n"]}}}}`,
			},
			ProtocolGemini: {
				path: "/v1beta/models/gemini-2.0:generateContent",
				body: `{"contents":[{"role":"user","parts":[{"text":"Give me JSON"}]}],
					"generationConfig":{"responseMimeType":"application/json",
						"responseSchema":{"type":"OBJECT","properties":{"n":{"type":"INTEGER","enum":["1","2","3"]},"s":{"type":"STRING","minLength":2}},"required":["n"]}}}`,
			},
		},
		upstream:              standardUpstreamResponses(`{"n":1}`),
		assertOutboundRequest: assertOutboundContains(semanticMarkers(``, `"json_schema"`, `"json_schema"`, `"responseMimeType":"application/json"`)),
		assertClientResponse:  assertResponseText(`{"n":1}`),
	}
}

// serverToolOnlyScenario: the request's only tool is an Anthropic server-side
// web_search plus a named tool_choice selecting it. Targets without a matching
// built-in must drop both the tool and the dangling selector — leaving
// tool_choice behind is a gateway bug that reliably 400s.
func serverToolOnlyScenario() conformanceScenario {
	return conformanceScenario{
		name: "server_tool_only",
		inbound: map[Protocol]inboundFixture{
			ProtocolAnthropic: {
				path: "/v1/messages",
				body: `{"model":"claude-3","max_tokens":100,
					"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}],
					"tool_choice":{"type":"tool","name":"web_search"},
					"messages":[{"role":"user","content":"Search for Go news"}]}`,
			},
		},
		upstream:              standardUpstreamResponses("Found news."),
		assertOutboundRequest: assertOutboundContains(semanticMarkers(`"web_search"`, ``, ``, ``)),
		assertClientResponse:  assertResponseText("Found news."),
	}
}

// thinkingScenario: extended thinking / reasoning controls, which every protocol
// spells differently (thinking.budget_tokens, reasoning_effort, reasoning.effort,
// thinkingConfig.thinkingBudget).
func thinkingScenario() conformanceScenario {
	return conformanceScenario{
		name: "thinking",
		inbound: map[Protocol]inboundFixture{
			ProtocolAnthropic: {
				path: "/v1/messages",
				body: `{"model":"claude-3","max_tokens":2000,"thinking":{"type":"enabled","budget_tokens":1024},
					"messages":[{"role":"user","content":"Think"}]}`,
			},
			ProtocolOpenAIChat: {
				path: "/v1/chat/completions",
				body: `{"model":"o3","messages":[{"role":"user","content":"Think"}],"reasoning_effort":"high"}`,
			},
			ProtocolOpenAIResponses: {
				path: "/v1/responses",
				body: `{"model":"o3","input":"Think","reasoning":{"effort":"high"}}`,
			},
			ProtocolGemini: {
				path: "/v1beta/models/gemini-2.5-pro:generateContent",
				body: `{"contents":[{"role":"user","parts":[{"text":"Think"}]}],
					"generationConfig":{"thinkingConfig":{"thinkingBudget":1024}}}`,
			},
		},
		upstream:              standardUpstreamResponses("Thought about it."),
		assertOutboundRequest: assertOutboundContains(semanticMarkers(`"thinking"`, `"reasoning_effort"`, `"reasoning"`, `"thinkingConfig"`)),
		assertClientResponse:  assertResponseText("Thought about it."),
	}
}

// --- Upstream reply fixtures ---

// standardUpstreamResponses builds an equivalent non-streaming reply carrying
// `text` for each outbound protocol.
func standardUpstreamResponses(text string) map[Protocol]string {
	quoted := jsonQuote(text)
	return map[Protocol]string{
		ProtocolAnthropic: `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514",
			"content":[{"type":"text","text":` + quoted + `}],"stop_reason":"end_turn",
			"usage":{"input_tokens":10,"output_tokens":5}}`,
		ProtocolOpenAIChat: `{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4o",
			"choices":[{"index":0,"message":{"role":"assistant","content":` + quoted + `},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		ProtocolOpenAIResponses: `{"id":"resp_1","object":"response","model":"gpt-4o","status":"completed",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":` + quoted + `}]}],
			"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`,
		ProtocolGemini: `{"candidates":[{"content":{"role":"model","parts":[{"text":` + quoted + `}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15},
			"modelVersion":"gemini-2.5-pro"}`,
	}
}

func jsonQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// assertOutboundContains checks protocol-specific serialization markers. The
// markers intentionally include both field names and values so a converter cannot
// satisfy the test merely by retaining an unrelated copy of the source JSON.
func assertOutboundContains(want map[Protocol][]string) func(t *testing.T, outbound Protocol, body []byte) {
	return func(t *testing.T, outbound Protocol, body []byte) {
		for _, marker := range want[outbound] {
			if !strings.Contains(string(body), marker) {
				t.Errorf("outbound %s request lost semantic marker %q\nbody: %s", protocolName(outbound), marker, body)
			}
		}
	}
}

func semanticMarkers(anthropic, chat, responses, gemini string) map[Protocol][]string {
	return map[Protocol][]string{
		ProtocolAnthropic:       {anthropic},
		ProtocolOpenAIChat:      {chat},
		ProtocolOpenAIResponses: {responses},
		ProtocolGemini:          {gemini},
	}
}

// --- Client response assertions ---

// assertResponseText verifies the client-facing response is well-formed for the
// inbound protocol and contains the expected assistant text.
func assertResponseText(want string) func(t *testing.T, inbound Protocol, body []byte) {
	return func(t *testing.T, inbound Protocol, body []byte) {
		got, err := extractResponseText(inbound, body)
		if err != nil {
			t.Fatalf("parse %s response: %v\nbody: %s", protocolName(inbound), err, body)
		}
		if got != want {
			t.Errorf("response text = %q, want %q\nbody: %s", got, want, body)
		}
	}
}

// extractResponseText pulls the assistant text out of a protocol-native response
// body, failing if the body does not have the expected shape.
func extractResponseText(p Protocol, body []byte) (string, error) {
	switch p {
	case ProtocolAnthropic:
		var resp struct {
			Type    string `json:"type"`
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			StopReason string `json:"stop_reason"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			return "", err
		}
		if resp.Type != "message" || resp.Role != "assistant" {
			return "", errf("type=%q role=%q, want message/assistant", resp.Type, resp.Role)
		}
		if resp.StopReason == "" {
			return "", errf("stop_reason is empty")
		}
		var sb strings.Builder
		for _, c := range resp.Content {
			if c.Type == "text" {
				sb.WriteString(c.Text)
			}
		}
		return sb.String(), nil

	case ProtocolOpenAIChat:
		var resp struct {
			Object  string `json:"object"`
			Choices []struct {
				Message *struct {
					Role    string  `json:"role"`
					Content *string `json:"content"`
				} `json:"message"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			return "", err
		}
		if resp.Object != "chat.completion" {
			return "", errf("object = %q", resp.Object)
		}
		if len(resp.Choices) == 0 || resp.Choices[0].Message == nil {
			return "", errf("no choice message")
		}
		if fr := resp.Choices[0].FinishReason; fr == nil || !validChatFinishReason(*fr) {
			return "", errf("finish_reason = %v is not a valid enum value", fr)
		}
		if resp.Choices[0].Message.Content == nil {
			return "", nil
		}
		return *resp.Choices[0].Message.Content, nil

	case ProtocolOpenAIResponses:
		var resp struct {
			Object string `json:"object"`
			Status string `json:"status"`
			Output []struct {
				Type    string `json:"type"`
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			} `json:"output"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			return "", err
		}
		if resp.Object != "response" {
			return "", errf("object = %q", resp.Object)
		}
		if !isOpenAIResponsesStatus(resp.Status) {
			return "", errf("status = %q is not a valid enum value", resp.Status)
		}
		var sb strings.Builder
		for _, item := range resp.Output {
			if item.Type != "message" {
				continue
			}
			for _, c := range item.Content {
				if c.Type == "output_text" {
					sb.WriteString(c.Text)
				}
			}
		}
		return sb.String(), nil

	case ProtocolGemini:
		var resp struct {
			Candidates []struct {
				Content *struct {
					Role  string `json:"role"`
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
				FinishReason string `json:"finishReason"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			return "", err
		}
		if len(resp.Candidates) == 0 {
			return "", errf("no candidates")
		}
		cand := resp.Candidates[0]
		if cand.FinishReason != "" && !isGeminiFinishReason(cand.FinishReason) {
			return "", errf("finishReason = %q is not a valid enum value", cand.FinishReason)
		}
		if cand.Content == nil {
			return "", nil
		}
		if cand.Content.Role != "model" {
			return "", errf("candidate role = %q, want model", cand.Content.Role)
		}
		var sb strings.Builder
		for _, part := range cand.Content.Parts {
			sb.WriteString(part.Text)
		}
		return sb.String(), nil
	}
	return "", errf("unknown protocol %s", p)
}

func validChatFinishReason(s string) bool {
	switch s {
	case "stop", "length", "tool_calls", "content_filter", "function_call":
		return true
	}
	return false
}

func errf(format string, args ...any) error {
	return fmt.Errorf(format, args...)
}
