package e2e_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"

	llmapimux "github.com/mahoushoujoarale/llmapimux"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	openai "github.com/openai/openai-go/v3"
	openaiopt "github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	openaishared "github.com/openai/openai-go/v3/shared"
	"google.golang.org/genai"
)

// ============================================================
// Test 1: Anthropic SDK → OpenAI Chat, multi-turn tool use (non-streaming)
// ============================================================

func TestE2E_AnthropicSDK_ToolUse_ToOpenAIChat(t *testing.T) {
	var mu sync.Mutex
	var requestCount int
	var round2Captured *e2eCapturedRequest

	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		mu.Lock()
		requestCount++
		round := requestCount
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_abc","type":"function","function":{"name":"multiply","arguments":"{\"a\":6,\"b\":7}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
		} else {
			mu.Lock()
			round2Captured = captured
			mu.Unlock()
			_, _ = w.Write([]byte(`{"id":"chatcmpl-2","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"The result of 6 × 7 is 42."},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}`))
		}
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/messages",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"claude-sonnet-4-20250514": "gpt-4o-mini"}).AnthropicHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		option.WithAPIKey("sk-anthropic-inbound"),
		option.WithBaseURL(muxServer.URL),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	// Round 1: send user message with tools
	round1, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:      anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens:  1024,
		Tools:      []anthropic.ToolUnionParam{anthropicMultiplyTool},
		ToolChoice: anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{Type: "any"}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("multiply 6*7")),
		},
	})
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}

	if round1.StopReason != anthropic.StopReasonToolUse {
		t.Fatalf("round 1 stop_reason = %q, want tool_use", round1.StopReason)
	}

	// Find tool_use block
	var toolUseID string
	for _, block := range round1.Content {
		if tu, ok := block.AsAny().(anthropic.ToolUseBlock); ok && tu.Name == "multiply" {
			toolUseID = tu.ID
		}
	}
	if toolUseID == "" {
		t.Fatal("no multiply tool_use block in round 1 response")
	}

	// Round 2: send full history with tool result
	round2, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 1024,
		Tools:     []anthropic.ToolUnionParam{anthropicMultiplyTool},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("multiply 6*7")),
			round1.ToParam(),
			anthropic.NewUserMessage(anthropic.NewToolResultBlock(toolUseID, "42", false)),
		},
	})
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}

	// Verify round 2 request body has a "tool" role message
	mu.Lock()
	r2cap := round2Captured
	mu.Unlock()
	if r2cap == nil {
		t.Fatal("round 2 upstream request not captured")
	}
	r2body := decodeCapturedJSONBody(t, r2cap)
	messages, _ := r2body["messages"].([]any)
	var foundToolMsg bool
	for _, m := range messages {
		msg, _ := m.(map[string]any)
		if msg["role"] == "tool" {
			foundToolMsg = true
			if tcid, _ := msg["tool_call_id"].(string); tcid == "" {
				t.Fatal("tool message missing tool_call_id")
			}
			content, _ := msg["content"].(string)
			if !strings.Contains(content, "42") {
				t.Fatalf("tool message content = %q, want to contain 42", content)
			}
		}
	}
	if !foundToolMsg {
		t.Fatal("round 2 request missing tool role message")
	}

	// Verify SDK receives TextBlock with "42"
	var found bool
	for _, block := range round2.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			if strings.Contains(tb.Text, "42") {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("round 2 response missing text with 42")
	}
}

// ============================================================
// Test 2: Anthropic SDK → OpenAI Chat, multi-turn tool use (streaming)
// ============================================================

func TestE2E_AnthropicSDK_ToolUse_ToOpenAIChat_Stream(t *testing.T) {
	var mu sync.Mutex
	var requestCount int

	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		mu.Lock()
		requestCount++
		round := requestCount
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		if round == 1 {
			// Tool call stream
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"multiply\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":6,\\\"b\\\":7}\"}}]},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		} else {
			// Text stream
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-2\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-2\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"42\"},\"finish_reason\":null}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-2\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			flusher.Flush()
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
		}
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/messages",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"claude-sonnet-4-20250514": "gpt-4o-mini"}).AnthropicHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		option.WithAPIKey("sk-anthropic-inbound"),
		option.WithBaseURL(muxServer.URL),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	// Round 1: streaming tool call
	stream1 := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Model:      anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens:  1024,
		Tools:      []anthropic.ToolUnionParam{anthropicMultiplyTool},
		ToolChoice: anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{Type: "any"}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("multiply 6*7")),
		},
	})

	message1 := anthropic.Message{}
	for stream1.Next() {
		event := stream1.Current()
		if err := message1.Accumulate(event); err != nil {
			t.Fatalf("round 1 accumulate: %v", err)
		}
	}
	if err := stream1.Err(); err != nil {
		t.Fatalf("round 1 stream: %v", err)
	}

	if message1.StopReason != anthropic.StopReasonToolUse {
		t.Fatalf("round 1 stop_reason = %q, want tool_use", message1.StopReason)
	}

	// Find tool_use block
	var toolUseID string
	for _, block := range message1.Content {
		if tu, ok := block.AsAny().(anthropic.ToolUseBlock); ok && tu.Name == "multiply" {
			toolUseID = tu.ID
		}
	}
	if toolUseID == "" {
		t.Fatal("no multiply tool_use block in round 1 streaming response")
	}

	// Round 2: streaming text
	stream2 := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 1024,
		Tools:     []anthropic.ToolUnionParam{anthropicMultiplyTool},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("multiply 6*7")),
			message1.ToParam(),
			anthropic.NewUserMessage(anthropic.NewToolResultBlock(toolUseID, "42", false)),
		},
	})

	message2 := anthropic.Message{}
	for stream2.Next() {
		event := stream2.Current()
		if err := message2.Accumulate(event); err != nil {
			t.Fatalf("round 2 accumulate: %v", err)
		}
	}
	if err := stream2.Err(); err != nil {
		t.Fatalf("round 2 stream: %v", err)
	}

	// Verify SDK receives text "42"
	var found bool
	for _, block := range message2.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			if strings.Contains(tb.Text, "42") {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("round 2 streaming response missing text with 42")
	}
}

// ============================================================
// Test 3: Anthropic SDK → OpenAI Responses, multi-turn tool use
// (validates the mixed user message with tool_result + text fix)
// ============================================================

func TestE2E_AnthropicSDK_ToolUse_ToOpenAIResponses(t *testing.T) {
	var mu sync.Mutex
	var requestCount int
	var round2Captured *e2eCapturedRequest

	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		mu.Lock()
		requestCount++
		round := requestCount
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = w.Write([]byte(`{"id":"resp-1","object":"response","model":"gpt-4o-mini","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"multiply","arguments":"{\"a\":6,\"b\":7}"}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`))
		} else {
			mu.Lock()
			round2Captured = captured
			mu.Unlock()
			_, _ = w.Write([]byte(`{"id":"resp-2","object":"response","model":"gpt-4o-mini","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"The result is 42."}]}],"usage":{"input_tokens":20,"output_tokens":10,"total_tokens":30}}`))
		}
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/messages",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, upstream.URL, "sk-openai-upstream", map[string]string{"claude-sonnet-4-20250514": "gpt-4o-mini"}).AnthropicHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		option.WithAPIKey("sk-anthropic-inbound"),
		option.WithBaseURL(muxServer.URL),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	// Round 1: tool call
	round1, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:      anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens:  1024,
		Tools:      []anthropic.ToolUnionParam{anthropicMultiplyTool},
		ToolChoice: anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{Type: "any"}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("multiply 6*7")),
		},
	})
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}

	if round1.StopReason != anthropic.StopReasonToolUse {
		t.Fatalf("round 1 stop_reason = %q, want tool_use", round1.StopReason)
	}

	var toolUseID string
	for _, block := range round1.Content {
		if tu, ok := block.AsAny().(anthropic.ToolUseBlock); ok && tu.Name == "multiply" {
			toolUseID = tu.ID
		}
	}
	if toolUseID == "" {
		t.Fatal("no multiply tool_use block in round 1 response")
	}

	// Round 2: send full history with tool result
	round2, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 1024,
		Tools:     []anthropic.ToolUnionParam{anthropicMultiplyTool},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("multiply 6*7")),
			round1.ToParam(),
			anthropic.NewUserMessage(anthropic.NewToolResultBlock(toolUseID, "42", false)),
		},
	})
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}

	// Verify round 2 request body has a function_call_output item with call_id
	mu.Lock()
	r2cap := round2Captured
	mu.Unlock()
	if r2cap == nil {
		t.Fatal("round 2 upstream request not captured")
	}
	r2body := decodeCapturedJSONBody(t, r2cap)

	// OpenAI Responses format: input is an array of items
	input, _ := r2body["input"].([]any)
	var foundFCOutput bool
	for _, item := range input {
		itemMap, _ := item.(map[string]any)
		if itemMap["type"] == "function_call_output" {
			foundFCOutput = true
			callID, _ := itemMap["call_id"].(string)
			if callID == "" {
				t.Fatal("function_call_output missing call_id")
			}
			output, _ := itemMap["output"].(string)
			if !strings.Contains(output, "42") {
				t.Fatalf("function_call_output output = %q, want to contain 42", output)
			}
		}
	}
	if !foundFCOutput {
		t.Fatalf("round 2 request missing function_call_output item, got input: %s", string(r2cap.Body))
	}

	// Verify SDK receives TextBlock with "42"
	var found bool
	for _, block := range round2.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			if strings.Contains(tb.Text, "42") {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("round 2 response missing text with 42")
	}
}

// ============================================================
// Test 4: OpenAI Chat SDK → Anthropic, multi-turn tool use
// ============================================================

func TestE2E_OpenAIChatSDK_ToolUse_ToAnthropic(t *testing.T) {
	var mu sync.Mutex
	var requestCount int

	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		mu.Lock()
		requestCount++
		round := requestCount
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = w.Write([]byte(`{"id":"msg-1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"tool_use","id":"toolu_abc","name":"multiply","input":{"a":6,"b":7}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`))
		} else {
			_, _ = w.Write([]byte(`{"id":"msg-2","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"The result of multiplying 6 by 7 is 42."}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":10}}`))
		}
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/chat/completions",
		newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, upstream.URL, "sk-ant-upstream", map[string]string{"gpt-4o-mini": "claude-sonnet-4-20250514"}).OpenAIChatHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := openai.NewClient(
		openaiopt.WithAPIKey("sk-openai-inbound"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		openaiopt.WithMaxRetries(0),
	)

	// Round 1: send user message with tools
	round1, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:      "gpt-4o-mini",
		Tools:      []openai.ChatCompletionToolUnionParam{openAIMultiplyTool},
		ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")},
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("multiply 6*7"),
		},
	})
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}
	if len(round1.Choices) == 0 {
		t.Fatal("round 1: no choices")
	}
	choice := round1.Choices[0]
	if len(choice.Message.ToolCalls) == 0 {
		t.Fatalf("round 1: no tool calls (finish_reason=%q)", choice.FinishReason)
	}
	toolCall := choice.Message.ToolCalls[0]
	funcCall := toolCall.AsFunction()
	if funcCall.Function.Name != "multiply" {
		t.Fatalf("tool call name = %q, want multiply", funcCall.Function.Name)
	}

	// Round 2: send full history with tool result
	round2, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: "gpt-4o-mini",
		Tools: []openai.ChatCompletionToolUnionParam{openAIMultiplyTool},
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("multiply 6*7"),
			choice.Message.ToParam(),
			openai.ToolMessage("42", toolCall.ID),
		},
	})
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}
	if len(round2.Choices) == 0 {
		t.Fatal("round 2: no choices")
	}
	if !strings.Contains(round2.Choices[0].Message.Content, "42") {
		t.Fatalf("round 2 content = %q, want to contain 42", round2.Choices[0].Message.Content)
	}
}

// ============================================================
// Test 5: Gemini SDK → OpenAI Chat, multi-turn tool use
// ============================================================

func TestE2E_GeminiSDK_ToolUse_ToOpenAIChat(t *testing.T) {
	var mu sync.Mutex
	var requestCount int

	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		mu.Lock()
		requestCount++
		round := requestCount
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_abc","type":"function","function":{"name":"multiply","arguments":"{\"a\":6,\"b\":7}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
		} else {
			_, _ = w.Write([]byte(`{"id":"chatcmpl-2","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"The result of 6 × 7 is 42."},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}`))
		}
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/models/",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"gemini-2.5-pro": "gpt-4o-mini"}).GeminiHandler(),
	)
	defer muxServer.Close()

	geminiClient, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "fake-gemini-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
		HTTPClient: newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL),
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := newE2EContext(t)
	defer cancel()

	userContent := genai.NewContentFromParts(
		[]*genai.Part{genai.NewPartFromText("multiply 6*7")},
		genai.RoleUser,
	)

	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{geminiMultiplyTool},
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny},
		},
	}

	// Round 1
	round1, err := geminiClient.Models.GenerateContent(ctx, "gemini-2.5-pro",
		[]*genai.Content{userContent}, cfg)
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}
	calls := round1.FunctionCalls()
	if len(calls) == 0 {
		t.Fatal("round 1: expected function call, got none")
	}
	call := calls[0]
	if call.Name != "multiply" {
		t.Fatalf("function call name = %q, want multiply", call.Name)
	}

	// Round 2
	if len(round1.Candidates) == 0 || round1.Candidates[0].Content == nil {
		t.Fatal("round 1: missing candidate content for history")
	}
	toolResultContent := genai.NewContentFromParts(
		[]*genai.Part{genai.NewPartFromFunctionResponse(call.Name, map[string]any{"result": "42"})},
		genai.RoleUser,
	)
	round2Contents := []*genai.Content{userContent, round1.Candidates[0].Content, toolResultContent}

	cfg2 := &genai.GenerateContentConfig{Tools: []*genai.Tool{geminiMultiplyTool}}
	round2, err := geminiClient.Models.GenerateContent(ctx, "gemini-2.5-pro", round2Contents, cfg2)
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}
	if !strings.Contains(round2.Text(), "42") {
		t.Fatalf("round 2 text = %q, want to contain 42", round2.Text())
	}
}

// ============================================================
// Test 6: Anthropic SDK → OpenAI Chat, MULTIPLE tool calls in one response
// ============================================================

func TestE2E_AnthropicSDK_MultipleToolCalls_ToOpenAIChat(t *testing.T) {
	var mu sync.Mutex
	var requestCount int
	var round2Captured *e2eCapturedRequest

	upstream := newE2EUpstreamServer(t, func(w http.ResponseWriter, r *http.Request, captured *e2eCapturedRequest) {
		mu.Lock()
		requestCount++
		round := requestCount
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			// Two tool calls in one response
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_abc","type":"function","function":{"name":"multiply","arguments":"{\"a\":6,\"b\":7}"}},{"id":"call_def","type":"function","function":{"name":"multiply","arguments":"{\"a\":3,\"b\":4}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
		} else {
			mu.Lock()
			round2Captured = captured
			mu.Unlock()
			_, _ = w.Write([]byte(`{"id":"chatcmpl-2","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"6*7=42 and 3*4=12"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}`))
		}
	})
	defer upstream.Close()

	muxServer := newE2EMuxServer(t,
		"/v1/messages",
		newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, upstream.URL, "sk-openai-upstream", map[string]string{"claude-sonnet-4-20250514": "gpt-4o-mini"}).AnthropicHandler(),
	)
	defer muxServer.Close()

	ctx, cancel := newE2EContext(t)
	defer cancel()

	client := anthropic.NewClient(
		option.WithAPIKey("sk-anthropic-inbound"),
		option.WithBaseURL(muxServer.URL),
		option.WithHTTPClient(newLocalOnlyHTTPClient(t, muxServer.URL, upstream.URL)),
		option.WithMaxRetries(0),
	)

	// Round 1: send user message with tools
	round1, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:      anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens:  1024,
		Tools:      []anthropic.ToolUnionParam{anthropicMultiplyTool},
		ToolChoice: anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{Type: "any"}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("multiply 6*7 and 3*4")),
		},
	})
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}

	if round1.StopReason != anthropic.StopReasonToolUse {
		t.Fatalf("round 1 stop_reason = %q, want tool_use", round1.StopReason)
	}

	// Find both tool_use blocks
	var toolUseIDs []string
	for _, block := range round1.Content {
		if tu, ok := block.AsAny().(anthropic.ToolUseBlock); ok && tu.Name == "multiply" {
			toolUseIDs = append(toolUseIDs, tu.ID)
		}
	}
	if len(toolUseIDs) != 2 {
		t.Fatalf("expected 2 tool_use blocks, got %d", len(toolUseIDs))
	}

	// Round 2: send full history with two tool results plus a text block.
	// Including a text block alongside tool_results exercises the mixed-content
	// user message path (RoleUser) which correctly splits tool_results into
	// separate "tool" messages in the outbound OpenAI Chat encoding.
	round2, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeSonnet4_20250514,
		MaxTokens: 1024,
		Tools:     []anthropic.ToolUnionParam{anthropicMultiplyTool},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("multiply 6*7 and 3*4")),
			round1.ToParam(),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(toolUseIDs[0], "42", false),
				anthropic.NewToolResultBlock(toolUseIDs[1], "12", false),
				anthropic.NewTextBlock("Here are both results."),
			),
		},
	})
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}

	// Verify round 2 request body has TWO tool messages
	mu.Lock()
	r2cap := round2Captured
	mu.Unlock()
	if r2cap == nil {
		t.Fatal("round 2 upstream request not captured")
	}
	r2body := decodeCapturedJSONBody(t, r2cap)
	messages, _ := r2body["messages"].([]any)
	var toolMsgCount int
	for _, m := range messages {
		msg, _ := m.(map[string]any)
		if msg["role"] == "tool" {
			toolMsgCount++
		}
	}
	if toolMsgCount != 2 {
		// Dump for debugging
		bodyJSON, _ := json.MarshalIndent(r2body, "", "  ")
		t.Fatalf("expected 2 tool messages, got %d; body: %s", toolMsgCount, string(bodyJSON))
	}

	// Verify SDK receives text with both results
	var found bool
	for _, block := range round2.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			if strings.Contains(tb.Text, "42") && strings.Contains(tb.Text, "12") {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("round 2 response missing text with both 42 and 12")
	}
}

// Ensure imports are used.
var (
	_ = fmt.Sprintf
	_ = json.Marshal
	_ openaishared.FunctionParameters
	_ param.Opt[string]
	_ = responses.ResponseNewParams{}
)
