package e2e_test

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
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

// --- Tool definitions ---

var anthropicMultiplyTool = anthropic.ToolUnionParam{
	OfTool: &anthropic.ToolParam{
		Name:        "multiply",
		Description: anthropic.String("Multiply two integers and return the result."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"a": map[string]any{"type": "integer"},
				"b": map[string]any{"type": "integer"},
			},
			Required: []string{"a", "b"},
		},
	},
}

var openAIMultiplyTool = openai.ChatCompletionFunctionTool(openaishared.FunctionDefinitionParam{
	Name:        "multiply",
	Description: openai.String("Multiply two integers and return the result."),
	Parameters: openaishared.FunctionParameters{
		"type": "object",
		"properties": map[string]any{
			"a": map[string]any{"type": "integer"},
			"b": map[string]any{"type": "integer"},
		},
		"required": []string{"a", "b"},
	},
})

var geminiMultiplyTool = &genai.Tool{
	FunctionDeclarations: []*genai.FunctionDeclaration{
		{
			Name:        "multiply",
			Description: "Multiply two integers and return the result.",
			Parameters: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"a": {Type: genai.TypeInteger},
					"b": {Type: genai.TypeInteger},
				},
				Required: []string{"a", "b"},
			},
		},
	},
}

var openAIResponsesMultiplyTool = responses.ToolParamOfFunction(
	"multiply",
	map[string]any{
		"type": "object",
		"properties": map[string]any{
			"a": map[string]any{"type": "integer"},
			"b": map[string]any{"type": "integer"},
		},
		"required": []string{"a", "b"},
	},
	false,
)

// --- Shared helpers ---

// extractMultiplyArgsJSON parses multiply tool args from a JSON string (Anthropic/OpenAI).
func extractMultiplyArgsJSON(t *testing.T, argsJSON string) (int, int) {
	t.Helper()
	var args struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		t.Fatalf("unmarshal multiply args: %v", err)
	}
	return args.A, args.B
}

// extractMultiplyArgsMap parses multiply tool args from map[string]any (Gemini; values are float64).
func extractMultiplyArgsMap(t *testing.T, args map[string]any) (int, int) {
	t.Helper()
	toInt := func(v any) int {
		switch x := v.(type) {
		case float64:
			return int(x)
		case int:
			return x
		}
		t.Fatalf("unexpected arg type %T", v)
		return 0
	}
	return toInt(args["a"]), toInt(args["b"])
}

func assertMultiplyArgs(t *testing.T, a, b int) {
	t.Helper()
	if !((a == 6 && b == 7) || (a == 7 && b == 6)) {
		t.Fatalf("expected multiply args (6,7) or (7,6), got (%d,%d)", a, b)
	}
}

func assertContains42(t *testing.T, text string) {
	t.Helper()
	if !strings.Contains(text, "42") {
		t.Fatalf("expected response to contain '42', got: %s", text)
	}
}

// --- Shared runner functions ---

func runAnthropicToolCall(t *testing.T, client anthropic.Client, model anthropic.Model) {
	t.Helper()

	ctx, cancel := newRealAPIContext(t)
	defer cancel()

	// Round 1 — force tool use so cross-protocol paths reliably call the tool.
	round1, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:      model,
		MaxTokens:  1024,
		Tools:      []anthropic.ToolUnionParam{anthropicMultiplyTool},
		ToolChoice: anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{Type: "any"}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(
				"What is 6 multiplied by 7? You must use the multiply tool.",
			)),
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
	var toolUseArgs string
	for _, block := range round1.Content {
		if tu, ok := block.AsAny().(anthropic.ToolUseBlock); ok && tu.Name == "multiply" {
			toolUseID = tu.ID
			toolUseArgs = string(tu.Input)
		}
	}
	if toolUseID == "" {
		t.Fatal("no multiply tool_use block in round 1 response")
	}
	a, b := extractMultiplyArgsJSON(t, toolUseArgs)
	assertMultiplyArgs(t, a, b)
	t.Logf("round 1: multiply(a=%d, b=%d)", a, b)

	// Round 2
	ctx2, cancel2 := newRealAPIContext(t)
	defer cancel2()

	// Build round-2 messages: original user msg + assistant round-1 + tool result
	userMsg := anthropic.NewUserMessage(anthropic.NewTextBlock(
		"What is 6 multiplied by 7? You must use the multiply tool.",
	))
	assistantMsg := round1.ToParam()
	toolResultMsg := anthropic.NewUserMessage(anthropic.NewToolResultBlock(toolUseID, "42", false))

	round2, err := client.Messages.New(ctx2, anthropic.MessageNewParams{
		Model:     model,
		MaxTokens: 1024,
		Tools:     []anthropic.ToolUnionParam{anthropicMultiplyTool},
		Messages:  []anthropic.MessageParam{userMsg, assistantMsg, toolResultMsg},
	})
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}
	var found bool
	for _, block := range round2.Content {
		if tb, ok := block.AsAny().(anthropic.TextBlock); ok {
			t.Logf("round 2 response: %s", tb.Text)
			assertContains42(t, tb.Text)
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("round 2: no TextBlock in %d content blocks", len(round2.Content))
	}
}

func runOpenAIToolCall(t *testing.T, client openai.Client, model string) {
	t.Helper()

	ctx, cancel := newRealAPIContext(t)
	defer cancel()

	// Round 1 — force tool use so cross-protocol paths reliably call the tool.
	round1, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:      model,
		Tools:      []openai.ChatCompletionToolUnionParam{openAIMultiplyTool},
		ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")},
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("What is 6 multiplied by 7? You must use the multiply tool."),
		},
	})
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}
	if len(round1.Choices) == 0 {
		t.Fatal("round 1: no choices")
	}
	choice := round1.Choices[0]
	// Some providers return finish_reason "stop" even when tool calls are present;
	// check for tool calls directly rather than relying on finish_reason alone.
	if len(choice.Message.ToolCalls) == 0 {
		t.Fatalf("round 1: no tool calls (finish_reason=%q)", choice.FinishReason)
	}
	toolCall := choice.Message.ToolCalls[0]
	funcCall := toolCall.AsFunction()
	if funcCall.Function.Name != "multiply" {
		t.Fatalf("tool call name = %q, want multiply", funcCall.Function.Name)
	}
	a, b := extractMultiplyArgsJSON(t, funcCall.Function.Arguments)
	assertMultiplyArgs(t, a, b)
	t.Logf("round 1: multiply(a=%d, b=%d)", a, b)

	// Round 2
	ctx2, cancel2 := newRealAPIContext(t)
	defer cancel2()

	round2, err := client.Chat.Completions.New(ctx2, openai.ChatCompletionNewParams{
		Model: model,
		Tools: []openai.ChatCompletionToolUnionParam{openAIMultiplyTool},
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("What is 6 multiplied by 7? You must use the multiply tool."),
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
	t.Logf("round 2 response: %s", round2.Choices[0].Message.Content)
	assertContains42(t, round2.Choices[0].Message.Content)
}

func runGeminiToolCall(t *testing.T, client *genai.Client, model string) {
	t.Helper()

	ctx, cancel := newRealAPIContext(t)
	defer cancel()

	userContent := genai.NewContentFromParts(
		[]*genai.Part{genai.NewPartFromText("What is 6 multiplied by 7? You must use the multiply tool.")},
		genai.RoleUser,
	)
	round1Contents := []*genai.Content{userContent}
	// Force tool use so cross-protocol paths reliably call the tool.
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{geminiMultiplyTool},
		ToolConfig: &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny},
		},
	}

	// Round 1
	round1, err := client.Models.GenerateContent(ctx, model, round1Contents, cfg)
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
	a, b := extractMultiplyArgsMap(t, call.Args)
	assertMultiplyArgs(t, a, b)
	t.Logf("round 1: multiply(a=%d, b=%d)", a, b)

	// Round 2
	ctx2, cancel2 := newRealAPIContext(t)
	defer cancel2()

	if len(round1.Candidates) == 0 || round1.Candidates[0].Content == nil {
		t.Fatal("round 1: missing candidate content for history")
	}
	toolResultContent := genai.NewContentFromParts(
		[]*genai.Part{genai.NewPartFromFunctionResponse(call.Name, map[string]any{"result": "42"})},
		genai.RoleUser,
	)
	round2Contents := []*genai.Content{userContent, round1.Candidates[0].Content, toolResultContent}

	// Round 2 — do NOT force tool use; model should respond with text using the function result
	cfg2 := &genai.GenerateContentConfig{Tools: []*genai.Tool{geminiMultiplyTool}}
	round2, err := client.Models.GenerateContent(ctx2, model, round2Contents, cfg2)
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}
	t.Logf("round 2 response: %s", round2.Text())
	assertContains42(t, round2.Text())
}

func runOpenAIResponsesToolCall(t *testing.T, client openai.Client, model string) {
	t.Helper()

	ctx, cancel := newRealAPIContext(t)
	defer cancel()

	// Round 1 — force tool use.
	round1, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: model,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.NewOpt("What is 6 multiplied by 7? You must use the multiply tool."),
		},
		Tools: []responses.ToolUnionParam{openAIResponsesMultiplyTool},
		ToolChoice: responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
		},
	})
	if err != nil {
		t.Fatalf("round 1: %v", err)
	}

	// Find function_call output item.
	var funcCall responses.ResponseFunctionToolCall
	var found bool
	for _, item := range round1.Output {
		if item.Type == "function_call" {
			fc := item.AsFunctionCall()
			if fc.Name == "multiply" {
				funcCall = fc
				found = true
				break
			}
		}
	}
	if !found {
		t.Fatal("round 1: no multiply function_call in output")
	}
	a, b := extractMultiplyArgsJSON(t, funcCall.Arguments)
	assertMultiplyArgs(t, a, b)
	t.Logf("round 1: multiply(a=%d, b=%d)", a, b)

	// Round 2 — provide the function result.
	ctx2, cancel2 := newRealAPIContext(t)
	defer cancel2()

	round2, err := client.Responses.New(ctx2, responses.ResponseNewParams{
		Model: model,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage(
					"What is 6 multiplied by 7? You must use the multiply tool.",
					responses.EasyInputMessageRoleUser,
				),
				responses.ResponseInputItemParamOfFunctionCall(funcCall.Arguments, funcCall.CallID, funcCall.Name),
				responses.ResponseInputItemParamOfFunctionCallOutput(funcCall.CallID, "42"),
			},
		},
		Tools: []responses.ToolUnionParam{openAIResponsesMultiplyTool},
	})
	if err != nil {
		t.Fatalf("round 2: %v", err)
	}

	text := collectResponsesOutputText(round2)
	t.Logf("round 2 response: %s", text)
	assertContains42(t, text)
}

// ============================================================
// Passthrough tests
// ============================================================

func TestRealAPI_Tool_Passthrough_Anthropic(t *testing.T) {
	baseURL := anthropicBaseURL(t)
	model := anthropicModel()

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/messages": newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, baseURL, anthropicAPIKey(), map[string]string{model: model}).AnthropicHandler(),
	})
	defer muxServer.Close()

	client := anthropic.NewClient(
		option.WithAPIKey("unused-inbound-key"),
		option.WithBaseURL(muxServer.URL),
		option.WithMaxRetries(0),
	)

	runAnthropicToolCall(t, client, anthropic.Model(model))
}

func TestRealAPI_Tool_Passthrough_OpenAI(t *testing.T) {
	baseURL := openAIBaseURL(t)
	model := openAIModel()

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/chat/completions": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, baseURL, openAIAPIKey(), map[string]string{model: model}).OpenAIChatHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	runOpenAIToolCall(t, client, model)
}

func TestRealAPI_Tool_Passthrough_Gemini(t *testing.T) {
	baseURL := geminiBaseURL(t)
	model := geminiModel()

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/models/": newTestMuxWithModelMap(llmapimux.ProtocolGemini, baseURL, geminiAPIKey(), map[string]string{model: model}).GeminiHandler(),
	})
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "unused-inbound-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	runGeminiToolCall(t, client, model)
}

// ============================================================
// Cross-protocol tests
// ============================================================

func TestRealAPI_Tool_Cross_AnthropicToOpenAI(t *testing.T) {
	baseURL := openAIBaseURL(t)
	const inboundModel = "tool-ant-to-oai"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/messages": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).AnthropicHandler(),
	})
	defer muxServer.Close()

	client := anthropic.NewClient(
		option.WithAPIKey("unused-inbound-key"),
		option.WithBaseURL(muxServer.URL),
		option.WithMaxRetries(0),
	)

	runAnthropicToolCall(t, client, anthropic.Model(inboundModel))
}

func TestRealAPI_Tool_Cross_AnthropicToGemini(t *testing.T) {
	baseURL := geminiBaseURL(t)
	const inboundModel = "tool-ant-to-gem"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/messages": newTestMuxWithModelMap(llmapimux.ProtocolGemini, baseURL, geminiAPIKey(), map[string]string{inboundModel: geminiModel()}).AnthropicHandler(),
	})
	defer muxServer.Close()

	client := anthropic.NewClient(
		option.WithAPIKey("unused-inbound-key"),
		option.WithBaseURL(muxServer.URL),
		option.WithMaxRetries(0),
	)

	runAnthropicToolCall(t, client, anthropic.Model(inboundModel))
}

func TestRealAPI_Tool_Cross_OpenAIToAnthropic(t *testing.T) {
	baseURL := anthropicBaseURL(t)
	const inboundModel = "tool-oai-to-ant"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/chat/completions": newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, baseURL, anthropicAPIKey(), map[string]string{inboundModel: anthropicModel()}).OpenAIChatHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	runOpenAIToolCall(t, client, inboundModel)
}

func TestRealAPI_Tool_Cross_OpenAIToGemini(t *testing.T) {
	baseURL := geminiBaseURL(t)
	const inboundModel = "tool-oai-to-gem"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/chat/completions": newTestMuxWithModelMap(llmapimux.ProtocolGemini, baseURL, geminiAPIKey(), map[string]string{inboundModel: geminiModel()}).OpenAIChatHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	runOpenAIToolCall(t, client, inboundModel)
}

func TestRealAPI_Tool_Cross_GeminiToAnthropic(t *testing.T) {
	baseURL := anthropicBaseURL(t)
	const inboundModel = "tool-gem-to-ant"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/models/": newTestMuxWithModelMap(llmapimux.ProtocolAnthropic, baseURL, anthropicAPIKey(), map[string]string{inboundModel: anthropicModel()}).GeminiHandler(),
	})
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "unused-inbound-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	runGeminiToolCall(t, client, inboundModel)
}

func TestRealAPI_Tool_Cross_GeminiToOpenAI(t *testing.T) {
	baseURL := openAIBaseURL(t)
	const inboundModel = "tool-gem-to-oai"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/models/": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIChat, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).GeminiHandler(),
	})
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "unused-inbound-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	runGeminiToolCall(t, client, inboundModel)
}

func TestRealAPI_Tool_Passthrough_OpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	model := openAIModel()

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/responses": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, baseURL, openAIAPIKey(), map[string]string{model: model}).OpenAIResponsesHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	runOpenAIResponsesToolCall(t, client, model)
}

func TestRealAPI_Tool_Cross_AnthropicToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "tool-ant-to-oairesp"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/messages": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).AnthropicHandler(),
	})
	defer muxServer.Close()

	client := anthropic.NewClient(
		option.WithAPIKey("unused-inbound-key"),
		option.WithBaseURL(muxServer.URL),
		option.WithMaxRetries(0),
	)

	runAnthropicToolCall(t, client, anthropic.Model(inboundModel))
}

func TestRealAPI_Tool_Cross_GeminiToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "tool-gem-to-oairesp"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/models/": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).GeminiHandler(),
	})
	defer muxServer.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "unused-inbound-key",
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    muxServer.URL + "/",
			APIVersion: "v1",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	runGeminiToolCall(t, client, inboundModel)
}

func TestRealAPI_Tool_Cross_OpenAIChatToOpenAIResponses(t *testing.T) {
	baseURL := openAIResponsesBaseURL(t)
	const inboundModel = "tool-oai-to-oairesp"

	muxServer := newRealMuxServer(t, map[string]http.Handler{
		"/v1/chat/completions": newTestMuxWithModelMap(llmapimux.ProtocolOpenAIResponses, baseURL, openAIAPIKey(), map[string]string{inboundModel: openAIModel()}).OpenAIChatHandler(),
	})
	defer muxServer.Close()

	client := openai.NewClient(
		openaiopt.WithAPIKey("unused-inbound-key"),
		openaiopt.WithBaseURL(muxServer.URL+"/v1"),
		openaiopt.WithMaxRetries(0),
	)

	runOpenAIToolCall(t, client, inboundModel)
}
