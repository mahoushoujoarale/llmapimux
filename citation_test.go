package llmapimux

import (
	"encoding/json"
	"testing"
)

// --- Anthropic citation decode ---

func TestDecodeAnthropicResponse_Citations_CharLocation(t *testing.T) {
	body := []byte(`{
		"id": "msg_123",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{
			"type": "text",
			"text": "According to the document, the answer is 42.",
			"citations": [
				{
					"type": "char_location",
					"cited_text": "the answer is 42",
					"document_index": 0,
					"document_title": "Guide to Everything",
					"start_char_index": 10,
					"end_char_index": 50
				}
			]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 100, "output_tokens": 50}
	}`)

	resp, err := DecodeAnthropicResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	part := resp.Content[0]
	if part.Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q, want %q", part.Type, ContentTypeText)
	}
	if part.Text.Text != "According to the document, the answer is 42." {
		t.Errorf("Content[0].Text = %q", part.Text.Text)
	}
	if len(part.Citations) != 1 {
		t.Fatalf("Citations len = %d, want 1", len(part.Citations))
	}
	c := part.Citations[0]
	if c.Kind != CitationKindCharLocation {
		t.Errorf("Citation.Kind = %q, want %q", c.Kind, CitationKindCharLocation)
	}
	if c.Title != "Guide to Everything" {
		t.Errorf("Citation.Title = %q, want %q", c.Title, "Guide to Everything")
	}
	if c.Start == nil || *c.Start != 10 {
		t.Errorf("Citation.Start = %v, want 10", c.Start)
	}
	if c.End == nil || *c.End != 50 {
		t.Errorf("Citation.End = %v, want 50", c.End)
	}
	if c.SourceID != "0" {
		t.Errorf("Citation.SourceID = %q, want %q", c.SourceID, "0")
	}
}

func TestDecodeAnthropicResponse_Citations_WebSearchResult(t *testing.T) {
	body := []byte(`{
		"id": "msg_456",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{
			"type": "text",
			"text": "The capital of France is Paris.",
			"citations": [
				{
					"type": "web_search_result",
					"url": "https://example.com/france",
					"title": "France Facts",
					"cited_text": "Paris is the capital"
				}
			]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 50, "output_tokens": 20}
	}`)

	resp, err := DecodeAnthropicResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	c := resp.Content[0].Citations[0]
	if c.Kind != CitationKindWebSearchResult {
		t.Errorf("Citation.Kind = %q, want %q", c.Kind, CitationKindWebSearchResult)
	}
	if c.URL != "https://example.com/france" {
		t.Errorf("Citation.URL = %q, want %q", c.URL, "https://example.com/france")
	}
	if c.Title != "France Facts" {
		t.Errorf("Citation.Title = %q, want %q", c.Title, "France Facts")
	}
}

// --- Anthropic citation round-trip ---

func TestAnthropicCitation_RoundTrip(t *testing.T) {
	body := []byte(`{
		"id": "msg_rt",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{
			"type": "text",
			"text": "Cited text here.",
			"citations": [
				{
					"type": "char_location",
					"cited_text": "some text",
					"document_index": 2,
					"document_title": "Doc Title",
					"start_char_index": 5,
					"end_char_index": 25
				},
				{
					"type": "web_search_result",
					"url": "https://example.com",
					"title": "Web Result",
					"cited_text": "web text"
				}
			]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`)

	// Decode
	resp, err := DecodeAnthropicResponse(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if len(resp.Content[0].Citations) != 2 {
		t.Fatalf("Citations len = %d, want 2", len(resp.Content[0].Citations))
	}

	// Encode back to Anthropic
	encoded, err := EncodeAnthropicResponse(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Decode the encoded response
	var rawOut map[string]interface{}
	if err := json.Unmarshal(encoded, &rawOut); err != nil {
		t.Fatalf("unmarshal encoded: %v", err)
	}

	// Verify content has citations
	content, ok := rawOut["content"].([]interface{})
	if !ok || len(content) != 1 {
		t.Fatalf("content = %v", rawOut["content"])
	}
	textBlock, ok := content[0].(map[string]interface{})
	if !ok {
		t.Fatalf("content[0] not a map")
	}
	citations, ok := textBlock["citations"].([]interface{})
	if !ok || len(citations) != 2 {
		t.Fatalf("citations = %v", textBlock["citations"])
	}

	// Verify first citation (char_location)
	c1, ok := citations[0].(map[string]interface{})
	if !ok {
		t.Fatalf("citations[0] not a map")
	}
	if c1["type"] != "char_location" {
		t.Errorf("citations[0].type = %v, want char_location", c1["type"])
	}
	if c1["document_title"] != "Doc Title" {
		t.Errorf("citations[0].document_title = %v, want Doc Title", c1["document_title"])
	}
	if c1["document_index"] != float64(2) {
		t.Errorf("citations[0].document_index = %v, want 2", c1["document_index"])
	}
	if c1["start_char_index"] != float64(5) {
		t.Errorf("citations[0].start_char_index = %v, want 5", c1["start_char_index"])
	}
	if c1["end_char_index"] != float64(25) {
		t.Errorf("citations[0].end_char_index = %v, want 25", c1["end_char_index"])
	}

	// Verify second citation (web_search_result)
	c2, ok := citations[1].(map[string]interface{})
	if !ok {
		t.Fatalf("citations[1] not a map")
	}
	if c2["type"] != "web_search_result" {
		t.Errorf("citations[1].type = %v, want web_search_result", c2["type"])
	}
	if c2["url"] != "https://example.com" {
		t.Errorf("citations[1].url = %v, want https://example.com", c2["url"])
	}
	if c2["title"] != "Web Result" {
		t.Errorf("citations[1].title = %v, want Web Result", c2["title"])
	}
}

// --- OpenAI Chat annotation decode ---

func TestDecodeOpenAIChatResponse_Annotations(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "The answer is documented here.",
				"annotations": [
					{
						"type": "url_citation",
						"url": "https://example.com/doc",
						"title": "Reference Doc",
						"start_index": 0,
						"end_index": 29
					}
				]
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	part := resp.Content[0]
	if part.Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q, want %q", part.Type, ContentTypeText)
	}
	if len(part.Citations) != 1 {
		t.Fatalf("Citations len = %d, want 1", len(part.Citations))
	}
	c := part.Citations[0]
	if c.Kind != CitationKindURLCitation {
		t.Errorf("Citation.Kind = %q, want %q", c.Kind, CitationKindURLCitation)
	}
	if c.URL != "https://example.com/doc" {
		t.Errorf("Citation.URL = %q, want %q", c.URL, "https://example.com/doc")
	}
	if c.Title != "Reference Doc" {
		t.Errorf("Citation.Title = %q, want %q", c.Title, "Reference Doc")
	}
	if c.Start == nil || *c.Start != 0 {
		t.Errorf("Citation.Start = %v, want 0", c.Start)
	}
	if c.End == nil || *c.End != 29 {
		t.Errorf("Citation.End = %v, want 29", c.End)
	}
}

func TestDecodeOpenAIChatResponse_NoAnnotations(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-456",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Hello!"
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
	}`)

	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if len(resp.Content[0].Citations) != 0 {
		t.Errorf("Citations len = %d, want 0", len(resp.Content[0].Citations))
	}
}

// --- OpenAI Responses annotation decode ---

func TestDecodeOpenAIResponsesResponse_Annotations(t *testing.T) {
	body := []byte(`{
		"id": "resp_123",
		"object": "response",
		"model": "gpt-4o",
		"status": "completed",
		"output": [{
			"type": "message",
			"role": "assistant",
			"content": [{
				"type": "output_text",
				"text": "See the reference.",
				"annotations": [
					{
						"type": "url_citation",
						"url": "https://example.com/ref",
						"title": "The Reference",
						"start_index": 4,
						"end_index": 18
					}
				]
			}]
		}],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
	}`)

	resp, err := DecodeOpenAIResponsesResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	part := resp.Content[0]
	if part.Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q, want %q", part.Type, ContentTypeText)
	}
	if len(part.Citations) != 1 {
		t.Fatalf("Citations len = %d, want 1", len(part.Citations))
	}
	c := part.Citations[0]
	if c.Kind != CitationKindURLCitation {
		t.Errorf("Citation.Kind = %q, want %q", c.Kind, CitationKindURLCitation)
	}
	if c.URL != "https://example.com/ref" {
		t.Errorf("Citation.URL = %q, want %q", c.URL, "https://example.com/ref")
	}
	if c.Title != "The Reference" {
		t.Errorf("Citation.Title = %q, want %q", c.Title, "The Reference")
	}
	if c.Start == nil || *c.Start != 4 {
		t.Errorf("Citation.Start = %v, want 4", c.Start)
	}
	if c.End == nil || *c.End != 18 {
		t.Errorf("Citation.End = %v, want 18", c.End)
	}
}

// --- Gemini weak-mapping citation decode ---

func TestDecodeGeminiResponse_CitationMetadata(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {
				"role": "model",
				"parts": [{"text": "The answer is documented."}]
			},
			"finishReason": "STOP",
			"citationMetadata": {
				"citationSources": [
					{
						"startIndex": 4,
						"endIndex": 24,
						"uri": "https://example.com/source",
						"title": "Source Document"
					}
				]
			}
		}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	part := resp.Content[0]
	if part.Type != ContentTypeText {
		t.Errorf("Content[0].Type = %q, want %q", part.Type, ContentTypeText)
	}
	if len(part.Citations) != 1 {
		t.Fatalf("Citations len = %d, want 1", len(part.Citations))
	}
	c := part.Citations[0]
	if c.Kind != CitationKindGemini {
		t.Errorf("Citation.Kind = %q, want %q", c.Kind, CitationKindGemini)
	}
	if c.URL != "https://example.com/source" {
		t.Errorf("Citation.URL = %q, want %q", c.URL, "https://example.com/source")
	}
	if c.Title != "Source Document" {
		t.Errorf("Citation.Title = %q, want %q", c.Title, "Source Document")
	}
	if c.Start == nil || *c.Start != 4 {
		t.Errorf("Citation.Start = %v, want 4", c.Start)
	}
	if c.End == nil || *c.End != 24 {
		t.Errorf("Citation.End = %v, want 24", c.End)
	}
}

func TestDecodeGeminiResponse_CitationMetadata_NoCitationSources(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {
				"role": "model",
				"parts": [{"text": "No citations here."}]
			},
			"finishReason": "STOP"
		}],
		"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if len(resp.Content[0].Citations) != 0 {
		t.Errorf("Citations len = %d, want 0", len(resp.Content[0].Citations))
	}
}

func TestDecodeGeminiResponse_CitationMetadata_MultipleSources(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {
				"role": "model",
				"parts": [{"text": "Multiple sources referenced here."}]
			},
			"finishReason": "STOP",
			"citationMetadata": {
				"citationSources": [
					{"startIndex": 0, "endIndex": 10, "uri": "https://a.com", "title": "Source A"},
					{"startIndex": 15, "endIndex": 30, "uri": "https://b.com", "title": "Source B"}
				]
			}
		}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if len(resp.Content[0].Citations) != 2 {
		t.Fatalf("Citations len = %d, want 2", len(resp.Content[0].Citations))
	}

	c1 := resp.Content[0].Citations[0]
	if c1.URL != "https://a.com" {
		t.Errorf("Citations[0].URL = %q, want %q", c1.URL, "https://a.com")
	}
	c2 := resp.Content[0].Citations[1]
	if c2.URL != "https://b.com" {
		t.Errorf("Citations[1].URL = %q, want %q", c2.URL, "https://b.com")
	}
}

// --- Gemini encode silently drops citations ---

func TestEncodeGeminiResponse_CitationsDropped(t *testing.T) {
	resp := &Response{
		Model:      "gemini-2.5-pro",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{
				Type: ContentTypeText,
				Text: &TextContent{Text: "Cited text."},
				Citations: []Citation{
					{Kind: CitationKindGemini, URL: "https://example.com", Title: "Source", Start: intPtr(0), End: intPtr(10)},
				},
			},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	encoded, err := EncodeGeminiResponse(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Verify no citationMetadata in the encoded response (silently dropped)
	var rawOut map[string]interface{}
	if err := json.Unmarshal(encoded, &rawOut); err != nil {
		t.Fatalf("unmarshal encoded: %v", err)
	}

	candidates, ok := rawOut["candidates"].([]interface{})
	if !ok || len(candidates) != 1 {
		t.Fatalf("candidates = %v", rawOut["candidates"])
	}
	cand, ok := candidates[0].(map[string]interface{})
	if !ok {
		t.Fatalf("candidates[0] not a map")
	}
	if _, ok := cand["citationMetadata"]; ok {
		t.Error("citationMetadata should not appear in encoded Gemini response (silently dropped)")
	}
}

// --- Cross-protocol: Anthropic → OpenAI Chat citation preservation ---

func TestCrossProtocol_AnthropicToOpenAIChat_CitationPreservation(t *testing.T) {
	// Decode an Anthropic response with citations
	anthropicBody := []byte(`{
		"id": "msg_cross",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{
			"type": "text",
			"text": "The answer is documented.",
			"citations": [
				{
					"type": "char_location",
					"cited_text": "answer is documented",
					"document_index": 1,
					"document_title": "Reference Guide",
					"start_char_index": 4,
					"end_char_index": 24
				}
			]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`)

	resp, err := DecodeAnthropicResponse(anthropicBody)
	if err != nil {
		t.Fatalf("decode anthropic: %v", err)
	}

	// Verify citations in IR
	if len(resp.Content) != 1 {
		t.Fatalf("Content len = %d, want 1", len(resp.Content))
	}
	if len(resp.Content[0].Citations) != 1 {
		t.Fatalf("Citations len = %d, want 1", len(resp.Content[0].Citations))
	}

	// Encode as OpenAI Chat response
	openaiBody, err := EncodeOpenAIChatResponse(resp)
	if err != nil {
		t.Fatalf("encode openai chat: %v", err)
	}

	// Decode the OpenAI Chat response to verify annotations are present
	var rawOut map[string]interface{}
	if err := json.Unmarshal(openaiBody, &rawOut); err != nil {
		t.Fatalf("unmarshal encoded: %v", err)
	}

	choices, ok := rawOut["choices"].([]interface{})
	if !ok || len(choices) != 1 {
		t.Fatalf("choices = %v", rawOut["choices"])
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		t.Fatalf("choices[0] not a map")
	}
	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		t.Fatalf("message not a map")
	}
	annotations, ok := message["annotations"].([]interface{})
	if !ok || len(annotations) != 1 {
		t.Fatalf("annotations = %v", message["annotations"])
	}
	ann, ok := annotations[0].(map[string]interface{})
	if !ok {
		t.Fatalf("annotations[0] not a map")
	}
	if ann["type"] != "char_location" {
		t.Errorf("annotation.type = %v, want char_location", ann["type"])
	}
	if ann["title"] != "Reference Guide" {
		t.Errorf("annotation.title = %v, want Reference Guide", ann["title"])
	}
	if ann["start_index"] != float64(4) {
		t.Errorf("annotation.start_index = %v, want 4", ann["start_index"])
	}
	if ann["end_index"] != float64(24) {
		t.Errorf("annotation.end_index = %v, want 24", ann["end_index"])
	}
}

// --- Cross-protocol: Anthropic → OpenAI Responses citation preservation ---

func TestCrossProtocol_AnthropicToOpenAIResponses_CitationPreservation(t *testing.T) {
	// Decode an Anthropic response with citations
	anthropicBody := []byte(`{
		"id": "msg_cross2",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{
			"type": "text",
			"text": "Referenced text.",
			"citations": [
				{
					"type": "web_search_result",
					"url": "https://example.com",
					"title": "Example",
					"cited_text": "Referenced text"
				}
			]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`)

	resp, err := DecodeAnthropicResponse(anthropicBody)
	if err != nil {
		t.Fatalf("decode anthropic: %v", err)
	}

	// Encode as OpenAI Responses response
	oaiRespBody, err := EncodeOpenAIResponsesResponse(resp)
	if err != nil {
		t.Fatalf("encode openai responses: %v", err)
	}

	// Decode and verify annotations
	var rawOut map[string]interface{}
	if err := json.Unmarshal(oaiRespBody, &rawOut); err != nil {
		t.Fatalf("unmarshal encoded: %v", err)
	}

	output, ok := rawOut["output"].([]interface{})
	if !ok || len(output) != 1 {
		t.Fatalf("output = %v", rawOut["output"])
	}
	msg, ok := output[0].(map[string]interface{})
	if !ok {
		t.Fatalf("output[0] not a map")
	}
	content, ok := msg["content"].([]interface{})
	if !ok || len(content) != 1 {
		t.Fatalf("content = %v", msg["content"])
	}
	textBlock, ok := content[0].(map[string]interface{})
	if !ok {
		t.Fatalf("content[0] not a map")
	}
	annotations, ok := textBlock["annotations"].([]interface{})
	if !ok || len(annotations) != 1 {
		t.Fatalf("annotations = %v", textBlock["annotations"])
	}
	ann, ok := annotations[0].(map[string]interface{})
	if !ok {
		t.Fatalf("annotations[0] not a map")
	}
	if ann["type"] != "web_search_result" {
		t.Errorf("annotation.type = %v, want web_search_result", ann["type"])
	}
	if ann["url"] != "https://example.com" {
		t.Errorf("annotation.url = %v, want https://example.com", ann["url"])
	}
	if ann["title"] != "Example" {
		t.Errorf("annotation.title = %v, want Example", ann["title"])
	}
}

// --- Cross-protocol: Anthropic → Gemini silently drops citations ---

func TestCrossProtocol_AnthropicToGemini_CitationsDropped(t *testing.T) {
	// Decode an Anthropic response with citations
	anthropicBody := []byte(`{
		"id": "msg_drop",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [{
			"type": "text",
			"text": "Cited text.",
			"citations": [
				{
					"type": "char_location",
					"cited_text": "Cited",
					"document_index": 0,
					"document_title": "Doc",
					"start_char_index": 0,
					"end_char_index": 5
				}
			]
		}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`)

	resp, err := DecodeAnthropicResponse(anthropicBody)
	if err != nil {
		t.Fatalf("decode anthropic: %v", err)
	}

	// Encode as Gemini response
	geminiBody, err := EncodeGeminiResponse(resp)
	if err != nil {
		t.Fatalf("encode gemini: %v", err)
	}

	// Verify no citationMetadata in Gemini output
	var rawOut map[string]interface{}
	if err := json.Unmarshal(geminiBody, &rawOut); err != nil {
		t.Fatalf("unmarshal encoded: %v", err)
	}

	candidates, ok := rawOut["candidates"].([]interface{})
	if !ok || len(candidates) != 1 {
		t.Fatalf("candidates = %v", rawOut["candidates"])
	}
	cand, ok := candidates[0].(map[string]interface{})
	if !ok {
		t.Fatalf("candidates[0] not a map")
	}
	if _, ok := cand["citationMetadata"]; ok {
		t.Error("citationMetadata should not appear in Gemini response when encoding from IR")
	}

	// The text should still be present
	content, ok := cand["content"].(map[string]interface{})
	if !ok {
		t.Fatalf("content not a map")
	}
	parts, ok := content["parts"].([]interface{})
	if !ok || len(parts) != 1 {
		t.Fatalf("parts = %v", content["parts"])
	}
	partMap, ok := parts[0].(map[string]interface{})
	if !ok {
		t.Fatalf("parts[0] not a map")
	}
	if partMap["text"] != "Cited text." {
		t.Errorf("text = %v, want 'Cited text.'", partMap["text"])
	}
}

// --- OpenAI Chat annotation encode ---

func TestEncodeOpenAIChatResponse_Annotations(t *testing.T) {
	resp := &Response{
		ID:         "chatcmpl-enc",
		Model:      "gpt-4o",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{
				Type: ContentTypeText,
				Text: &TextContent{Text: "The answer is here."},
				Citations: []Citation{
					{Kind: CitationKindURLCitation, URL: "https://example.com", Title: "Source", Start: intPtr(0), End: intPtr(19)},
				},
			},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	encoded, err := EncodeOpenAIChatResponse(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var rawOut map[string]interface{}
	if err := json.Unmarshal(encoded, &rawOut); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	choices := rawOut["choices"].([]interface{})
	message := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	annotations, ok := message["annotations"].([]interface{})
	if !ok || len(annotations) != 1 {
		t.Fatalf("annotations = %v", message["annotations"])
	}
	ann := annotations[0].(map[string]interface{})
	if ann["type"] != "url_citation" {
		t.Errorf("annotation.type = %v, want url_citation", ann["type"])
	}
	if ann["url"] != "https://example.com" {
		t.Errorf("annotation.url = %v, want https://example.com", ann["url"])
	}
}

// --- OpenAI Responses annotation encode ---

func TestEncodeOpenAIResponsesResponse_Annotations(t *testing.T) {
	resp := &Response{
		ID:         "resp_enc",
		Model:      "gpt-4o",
		StopReason: StopReasonEndTurn,
		Content: []ContentPart{
			{
				Type: ContentTypeText,
				Text: &TextContent{Text: "Referenced."},
				Citations: []Citation{
					{Kind: CitationKindURLCitation, URL: "https://example.com/ref", Title: "Ref"},
				},
			},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	encoded, err := EncodeOpenAIResponsesResponse(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	var rawOut map[string]interface{}
	if err := json.Unmarshal(encoded, &rawOut); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	output := rawOut["output"].([]interface{})
	msg := output[0].(map[string]interface{})
	content := msg["content"].([]interface{})
	textBlock := content[0].(map[string]interface{})
	annotations, ok := textBlock["annotations"].([]interface{})
	if !ok || len(annotations) != 1 {
		t.Fatalf("annotations = %v", textBlock["annotations"])
	}
	ann := annotations[0].(map[string]interface{})
	if ann["type"] != "url_citation" {
		t.Errorf("annotation.type = %v, want url_citation", ann["type"])
	}
	if ann["url"] != "https://example.com/ref" {
		t.Errorf("annotation.url = %v, want https://example.com/ref", ann["url"])
	}
}

// --- OpenAI Chat annotation round-trip ---

func TestOpenAIChatAnnotation_RoundTrip(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-rt",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "See the docs.",
				"annotations": [
					{
						"type": "url_citation",
						"url": "https://docs.example.com",
						"title": "Documentation",
						"start_index": 4,
						"end_index": 12
					}
				]
			},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`)

	// Decode
	resp, err := DecodeOpenAIChatResponse(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	if len(resp.Content[0].Citations) != 1 {
		t.Fatalf("Citations len = %d, want 1", len(resp.Content[0].Citations))
	}

	// Encode back
	encoded, err := EncodeOpenAIChatResponse(resp)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	// Decode again to verify
	resp2, err := DecodeOpenAIChatResponse(encoded)
	if err != nil {
		t.Fatalf("decode round-trip: %v", err)
	}

	if len(resp2.Content[0].Citations) != 1 {
		t.Fatalf("Round-trip Citations len = %d, want 1", len(resp2.Content[0].Citations))
	}
	c := resp2.Content[0].Citations[0]
	if c.Kind != CitationKindURLCitation {
		t.Errorf("Round-trip Citation.Kind = %q, want %q", c.Kind, CitationKindURLCitation)
	}
	if c.URL != "https://docs.example.com" {
		t.Errorf("Round-trip Citation.URL = %q, want %q", c.URL, "https://docs.example.com")
	}
	if c.Title != "Documentation" {
		t.Errorf("Round-trip Citation.Title = %q, want %q", c.Title, "Documentation")
	}
	if c.Start == nil || *c.Start != 4 {
		t.Errorf("Round-trip Citation.Start = %v, want 4", c.Start)
	}
	if c.End == nil || *c.End != 12 {
		t.Errorf("Round-trip Citation.End = %v, want 12", c.End)
	}
}

// --- Gemini weak-mapping behavior: citations attached to first text part ---

func TestDecodeGeminiResponse_CitationMetadata_AttachedToFirstTextPart(t *testing.T) {
	body := []byte(`{
		"candidates": [{
			"content": {
				"role": "model",
				"parts": [
					{"functionCall": {"name": "get_weather", "args": {"city": "London"}}},
					{"text": "First text part."},
					{"text": "Second text part."}
				]
			},
			"finishReason": "STOP",
			"citationMetadata": {
				"citationSources": [
					{"startIndex": 0, "endIndex": 15, "uri": "https://weather.com", "title": "Weather"}
				]
			}
		}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20}
	}`)

	resp, err := DecodeGeminiResponse(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have 3 parts: tool_use, text, text
	if len(resp.Content) != 3 {
		t.Fatalf("Content len = %d, want 3", len(resp.Content))
	}
	if resp.Content[0].Type != ContentTypeToolUse {
		t.Errorf("Content[0].Type = %q, want %q", resp.Content[0].Type, ContentTypeToolUse)
	}
	if resp.Content[1].Type != ContentTypeText {
		t.Errorf("Content[1].Type = %q, want %q", resp.Content[1].Type, ContentTypeText)
	}
	if resp.Content[2].Type != ContentTypeText {
		t.Errorf("Content[2].Type = %q, want %q", resp.Content[2].Type, ContentTypeText)
	}

	// Citations should be attached to the FIRST text part (Content[1]), not the tool_use
	if len(resp.Content[0].Citations) != 0 {
		t.Errorf("Content[0] (tool_use) Citations len = %d, want 0", len(resp.Content[0].Citations))
	}
	if len(resp.Content[1].Citations) != 1 {
		t.Fatalf("Content[1] (first text) Citations len = %d, want 1", len(resp.Content[1].Citations))
	}
	if len(resp.Content[2].Citations) != 0 {
		t.Errorf("Content[2] (second text) Citations len = %d, want 0", len(resp.Content[2].Citations))
	}

	c := resp.Content[1].Citations[0]
	if c.Kind != CitationKindGemini {
		t.Errorf("Citation.Kind = %q, want %q", c.Kind, CitationKindGemini)
	}
	if c.URL != "https://weather.com" {
		t.Errorf("Citation.URL = %q, want %q", c.URL, "https://weather.com")
	}
}

// --- No-citation content doesn't get Citations field ---

func TestContentPart_NoCitations_OmittedInJSON(t *testing.T) {
	part := ContentPart{
		Type: ContentTypeText,
		Text: &TextContent{Text: "hello"},
	}

	data, err := json.Marshal(part)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if _, ok := raw["citations"]; ok {
		t.Error("citations field should be omitted when empty")
	}
}
