package llmapimux

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	gemini "github.com/mahoushoujoarale/llmapimux/protocol/gemini"
)

// --- Private intermediate structs for Gemini GenerateContent JSON ---

// Field names use camelCase to match the official Gemini REST API (proto3 JSON encoding)
// and the google.golang.org/genai SDK serialization format.

// --- Response types ---

// --- Gemini Schema types ---

// --- Schema conversion helpers ---

// geminiTypeToJSONType converts a Gemini uppercase type name to a JSON Schema lowercase type.
func geminiTypeToJSONType(t string) string {
	switch strings.ToUpper(t) {
	case "STRING":
		return "string"
	case "NUMBER":
		return "number"
	case "INTEGER":
		return "integer"
	case "BOOLEAN":
		return "boolean"
	case "ARRAY":
		return "array"
	case "OBJECT":
		return "object"
	default:
		return strings.ToLower(t)
	}
}

// jsonTypeToGeminiType converts a JSON Schema lowercase type to a Gemini uppercase type name.
func jsonTypeToGeminiType(t string) string {
	switch strings.ToLower(t) {
	case "string":
		return "STRING"
	case "number":
		return "NUMBER"
	case "integer":
		return "INTEGER"
	case "boolean":
		return "BOOLEAN"
	case "array":
		return "ARRAY"
	case "object":
		return "OBJECT"
	default:
		return strings.ToUpper(t)
	}
}

// jsonSchemaMap is a generic map representation of a JSON Schema.
type jsonSchemaMap = map[string]interface{}

// geminiSchemaToJSONSchema converts a Gemini Schema (as raw JSON) to a JSON Schema (as raw JSON).
func geminiSchemaToJSONSchema(geminiRaw json.RawMessage) (json.RawMessage, error) {
	if len(geminiRaw) == 0 {
		return nil, nil
	}
	var gs gemini.Schema
	if err := json.Unmarshal(geminiRaw, &gs); err != nil {
		return nil, fmt.Errorf("unmarshal gemini schema: %w", err)
	}
	js := convertGeminiSchemaToJSON(gs)
	result, err := json.Marshal(js)
	if err != nil {
		return nil, fmt.Errorf("marshal json schema: %w", err)
	}
	return result, nil
}

// convertGeminiSchemaToJSON recursively converts a gemini.Schema to a jsonSchemaMap.
func convertGeminiSchemaToJSON(gs gemini.Schema) jsonSchemaMap {
	js := jsonSchemaMap{}
	if gs.Type != "" {
		js["type"] = geminiTypeToJSONType(gs.Type)
	}
	if gs.Format != "" {
		js["format"] = gs.Format
	}
	if gs.Description != "" {
		js["description"] = gs.Description
	}
	if gs.Nullable != nil {
		js["nullable"] = *gs.Nullable
	}
	if len(gs.Properties) > 0 {
		props := jsonSchemaMap{}
		for k, v := range gs.Properties {
			props[k] = convertGeminiSchemaToJSON(v)
		}
		js["properties"] = props
	}
	if len(gs.Required) > 0 {
		js["required"] = gs.Required
	}
	if gs.Items != nil {
		js["items"] = convertGeminiSchemaToJSON(*gs.Items)
	}
	if len(gs.Enum) > 0 {
		js["enum"] = geminiEnumToJSON(gs.Enum, gs.Type)
	}
	if gs.MinLength != nil {
		js["minLength"] = *gs.MinLength
	}
	if gs.MaxLength != nil {
		js["maxLength"] = *gs.MaxLength
	}
	if gs.Pattern != "" {
		js["pattern"] = gs.Pattern
	}
	if gs.Minimum != nil {
		js["minimum"] = *gs.Minimum
	}
	if gs.Maximum != nil {
		js["maximum"] = *gs.Maximum
	}
	if gs.MinItems != nil {
		js["minItems"] = *gs.MinItems
	}
	if gs.MaxItems != nil {
		js["maxItems"] = *gs.MaxItems
	}
	if len(gs.AnyOf) > 0 {
		variants := make([]jsonSchemaMap, 0, len(gs.AnyOf))
		for _, v := range gs.AnyOf {
			variants = append(variants, convertGeminiSchemaToJSON(v))
		}
		js["anyOf"] = variants
	}
	// If nothing at all was set, still emit a type so the schema is well-formed.
	if len(js) == 0 {
		js["type"] = geminiTypeToJSONType(gs.Type)
	}
	return js
}

// geminiEnumToJSON converts Gemini's always-string enum values back to their JSON
// Schema types. Gemini serialises integer and number enums as strings, so an
// INTEGER schema's ["1","2"] must become [1,2] to remain a valid JSON Schema.
func geminiEnumToJSON(enum []string, geminiType string) []interface{} {
	values := make([]interface{}, 0, len(enum))
	switch strings.ToUpper(geminiType) {
	case "INTEGER":
		for _, e := range enum {
			if n, err := strconv.ParseInt(e, 10, 64); err == nil {
				values = append(values, n)
				continue
			}
			values = append(values, e)
		}
	case "NUMBER":
		for _, e := range enum {
			if f, err := strconv.ParseFloat(e, 64); err == nil {
				values = append(values, f)
				continue
			}
			values = append(values, e)
		}
	case "BOOLEAN":
		for _, e := range enum {
			if b, err := strconv.ParseBool(e); err == nil {
				values = append(values, b)
				continue
			}
			values = append(values, e)
		}
	default:
		for _, e := range enum {
			values = append(values, e)
		}
	}
	return values
}

// jsonSchemaToGeminiSchema converts a JSON Schema (as raw JSON) to a Gemini Schema (as raw JSON).
func jsonSchemaToGeminiSchema(jsonRaw json.RawMessage) (json.RawMessage, error) {
	if len(jsonRaw) == 0 {
		return nil, nil
	}
	var js jsonSchemaMap
	if err := json.Unmarshal(jsonRaw, &js); err != nil {
		return nil, fmt.Errorf("unmarshal json schema: %w", err)
	}
	gs := convertJSONSchemaToGemini(js)
	result, err := json.Marshal(gs)
	if err != nil {
		return nil, fmt.Errorf("marshal gemini schema: %w", err)
	}
	return result, nil
}

// convertJSONSchemaToGemini recursively converts a jsonSchemaMap to a gemini.Schema.
//
// Keywords Gemini understands (format, nullable, enum, the string/number/array
// constraints, anyOf) are carried across. oneOf is mapped onto anyOf since Gemini
// has no exclusive-choice keyword. Keywords with no Gemini equivalent
// ($defs/$ref, additionalProperties, allOf, not, const) are dropped — Gemini
// rejects unknown schema fields.
func convertJSONSchemaToGemini(js jsonSchemaMap) gemini.Schema {
	gs := gemini.Schema{}
	if t, ok := js["type"].(string); ok {
		gs.Type = jsonTypeToGeminiType(t)
	} else if types, ok := js["type"].([]interface{}); ok {
		// JSON Schema union type, e.g. ["string","null"] → STRING + nullable.
		for _, raw := range types {
			s, ok := raw.(string)
			if !ok {
				continue
			}
			if s == "null" {
				nullable := true
				gs.Nullable = &nullable
				continue
			}
			if gs.Type == "" {
				gs.Type = jsonTypeToGeminiType(s)
			}
		}
	}
	if f, ok := js["format"].(string); ok {
		gs.Format = f
	}
	if d, ok := js["description"].(string); ok {
		gs.Description = d
	}
	if n, ok := js["nullable"].(bool); ok {
		gs.Nullable = &n
	}
	if props, ok := js["properties"].(map[string]interface{}); ok {
		gs.Properties = make(map[string]gemini.Schema, len(props))
		for k, v := range props {
			if vm, ok := v.(map[string]interface{}); ok {
				gs.Properties[k] = convertJSONSchemaToGemini(vm)
			}
		}
	}
	if req, ok := js["required"].([]interface{}); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				gs.Required = append(gs.Required, s)
			}
		}
	}
	if items, ok := js["items"].(map[string]interface{}); ok {
		converted := convertJSONSchemaToGemini(items)
		gs.Items = &converted
	}
	if enum, ok := js["enum"].([]interface{}); ok {
		// Gemini requires enum entries to be strings even for numeric types, so
		// non-string values are stringified rather than dropped.
		for _, e := range enum {
			if s, ok := jsonScalarToString(e); ok {
				gs.Enum = append(gs.Enum, s)
			}
		}
		// An INTEGER/NUMBER schema with a stringified enum needs no type change:
		// Gemini validates the enum against the declared type.
	}
	if v, ok := jsonSchemaInt(js["minLength"]); ok {
		gs.MinLength = &v
	}
	if v, ok := jsonSchemaInt(js["maxLength"]); ok {
		gs.MaxLength = &v
	}
	if p, ok := js["pattern"].(string); ok {
		gs.Pattern = p
	}
	if v, ok := jsonSchemaFloat(js["minimum"]); ok {
		gs.Minimum = &v
	}
	if v, ok := jsonSchemaFloat(js["maximum"]); ok {
		gs.Maximum = &v
	}
	if v, ok := jsonSchemaInt(js["minItems"]); ok {
		gs.MinItems = &v
	}
	if v, ok := jsonSchemaInt(js["maxItems"]); ok {
		gs.MaxItems = &v
	}
	// anyOf, and oneOf folded onto it.
	for _, key := range []string{"anyOf", "oneOf"} {
		variants, ok := js[key].([]interface{})
		if !ok {
			continue
		}
		for _, v := range variants {
			if vm, ok := v.(map[string]interface{}); ok {
				gs.AnyOf = append(gs.AnyOf, convertJSONSchemaToGemini(vm))
			}
		}
	}
	return gs
}

// jsonScalarToString renders a JSON scalar as a string for Gemini's string-only
// enum representation. Objects and arrays are not valid enum entries.
func jsonScalarToString(v interface{}) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, true
	case bool:
		return strconv.FormatBool(t), true
	case float64:
		// Render integral floats without a trailing ".0" so an integer enum
		// round-trips cleanly.
		if t == float64(int64(t)) {
			return strconv.FormatInt(int64(t), 10), true
		}
		return strconv.FormatFloat(t, 'g', -1, 64), true
	case json.Number:
		return t.String(), true
	default:
		return "", false
	}
}

// jsonSchemaInt extracts an int from a decoded JSON number.
func jsonSchemaInt(v interface{}) (int, bool) {
	f, ok := jsonSchemaFloat(v)
	if !ok {
		return 0, false
	}
	return int(f), true
}

// jsonSchemaFloat extracts a float64 from a decoded JSON number.
func jsonSchemaFloat(v interface{}) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case int:
		return float64(t), true
	case json.Number:
		f, err := t.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

// generateSyntheticID generates a synthetic UUID-like ID for Gemini function calls/responses
// that lack an explicit ID.
func generateSyntheticID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return fmt.Sprintf("call_%x", b)
}

// distributeCitationsToTextParts assigns candidate-level citations to the text content parts
// they belong to, based on character offset ranges. Gemini citations are candidate-level;
// other protocols (Anthropic, OpenAI) expect them at the part level.
func distributeCitationsToTextParts(content []ContentPart, citations []Citation) {
	if len(citations) == 0 {
		return
	}
	offset := 0
	for i := range content {
		if content[i].Type != ContentTypeText || content[i].Text == nil {
			continue
		}
		partLen := len(content[i].Text.Text)
		partStart := offset
		partEnd := offset + partLen

		var partCitations []Citation
		for _, c := range citations {
			if c.Start != nil && c.End != nil &&
				*c.Start >= partStart && *c.End <= partEnd {
				adjusted := c
				s := *c.Start - partStart
				e := *c.End - partStart
				adjusted.Start = &s
				adjusted.End = &e
				partCitations = append(partCitations, adjusted)
			}
		}
		content[i].Citations = partCitations
		offset = partEnd
	}
}

// --- URL path parsing ---

// parseGeminiModelFromURL extracts the model name from a Gemini API URL path.
// Expected formats:
//
//	/v1/models/{model}:generateContent
//	/v1/models/{model}:streamGenerateContent
//	/v1beta/models/{model}:generateContent
func parseGeminiModelFromURL(urlPath string) (string, error) {
	// Find "models/" in the path
	idx := strings.Index(urlPath, "models/")
	if idx == -1 {
		return "", fmt.Errorf("parse gemini model from URL: no 'models/' segment in %q", urlPath)
	}
	rest := urlPath[idx+len("models/"):]
	// The model name is everything up to the colon
	colonIdx := strings.Index(rest, ":")
	if colonIdx == -1 {
		return "", fmt.Errorf("parse gemini model from URL: no ':' found after model name in %q", urlPath)
	}
	model := rest[:colonIdx]
	if model == "" {
		return "", fmt.Errorf("parse gemini model from URL: empty model name in %q", urlPath)
	}
	return model, nil
}

// --- Decode functions ---

// DecodeGeminiRequest decodes a Gemini GenerateContent API request into the unified IR Request type.
// The model is extracted from the URL path, not the request body.
func DecodeGeminiRequest(urlPath string, body []byte) (*Request, error) {
	model, err := parseGeminiModelFromURL(urlPath)
	if err != nil {
		return nil, err
	}
	if err := requireJSONObject(body); err != nil {
		return nil, fmt.Errorf("decode gemini request: %w", err)
	}

	var raw gemini.Request
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode gemini request: %w", err)
	}

	req := &Request{
		Model: model,
	}

	// System instruction
	if raw.SystemInstruction != nil && len(raw.SystemInstruction.Parts) > 0 {
		parts, err := convertGeminiPartsToIR(raw.SystemInstruction.Parts, nil)
		if err != nil {
			return nil, fmt.Errorf("decode gemini request system_instruction: %w", err)
		}
		req.SystemPrompt = parts
	}

	// Contents → Messages
	if len(raw.Contents) > 0 {
		messages := make([]Message, 0, len(raw.Contents))
		correlator := newGeminiToolCorrelator()
		for i, c := range raw.Contents {
			msg, err := convertGeminiContentToMessage(c, correlator)
			if err != nil {
				return nil, fmt.Errorf("decode gemini request contents[%d]: %w", i, err)
			}
			messages = append(messages, msg)
		}
		req.Messages = messages
	}

	// Tools
	if len(raw.Tools) > 0 {
		var tools []Tool
		for _, td := range raw.Tools {
			for _, fd := range td.FunctionDeclarations {
				t := Tool{
					Name:        fd.Name,
					Description: fd.Description,
				}
				if len(fd.Parameters) > 0 {
					jsonParams, err := geminiSchemaToJSONSchema(fd.Parameters)
					if err != nil {
						return nil, fmt.Errorf("decode gemini request tool %q parameters: %w", fd.Name, err)
					}
					t.Parameters = jsonParams
				}
				tools = append(tools, t)
			}
		}
		if len(tools) > 0 {
			req.Tools = tools
		}
	}

	// Tool config
	if raw.ToolConfig != nil && raw.ToolConfig.FunctionCallingConfig != nil {
		fcc := raw.ToolConfig.FunctionCallingConfig
		tc := &ToolChoice{}
		switch strings.ToUpper(fcc.Mode) {
		case "AUTO":
			tc.Type = "auto"
		case "NONE":
			tc.Type = "none"
		case "ANY":
			tc.Type = "required"
		default:
			tc.Type = strings.ToLower(fcc.Mode)
		}
		// allowedFunctionNames → IR AllowedToolNames
		if len(fcc.AllowedFunctionNames) > 0 {
			tc.AllowedToolNames = fcc.AllowedFunctionNames
		}
		req.ToolChoice = tc
	}

	// Generation config
	if raw.GenerationConfig != nil {
		gc := raw.GenerationConfig
		req.Temperature = gc.Temperature
		req.TopP = gc.TopP
		req.TopK = gc.TopK
		if gc.MaxOutputTokens != nil {
			req.MaxTokens = *gc.MaxOutputTokens
		}
		req.StopSequences = gc.StopSequences

		// Response format
		if gc.ResponseMimeType != "" {
			rf := &ResponseFormat{}
			switch gc.ResponseMimeType {
			case "text/plain":
				rf.Type = "text"
			case "application/json":
				if len(gc.ResponseSchema) > 0 {
					rf.Type = "json_schema"
					jsonSchema, err := geminiSchemaToJSONSchema(gc.ResponseSchema)
					if err != nil {
						return nil, fmt.Errorf("decode gemini request response schema: %w", err)
					}
					rf.JSONSchema = jsonSchema
				} else {
					rf.Type = "json_object"
				}
			default:
				rf.Type = gc.ResponseMimeType
			}
			req.ResponseFormat = rf
		}
	}

	// Thinking config (nested inside generationConfig)
	if raw.GenerationConfig != nil && raw.GenerationConfig.ThinkingConfig != nil {
		tc := raw.GenerationConfig.ThinkingConfig
		switch {
		case tc.Budget() > 0:
			req.Thinking = &ThinkingConfig{
				Mode:         "enabled",
				BudgetTokens: tc.Budget(),
			}
		case tc.Budget() < 0:
			// Gemini uses -1 for dynamic thinking: the model picks the budget.
			req.Thinking = &ThinkingConfig{Mode: "adaptive"}
		case tc.ThinkingBudget != nil:
			// An explicit thinkingBudget of 0 turns thinking off in Gemini.
			req.Thinking = &ThinkingConfig{Mode: "disabled"}
		default:
			// thinkingConfig present but no budget — let the model decide.
			req.Thinking = &ThinkingConfig{
				Mode:         "adaptive",
				BudgetTokens: 0,
			}
		}
		// Phase 2: includeThoughts and thinkingLevel
		if tc.IncludeThoughts != nil {
			req.Thinking.IncludeThoughts = tc.IncludeThoughts
		}
		if tc.ThinkingLevel != "" {
			req.Thinking.Level = tc.ThinkingLevel
		}
	}

	return req, nil
}

// geminiToolCorrelator tracks synthetic tool IDs across Gemini contents in a single request.
type geminiToolCorrelator struct {
	idsByName  map[string][]string
	nextByName map[string]int
}

func newGeminiToolCorrelator() *geminiToolCorrelator {
	return &geminiToolCorrelator{
		idsByName:  make(map[string][]string),
		nextByName: make(map[string]int),
	}
}

func (c *geminiToolCorrelator) record(name, id string) {
	if c == nil || name == "" || id == "" {
		return
	}
	c.idsByName[name] = append(c.idsByName[name], id)
}

func (c *geminiToolCorrelator) correlate(name string) string {
	if c == nil || name == "" {
		return ""
	}
	ids := c.idsByName[name]
	offset := c.nextByName[name]
	if offset >= len(ids) {
		return ""
	}
	c.nextByName[name] = offset + 1
	return ids[offset]
}

// convertGeminiContentToMessage converts a gemini.Content to an IR Message.
func convertGeminiContentToMessage(c gemini.Content, correlator *geminiToolCorrelator) (Message, error) {
	var role Role
	switch c.Role {
	case "user":
		role = RoleUser
	case "model":
		role = RoleAssistant
	default:
		role = Role(c.Role)
	}

	// Check if this content has FunctionResponse parts — those should become RoleTool
	hasFuncResponse := false
	for _, p := range c.Parts {
		if p.FunctionResponse != nil {
			hasFuncResponse = true
			break
		}
	}
	if hasFuncResponse {
		role = RoleTool
	}

	parts, err := convertGeminiPartsToIR(c.Parts, correlator)
	if err != nil {
		return Message{}, err
	}

	return Message{
		Role:    role,
		Content: parts,
	}, nil
}

// convertGeminiPartsToIR converts a slice of gemini.Part to IR ContentParts.
func convertGeminiPartsToIR(parts []gemini.Part, correlator *geminiToolCorrelator) ([]ContentPart, error) {
	result := make([]ContentPart, 0, len(parts))
	for i, p := range parts {
		cp, err := convertGeminiPartToIR(p)
		if err != nil {
			return nil, fmt.Errorf("part[%d]: %w", i, err)
		}
		if p.FunctionCall != nil {
			correlator.record(p.FunctionCall.Name, cp.ToolUse.ID)
		}
		if p.FunctionResponse != nil && p.FunctionResponse.ID == "" {
			if correlatedID := correlator.correlate(p.FunctionResponse.Name); correlatedID != "" {
				cp.ToolResult.ToolUseID = correlatedID
			}
		}
		result = append(result, cp)
	}
	return result, nil
}

// convertGeminiPartToIR converts a single gemini.Part to an IR ContentPart.
func convertGeminiPartToIR(p gemini.Part) (ContentPart, error) {
	switch {
	case p.FunctionCall != nil:
		id := p.FunctionCall.ID
		if id == "" {
			id = generateSyntheticID()
		}
		return ContentPart{
			Type: ContentTypeToolUse,
			ToolUse: &ToolUseContent{
				ID:        id,
				Name:      p.FunctionCall.Name,
				Arguments: p.FunctionCall.Args,
			},
		}, nil

	case p.FunctionResponse != nil:
		id := p.FunctionResponse.ID
		if id == "" {
			id = generateSyntheticID()
		}
		// Wrap the response JSON as text content inside the tool result
		var content []ContentPart
		if len(p.FunctionResponse.Response) > 0 {
			content = []ContentPart{
				{Type: ContentTypeText, Text: &TextContent{Text: string(p.FunctionResponse.Response)}},
			}
		}
		return ContentPart{
			Type: ContentTypeToolResult,
			ToolResult: &ToolResultContent{
				ToolUseID: id,
				Content:   content,
			},
		}, nil

	case p.InlineData != nil:
		data, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil {
			return ContentPart{}, fmt.Errorf("decode inline data base64: %w", err)
		}
		if strings.HasPrefix(p.InlineData.MimeType, "application/pdf") {
			return ContentPart{
				Type: ContentTypeDocument,
				Document: &DocumentContent{
					Data:      data,
					MediaType: p.InlineData.MimeType,
				},
			}, nil
		}
		if strings.HasPrefix(p.InlineData.MimeType, "video/") {
			return ContentPart{
				Type: ContentTypeVideo,
				Video: &VideoContent{
					Data:      data,
					MediaType: p.InlineData.MimeType,
				},
			}, nil
		}
		return ContentPart{
			Type: ContentTypeImage,
			Image: &ImageContent{
				Data:      data,
				MediaType: p.InlineData.MimeType,
			},
		}, nil

	case p.FileData != nil:
		if strings.HasPrefix(p.FileData.MimeType, "application/pdf") {
			return ContentPart{
				Type: ContentTypeDocument,
				Document: &DocumentContent{
					URL:       p.FileData.FileURI,
					MediaType: p.FileData.MimeType,
				},
			}, nil
		}
		if strings.HasPrefix(p.FileData.MimeType, "video/") {
			return ContentPart{
				Type: ContentTypeVideo,
				Video: &VideoContent{
					URL:       p.FileData.FileURI,
					MediaType: p.FileData.MimeType,
				},
			}, nil
		}
		return ContentPart{
			Type: ContentTypeImage,
			Image: &ImageContent{
				URL:       p.FileData.FileURI,
				MediaType: p.FileData.MimeType,
			},
		}, nil

	case p.Thought != nil && *p.Thought:
		return ContentPart{
			Type:     ContentTypeThinking,
			Thinking: &ThinkingContent{Thinking: p.Text},
		}, nil

	case p.Text != "":
		return ContentPart{
			Type: ContentTypeText,
			Text: &TextContent{Text: p.Text},
		}, nil

	default:
		// Empty text part — still valid
		return ContentPart{
			Type: ContentTypeText,
			Text: &TextContent{Text: ""},
		}, nil
	}
}

// --- Encode functions ---

// EncodeGeminiRequest encodes a unified IR Request into a Gemini GenerateContent API request.
// Returns the model name (for URL construction) and the JSON body separately.
func EncodeGeminiRequest(req *Request) (model string, body []byte, err error) {
	raw := gemini.Request{}

	// Build tool-use ID → name map for function_response.name lookup.
	// Protocols like Anthropic omit the function name in tool_result blocks;
	// Gemini requires function_response.name to be set. Lazily allocated.
	var toolNameByID map[string]string
	for _, msg := range req.Messages {
		for _, p := range msg.Content {
			if p.Type == ContentTypeToolUse && p.ToolUse != nil && p.ToolUse.ID != "" {
				if toolNameByID == nil {
					toolNameByID = make(map[string]string)
				}
				toolNameByID[p.ToolUse.ID] = p.ToolUse.Name
			}
		}
	}

	// System prompt → systemInstruction
	if len(req.SystemPrompt) > 0 {
		parts := convertIRPartsToGemini(req.SystemPrompt)
		raw.SystemInstruction = &gemini.Content{
			Parts: parts,
		}
	}

	// Messages → contents
	if len(req.Messages) > 0 {
		contents := make([]gemini.Content, 0, len(req.Messages))
		for _, m := range req.Messages {
			c := encodeIRMessageToGemini(m, toolNameByID)
			contents = append(contents, c)
		}
		raw.Contents = contents
	}

	// Tools — only function tools round-trip into Gemini's FunctionDeclarations.
	// Server-side tools (Anthropic web_search, bash, computer, text_editor) have
	// no function-declaration representation and must be dropped so Gemini does
	// not receive a function with no parameters and 400 the request.
	encodedToolNames := map[string]bool{}
	if len(req.Tools) > 0 {
		decls := make([]gemini.FunctionDeclaration, 0, len(req.Tools))
		for _, t := range req.Tools {
			if !isGeminiSupportedToolType(t.Type) {
				continue
			}
			fd := gemini.FunctionDeclaration{
				Name:        t.Name,
				Description: t.Description,
			}
			if len(t.Parameters) > 0 {
				gemParams, err := jsonSchemaToGeminiSchema(t.Parameters)
				if err != nil {
					return "", nil, fmt.Errorf("encode gemini request tool %q parameters: %w", t.Name, err)
				}
				fd.Parameters = gemParams
			}
			decls = append(decls, fd)
			if t.Name != "" {
				encodedToolNames[t.Name] = true
			}
		}
		if len(decls) > 0 {
			raw.Tools = []gemini.ToolDeclaration{{FunctionDeclarations: decls}}
		}
	}

	// Tool choice — sanitize against the surviving tool set. Gemini requires
	// tool_config to reference known function names; degrading here avoids
	// provider-side errors when an Anthropic named selector points at a
	// dropped server tool.
	if req.ToolChoice != nil {
		hasTools := len(raw.Tools) > 0 && len(raw.Tools[0].FunctionDeclarations) > 0
		toolCount := 0
		if hasTools {
			toolCount = len(raw.Tools[0].FunctionDeclarations)
		}
		effective := sanitizeToolChoiceForEncode(req.ToolChoice, encodedToolNames, toolCount, len(req.Tools))
		if effective != nil {
			var mode string
			switch effective.Type {
			case "auto":
				mode = "AUTO"
			case "none":
				mode = "NONE"
			case "required":
				mode = "ANY"
			default:
				mode = strings.ToUpper(effective.Type)
			}
			fcc := &gemini.FunctionCallingConfig{Mode: mode}
			if len(effective.AllowedToolNames) > 0 {
				fcc.AllowedFunctionNames = effective.AllowedToolNames
			}
			// AllowParallelCalls: Gemini has no parallel_tool_calls equivalent — silently drop.
			raw.ToolConfig = &gemini.ToolConfig{
				FunctionCallingConfig: fcc,
			}
		}
	}

	// Generation config
	hasGenConfig := req.Temperature != nil || req.TopP != nil || req.TopK != nil ||
		req.MaxTokens > 0 || len(req.StopSequences) > 0 || req.ResponseFormat != nil
	if hasGenConfig {
		gc := &gemini.GenerationConfig{
			Temperature:   req.Temperature,
			TopP:          req.TopP,
			TopK:          req.TopK,
			StopSequences: req.StopSequences,
		}
		if req.MaxTokens > 0 {
			mt := req.MaxTokens
			gc.MaxOutputTokens = &mt
		}
		// Response format
		if req.ResponseFormat != nil {
			switch req.ResponseFormat.Type {
			case "text":
				gc.ResponseMimeType = "text/plain"
			case "json_object":
				gc.ResponseMimeType = "application/json"
			case "json_schema":
				gc.ResponseMimeType = "application/json"
				if len(req.ResponseFormat.JSONSchema) > 0 {
					gemSchema, err := jsonSchemaToGeminiSchema(req.ResponseFormat.JSONSchema)
					if err != nil {
						return "", nil, fmt.Errorf("encode gemini request response schema: %w", err)
					}
					gc.ResponseSchema = gemSchema
				}
			}
		}
		raw.GenerationConfig = gc
	}

	// Thinking config (nested inside generationConfig)
	if req.Thinking != nil {
		if raw.GenerationConfig == nil {
			raw.GenerationConfig = &gemini.GenerationConfig{}
		}
		var gtc *gemini.ThinkingConfig
		switch req.Thinking.Mode {
		case "enabled":
			if req.Thinking.BudgetTokens > 0 {
				gtc = &gemini.ThinkingConfig{}
				gtc.SetThinkingBudget(req.Thinking.BudgetTokens)
			} else {
				// codeflicker-fix: LOGIC-Issue-001/tb3m3jp0fdxv42afyew5
				// OpenAI reasoning_effort has no Gemini budget equivalent. Preserve
				// enabled intent with an adaptive thinking config instead of dropping it.
				gtc = &gemini.ThinkingConfig{}
			}
		case "adaptive":
			// No explicit budget — the model chooses how much to think.
			gtc = &gemini.ThinkingConfig{}
		case "disabled":
			// Gemini enables thinking by default on models that support it, so an
			// explicit disable must be forwarded as a zero budget. Emitting no
			// thinkingConfig at all would invert the caller's intent.
			gtc = &gemini.ThinkingConfig{}
			gtc.SetThinkingBudget(0)
		}
		// Phase 2: IncludeThoughts and Level always propagate when ThinkingConfig is set,
		// even if Mode doesn't match "enabled"/"adaptive" exactly.
		if gtc == nil && (req.Thinking.IncludeThoughts != nil || req.Thinking.Level != "") {
			gtc = &gemini.ThinkingConfig{}
		}
		if gtc != nil {
			if req.Thinking.IncludeThoughts != nil {
				gtc.IncludeThoughts = req.Thinking.IncludeThoughts
			}
			if req.Thinking.Level != "" {
				gtc.ThinkingLevel = req.Thinking.Level
			}
			raw.GenerationConfig.ThinkingConfig = gtc
		}
	}

	bodyBytes, err := json.Marshal(raw)
	if err != nil {
		return "", nil, fmt.Errorf("encode gemini request: %w", err)
	}

	return req.Model, bodyBytes, nil
}

// geminiFunctionResponsePayload builds the function_response.response value.
//
// Gemini requires this field to be a google.protobuf.Struct — i.e. a JSON object,
// never a bare string or scalar. Two cases matter:
//
//   - The tool result text is already a JSON object (the common case, and what
//     DecodeGeminiRequest produces when it stringifies an inbound Struct). It is
//     passed through verbatim so a Gemini→Gemini round-trip preserves structure
//     instead of nesting the object inside a JSON string under "result".
//   - Anything else (plain text, a JSON array, a scalar) is wrapped as
//     {"result": <text>} because it cannot be a Struct on its own.
//
// When the result is an error, an "error" key is added so the model can tell a
// failure from a success — Gemini has no is_error flag.
func geminiFunctionResponsePayload(result *ToolResultContent) json.RawMessage {
	if len(result.Content) == 0 {
		if !result.IsError {
			return nil
		}
		wrapped, _ := json.Marshal(map[string]any{"error": true})
		return json.RawMessage(wrapped)
	}

	text := toolResultText(result)

	// Pass through an existing JSON object unchanged.
	trimmed := strings.TrimSpace(text)
	if strings.HasPrefix(trimmed, "{") && json.Valid([]byte(trimmed)) {
		if !result.IsError {
			return json.RawMessage(trimmed)
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal([]byte(trimmed), &obj); err == nil {
			if _, exists := obj["error"]; !exists {
				obj["error"] = json.RawMessage("true")
			}
			if merged, err := json.Marshal(obj); err == nil {
				return json.RawMessage(merged)
			}
		}
		return json.RawMessage(trimmed)
	}

	payload := map[string]any{"result": text}
	if result.IsError {
		payload["error"] = true
	}
	wrapped, _ := json.Marshal(payload)
	return json.RawMessage(wrapped)
}

// encodeIRMessageToGemini converts an IR Message to a gemini.Content.
func encodeIRMessageToGemini(m Message, toolNameByID map[string]string) gemini.Content {
	var role string
	switch m.Role {
	case RoleUser:
		role = "user"
	case RoleAssistant:
		role = "model"
	case RoleTool:
		// Gemini has no native "tool" role — use "user" with FunctionResponse parts
		role = "user"
	default:
		role = string(m.Role)
	}

	// Resolve missing function_response names before encoding.
	// Protocols like Anthropic omit the name in tool_result; Gemini requires it.
	content := m.Content
	if len(toolNameByID) > 0 {
		var copied bool
		for i, p := range content {
			if p.Type == ContentTypeToolResult && p.ToolResult != nil && p.ToolResult.Name == "" {
				if name := toolNameByID[p.ToolResult.ToolUseID]; name != "" {
					if !copied {
						content = append([]ContentPart(nil), m.Content...)
						copied = true
					}
					tr := *content[i].ToolResult
					tr.Name = name
					content[i].ToolResult = &tr
				}
			}
		}
	}

	return gemini.Content{
		Role:  role,
		Parts: ensureNonEmptyGeminiParts(convertIRPartsToGemini(content)),
	}
}

// ensureNonEmptyGeminiParts guarantees at least one part. Gemini rejects a
// content entry with an empty parts array (400 INVALID_ARGUMENT), which would
// otherwise happen for a message whose every part was unrepresentable or empty.
func ensureNonEmptyGeminiParts(parts []gemini.Part) []gemini.Part {
	if len(parts) > 0 {
		return parts
	}
	return []gemini.Part{{Text: ""}}
}

// convertIRPartsToGemini converts IR ContentParts to gemini.Parts.
//
// Parts with no Gemini representation are rendered as a short text placeholder
// instead of being dropped: Gemini rejects a content entry whose parts array is
// empty, and a silent drop hides information from the model.
func convertIRPartsToGemini(parts []ContentPart) []gemini.Part {
	result := make([]gemini.Part, 0, len(parts))
	for _, p := range parts {
		gp := convertIRPartToGemini(p)
		if isEmptyGeminiPart(gp) {
			// Empty text is legitimate for an explicit empty text part; anything
			// else means the content type had no mapping.
			if p.Type == ContentTypeText {
				continue
			}
			if text := geminiPlaceholderText(p); text != "" {
				result = append(result, gemini.Part{Text: text})
			}
			continue
		}
		result = append(result, gp)
	}
	return result
}

// isEmptyGeminiPart reports whether a part would serialize to a bare {} object.
func isEmptyGeminiPart(gp gemini.Part) bool {
	return gp.Text == "" && gp.InlineData == nil && gp.FileData == nil &&
		gp.FunctionCall == nil && gp.FunctionResponse == nil && gp.Thought == nil
}

// geminiPlaceholderText renders an IR part that Gemini cannot express as text.
func geminiPlaceholderText(p ContentPart) string {
	if p.Type == ContentTypeDocument {
		return documentPlaceholderText(p.Document)
	}
	return unrepresentablePlaceholderText(p)
}

// convertIRPartToGemini converts a single IR ContentPart to a gemini.Part.
func convertIRPartToGemini(p ContentPart) gemini.Part {
	switch p.Type {
	case ContentTypeText:
		gp := gemini.Part{}
		if p.Text != nil {
			gp.Text = p.Text.Text
		}
		return gp

	case ContentTypeImage:
		if p.Image != nil {
			if len(p.Image.Data) > 0 {
				return gemini.Part{
					InlineData: &gemini.InlineData{
						MimeType: p.Image.MediaType,
						Data:     base64.StdEncoding.EncodeToString(p.Image.Data),
					},
				}
			}
			if p.Image.URL != "" {
				return gemini.Part{
					FileData: &gemini.FileData{
						MimeType: p.Image.MediaType,
						FileURI:  p.Image.URL,
					},
				}
			}
		}
		return gemini.Part{}

	case ContentTypeDocument:
		if p.Document != nil {
			if len(p.Document.Data) > 0 {
				return gemini.Part{
					InlineData: &gemini.InlineData{
						MimeType: p.Document.MediaType,
						Data:     base64.StdEncoding.EncodeToString(p.Document.Data),
					},
				}
			}
			if p.Document.URL != "" {
				return gemini.Part{
					FileData: &gemini.FileData{
						MimeType: p.Document.MediaType,
						FileURI:  p.Document.URL,
					},
				}
			}
		}
		return gemini.Part{}

	case ContentTypeVideo:
		if p.Video != nil {
			if len(p.Video.Data) > 0 {
				return gemini.Part{
					InlineData: &gemini.InlineData{
						MimeType: p.Video.MediaType,
						Data:     base64.StdEncoding.EncodeToString(p.Video.Data),
					},
				}
			}
			if p.Video.URL != "" {
				return gemini.Part{
					FileData: &gemini.FileData{
						MimeType: p.Video.MediaType,
						FileURI:  p.Video.URL,
					},
				}
			}
		}
		return gemini.Part{}

	case ContentTypeToolUse:
		if p.ToolUse != nil {
			return gemini.Part{
				FunctionCall: &gemini.FunctionCall{
					Name: p.ToolUse.Name,
					Args: p.ToolUse.Arguments,
					ID:   p.ToolUse.ID,
				},
			}
		}
		return gemini.Part{}

	case ContentTypeToolResult:
		if p.ToolResult != nil {
			return gemini.Part{
				FunctionResponse: &gemini.FunctionResponse{
					Name:     p.ToolResult.Name,
					Response: geminiFunctionResponsePayload(p.ToolResult),
					ID:       p.ToolResult.ToolUseID,
				},
			}
		}
		return gemini.Part{}

	case ContentTypeThinking:
		if p.Thinking != nil {
			thought := true
			return gemini.Part{
				Text:    p.Thinking.Thinking,
				Thought: &thought,
			}
		}
		return gemini.Part{}

	default:
		// Skip unsupported content types — return empty part with no text,
		// which will be filtered out by the caller if needed.
		return gemini.Part{}
	}
}

// --- Response decode/encode ---

// DecodeGeminiResponse decodes a Gemini GenerateContent API JSON response body
// into the unified IR Response type.
func DecodeGeminiResponse(body []byte) (*Response, error) {
	var raw gemini.Response
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("decode gemini response: %w", err)
	}

	resp := &Response{
		Model: raw.ModelVersion,
	}

	// Usage
	if raw.UsageMetadata != nil {
		resp.Usage = Usage{
			PromptTokens:              raw.UsageMetadata.PromptTokenCount,
			PromptCacheHitTokens:      raw.UsageMetadata.CachedContentTokenCount,
			CompletionTokens:          raw.UsageMetadata.CandidatesTokenCount,
			CompletionReasoningTokens: raw.UsageMetadata.ThoughtsTokenCount,
			ServerToolUseTokens:       raw.UsageMetadata.ToolUsePromptTokenCount,
			TotalTokens:               raw.UsageMetadata.TotalTokenCount,
		}
	}

	// Candidates (use first)
	if len(raw.Candidates) > 0 {
		cand := raw.Candidates[0]

		if cand.Content != nil && len(cand.Content.Parts) > 0 {
			parts, err := convertGeminiPartsToIR(cand.Content.Parts, nil)
			if err != nil {
				return nil, fmt.Errorf("decode gemini response content: %w", err)
			}
			resp.Content = parts
		}

		// Parse candidate-level citation metadata and distribute to text content parts
		// by character offset range.
		if cand.CitationMetadata != nil && len(cand.CitationMetadata.CitationSources) > 0 {
			citations := make([]Citation, 0, len(cand.CitationMetadata.CitationSources))
			for _, cs := range cand.CitationMetadata.CitationSources {
				c := Citation{
					Kind:  CitationKindGemini,
					Title: cs.Title,
					URL:   cs.URI,
				}
				start := cs.StartIndex
				c.Start = &start
				end := cs.EndIndex
				c.End = &end
				citations = append(citations, c)
			}
			distributeCitationsToTextParts(resp.Content, citations)
		}

		// Check for FunctionCall parts to infer StopReasonToolUse
		hasFunctionCall := false
		for _, p := range resp.Content {
			if p.Type == ContentTypeToolUse {
				hasFunctionCall = true
				break
			}
		}

		// Map finishReason to StopReason. Only infer tool_use when finishReason
		// is ambiguous (STOP or empty); explicit reasons like SAFETY take priority.
		switch cand.FinishReason {
		case "STOP":
			if hasFunctionCall {
				resp.StopReason = StopReasonToolUse
			} else {
				resp.StopReason = StopReasonEndTurn
			}
		case "MAX_TOKENS":
			resp.StopReason = StopReasonMaxTokens
		case "SAFETY":
			resp.StopReason = StopReasonContentFilter
		case "STOP_SEQUENCE":
			resp.StopReason = StopReasonStopSequence
		default:
			if cand.FinishReason != "" {
				resp.StopReason = StopReason(cand.FinishReason)
			} else if hasFunctionCall {
				resp.StopReason = StopReasonToolUse
			}
		}
	}

	return resp, nil
}

// stopReasonToGeminiFinishReason converts a unified IR StopReason to the Gemini finishReason string.
//
// Gemini's finishReason is a closed enum, so unrecognised IR reasons (e.g.
// Anthropic's "refusal") are bucketed onto the closest valid value rather than
// being forwarded verbatim.
func stopReasonToGeminiFinishReason(r StopReason) string {
	switch r {
	case StopReasonEndTurn:
		return "STOP"
	case StopReasonMaxTokens:
		return "MAX_TOKENS"
	case StopReasonContentFilter:
		return "SAFETY"
	case StopReasonStopSequence:
		return "STOP_SEQUENCE"
	case StopReasonToolUse:
		return "STOP"
	case StopReasonPauseTurn:
		return "MAX_TOKENS"
	case "":
		return ""
	default:
		// Already a Gemini enum value (round-trip of an unmapped reason such as
		// RECITATION or BLOCKLIST) — keep it. Otherwise bucket via the shared
		// OpenAI heuristic and translate.
		if isGeminiFinishReason(string(r)) {
			return string(r)
		}
		switch openAIFinishReasonFallback(string(r)) {
		case "length":
			return "MAX_TOKENS"
		case "content_filter":
			return "SAFETY"
		default:
			return "STOP"
		}
	}
}

// isGeminiFinishReason reports whether s is already a documented Gemini
// finishReason value, which happens when a Gemini→Gemini round-trip carries a
// reason the IR does not model.
func isGeminiFinishReason(s string) bool {
	switch s {
	case "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "LANGUAGE", "OTHER",
		"BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL",
		"IMAGE_SAFETY", "UNEXPECTED_TOOL_CALL", "STOP_SEQUENCE",
		"FINISH_REASON_UNSPECIFIED":
		return true
	}
	return false
}

// EncodeGeminiResponse encodes a unified IR Response into a Gemini GenerateContent API JSON body.
func EncodeGeminiResponse(resp *Response) ([]byte, error) {
	raw := gemini.Response{
		ModelVersion: resp.Model,
	}

	// Content
	var parts []gemini.Part
	if len(resp.Content) > 0 {
		parts = convertIRPartsToGemini(resp.Content)
	}

	finishReason := stopReasonToGeminiFinishReason(resp.StopReason)

	candidate := gemini.Candidate{
		Content: &gemini.Content{
			Role:  "model",
			Parts: parts,
		},
		FinishReason: finishReason,
	}

	// Citations are candidate-level in Gemini; collect them back from the
	// per-part IR representation and re-base their offsets onto the concatenated
	// candidate text. This is the inverse of distributeCitationsToTextParts, so
	// citations survive a round-trip instead of being decode-only.
	if cm := collectGeminiCitationMetadata(resp.Content); cm != nil {
		candidate.CitationMetadata = cm
	}

	raw.Candidates = []gemini.Candidate{candidate}

	// Usage
	raw.UsageMetadata = &gemini.UsageMetadata{
		PromptTokenCount:        resp.Usage.PromptTokens,
		CandidatesTokenCount:    resp.Usage.CompletionTokens,
		TotalTokenCount:         resp.Usage.TotalTokens,
		ThoughtsTokenCount:      resp.Usage.CompletionReasoningTokens,
		CachedContentTokenCount: resp.Usage.PromptCacheHitTokens,
		ToolUsePromptTokenCount: resp.Usage.ServerToolUseTokens,
	}

	return json.Marshal(raw)
}

// collectGeminiCitationMetadata is the inverse of distributeCitationsToTextParts:
// it gathers per-part IR citations back into candidate-level citationMetadata,
// re-basing each offset onto the concatenated candidate text. Returns nil when
// there are no citations.
func collectGeminiCitationMetadata(content []ContentPart) *gemini.CitationMetadata {
	var sources []gemini.CitationSource
	offset := 0
	for _, p := range content {
		if p.Type != ContentTypeText || p.Text == nil {
			continue
		}
		for _, c := range p.Citations {
			src := gemini.CitationSource{
				URI:   c.URL,
				Title: c.Title,
			}
			if c.Start != nil {
				src.StartIndex = *c.Start + offset
			}
			if c.End != nil {
				src.EndIndex = *c.End + offset
			}
			sources = append(sources, src)
		}
		offset += len(p.Text.Text)
	}
	if len(sources) == 0 {
		return nil
	}
	return &gemini.CitationMetadata{CitationSources: sources}
}

// --- Streaming decode/encode ---

// DecodeGeminiStreamChunk decodes a Gemini streaming chunk JSON (the data from an SSE "data:" line)
// into the unified IR StreamEvent.
// Gemini sends complete GenerateContentResponse objects per chunk.
func DecodeGeminiStreamChunk(data []byte) ([]*StreamEvent, error) {
	var raw gemini.Response
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("decode gemini stream chunk: %w", err)
	}

	// Check for finish reason in the first candidate
	var finishReason string
	var contentParts []gemini.Part
	if len(raw.Candidates) > 0 {
		cand := raw.Candidates[0]
		finishReason = cand.FinishReason
		if cand.Content != nil {
			contentParts = cand.Content.Parts
		}
	}

	// Convert parts to IR
	var irParts []ContentPart
	if len(contentParts) > 0 {
		var err error
		irParts, err = convertGeminiPartsToIR(contentParts, nil)
		if err != nil {
			return nil, fmt.Errorf("decode gemini stream chunk parts: %w", err)
		}
	}

	// Check for function calls
	hasFunctionCall := false
	for _, p := range irParts {
		if p.Type == ContentTypeToolUse {
			hasFunctionCall = true
			break
		}
	}

	// Determine stop reason. Mirror the non-streaming DecodeGeminiResponse
	// precedence: an explicit finishReason such as MAX_TOKENS or SAFETY always
	// wins, and tool_use is only inferred when the reason is ambiguous (STOP or
	// absent). Previously any functionCall unconditionally forced tool_use, which
	// masked truncation and safety blocks in the streaming path.
	var stopReason *StopReason
	if finishReason != "" || hasFunctionCall {
		var sr StopReason
		switch finishReason {
		case "STOP":
			if hasFunctionCall {
				sr = StopReasonToolUse
			} else {
				sr = StopReasonEndTurn
			}
		case "MAX_TOKENS":
			sr = StopReasonMaxTokens
		case "SAFETY":
			sr = StopReasonContentFilter
		case "STOP_SEQUENCE":
			sr = StopReasonStopSequence
		case "":
			sr = StopReasonToolUse // hasFunctionCall is necessarily true here
		default:
			sr = StopReason(finishReason)
		}
		stopReason = &sr
	}

	// Usage
	var usage *Usage
	if raw.UsageMetadata != nil {
		usage = &Usage{
			PromptTokens:              raw.UsageMetadata.PromptTokenCount,
			PromptCacheHitTokens:      raw.UsageMetadata.CachedContentTokenCount,
			CompletionTokens:          raw.UsageMetadata.CandidatesTokenCount,
			CompletionReasoningTokens: raw.UsageMetadata.ThoughtsTokenCount,
			ServerToolUseTokens:       raw.UsageMetadata.ToolUsePromptTokenCount,
			TotalTokens:               raw.UsageMetadata.TotalTokenCount,
		}
	}

	// Build events
	if len(irParts) > 0 {
		events := make([]*StreamEvent, len(irParts))
		for i := range irParts {
			events[i] = &StreamEvent{
				Type:  StreamEventDelta,
				Index: i,
				Delta: &irParts[i],
			}
		}
		// Attach stop reason and usage to the last delta event
		last := events[len(events)-1]
		last.StopReason = stopReason
		last.Usage = usage
		return events, nil
	} else if stopReason != nil {
		event := &StreamEvent{
			Type:       StreamEventStop,
			StopReason: stopReason,
			Usage:      usage,
		}
		return []*StreamEvent{event}, nil
	} else {
		event := &StreamEvent{
			Type: StreamEventStart,
			Response: &Response{
				Model: raw.ModelVersion,
			},
			Usage: usage,
		}
		return []*StreamEvent{event}, nil
	}
}

// EncodeGeminiStreamChunk encodes a unified IR StreamEvent into a Gemini
// GenerateContentResponse JSON chunk (suitable for an SSE "data:" line).
func EncodeGeminiStreamChunk(event *StreamEvent) ([]byte, error) {
	// A nil event is a "nothing to emit" signal from an upstream decoder that saw a
	// chunk it does not map. Treat it as a skip rather than dereferencing it.
	if event == nil {
		return nil, nil
	}
	raw := gemini.Response{}

	switch event.Type {
	case StreamEventStart:
		if event.Response != nil {
			raw.ModelVersion = event.Response.Model
		}
		// Empty candidates for start
		raw.Candidates = []gemini.Candidate{
			{
				Content: &gemini.Content{
					Role:  "model",
					Parts: []gemini.Part{},
				},
			},
		}

	case StreamEventDelta:
		var parts []gemini.Part
		if event.Delta != nil {
			gp := convertIRPartToGemini(*event.Delta)
			parts = []gemini.Part{gp}
		}

		cand := gemini.Candidate{
			Content: &gemini.Content{
				Role:  "model",
				Parts: parts,
			},
		}

		// If there's a stop reason on this delta, include finish reason
		if event.StopReason != nil {
			cand.FinishReason = stopReasonToGeminiFinishReason(*event.StopReason)
		}

		raw.Candidates = []gemini.Candidate{cand}

	case StreamEventStop:
		cand := gemini.Candidate{
			Content: &gemini.Content{
				Role:  "model",
				Parts: []gemini.Part{},
			},
		}
		if event.StopReason != nil {
			cand.FinishReason = stopReasonToGeminiFinishReason(*event.StopReason)
		} else {
			cand.FinishReason = "STOP"
		}
		raw.Candidates = []gemini.Candidate{cand}

	case StreamEventContentBlockStart, StreamEventContentBlockStop:
		// Block lifecycle has no Gemini representation. Tool-call name/args
		// assembly is handled by geminiCodec.WriteStreamingResponse, which buffers
		// the fragments and emits one complete functionCall part.
		return nil, nil

	case StreamEventError:
		// Error events have no direct equivalent in Gemini streaming; skip silently.
		return nil, nil

	default:
		return nil, fmt.Errorf("unknown IR stream event type: %q", event.Type)
	}

	// Usage
	if event.Usage != nil {
		raw.UsageMetadata = &gemini.UsageMetadata{
			PromptTokenCount:        event.Usage.PromptTokens,
			CandidatesTokenCount:    event.Usage.CompletionTokens,
			TotalTokenCount:         event.Usage.TotalTokens,
			ThoughtsTokenCount:      event.Usage.CompletionReasoningTokens,
			CachedContentTokenCount: event.Usage.PromptCacheHitTokens,
			ToolUsePromptTokenCount: event.Usage.ServerToolUseTokens,
		}
	}

	return json.Marshal(raw)
}
