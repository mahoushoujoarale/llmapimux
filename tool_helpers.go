package llmapimux

import (
	"encoding/json"
	"fmt"
)

func cloneRawMessage(src json.RawMessage) json.RawMessage {
	if src == nil {
		return nil
	}
	dst := make([]byte, len(src))
	copy(dst, src)
	return json.RawMessage(dst)
}

func cloneRawMessageMap(src map[string]json.RawMessage) map[string]json.RawMessage {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]json.RawMessage, len(src))
	for k, v := range src {
		dst[k] = cloneRawMessage(v)
	}
	return dst
}

func normalizeAnthropicToolType(raw string) string {
	switch raw {
	case "", "custom":
		return "function"
	case "web_search_20250305":
		return "web_search"
	default:
		return raw
	}
}

func anthropicToolTypeFromIR(raw string) string {
	switch raw {
	case "", "function", "custom":
		return "custom"
	case "web_search":
		return "web_search_20250305"
	default:
		return raw
	}
}

func normalizeOpenAIResponsesToolType(raw string) string {
	switch raw {
	case "", "function":
		return "function"
	case "web_search_preview", "web_search_preview_2025_03_11":
		return "web_search"
	default:
		return raw
	}
}

func openAIResponsesToolTypeFromIR(raw string) string {
	switch raw {
	case "", "function", "custom":
		return "function"
	case "web_search":
		return "web_search"
	default:
		return raw
	}
}

func isFunctionToolType(raw string) bool {
	switch raw {
	case "", "function", "custom":
		return true
	default:
		return false
	}
}

// isOpenAIResponsesSupportedToolType reports whether an IR tool type has a
// known representation in the OpenAI Responses API. Anthropic server-side tools
// such as bash_*, computer_*, and text_editor_* are preserved in IR so that
// Anthropic-to-Anthropic passthroughs keep them, but they have no OpenAI
// Responses equivalent and must be dropped before the request is sent.
func isOpenAIResponsesSupportedToolType(irType string) bool {
	if isFunctionToolType(irType) {
		return true
	}
	switch openAIResponsesToolTypeFromIR(irType) {
	case "web_search",
		"web_search_preview",
		"file_search",
		"code_interpreter",
		"computer_use",
		"mcp":
		return true
	}
	return false
}

// isOpenAIChatSupportedToolType reports whether an IR tool type is
// representable in the OpenAI Chat Completions API. Chat Completions only
// supports function tools.
func isOpenAIChatSupportedToolType(irType string) bool {
	return isFunctionToolType(irType)
}

// isGeminiSupportedToolType reports whether an IR tool type is representable
// as a Gemini function declaration. Gemini has distinct request shapes for its
// own built-in tools (google_search, code_execution) which do not round-trip
// through function declarations, so anything non-function is dropped here.
func isGeminiSupportedToolType(irType string) bool {
	return isFunctionToolType(irType)
}

func defaultToolNameForType(raw string) string {
	switch raw {
	case "", "function", "custom":
		return ""
	default:
		return raw
	}
}

func findToolByName(tools []Tool, name string) *Tool {
	for i := range tools {
		if tools[i].Name == name {
			return &tools[i]
		}
	}
	return nil
}

func findToolByType(tools []Tool, typ string) *Tool {
	for i := range tools {
		if tools[i].Type == typ {
			return &tools[i]
		}
	}
	return nil
}

func selectToolsByName(tools []Tool, names []string) []Tool {
	if len(names) == 0 {
		return nil
	}
	selected := make([]Tool, 0, len(names))
	for _, name := range names {
		if tool := findToolByName(tools, name); tool != nil {
			selected = append(selected, *tool)
		}
	}
	return selected
}

// sanitizeToolChoiceForEncode adapts an IR ToolChoice against the tool set
// that survived encoding. It only intervenes when the choice is a hard
// reference to a specific tool:
//
//   - tc.Type == "tool" with ToolName not present in encodedToolNames →
//     degrade to "auto" so the outbound JSON does not reach the provider as
//     tool_choice={type:"function",name:X} with no matching tool, which
//     reproduces the original "Tool choice 'function' not found in 'tools'"
//     error class.
//
// Abstract choices (auto, none, required) and AllowedToolNames pass through
// unchanged — AllowedToolNames is a soft hint in Gemini and providers that
// strict-validate individual names generally tolerate a permissive list, so
// we defer to provider-side validation instead of over-sanitizing here.
// Returns nil when only AllowParallelCalls was set (empty Type, no
// AllowedToolNames), so callers skip emitting an empty tool_choice object.
func sanitizeToolChoiceForEncode(tc *ToolChoice, encodedToolNames map[string]bool, encodedToolCount int) *ToolChoice {
	if tc == nil {
		return nil
	}
	clone := *tc
	if clone.Type == "tool" && clone.ToolName != "" {
		if encodedToolCount == 0 || !encodedToolNames[clone.ToolName] {
			clone.Type = "auto"
			clone.ToolName = ""
		}
	}
	if clone.Type == "" && len(clone.AllowedToolNames) == 0 {
		return nil
	}
	return &clone
}

// nameSetFromIRTools returns the set of non-empty IR tool names, used by
// sanitizeToolChoiceForEncode to detect dropped selectors.
func nameSetFromIRTools(tools []Tool) map[string]bool {
	if len(tools) == 0 {
		return nil
	}
	names := make(map[string]bool, len(tools))
	for _, t := range tools {
		if t.Name != "" {
			names[t.Name] = true
		}
	}
	return names
}

func isNonEmptyJSONArray(raw json.RawMessage) (bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return false, nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return false, err
	}
	return len(items) > 0, nil
}

func decodeOpenAIResponsesToolExtraFields(toolType string, extra map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	dst := cloneRawMessageMap(extra)
	if len(dst) == 0 || isFunctionToolType(toolType) {
		return dst, nil
	}

	switch normalizeOpenAIResponsesToolType(toolType) {
	case "web_search":
		if filtersRaw, ok := dst["filters"]; ok && len(filtersRaw) > 0 && string(filtersRaw) != "null" {
			var filters map[string]json.RawMessage
			if err := json.Unmarshal(filtersRaw, &filters); err != nil {
				return nil, fmt.Errorf("decode web_search filters: %w", err)
			}
			if allowedDomains, ok := filters["allowed_domains"]; ok && len(allowedDomains) > 0 && string(allowedDomains) != "null" {
				nonEmpty, err := isNonEmptyJSONArray(allowedDomains)
				if err != nil {
					return nil, fmt.Errorf("decode web_search filters.allowed_domains: %w", err)
				}
				if nonEmpty {
					dst["allowed_domains"] = cloneRawMessage(allowedDomains)
				}
				delete(filters, "allowed_domains")
			}
			if len(filters) > 0 {
				sanitizedFilters, err := json.Marshal(filters)
				if err != nil {
					return nil, fmt.Errorf("decode sanitized web_search filters: %w", err)
				}
				dst["filters"] = sanitizedFilters
			} else {
				delete(dst, "filters")
			}
		}
	}

	if len(dst) == 0 {
		return nil, nil
	}
	return dst, nil
}

func encodeOpenAIResponsesToolExtraFields(toolType string, extra map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	dst := cloneRawMessageMap(extra)
	if len(dst) == 0 || isFunctionToolType(toolType) {
		return dst, nil
	}

	switch openAIResponsesToolTypeFromIR(toolType) {
	case "web_search":
		var filters map[string]json.RawMessage
		if filtersRaw, ok := dst["filters"]; ok && len(filtersRaw) > 0 && string(filtersRaw) != "null" {
			if err := json.Unmarshal(filtersRaw, &filters); err != nil {
				return nil, fmt.Errorf("encode web_search filters: %w", err)
			}
		}
		if filters == nil {
			filters = make(map[string]json.RawMessage)
		}
		if allowedDomains, ok := dst["allowed_domains"]; ok && len(allowedDomains) > 0 && string(allowedDomains) != "null" {
			nonEmpty, err := isNonEmptyJSONArray(allowedDomains)
			if err != nil {
				return nil, fmt.Errorf("encode web_search allowed_domains: %w", err)
			}
			if nonEmpty {
				filters["allowed_domains"] = cloneRawMessage(allowedDomains)
			}
		}
		delete(dst, "allowed_domains")
		delete(dst, "blocked_domains")
		delete(dst, "max_uses")

		if len(filters) > 0 {
			filtersRaw, err := json.Marshal(filters)
			if err != nil {
				return nil, fmt.Errorf("encode web_search filters object: %w", err)
			}
			dst["filters"] = filtersRaw
		} else {
			delete(dst, "filters")
		}
	}

	if len(dst) == 0 {
		return nil, nil
	}
	return dst, nil
}

func encodeAnthropicToolExtraFields(toolType string, extra map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	dst := cloneRawMessageMap(extra)
	if len(dst) == 0 || isFunctionToolType(toolType) {
		return dst, nil
	}

	switch anthropicToolTypeFromIR(toolType) {
	case "web_search_20250305":
		result := make(map[string]json.RawMessage, 3)
		if allowedDomains, ok := dst["allowed_domains"]; ok && len(allowedDomains) > 0 && string(allowedDomains) != "null" {
			result["allowed_domains"] = cloneRawMessage(allowedDomains)
		} else if filtersRaw, ok := dst["filters"]; ok && len(filtersRaw) > 0 && string(filtersRaw) != "null" {
			var filters map[string]json.RawMessage
			if err := json.Unmarshal(filtersRaw, &filters); err != nil {
				return nil, fmt.Errorf("encode anthropic web_search filters: %w", err)
			}
			if allowedDomains, ok := filters["allowed_domains"]; ok && len(allowedDomains) > 0 && string(allowedDomains) != "null" {
				result["allowed_domains"] = cloneRawMessage(allowedDomains)
			}
		}
		if blockedDomains, ok := dst["blocked_domains"]; ok && len(blockedDomains) > 0 && string(blockedDomains) != "null" {
			result["blocked_domains"] = cloneRawMessage(blockedDomains)
		}
		if maxUses, ok := dst["max_uses"]; ok && len(maxUses) > 0 && string(maxUses) != "null" {
			result["max_uses"] = cloneRawMessage(maxUses)
		}
		if len(result) == 0 {
			return nil, nil
		}
		return result, nil
	}

	return dst, nil
}
