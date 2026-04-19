package llmapimux

import "encoding/json"

func cloneRawMessageMap(src map[string]json.RawMessage) map[string]json.RawMessage {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]json.RawMessage, len(src))
	for k, v := range src {
		if v == nil {
			dst[k] = nil
			continue
		}
		buf := make([]byte, len(v))
		copy(buf, v)
		dst[k] = json.RawMessage(buf)
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
