package llmapimux

import "strings"

// ModelMaxOutputTokens returns the default max output tokens for a given model name.
// Returns 0 if the model is unknown. Used when encoding to protocols that require
// max_tokens (e.g., Anthropic) and the inbound request did not specify one.
func ModelMaxOutputTokens(model string) int {
	for prefix, val := range modelMaxTokens {
		if strings.HasPrefix(strings.ToLower(model), prefix) {
			return val
		}
	}
	return 0
}

// modelMaxTokens maps model name prefixes to their maximum output tokens.
// Based on Anthropic model documentation as of 2026-07.
// Prefix matching handles date-suffixed names like "claude-sonnet-4-20250514".
var modelMaxTokens = map[string]int{
	"claude-4-opus":    16384,
	"claude-opus-4":    16384,
	"claude-4-sonnet":  16384,
	"claude-sonnet-4":  16384,
	"claude-3-5-sonnet": 8192,
	"claude-3-5-haiku": 8192,
	"claude-3-opus":    4096,
	"claude-sonnet-3":  4096,
	"claude-3-haiku":   4096,
	"claude-haiku-3":   4096,
}

// FallbackMaxTokens is used when the model is unknown and max_tokens is not specified.
const FallbackMaxTokens = 16384
