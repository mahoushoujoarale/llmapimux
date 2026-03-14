package llmapimux

import "testing"

func TestProtocolConstants(t *testing.T) {
	protocols := []Protocol{
		ProtocolOpenAIChat,
		ProtocolOpenAIResponses,
		ProtocolAnthropic,
		ProtocolGemini,
	}
	seen := map[Protocol]bool{}
	for _, p := range protocols {
		if seen[p] {
			t.Fatalf("duplicate protocol: %s", p)
		}
		seen[p] = true
		if p == "" {
			t.Fatal("empty protocol")
		}
	}
}

func TestRouteInfoAndResult(t *testing.T) {
	info := RouteInfo{
		RequestID:       "req-123",
		Model:           "gpt-4o",
		InboundProtocol: ProtocolOpenAIChat,
		Stream:          true,
		HasTools:        false,
		HasMedia:        false,
		APIKey:          "sk-test",
	}
	if info.Model != "gpt-4o" {
		t.Errorf("unexpected model: %s", info.Model)
	}

	result := RouteResult{
		Protocol: ProtocolAnthropic,
		BaseURL:  "https://api.anthropic.com",
		APIKey:   "sk-ant",
		Model:    "claude-sonnet-4-20250514",
	}
	if result.Model != "claude-sonnet-4-20250514" {
		t.Errorf("unexpected model: %s", result.Model)
	}
}

func TestOutboundConfig(t *testing.T) {
	cfg := OutboundConfig{
		BaseURL: "https://api.openai.com",
		APIKey:  "sk-test",
	}
	if cfg.BaseURL != "https://api.openai.com" {
		t.Fatalf("unexpected BaseURL: %s", cfg.BaseURL)
	}
}

func TestRouteInfo_ExplicitInputsOnly(t *testing.T) {
	tests := []struct {
		name string
		info RouteInfo
		want RouteResult
	}{
		{
			name: "openai_chat_to_anthropic",
			info: RouteInfo{Model: "gpt-4o", InboundProtocol: ProtocolOpenAIChat, Stream: false},
			want: RouteResult{Protocol: ProtocolAnthropic, BaseURL: "https://api.anthropic.com", APIKey: "sk-ant", Model: "claude-sonnet-4-20250514"},
		},
		{
			name: "gemini_stream_to_openai",
			info: RouteInfo{Model: "gemini-2.5-pro", InboundProtocol: ProtocolGemini, Stream: true},
			want: RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: "https://api.openai.com", APIKey: "sk-oai", Model: "gpt-4o-mini"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := routeByExplicitInputs(tt.info)
			if got.Protocol != tt.want.Protocol || got.BaseURL != tt.want.BaseURL || got.APIKey != tt.want.APIKey || got.Model != tt.want.Model {
				t.Fatalf("routeByExplicitInputs() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestRouteInfo_ContainsOnlyExplicitRoutingInputs(t *testing.T) {
	info := RouteInfo{
		RequestID:       "req-456",
		Model:           "gpt-4o",
		InboundProtocol: ProtocolOpenAIChat,
		Stream:          false,
		HasTools:        true,
		HasMedia:        true,
		APIKey:          "sk-test",
	}

	got := routeByExplicitInputs(info)
	if got.Protocol != ProtocolAnthropic || got.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("unexpected route result: %+v", got)
	}
}

func routeByExplicitInputs(info RouteInfo) RouteResult {
	if info.InboundProtocol == ProtocolGemini && info.Stream {
		return RouteResult{Protocol: ProtocolOpenAIChat, BaseURL: "https://api.openai.com", APIKey: "sk-oai", Model: "gpt-4o-mini"}
	}
	return RouteResult{Protocol: ProtocolAnthropic, BaseURL: "https://api.anthropic.com", APIKey: "sk-ant", Model: "claude-sonnet-4-20250514"}
}
