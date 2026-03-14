package llmapimux

import (
	"context"
	"testing"
)

func TestNewMuxHandlers(t *testing.T) {
	router := &staticRouter{result: RouteResult{
		Protocol: ProtocolOpenAIChat,
		BaseURL:  "https://api.openai.com",
		APIKey:   "sk-test",
		Model:    "gpt-4o",
	}}
	mux := NewMux(router)

	if mux.OpenAIChatHandler() == nil {
		t.Error("OpenAIChatHandler returned nil")
	}
	if mux.OpenAIResponsesHandler() == nil {
		t.Error("OpenAIResponsesHandler returned nil")
	}
	if mux.AnthropicHandler() == nil {
		t.Error("AnthropicHandler returned nil")
	}
	if mux.GeminiHandler() == nil {
		t.Error("GeminiHandler returned nil")
	}
}

func TestNewMux_DefaultStatsReporter(t *testing.T) {
	router := &staticRouter{}
	mux := NewMux(router)

	h, ok := mux.OpenAIChatHandler().(*Handler)
	if !ok {
		t.Fatal("expected OpenAIChatHandler to return *Handler")
	}
	if h.stats == nil {
		t.Fatal("expected non-nil handler stats reporter")
	}
	if _, ok := h.stats.(NoopStatsReporter); !ok {
		t.Fatalf("expected NoopStatsReporter, got %T", h.stats)
	}
}

func TestWithStatsReporterNil_UsesNoop(t *testing.T) {
	router := &staticRouter{}
	mux := NewMux(router, WithStatsReporter(nil))

	h, ok := mux.OpenAIChatHandler().(*Handler)
	if !ok {
		t.Fatal("expected OpenAIChatHandler to return *Handler")
	}
	if h.stats == nil {
		t.Fatal("expected non-nil handler stats reporter")
	}
	if _, ok := h.stats.(NoopStatsReporter); !ok {
		t.Fatalf("expected NoopStatsReporter, got %T", h.stats)
	}
}

type typedNilStatsReporter struct{}

func (*typedNilStatsReporter) OnRequestStart(context.Context, RequestStartEvent)  {}
func (*typedNilStatsReporter) OnFirstByte(context.Context, FirstByteEvent)        {}
func (*typedNilStatsReporter) OnStreamChunk(context.Context, StreamChunkEvent)    {}
func (*typedNilStatsReporter) OnComplete(context.Context, CompleteEvent)          {}
func (*typedNilStatsReporter) OnAttemptError(context.Context, AttemptErrorEvent)  {}

func TestWithStatsReporterTypedNil_UsesNoop(t *testing.T) {
	router := &staticRouter{}
	var typedNil *typedNilStatsReporter
	mux := NewMux(router, WithStatsReporter(typedNil))

	h, ok := mux.OpenAIChatHandler().(*Handler)
	if !ok {
		t.Fatal("expected OpenAIChatHandler to return *Handler")
	}
	if h.stats == nil {
		t.Fatal("expected non-nil handler stats reporter")
	}
	if _, ok := h.stats.(NoopStatsReporter); !ok {
		t.Fatalf("expected NoopStatsReporter, got %T", h.stats)
	}
}
