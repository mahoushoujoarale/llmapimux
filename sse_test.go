package llmapimux

import (
	"bytes"
	"strings"
	"testing"
)

func TestReaderBasic(t *testing.T) {
	input := "data: {\"text\":\"hello\"}\n\ndata: {\"text\":\"world\"}\n\n"
	r := NewSSEReader(strings.NewReader(input))

	event, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(event) != `{"text":"hello"}` {
		t.Fatalf("unexpected event: %s", event)
	}

	event, err = r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(event) != `{"text":"world"}` {
		t.Fatalf("unexpected event: %s", event)
	}
}

func TestReaderDoneSignal(t *testing.T) {
	input := "data: {\"text\":\"hello\"}\n\ndata: [DONE]\n\n"
	r := NewSSEReader(strings.NewReader(input))

	_, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}

	event, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(event) != "[DONE]" {
		t.Fatalf("expected [DONE], got: %s", event)
	}
}

func TestReaderMultiLineData(t *testing.T) {
	// Some SSE implementations split data across multiple data: lines
	input := "data: {\"text\":\ndata: \"hello\"}\n\n"
	r := NewSSEReader(strings.NewReader(input))

	event, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(event) != "{\"text\":\n\"hello\"}" {
		t.Fatalf("unexpected event: %q", event)
	}
}

func TestReaderLargeEvent(t *testing.T) {
	payload := strings.Repeat("x", 70*1024)
	input := "data: " + payload + "\n\n"
	r := NewSSEReader(strings.NewReader(input))

	event, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(event) != payload {
		t.Fatalf("len(event) = %d, want %d", len(event), len(payload))
	}
}

func TestReaderEventField(t *testing.T) {
	// Anthropic uses "event:" lines
	input := "event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
	r := NewSSEReader(strings.NewReader(input))

	event, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(event) != `{"type":"message_start"}` {
		t.Fatalf("unexpected: %s", event)
	}
	if r.LastEventType() != "message_start" {
		t.Fatalf("unexpected event type: %s", r.LastEventType())
	}
}

func TestWriterBasic(t *testing.T) {
	var buf bytes.Buffer
	w := NewSSEWriter(&buf)

	w.WriteData([]byte(`{"text":"hello"}`))
	w.WriteData([]byte(`{"text":"world"}`))

	expected := "data: {\"text\":\"hello\"}\n\ndata: {\"text\":\"world\"}\n\n"
	if buf.String() != expected {
		t.Fatalf("unexpected output: %q", buf.String())
	}
}

func TestWriterWithEvent(t *testing.T) {
	var buf bytes.Buffer
	w := NewSSEWriter(&buf)

	w.WriteEvent("message_start", []byte(`{"type":"message_start"}`))

	expected := "event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
	if buf.String() != expected {
		t.Fatalf("unexpected output: %q", buf.String())
	}
}

func TestWriterDone(t *testing.T) {
	var buf bytes.Buffer
	w := NewSSEWriter(&buf)
	w.WriteDone()
	if buf.String() != "data: [DONE]\n\n" {
		t.Fatalf("unexpected: %q", buf.String())
	}
}
