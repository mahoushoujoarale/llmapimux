package llmapimux

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/mahoushoujoarale/llmapimux/protocol/openairesponses"
)

// openaiResponsesCodec implements inboundCodec for the OpenAI Responses protocol.
type openaiResponsesCodec struct{}

func (c *openaiResponsesCodec) Protocol() Protocol {
	return ProtocolOpenAIResponses
}

func (c *openaiResponsesCodec) KnownFields() map[string]bool {
	return openaiResponsesKnownFields
}

func (c *openaiResponsesCodec) ExtractAPIKey(r *http.Request) string {
	return parseBearerAPIKey(r)
}

func (c *openaiResponsesCodec) DecodeRequest(r *http.Request, body []byte) (*Request, error) {
	return decodeRequestWithInboundProtocol(body, ProtocolOpenAIResponses, DecodeOpenAIResponsesRequest)
}

func (c *openaiResponsesCodec) WriteError(w http.ResponseWriter, statusCode int, msg string) {
	writeOpenAIError(w, statusCode, msg)
}

func (c *openaiResponsesCodec) EncodeResponse(resp *Response) ([]byte, error) {
	return EncodeOpenAIResponsesResponse(resp)
}

// responsesStreamState normalizes IR streaming events into a complete OpenAI
// Responses SSE lifecycle. The IR deliberately does not carry protocol-specific
// item IDs, content indexes, or sequence numbers, so those values are owned by
// one outbound stream session.
type responsesStreamState struct {
	writer     *SSEWriter
	responseID string
	model      string
	sequence   int
	started    bool
	usage      Usage
	stopReason StopReason
	nextOutput int
	items      map[string]*responsesStreamItem
}

type responsesStreamItem struct {
	key          string
	sourceIndex  int
	outputIndex  int
	itemID       string
	kind         ContentType
	contentIndex int
	item         openairesponses.OutputItem
	text         strings.Builder
	announced    bool
	closed       bool
}

func newResponsesStreamState(writer *SSEWriter) *responsesStreamState {
	return &responsesStreamState{writer: writer, items: make(map[string]*responsesStreamItem)}
}

func (s *responsesStreamState) write(eventType string, payload *openairesponses.StreamEvent) error {
	s.sequence++
	payload.Type = eventType
	payload.SequenceNumber = intPtr(s.sequence)
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return s.writer.WriteEvent(eventType, data)
}

func (s *responsesStreamState) start(response *Response) error {
	if s.started {
		return nil
	}
	if response != nil {
		s.responseID = response.ID
		s.model = response.Model
		mergeStreamUsage(&s.usage, &response.Usage)
	}
	if s.responseID == "" {
		s.responseID = newResponsesStreamID("resp")
	}
	s.started = true
	created := &openairesponses.Response{ID: s.responseID, Object: "response", Model: s.model, Status: "in_progress"}
	if err := s.write("response.created", &openairesponses.StreamEvent{Response: created}); err != nil {
		return err
	}
	return s.write("response.in_progress", &openairesponses.StreamEvent{
		Response: &openairesponses.Response{ID: s.responseID, Object: "response", Model: s.model, Status: "in_progress"},
	})
}

func (s *responsesStreamState) ensureStarted() error {
	return s.start(nil)
}

func responsesItemKey(index int, kind ContentType) string {
	return fmt.Sprintf("%d:%s", index, kind)
}

func (s *responsesStreamState) ensureItem(index int, part *ContentPart) (*responsesStreamItem, error) {
	if part == nil {
		return nil, nil
	}
	key := responsesItemKey(index, part.Type)
	if item := s.items[key]; item != nil {
		return item, nil
	}

	item := &responsesStreamItem{
		key:          key,
		sourceIndex:  index,
		outputIndex:  s.nextOutput,
		itemID:       newResponsesStreamID("item"),
		kind:         part.Type,
		contentIndex: 0,
	}
	s.nextOutput++

	switch part.Type {
	case ContentTypeToolUse:
		item.item = openairesponses.OutputItem{Type: "function_call", ID: item.itemID, Status: "in_progress"}
		if part.ToolUse != nil {
			item.item.CallID = part.ToolUse.ID
			item.item.Name = part.ToolUse.Name
		}
	case ContentTypeThinking:
		item.item = openairesponses.OutputItem{Type: "reasoning", ID: item.itemID, Status: "in_progress"}
	case ContentTypeRefusal, ContentTypeText:
		item.item = openairesponses.OutputItem{Type: "message", ID: item.itemID, Role: "assistant", Status: "in_progress", Content: []openairesponses.OutputContent{}}
	default:
		return nil, nil
	}

	s.items[key] = item
	item.announced = true
	outputIndex := item.outputIndex
	if err := s.write("response.output_item.added", &openairesponses.StreamEvent{OutputIndex: &outputIndex, Item: &item.item}); err != nil {
		return nil, err
	}

	if item.kind == ContentTypeText || item.kind == ContentTypeRefusal {
		contentIndex := item.contentIndex
		partType := "output_text"
		if item.kind == ContentTypeRefusal {
			partType = "refusal"
		}
		if err := s.write("response.content_part.added", &openairesponses.StreamEvent{
			OutputIndex: &outputIndex, ItemID: item.itemID, ContentIndex: &contentIndex,
			Part: &openairesponses.OutputContent{Type: partType, Annotations: []json.RawMessage{}},
		}); err != nil {
			return nil, err
		}
	}
	return item, nil
}

func (s *responsesStreamState) writeDelta(index int, part *ContentPart) error {
	if part == nil {
		return nil
	}
	item, err := s.ensureItem(index, part)
	if err != nil || item == nil {
		return err
	}

	outputIndex, contentIndex := item.outputIndex, item.contentIndex
	switch part.Type {
	case ContentTypeText:
		if part.Text == nil || part.Text.Text == "" {
			return nil
		}
		item.text.WriteString(part.Text.Text)
		return s.write("response.output_text.delta", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, ContentIndex: &contentIndex, Delta: part.Text.Text})
	case ContentTypeRefusal:
		if part.Refusal == nil || part.Refusal.Refusal == "" {
			return nil
		}
		item.text.WriteString(part.Refusal.Refusal)
		return s.write("response.refusal.delta", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, ContentIndex: &contentIndex, Delta: part.Refusal.Refusal})
	case ContentTypeThinking:
		if part.Thinking == nil || part.Thinking.Thinking == "" {
			return nil
		}
		item.text.WriteString(part.Thinking.Thinking)
		return s.write("response.reasoning_summary_text.delta", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, ContentIndex: &contentIndex, Delta: part.Thinking.Thinking})
	case ContentTypeToolUse:
		if part.ToolUse == nil {
			return nil
		}
		if part.ToolUse.Name != "" {
			item.item.Name = part.ToolUse.Name
		}
		if part.ToolUse.ID != "" {
			item.item.CallID = part.ToolUse.ID
		}
		arguments := string(part.ToolUse.Arguments)
		if arguments == "" {
			return nil
		}
		item.text.WriteString(arguments)
		return s.write("response.function_call_arguments.delta", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, Delta: arguments})
	}
	return nil
}

func (s *responsesStreamState) closeItem(item *responsesStreamItem) error {
	if item == nil || item.closed {
		return nil
	}
	outputIndex, contentIndex := item.outputIndex, item.contentIndex
	text := item.text.String()

	switch item.kind {
	case ContentTypeText:
		if err := s.write("response.output_text.done", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, ContentIndex: &contentIndex, Text: text}); err != nil {
			return err
		}
		item.item.Content = []openairesponses.OutputContent{{Type: "output_text", Text: text, Annotations: []json.RawMessage{}}}
		if err := s.write("response.content_part.done", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, ContentIndex: &contentIndex, Part: &item.item.Content[0]}); err != nil {
			return err
		}
	case ContentTypeRefusal:
		if err := s.write("response.refusal.done", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, ContentIndex: &contentIndex, Text: text}); err != nil {
			return err
		}
		item.item.Content = []openairesponses.OutputContent{{Type: "refusal", Refusal: text}}
		if err := s.write("response.content_part.done", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, ContentIndex: &contentIndex, Part: &item.item.Content[0]}); err != nil {
			return err
		}
	case ContentTypeThinking:
		item.item.Summary = []openairesponses.ReasoningSummary{{Type: "summary_text", Text: text}}
	case ContentTypeToolUse:
		item.item.Arguments = text
		if err := s.write("response.function_call_arguments.done", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, Delta: text}); err != nil {
			return err
		}
	}

	item.item.Status = "completed"
	if err := s.write("response.output_item.done", &openairesponses.StreamEvent{OutputIndex: &outputIndex, ItemID: item.itemID, Item: &item.item}); err != nil {
		return err
	}
	item.closed = true
	return nil
}

func (s *responsesStreamState) closeIndex(index int) error {
	for _, item := range s.items {
		if item.sourceIndex == index {
			return s.closeItem(item)
		}
	}
	return nil
}

func (s *responsesStreamState) complete() error {
	if err := s.ensureStarted(); err != nil {
		return err
	}
	items := make([]*responsesStreamItem, 0, len(s.items))
	for _, item := range s.items {
		items = append(items, item)
	}
	sort.Slice(items, func(i, j int) bool { return items[i].outputIndex < items[j].outputIndex })
	output := make([]openairesponses.OutputItem, 0, len(items))
	for _, item := range items {
		if err := s.closeItem(item); err != nil {
			return err
		}
		output = append(output, item.item)
	}

	status := "completed"
	switch s.stopReason {
	case StopReasonMaxTokens, StopReasonPauseTurn:
		status = "incomplete"
	case StopReasonContentFilter:
		status = "failed"
	}
	return s.write("response.completed", &openairesponses.StreamEvent{Response: &openairesponses.Response{
		ID: s.responseID, Object: "response", Model: s.model, Status: status, Output: output, Usage: encodeOaiRespUsage(&s.usage),
	}})
}

func newResponsesStreamID(prefix string) string {
	var random [8]byte
	if _, err := rand.Read(random[:]); err == nil {
		return prefix + "_" + hex.EncodeToString(random[:])
	}
	return fmt.Sprintf("%s_fallback", prefix)
}

func (c *openaiResponsesCodec) WriteStreamingResponse(sseWriter *SSEWriter, ch <-chan StreamResult) {
	state := newResponsesStreamState(sseWriter)
	for result := range ch {
		if result.Err != nil {
			writeOpenAIResponsesStreamError(sseWriter, "server_error", "", result.Err.Error())
			return
		}
		event := result.Event
		if event == nil {
			continue
		}
		if event.Type == StreamEventStart {
			if err := state.start(event.Response); err != nil {
				return
			}
			continue
		}
		if err := state.ensureStarted(); err != nil {
			return
		}
		if event.Usage != nil {
			mergeStreamUsage(&state.usage, event.Usage)
		}
		if event.StopReason != nil {
			state.stopReason = *event.StopReason
		}
		if event.Type == StreamEventError {
			message, errType, code := "upstream stream error", "api_error", ""
			if event.Error != nil {
				if event.Error.Message != "" {
					message = event.Error.Message
				}
				if event.Error.Type != "" {
					errType = event.Error.Type
				}
				code = event.Error.Code
			}
			writeOpenAIResponsesStreamError(sseWriter, errType, code, message)
			return
		}
		switch event.Type {
		case StreamEventContentBlockStart:
			if _, err := state.ensureItem(event.Index, event.Delta); err != nil {
				return
			}
		case StreamEventDelta:
			if err := state.writeDelta(event.Index, event.Delta); err != nil {
				return
			}
		case StreamEventContentBlockStop:
			if err := state.closeIndex(event.Index); err != nil {
				return
			}
		case StreamEventStop:
			if err := state.complete(); err != nil {
				return
			}
			return
		}
	}
}

// writeOpenAIResponsesStreamError emits an in-band Responses `error` SSE event.
// Used when a stream fails after the HTTP 200 has already been committed.
func writeOpenAIResponsesStreamError(sseWriter *SSEWriter, errType, code, message string) {
	payload := map[string]any{
		"type":    "error",
		"message": message,
	}
	if code == "" {
		code = errType
	}
	if code != "" {
		payload["code"] = code
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	sseWriter.WriteEvent("error", data) //nolint:errcheck
}
