package llmapimux

import "net/http"

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

func (c *openaiResponsesCodec) WriteStreamingResponse(sseWriter *SSEWriter, ch <-chan StreamResult) {
	for result := range ch {
		if result.Err != nil {
			// Cannot change status code at this point — just stop.
			break
		}
		eventType, data, err := EncodeOpenAIResponsesStreamEvent(result.Event)
		if err != nil {
			break
		}
		if err := sseWriter.WriteEvent(eventType, data); err != nil {
			break
		}
	}
	// OpenAI Responses API does NOT use a [DONE] sentinel
}
