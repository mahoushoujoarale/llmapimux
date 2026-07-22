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
	var accumulatedUsage Usage
	var lastStopReason StopReason

	for result := range ch {
		if result.Err != nil {
			// Cannot change status code at this point — just stop.
			break
		}

		// Accumulate usage from early events (e.g. Anthropic message_start
		// carries PromptTokens in StreamEventStart.Response.Usage).
		// OpenAI Responses only emits usage in response.completed, so we
		// must defer it.
		if result.Event != nil {
			if result.Event.Usage != nil {
				mergeStreamUsage(&accumulatedUsage, result.Event.Usage)
			}
			if result.Event.Response != nil && result.Event.Response.Usage.PromptTokens != 0 {
				mergeStreamUsage(&accumulatedUsage, &result.Event.Response.Usage)
			}
			// Capture stop reasons from delta events (e.g. Anthropic message_delta).
			if result.Event.StopReason != nil {
				lastStopReason = *result.Event.StopReason
			}
		}

		// On stop event, inject accumulated usage and stop reason.
		if result.Event != nil && result.Event.Type == StreamEventStop {
			if result.Event.Usage == nil && accumulatedUsage.PromptTokens != 0 {
				result.Event.Usage = &Usage{}
				*result.Event.Usage = accumulatedUsage
			} else if result.Event.Usage != nil && result.Event.Usage.PromptTokens == 0 && accumulatedUsage.PromptTokens != 0 {
				mergeStreamUsage(result.Event.Usage, &accumulatedUsage)
			}
			if result.Event.StopReason == nil && lastStopReason != "" {
				result.Event.StopReason = &lastStopReason
			}
		}

		eventType, data, err := EncodeOpenAIResponsesStreamEvent(result.Event)
		if err != nil {
			// Some IR events (e.g. usage-only deltas) cannot be encoded into
			// the OpenAI Responses streaming format. Skip them silently — their
			// data has already been accumulated above.
			continue
		}
		if err := sseWriter.WriteEvent(eventType, data); err != nil {
			break
		}
	}
	// OpenAI Responses API does NOT use a [DONE] sentinel
}
