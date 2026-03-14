package openairesponses

// StreamEvent is the JSON payload for a Responses API streaming event.
type StreamEvent struct {
	// Required by the SDK to discriminate event types.
	Type string `json:"type,omitempty"`

	// response.created / response.completed
	Response *Response `json:"response,omitempty"`

	// response.output_item.added / response.output_item.done
	OutputIndex *int        `json:"output_index,omitempty"`
	Item        *OutputItem `json:"item,omitempty"`

	// response.content_part.added / response.content_part.done
	ContentIndex *int           `json:"content_index,omitempty"`
	Part         *OutputContent `json:"part,omitempty"`

	// response.output_text.delta / response.function_call_arguments.delta
	Delta  string `json:"delta,omitempty"`
	ItemID string `json:"item_id,omitempty"`
}
