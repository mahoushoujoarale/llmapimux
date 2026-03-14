package llmapimux

import "net/http"

// inboundCodec encapsulates protocol-specific behavior for the unified Handler.
// DecodeRequest must set Request.InboundProtocol and keep RawExtra unset.
type inboundCodec interface {
	Protocol() Protocol
	KnownFields() map[string]bool
	ExtractAPIKey(r *http.Request) string
	DecodeRequest(r *http.Request, body []byte) (*Request, error)
	WriteError(w http.ResponseWriter, statusCode int, msg string)
	EncodeResponse(resp *Response) ([]byte, error)
	WriteStreamingResponse(sseWriter *SSEWriter, ch <-chan StreamResult)
}
