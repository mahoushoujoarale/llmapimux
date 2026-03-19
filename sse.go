package llmapimux

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// SSEReader reads Server-Sent Events from an io.Reader.
type SSEReader struct {
	reader        *bufio.Reader
	lastEventType string
}

// NewSSEReader creates a new SSEReader wrapping the given io.Reader.
func NewSSEReader(r io.Reader) *SSEReader {
	return &SSEReader{
		reader: bufio.NewReader(r),
	}
}

// Read reads lines until an empty line delimiter, concatenating all data: fields.
// It returns the assembled data payload. Returns io.EOF when the stream ends.
func (r *SSEReader) Read() ([]byte, error) {
	var dataParts [][]byte
	r.lastEventType = ""

	for {
		line, err := r.readLine()
		if err != nil {
			if err == io.EOF {
				if len(dataParts) > 0 {
					return bytes.Join(dataParts, []byte("\n")), nil
				}
				return nil, io.EOF
			}
			return nil, err
		}

		// Empty line signals end of event
		if line == "" {
			if len(dataParts) > 0 {
				return bytes.Join(dataParts, []byte("\n")), nil
			}
			// Empty event; continue reading for the next one
			continue
		}

		if strings.HasPrefix(line, "data:") {
			payload := line[len("data:"):]
			if len(payload) > 0 && payload[0] == ' ' {
				payload = payload[1:]
			}
			dataParts = append(dataParts, []byte(payload))
			continue
		}

		if strings.HasPrefix(line, "event:") {
			value := line[len("event:"):]
			if len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}
			r.lastEventType = value
			continue
		}

		// Other fields (id:, retry:, comments) are ignored
	}
}

func (r *SSEReader) readLine() (string, error) {
	line, err := r.reader.ReadString('\n')
	if err != nil {
		if err != io.EOF {
			return "", err
		}
		if len(line) == 0 {
			return "", io.EOF
		}
	}

	line = strings.TrimSuffix(line, "\n")
	line = strings.TrimSuffix(line, "\r")
	return line, err
}

// LastEventType returns the event: field value from the most recently read event.
func (r *SSEReader) LastEventType() string {
	return r.lastEventType
}

// SSEWriter writes Server-Sent Events to an io.Writer.
type SSEWriter struct {
	w io.Writer
	f http.Flusher // cached at construction time; nil if w does not implement http.Flusher
}

// NewSSEWriter creates a new SSEWriter wrapping the given io.Writer.
func NewSSEWriter(w io.Writer) *SSEWriter {
	f, _ := w.(http.Flusher)
	return &SSEWriter{w: w, f: f}
}

// WriteData writes a data-only SSE event and flushes if possible.
func (w *SSEWriter) WriteData(data []byte) error {
	_, err := fmt.Fprintf(w.w, "data: %s\n\n", data)
	if err != nil {
		return err
	}
	w.flush()
	return nil
}

// WriteEvent writes an SSE event with both event: and data: fields, then flushes if possible.
func (w *SSEWriter) WriteEvent(event string, data []byte) error {
	_, err := fmt.Fprintf(w.w, "event: %s\ndata: %s\n\n", event, data)
	if err != nil {
		return err
	}
	w.flush()
	return nil
}

// WriteDone writes the [DONE] sentinel event and flushes if possible.
func (w *SSEWriter) WriteDone() error {
	_, err := fmt.Fprintf(w.w, "data: [DONE]\n\n")
	if err != nil {
		return err
	}
	w.flush()
	return nil
}

// flush flushes the underlying writer if it implements http.Flusher.
func (w *SSEWriter) flush() {
	if w.f != nil {
		w.f.Flush()
	}
}
