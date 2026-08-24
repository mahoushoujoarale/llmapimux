package llmapimux

import (
	"net/http"
	"net/url"
	"testing"
	"time"
)

func TestHTTPClientForProxyClonesInjectedTransport(t *testing.T) {
	injected := &http.Client{
		Timeout: 15 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns: 42,
		},
	}
	const proxyURL = "http://proxy.example:8080"

	got := httpClientForProxy(injected, proxyURL)
	if got == injected {
		t.Fatal("expected a derived client, got the same instance")
	}
	if got.Timeout != injected.Timeout {
		t.Fatalf("expected Timeout %v to be preserved, got %v", injected.Timeout, got.Timeout)
	}
	derived, ok := got.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected *http.Transport, got %T", got.Transport)
	}
	if derived.MaxIdleConns != 42 {
		t.Fatalf("expected MaxIdleConns 42 to be preserved, got %d", derived.MaxIdleConns)
	}
	if derived.Proxy == nil {
		t.Fatal("expected Proxy to be set on derived transport")
	}
	req := &http.Request{URL: &url.URL{Scheme: "https", Host: "api.example.com"}}
	target, err := derived.Proxy(req)
	if err != nil {
		t.Fatalf("Proxy func error: %v", err)
	}
	if target == nil || target.String() != proxyURL {
		t.Fatalf("expected proxy %q, got %v", proxyURL, target)
	}

	// The injected transport itself must not be mutated.
	orig := injected.Transport.(*http.Transport)
	if orig.Proxy != nil {
		t.Fatal("injected transport Proxy should not be mutated")
	}

	// Cached per (proxyURL, base transport): same key returns the same client.
	if again := httpClientForProxy(injected, proxyURL); again != got {
		t.Fatal("expected cached client to be reused for the same (proxyURL, transport)")
	}
	// A different base transport must get a different derived client.
	other := &http.Client{Transport: &http.Transport{}}
	if again := httpClientForProxy(other, proxyURL); again == got {
		t.Fatal("expected distinct cache entry for a different base transport")
	}
}

func TestHTTPClientForProxyDefaultHasKeepalive(t *testing.T) {
	// No base client: the default proxy transport must enable TCP keepalive.
	got := httpClientForProxy(nil, "http://proxy.example:8080")
	tr, ok := got.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected *http.Transport, got %T", got.Transport)
	}
	if tr.DialContext == nil {
		t.Fatal("expected DialContext with keepalive dialer")
	}
	if tr.Proxy == nil {
		t.Fatal("expected Proxy to be set")
	}
}

func TestHTTPClientForProxyEmptyURLReturnsBase(t *testing.T) {
	base := &http.Client{}
	if got := httpClientForProxy(base, ""); got != base {
		t.Fatal("expected base client for empty proxy URL")
	}
	if got := httpClientForProxy(nil, ""); got != http.DefaultClient {
		t.Fatal("expected DefaultClient for empty proxy URL and nil base")
	}
}
