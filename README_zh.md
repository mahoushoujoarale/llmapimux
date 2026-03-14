# llmapimux

[English](README.md)

一个 Go SDK，提供 `http.Handler` 实现，用于代理和多路复用 LLM API 请求。在 OpenAI、Anthropic 和 Google Gemini API 之间透明地进行协议转换。

## 支持的协议

| 协议 | 入站（接收） | 出站（发送） |
|---|---|---|
| OpenAI Chat Completions | 支持 | 支持 |
| OpenAI Responses API | 支持 | 支持 |
| Anthropic Messages | 支持 | 支持 |
| Gemini GenerateContent | 支持 | 支持 |

任意入站协议可以路由到任意出站协议 ——  llmapimux 通过统一的中间表示（IR）自动处理协议转换。

## 安装

```bash
go get github.com/llmapimux/llmapimux
```

需要 Go 1.21+。

## 快速开始

```go
package main

import (
	"context"
	"net/http"

	"github.com/llmapimux/llmapimux"
)

// SimpleRouter 将所有请求路由到单个 OpenAI 兼容的后端。
type SimpleRouter struct{}

func (r *SimpleRouter) Route(ctx context.Context, info llmapimux.RouteInfo) (llmapimux.RouteResult, error) {
	return llmapimux.RouteResult{
		Protocol: llmapimux.ProtocolOpenAIChat,
		BaseURL:  "https://api.openai.com",
		APIKey:   "sk-your-api-key",
		Model:    info.Model,
	}, nil
}

func main() {
	mux := llmapimux.NewMux(&SimpleRouter{})

	http.Handle("/v1/chat/completions", mux.OpenAIChatHandler())
	http.Handle("/v1/responses", mux.OpenAIResponsesHandler())
	http.Handle("/v1/messages", mux.AnthropicHandler())
	http.Handle("/v1/models/", mux.GeminiHandler()) // 尾部斜杠用于前缀匹配

	http.ListenAndServe(":8080", nil)
}
```

这会创建一个代理服务器，接受 4 种协议中任意一种的请求，并将其转发到 OpenAI 后端，按需转换协议。

## 架构

```
入站请求
    │
    ▼
┌─────────────────┐
│  协议处理器       │ (OpenAI Chat / Responses / Anthropic / Gemini)
│  解码 → IR       │
└────────┬────────┘
         │
    ┌────▼────┐
    │  路由器   │ (你的路由逻辑)
    └────┬────┘
         │
┌────────▼────────┐
│  出站客户端       │ (OpenAI Chat / Responses / Anthropic / Gemini)
│  IR → 编码       │
└─────────────────┘
         │
         ▼
   目标 LLM API
```

**核心设计：**
- **统一 IR + 2N 适配器** 覆盖 N² 种协议路径（4 种协议 = 8 个适配器覆盖 16 种组合）
- **无同协议直通** — 所有请求都经过 解码 → IR → 编码，确保行为一致
- **RawExtra 侧通道** 为同协议往返保留协议特有字段
- **无重试逻辑** — 错误按原样转发
- **上下文传播** — 客户端断开连接会取消上游请求
- **零外部依赖** — 仅使用 Go 标准库

## 核心概念

### Router（路由器）

`Router` 接口是唯一必需的组件，决定每个请求的去向：

```go
type Router interface {
    Route(ctx context.Context, info RouteInfo) (RouteResult, error)
}
```

`RouteInfo` 提供：`RequestID`、`Model`、`InboundProtocol`、`Stream`、`HasTools`、`HasMedia`、`APIKey`。

`RouteResult` 指定：`Protocol`、`BaseURL`、`APIKey`、`Model`。

### Authenticator（认证器）

可选的入站认证：

```go
type Authenticator interface {
    Authenticate(ctx context.Context, apiKey string) error
}

mux := llmapimux.NewMux(router, llmapimux.WithAuthenticator(myAuth))
```

### StatsReporter（统计报告器）

可选的请求生命周期观测钩子：

```go
type StatsReporter interface {
    OnRequestStart(ctx context.Context, e RequestStartEvent)
    OnFirstByte(ctx context.Context, e FirstByteEvent)
    OnStreamChunk(ctx context.Context, e StreamChunkEvent)
    OnComplete(ctx context.Context, e CompleteEvent)
}

mux := llmapimux.NewMux(router, llmapimux.WithStatsReporter(myReporter))
```

嵌入 `NoopStatsReporter` 可以只实现你需要的方法。

## 协议注意事项

- **Gemini**：模型从 URL 路径中提取（非请求体）；入站处理器必须使用尾部斜杠路径注册（如 `/v1/models/`）以进行前缀匹配
- **Anthropic**：`redacted_thinking` 内容精确往返传递；入站认证同时支持 `x-api-key` 请求头和 `Authorization: Bearer` token（x-api-key 优先）
- **OpenAI Chat**：`system` 和 `developer` 角色都映射为 IR 系统提示；出站使用 `developer` 角色
- **OpenAI Responses**：无状态代理（不支持 `previous_response_id`）；内置工具会被静默丢弃
- 跨协议时，目标协议无法表示的字段会被静默丢弃

## 流式传输

所有协议组合均支持流式和非流式模式。流式响应使用 Server-Sent Events (SSE)。`RouteInfo` 中的 `Stream` 字段允许路由器做出协议感知的决策。

## 测试

```bash
go test ./...                          # 单元测试 + 集成测试
cd tests/e2e && go test ./...          # 使用真实 SDK 客户端的端到端测试
```

端到端测试使用真实 SDK 客户端（`anthropic-sdk-go`、`openai-go/v3`、`google.golang.org/genai`）对本地假服务器进行测试。真实 API 测试需要 `.env` 文件，缺少凭证时会自动跳过。

## 许可证

[MIT](LICENSE)
