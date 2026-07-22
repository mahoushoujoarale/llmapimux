# llmapimux 协议转换缺陷修复方案

> **文档版本**: v2.0（无向后兼容约束）  
> **日期**: 2026-07-22  
> **预估总工作量**: 5-7 人天  

---

## 目录

1. [概述](#1-概述)
2. [缺陷总览](#2-缺陷总览)
3. [IR Usage 重新设计](#3-ir-usage-重新设计)
4. [max_tokens 修复](#4-max_tokens-修复)
5. [StopReason 映射修复](#5-stopreason-映射修复)
6. [Gemini tool_use 推断修复](#6-gemini-tool_use-推断修复)
7. [Gemini citation 分配修复](#7-gemini-citation-分配修复)
8. [Refusal 弱映射修复](#8-refusal-弱映射修复)
9. [Stats 新增指标](#9-stats-新增指标)
10. [不改项说明](#10-不改项说明)
11. [修改文件清单](#11-修改文件清单)

---

## 1. 概述

llmapimux 在 4 种 LLM 协议间做代理转换，存在若干"协议不对等"缺陷。既然**不需要向后兼容**，我们可以直接做干净的 IR 重新设计，而非在旧结构上打补丁。

**核心思路**：

1. **IR Usage 彻底拆分** — 不再用一个 `InputTokens` 试图兼容 4 种协议的不同语义，而是让每个维度都有自己的字段
2. **消除硬编码默认值** — 用模型感知逻辑替代
3. **纠正错误映射** — `pause_turn` → `length`（而非 `stop`），Gemini 推断逻辑加条件
4. **补全丢失信息** — 新增 TPOT 指标、refusal 原始类型保留

---

## 2. 缺陷总览

| # | 严重度 | 缺陷 | 本文档章节 |
|---|--------|------|-----------|
| 1 | 🔴 | `input_tokens` ≠ `prompt_tokens` 语义不等价（OpenAI 含 cache，Anthropic 不含） | §3 |
| 2 | 🔴 | Anthropic `ServerToolUseTokens` 丢失 | §3 |
| 3 | 🔴 | OpenAI `audio_tokens` / `prediction_tokens` 丢失 | §3 |
| 4 | 🔴 | `prompt_tokens` 含 `cached_tokens` 导致 OpenAI→Anthropic 双计数 | §3 |
| 5 | 🔴 | `max_tokens` 默认硬编码 4096 | §4 |
| 6 | 🔴 | Gemini tool_use 停止原因靠推断覆盖 finishReason | §6 |
| 7 | 🔴 | Gemini 无 tool 角色，RoleTool 伪装为 "user" | §10(不改) |
| 8 | 🟠 | `pause_turn` → `stop` 降级（应为 `length`） | §5 |
| 9 | 🟠 | Refusal → text 弱映射（Anthropic 无原生 refusal） | §8 |
| 10 | 🟠 | Gemini citation 只挂到第一个 text 块 | §7 |
| 11 | 🟠 | Phase 2 Thinking 字段（IncludeThoughts/Level）跨协议丢弃 | §10(不改) |
| 12 | 🟢 | Gemini 合成 ID 与原始 ID 不对应 | §10(不改) |
| 13 | 🟢 | Anthropic 多工具白名单丢弃 | §10(不改) |
| 14 | 🟢 | Gemini AllowParallelCalls 丢弃 | §10(不改) |
| 15 | 🟢 | 流式生命周期/错误事件丢弃 | §10(不改) |

---

## 3. IR Usage 重新设计

### 3.1 问题

当前 `Usage` 结构用一个 `InputTokens` 字段映射 4 种协议的"输入 token"，但它们的语义根本不同：

| 协议 | 字段 | 含义 |
|------|------|------|
| Anthropic `input_tokens` | 文本 + cache creation token（**不含** cache read） | `input_tokens` = total_input - cache_read |
| Anthropic `cache_creation_input_tokens` | 写入 prompt cache 的 token | 独立字段 |
| Anthropic `cache_read_input_tokens` | 命中 prompt cache 的 token | 独立字段 |
| OpenAI Chat `prompt_tokens` | 所有输入 token | **含** `cached_tokens` |
| OpenAI Chat `prompt_tokens_details.cached_tokens` | 其中 cache 命中的部分 | 子字段 |
| OpenAI Chat `completion_tokens_details.audio_tokens` | 音频输出 token | 丢失！ |
| Gemini `promptTokenCount` | 所有输入 token | 含义浮动 |

**导致的坑**：
- OpenAI → Anthropic：`input_tokens = prompt_tokens`（偏大，因为 prompt_tokens 含 cache 但 Anthropic 的 input_tokens 是三项互斥之一）
- Anthropic → OpenAI：`prompt_tokens = input_tokens`（偏小，因为 Anthropic 的 input_tokens 只含 breakpoint 之后的 token，不含 cache_creation 和 cache_read）
- `ServerToolUseTokens` / `audio_tokens` / `prediction_tokens` 完全丢失

### 3.2 方案：拆分为原子维度

**不再用 `InputTokens` 做模糊映射**，而是让每个原子维度有自己明确的字段：

```go
// Usage tracks token consumption for a request/response pair.
type Usage struct {
    // === 输入侧 ===
    PromptTokens          int `json:"prompt_tokens,omitempty"`           // 总输入 token（含 cache）
    PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens,omitempty"` // cache 命中 token（Anthropic: cache_read_input_tokens, OpenAI: cached_tokens）
    PromptCacheMissTokens int `json:"prompt_cache_miss_tokens,omitempty"` // cache 未命中 = PromptTokens - PromptCacheHitTokens
    PromptAudioTokens     int `json:"prompt_audio_tokens,omitempty"`      // 输入音频 token（OpenAI audio_tokens）
    PromptCacheWriteTokens int `json:"prompt_cache_write_tokens,omitempty"` // cache 写入 token（Anthropic: cache_creation_input_tokens）

    // === 输出侧 ===
    CompletionTokens              int `json:"completion_tokens,omitempty"`               // 总输出 token
    CompletionReasoningTokens     int `json:"completion_reasoning_tokens,omitempty"`     // 推理/思考 token（OpenAI: reasoning_tokens, Anthropic: 无, Gemini: thoughtsTokenCount）
    CompletionAudioTokens         int `json:"completion_audio_tokens,omitempty"`         // 输出音频 token
    CompletionAcceptedPrediction  int `json:"completion_accepted_prediction,omitempty"` // 接受的预测 token
    CompletionRejectedPrediction  int `json:"completion_rejected_prediction,omitempty"` // 拒绝的预测 token
    ServerToolUseTokens           int `json:"server_tool_use_tokens,omitempty"`          // 服务端工具 token（Anthropic 专有）

    // === 汇总 ===
    TotalTokens int `json:"total_tokens,omitempty"` // = PromptTokens + CompletionTokens（一般由上游给出）
}
```

**核心变化**：
- `InputTokens` → `PromptTokens`（重命名，语义明确：总输入，含 cache）
- `CacheReadTokens` → `PromptCacheHitTokens`（更明确的命名）
- `CacheCreationTokens` → `PromptCacheWriteTokens`（更明确的命名）
- `OutputTokens` → `CompletionTokens`（与 OpenAI 对齐）
- `ThinkingTokens` → `CompletionReasoningTokens`（更通用）
- 新增 `PromptAudioTokens`、`CompletionAudioTokens`、`CompletionAcceptedPrediction`、`CompletionRejectedPrediction`、`ServerToolUseTokens`

### 3.3 各协议映射

#### Anthropic ↔ IR

| Anthropic 字段 | IR 字段 | 方向 |
|---------------|---------|------|
| `input_tokens` | `PromptTokens - PromptCacheWriteTokens - PromptCacheHitTokens`（三项互斥分类，input_tokens = breakpoint 之后的 token） | Decode: `PromptTokens = input_tokens + cache_creation + cache_read` |
| `cache_creation_input_tokens` | `PromptCacheWriteTokens` | 双向 |
| `cache_read_input_tokens` | `PromptCacheHitTokens` | 双向 |
| `server_tool_use_tokens` | `ServerToolUseTokens` | 双向 |
| `output_tokens` | `CompletionTokens` | 双向 |

**Decode 公式**：
```go
usage := Usage{
    PromptTokens:          raw.Usage.InputTokens + raw.Usage.CacheCreationInputTokens + raw.Usage.CacheReadInputTokens,
    PromptCacheWriteTokens: raw.Usage.CacheCreationInputTokens,
    PromptCacheHitTokens:  raw.Usage.CacheReadInputTokens,
    ServerToolUseTokens:   raw.Usage.ServerToolUseTokens,
    CompletionTokens:      raw.Usage.OutputTokens,
    TotalTokens:           raw.Usage.InputTokens + raw.Usage.OutputTokens,
}
```

**Encode 公式**：
```go
anthropic.Usage{
    InputTokens:              u.PromptTokens - u.PromptCacheWriteTokens - u.PromptCacheHitTokens,
    OutputTokens:             u.CompletionTokens,
    CacheCreationInputTokens: u.PromptCacheWriteTokens,
    CacheReadInputTokens:     u.PromptCacheHitTokens,
    ServerToolUseTokens:      u.ServerToolUseTokens,
}
```

#### OpenAI Chat ↔ IR

| OpenAI Chat 字段 | IR 字段 | 方向 |
|-----------------|---------|------|
| `prompt_tokens` | `PromptTokens` | 双向（直接映射） |
| `prompt_tokens_details.cached_tokens` | `PromptCacheHitTokens` | 双向 |
| `prompt_tokens_details.audio_tokens` | `PromptAudioTokens` | 双向 |
| `completion_tokens` | `CompletionTokens` | 双向 |
| `completion_tokens_details.reasoning_tokens` | `CompletionReasoningTokens` | 双向 |
| `completion_tokens_details.audio_tokens` | `CompletionAudioTokens` | 双向 |
| `completion_tokens_details.accepted_prediction_tokens` | `CompletionAcceptedPrediction` | 双向 |
| `completion_tokens_details.rejected_prediction_tokens` | `CompletionRejectedPrediction` | 双向 |
| `total_tokens` | `TotalTokens` | 双向 |

**Decode**：
```go
usage := Usage{
    PromptTokens:               u.PromptTokens,
    PromptCacheHitTokens:       u.PromptTokensDetails.CachedTokens,
    PromptAudioTokens:          u.PromptTokensDetails.AudioTokens,
    CompletionTokens:           u.CompletionTokens,
    CompletionReasoningTokens:  u.CompletionTokensDetails.ReasoningTokens,
    CompletionAudioTokens:      u.CompletionTokensDetails.AudioTokens,
    CompletionAcceptedPrediction: u.CompletionTokensDetails.AcceptedPredictionTokens,
    CompletionRejectedPrediction: u.CompletionTokensDetails.RejectedPredictionTokens,
    TotalTokens:                u.TotalTokens,
}
```

**Encode**：
```go
raw := &openaichat.ChatUsage{
    PromptTokens:     u.PromptTokens,
    CompletionTokens: u.CompletionTokens,
    TotalTokens:      u.TotalTokens,
}
if u.PromptCacheHitTokens != 0 || u.PromptAudioTokens != 0 {
    raw.PromptTokensDetails = &openaichat.ChatPromptDetails{
        CachedTokens: u.PromptCacheHitTokens,
        AudioTokens:  u.PromptAudioTokens,
    }
}
if u.CompletionReasoningTokens != 0 || u.CompletionAudioTokens != 0 || u.CompletionAcceptedPrediction != 0 || u.CompletionRejectedPrediction != 0 {
    raw.CompletionTokensDetails = &openaichat.ChatCompletionDetails{
        ReasoningTokens:          u.CompletionReasoningTokens,
        AudioTokens:              u.CompletionAudioTokens,
        AcceptedPredictionTokens: u.CompletionAcceptedPrediction,
        RejectedPredictionTokens: u.CompletionRejectedPrediction,
    }
}
```

#### OpenAI Responses ↔ IR

与 OpenAI Chat 类似，字段名对应关系：
| Responses 字段 | IR 字段 |
|---------------|---------|
| `input_tokens` | `PromptTokens` |
| `input_tokens_details.cached_tokens` | `PromptCacheHitTokens` |
| `output_tokens` | `CompletionTokens` |
| `output_tokens_details.reasoning_tokens` | `CompletionReasoningTokens` |

#### Gemini ↔ IR

| Gemini 字段 | IR 字段 | 方向 |
|-------------|---------|------|
| `promptTokenCount` | `PromptTokens` | 双向 |
| `candidatesTokenCount` | `CompletionTokens` | 双向 |
| `totalTokenCount` | `TotalTokens` | 双向 |
| `thoughtsTokenCount` | `CompletionReasoningTokens` | 双向 |
| `cachedContentTokenCount` | `PromptCacheHitTokens` | 双向 |

### 3.4 核心语义修正

**OpenAI → Anthropic 时**：
```go
// 之前（错误）：
Anthropic.input_tokens = OpenAI.prompt_tokens  // 含 cache，偏大！

// 之后（正确）：
// Anthropic 三项互斥：input_tokens + cache_creation + cache_read = 总输入
// IR.PromptTokens = 总输入（= OpenAI prompt_tokens）
// 所以：input_tokens = PromptTokens - cache_creation - cache_read
Anthropic.input_tokens = IR.PromptTokens - IR.PromptCacheWriteTokens - IR.PromptCacheHitTokens
```

**Anthropic → OpenAI 时**：
```go
// 之前（错误）：
OpenAI.prompt_tokens = Anthropic.input_tokens  // 不含 cache，偏小！

// 之后（正确）：
// Anthropic 三项互斥：total_input = input_tokens + cache_creation + cache_read
// OpenAI prompt_tokens = total input (含 cache)
OpenAI.prompt_tokens = IR.PromptTokens = Anthropic.input_tokens + cache_creation_input_tokens + cache_read_input_tokens
```

### 3.5 流式 Usage 合并

`mergeStreamUsage` 需要更新所有新字段：

```go
func mergeStreamUsage(dst *Usage, src *Usage) {
    if src.PromptTokens != 0 {
        dst.PromptTokens = src.PromptTokens
    }
    if src.PromptCacheHitTokens != 0 {
        dst.PromptCacheHitTokens = src.PromptCacheHitTokens
    }
    if src.PromptCacheWriteTokens != 0 {
        dst.PromptCacheWriteTokens = src.PromptCacheWriteTokens
    }
    if src.PromptAudioTokens != 0 {
        dst.PromptAudioTokens = src.PromptAudioTokens
    }
    if src.CompletionTokens != 0 {
        dst.CompletionTokens = src.CompletionTokens
    }
    if src.CompletionReasoningTokens != 0 {
        dst.CompletionReasoningTokens = src.CompletionReasoningTokens
    }
    if src.CompletionAudioTokens != 0 {
        dst.CompletionAudioTokens = src.CompletionAudioTokens
    }
    if src.ServerToolUseTokens != 0 {
        dst.ServerToolUseTokens = src.ServerToolUseTokens
    }
    if src.CompletionAcceptedPrediction != 0 {
        dst.CompletionAcceptedPrediction = src.CompletionAcceptedPrediction
    }
    if src.CompletionRejectedPrediction != 0 {
        dst.CompletionRejectedPrediction = src.CompletionRejectedPrediction
    }
    if src.TotalTokens != 0 {
        dst.TotalTokens = src.TotalTokens
    }
}
```

### 3.6 Stats 事件影响

`CompleteEvent.Usage` 和 `StreamChunkEvent.IREvent.Usage` 字段名变更，下游 Stats 消费者需同步更新。

---

## 4. max_tokens 修复

### 4.1 新增 `model_defaults.go`

```go
package llmapimux

import "strings"

// ModelMaxOutputTokens returns the default max output tokens for a model.
// Returns 0 if the model is unknown.
func ModelMaxOutputTokens(model string) int {
    for prefix, val := range modelMaxTokens {
        if strings.HasPrefix(strings.ToLower(model), prefix) {
            return val
        }
    }
    return 0
}

var modelMaxTokens = map[string]int{
    "claude-4-opus":       16384,
    "claude-opus-4":       16384,
    "claude-4-sonnet":     16384,
    "claude-sonnet-4":     16384,
    "claude-3-5-sonnet":   8192,
    "claude-3-5-haiku":   8192,
    "claude-3-opus":       4096,
    "claude-3-sonnet":     4096,
    "claude-3-haiku":      4096,
}

const fallbackMaxTokens = 16384
```

### 4.2 修改 `convert_anthropic.go` — EncodeAnthropicRequest

```go
func EncodeAnthropicRequest(req *Request) ([]byte, error) {
    maxTokens := req.MaxTokens
    if maxTokens == 0 {
        if m := ModelMaxOutputTokens(req.Model); m > 0 {
            maxTokens = m
        } else {
            maxTokens = fallbackMaxTokens // 从 4096 提高到 16384
        }
    }
    // ... 其余不变
}
```

### 4.3 新增 MuxOption

```go
// WithDefaultMaxTokens sets a function that returns the default max_tokens
// for a model when not specified in the request.
func WithDefaultMaxTokens(fn func(model string) int) MuxOption {
    return func(m *Mux) { m.defaultMaxTokens = fn }
}
```

---

## 5. StopReason 映射修复

### 5.1 pause_turn 映射

**当前**（错误）：
| IR → 目标 | OpenAI Chat | OpenAI Responses | Gemini |
|-----------|-------------|------------------|--------|
| `pause_turn` | `stop` | `completed` | `STOP` |

**修复后**（正确）：
| IR → 目标 | OpenAI Chat | OpenAI Responses | Gemini |
|-----------|-------------|------------------|--------|
| `pause_turn` | `length` | `max_output_tokens` | `MAX_TOKENS` |

**理由**：`pause_turn` = "模型想继续但被 thinking budget 截断"，语义更接近"达到上限"而非"正常结束"。

### 5.2 修改文件

**`convert_openai_chat.go`**:
```go
func encodeOpenAIChatFinishReason(r StopReason) string {
    // ...
    case StopReasonPauseTurn:
        return "length"  // 改自 "stop"
}
```

**`convert_openai_responses.go`**:
```go
// 编码 StopReason 时
case StopReasonPauseTurn:
    return "max_output_tokens"  // 改自 "completed"
```

**`convert_gemini.go`**:
```go
func stopReasonToGeminiFinishReason(r StopReason) string {
    // ...
    case StopReasonPauseTurn:
        return "MAX_TOKENS"  // 改自 "STOP"
}
```

---

## 6. Gemini tool_use 推断修复

### 6.1 当前逻辑（有缺陷）

```go
if hasFunctionCall {
    resp.StopReason = StopReasonToolUse  // 无条件覆盖
} else {
    switch cand.FinishReason { ... }
}
```

**问题**：当 `finishReason="SAFETY"` 且有 FunctionCall 时，`SAFETY` 被覆盖为 `tool_use`。

### 6.2 修复后逻辑

```go
switch cand.FinishReason {
case "STOP":
    if hasFunctionCall {
        resp.StopReason = StopReasonToolUse
    } else {
        resp.StopReason = StopReasonEndTurn
    }
case "MAX_TOKENS":
    resp.StopReason = StopReasonMaxTokens
case "SAFETY":
    resp.StopReason = StopReasonContentFilter  // 即使有 FunctionCall 也用 SAFETY
case "STOP_SEQUENCE":
    resp.StopReason = StopReasonStopSequence
default:
    if cand.FinishReason != "" {
        resp.StopReason = StopReason(cand.FinishReason)
    } else if hasFunctionCall {
        resp.StopReason = StopReasonToolUse  // finishReason 为空才推断
    }
}
```

---

## 7. Gemini citation 分配修复

### 7.1 当前逻辑（有缺陷）

```go
// Attach to first text content part
for i := range resp.Content {
    if resp.Content[i].Type == ContentTypeText {
        resp.Content[i].Citations = citations
        break  // 只挂第一个
    }
}
```

### 7.2 修复后逻辑

按字符偏移范围将 citation 分配到对应的 text 块：

```go
func distributeCitationsToTextParts(content []ContentPart, citations []Citation) {
    if len(citations) == 0 {
        return
    }
    offset := 0
    for i := range content {
        if content[i].Type != ContentTypeText || content[i].Text == nil {
            continue
        }
        partLen := len(content[i].Text.Text)
        partStart := offset
        partEnd := offset + partLen

        var partCitations []Citation
        for _, c := range citations {
            if c.Start != nil && c.End != nil &&
                *c.Start >= partStart && *c.End <= partEnd {
                adjusted := c
                s := *c.Start - partStart
                e := *c.End - partStart
                adjusted.Start = &s
                adjusted.End = &e
                partCitations = append(partCitations, adjusted)
            }
        }
        content[i].Citations = partCitations
        offset = partEnd
    }
}
```

在 `DecodeGeminiResponse` 中替换原来的单块分配逻辑：

```go
if cand.CitationMetadata != nil && len(cand.CitationMetadata.CitationSources) > 0 {
    citations := /* ... 构建 Citation 列表 ... */
    distributeCitationsToTextParts(resp.Content, citations)  // 替换原来的 break 逻辑
}
```

---

## 8. Refusal 弱映射修复

### 8.1 问题

Anthropic 无原生 `refusal` 类型，IR `ContentTypeRefusal` 编码为 `type: "text"`，丢失语义。

### 8.2 方案：在 IR ContentPart 加 SourceType 元数据

```go
type ContentPart struct {
    Type      ContentType `json:"type"`
    // ... 现有字段不变 ...
    
    // 新增：记录跨协议转换前的原始类型
    // 用于同协议往返时恢复原始 ContentPart.Type
    SourceType ContentType `json:"source_type,omitempty"`
}
```

**工作流**：

1. **Anthropic Decode** → IR `ContentTypeRefusal`（从 OpenAI 来）→ `SourceType = "refusal"`
2. **Anthropic Encode** → 检查 `SourceType`，如果是 `"refusal"` 且目标也是 Anthropic，用 RawExtra 恢复
3. **跨协议 Encode** → `ContentTypeRefusal` 编码为 `type: "text"`（保留弱映射），但 `SourceType` 被带在 IR 中，Stats 消费者可识别

**实际改动**：这个方案仍然无法让 Anthropic 协议原生表达 refusal（因为 API 不支持），但 `SourceType` 让下游知道"这个 text 块原本是 refusal"。

### 8.3 编码改动

**`convert_anthropic.go`** — `encodeAnthropicContentPart`:
```go
case ContentTypeRefusal:
    // Anthropic 无原生 refusal 类型，降级为 text
    b := anthropic.ContentBlock{Type: "text"}
    if p.Refusal != nil {
        b.Text = p.Refusal.Refusal
    }
    return b, nil
// 行为不变，但调用者/Stats 可通过 IR 上的 SourceType 识别
```

**各 Decode 函数** — 设置 SourceType：
```go
// OpenAI Chat/Responses decode refusal 时
part := ContentPart{
    Type:      ContentTypeRefusal,
    Refusal:   &RefusalContent{Refusal: refusalText},
    SourceType: ContentTypeRefusal,  // 新增
}
```

---

## 9. Stats 新增指标

### 9.1 CompleteEvent 新增字段

```go
type CompleteEvent struct {
    // ... 现有字段不变 ...
    
    // 新增
    TPOT   time.Duration // Time Per Output Token = (TotalLatency - TTFB) / CompletionTokens
    Chunks int           // 流式 chunk 总数（non-streaming 为 0）
}
```

### 9.2 计算逻辑

**`handleNonStreaming`**:
```go
tpot := time.Duration(0)
chunks := 0
// non-streaming: TTFB ≈ TotalLatency, TPOT 无意义，设 0
```

**`handleStreaming`**:
```go
tpot := time.Duration(0)
if summary.usage.CompletionTokens > 0 && totalLatency > summary.ttfb {
    decodeTime := totalLatency - summary.ttfb
    tpot = decodeTime / time.Duration(summary.usage.CompletionTokens)
}
chunks = seq  // 从 wrapStreamForStats 传出
```

---

## 10. 不改项说明

| # | 缺陷 | 不改原因 |
|---|------|---------|
| 7 | Gemini 无 tool 角色，RoleTool → "user" | **API 硬限制**：Gemini 确实没有 `tool` 角色，SDK 和文档都使用 `user` + `functionResponse`。代码已通过 `toolNameByID` 确保 `FunctionResponse.name` 正确解析 |
| 11 | Phase 2 Thinking 字段跨协议丢弃 | **协议固有限制**：OpenAI Chat/Responses/Anthropic 都没有 `IncludeThoughts`/`Level` 的原生等效。同协议往返时通过 RawExtra 自动保留（这些字段不在 knownFields 中） |
| 12 | Gemini 合成 ID | **正确设计**：`geminiToolCorrelator` 确保 FunctionCall/FunctionResponse ID 配对正确 |
| 13 | Anthropic 多工具白名单丢弃 | **API 硬限制**：Anthropic `tool_choice` 只支持 auto/any/none/单工具 |
| 14 | Gemini AllowParallelCalls 丢弃 | **API 硬限制**：Gemini 无并行工具调用控制 |
| 15 | 流式生命周期/错误事件丢弃 | **协议固有限制**：Anthropic codec 已做合成处理，OpenAI/Gemini 确实不支持 |

---

## 11. 修改文件清单

| 文件 | 改动 | 章节 |
|------|------|------|
| `message.go` | Usage 结构重新设计 + ContentPart 加 SourceType | §3, §8 |
| `stats.go` | CompleteEvent 新增 TPOT/Chunks | §9 |
| `handler.go` | mergeStreamUsage 重写 + TPOT 计算 | §3, §9 |
| `convert_anthropic.go` | Decode/Encode Usage 全新映射 + pause_turn | §3, §5 |
| `convert_openai_chat.go` | Decode/Encode Usage 全新映射 + pause_turn | §3, §5 |
| `convert_openai_responses.go` | Decode/Encode Usage 全新映射 + pause_turn | §3, §5 |
| `convert_gemini.go` | Decode/Encode Usage + tool_use 推断 + citation + pause_turn | §3, §5, §6, §7 |
| `mux.go` | 新增 WithDefaultMaxTokens MuxOption | §4 |
| `model_defaults.go` | **新建**：模型默认值表 | §4 |
| `*_test.go`（约 10 个） | 同步更新所有 Usage 相关测试 | 全部 |

### 测试策略

每个 Usage 映射改动需验证：
1. **Decode 正确性**：协议 JSON → IR Usage，各字段值正确
2. **Encode 正确性**：IR Usage → 协议 JSON，各字段值正确
3. **Round-trip**：Decode → Encode → Decode，值一致
4. **跨协议修正**：OpenAI Usage → IR → Anthropic Usage，input_tokens 语义正确
5. **流式合并**：mergeStreamUsage 各字段正确合并
