---
name: adr-writer
description: "Guides architecture decision discussions and drafts Architecture Decision Records (ADRs) for Modular RAG MCP Server. Explores context from DEV_SPEC and codebase, compares options with tradeoffs, drafts ADR in standard format, and writes to docs/adr/ only after user approval at checkpoints. Use when user says '写 ADR', '架构决策', 'ADR', 'architecture decision', '技术方案记录', '决策记录', or wants to document and decide between design alternatives."
---

# ADR Writer — 架构决策记录

半自动工作流：先对齐问题与约束，再对比方案，起草 ADR，**经你确认后才落盘**。

用户交互语言：**中文**。ADR 正文可用中文（默认）或英文（用户指定时）。

## Pipeline

```
Phase 0 静默摸底 → Phase 1 对齐决策问题 [Gate A]
→ Phase 2 方案对比 [Gate B] → Phase 3 起草 ADR [Gate C]
→ Phase 4 落盘与索引 [Gate D 可选]
```

**硬性规则**

- 未经对应 Gate 确认，**不得**创建或修改 `docs/adr/` 下的文件。
- 不得替用户做最终决策；只整理选项、利弊与推荐倾向（须标明「倾向」而非「定论」）。
- 新 ADR 必须引用相关代码路径、`DEV_SPEC.md` 章节，以及已存在的相关 ADR（若有）。

---

## Phase 0: 静默摸底（不向用户提问）

1. 读 `DEV_SPEC.md` 中与当前主题相关的架构、模块边界、非功能需求。
2. 用搜索/阅读定位相关实现：`src/`、`config/settings.yaml`、已有测试。
3. 列出 `docs/adr/*.md`（若目录不存在则记下「尚无 ADR」），避免重复决策。
4. 内部整理：**当前行为**、**痛点**、**约束**（兼容性、性能、可观测性、可插拔 Provider 等）。

---

## Phase 1: 对齐决策问题 — Gate A

用简短中文呈现：

| 项 | 内容 |
|----|------|
| 决策标题（草案） | 一句话 |
| 背景与问题 | 为什么要现在决定 |
| 约束与边界 | 必须满足 / 明确不做 |
| 相关代码与文档 | 路径列表 |
| 成功标准 | 怎样算「决定对了」 |

然后 **停止**，请用户确认或修正。用户明确说「继续」「对了」「OK」后再进入 Phase 2。

---

## Phase 2: 方案对比 — Gate B

至少给出 **2 个**可行方案（含「维持现状」若合理）。每个方案包含：

- 概要做法
- 优点 / 缺点
- 对 ingestion、检索、MCP、配置、测试、运维的影响
- 实现复杂度（粗估：低 / 中 / 高）

用表格汇总，并给出 **推荐倾向**（一句话 + 理由），标注为「倾向，非最终结论」。

**停止**，等待用户选择方案、组合方案，或要求补充新方案。未收到选择前不起草全文。

---

## Phase 3: 起草 ADR — Gate C

按 [assets/adr-template.md](assets/adr-template.md) 生成完整草稿：

- 状态默认为 `Proposed`（用户可指定 `Accepted` 等）
- **Decision** 只写用户已确认的内容
- **Consequences** 分正面 / 负面 / 中性，写可操作的后续项（改哪些模块、补哪些测试）

将草稿以 Markdown 展示在对话中（不写入磁盘）。

**停止**，等待用户反馈：修改段落、改标题、改状态、中英切换等。用户说「可以落盘」「写入」「approve」等明确同意后进入 Phase 4。

---

## Phase 4: 落盘与索引

1. **编号**：扫描 `docs/adr/`，取最大四位编号 +1；若无文件则从 `0001` 开始。
2. **文件名**：`docs/adr/NNNN-<kebab-case-title>.md`（仅小写字母、数字、连字符）。
3. **写入**确认后的正文；若目录不存在则创建 `docs/adr/`。
4. **索引（可选）**：若存在 `docs/adr/README.md`，在表格中追加一行；若不存在且已有 ≥2 篇 ADR，可询问是否创建 README 索引（须用户同意）。

完成后告知：文件路径、编号、状态，以及建议的后续动作（实现任务、更新 DEV_SPEC、开 PR 等）。

---

## Gate 速查

| Gate | 触发句示例 | Agent 允许的动作 |
|------|------------|------------------|
| A | 「继续」「背景对了」 | 进入方案对比 |
| B | 「选方案 2」「按 B+C」 | 起草 ADR |
| C | 「落盘」「approve」 | 写入 `docs/adr/` |
| D | 「更新索引」 | 改 `docs/adr/README.md` |

---

## 附加资源

- ADR 写作要点与状态流转： [references/adr-guide.md](references/adr-guide.md)
- 正文结构模板： [assets/adr-template.md](assets/adr-template.md)

## 示例触发

用户：「我们在纠结向量库要不要换 Chroma，帮我写个 ADR。」

Agent：执行 Phase 0 → Phase 1 表格 → **等待** → Phase 2 对比 → **等待** → Phase 3 草稿 → **等待** → Phase 4 写入 `docs/adr/0003-chroma-vector-store.md`（编号以仓库实际为准）。
