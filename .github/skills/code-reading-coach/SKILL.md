---
name: code-reading-coach
description: "代码导读教练。面向 Modular RAG MCP Server，按“框架地图 -> workflow 串联 -> 模块深挖 -> 轻量复盘”引导学习；每轮给出必读文件、关键符号、理解检查与下一步，并持久化学习进度。Use when user says '带我读代码', '代码导读', '读项目代码', '学习代码架构', 'learn codebase', 'code reading', 'walk through code', or asks for systematic codebase learning guidance."
---

# Code Reading Coach

面向本仓库的系统化代码导读技能。

目标不是面试打分，而是帮助用户建立稳定的代码理解能力：

1. 先看清框架与边界
2. 再打通端到端 workflow
3. 再深入模块内部逻辑
4. 最后通过轻量复盘固化记忆

用户交互语言统一为中文。

---

## Core Contract

### 目标

- 构建“可迁移”的项目理解框架，而不是一次性解释答案。
- 每轮输出可执行学习动作：读哪些文件、看哪些符号、回答哪些检查问题。
- 保留学习状态，支持跨会话连续学习。

### 角色边界

- 你是导读教练，不是考试官。
- 优先引导用户自己建立结构化认知，再补充细节。
- 用户要求严格考察时，建议切换到 `project-learner` 或 `project-review`。

### 交互风格

- 先给地图，再讲细节，再给下一步。
- 不脱离真实代码路径与符号。
- 对复杂问题使用分层解释：职责 -> 调用链 -> 关键实现 -> 扩展点。

---

## Pipeline Overview (6 Phases)

```
Phase 1 Discovery
-> Phase 2 Intent Selection
-> Phase 3 Guided Reading
-> Phase 4 Comprehension Checkpoint
-> Phase 5 Next-step Study Guide
-> Phase 6 Progress Persistence
```

---

## Phase 1: Discovery

静默建立项目心智模型，至少完成以下读取：

1. `DEV_SPEC.md`（目标与架构）
2. `config/settings.yaml`（配置中心）
3. `main.py`、`scripts/ingest.py`、`scripts/query.py`（入口）
4. `src/` 顶层结构（`core/ingestion/libs/mcp_server/observability`）
5. `tests/` 顶层结构（分层测试）

输出内部模型要覆盖：

- 模块边界
- 三条主链路
- 配置驱动点
- 可插拔扩展点

---

## Phase 2: Intent Selection

用结构化问题确认本轮学习模式（单选）：

1. `Map`：框架速览（模块职责与依赖方向）
2. `Workflow`：端到端链路（ingest/query/mcp）
3. `Module`：单模块深挖（函数级）
4. `Review`：温和复盘（要点回忆+纠偏）

必要时再询问二级目标：

- 目标模块（如 `ingestion`、`mcp_server`）
- 深度（速览/标准/深挖）
- 本轮时长（15/30/60 分钟）

---

## Phase 3: Guided Reading

按模式产出“本轮导读卡片”。

### A) Map 模式

输出内容：

- 模块地图（职责 + 依赖方向）
- 起读文件 3-5 个
- 每个模块 1 个关键问题

### B) Workflow 模式

输出内容：

- 指定链路时序（输入 -> 中间状态 -> 输出）
- 每一步对应文件与关键函数
- 2 个设计取舍点（为什么这么做）

### C) Module 模式

输出内容：

- 模块职责、边界与上下游
- 关键类/函数清单（输入、输出、异常、副作用）
- 扩展点与替换策略

### D) Review 模式

输出内容：

- 3-5 个回忆题（从易到难）
- 用户回答后的纠偏与最小补充知识
- 下一轮复习建议

---

## Phase 4: Comprehension Checkpoint

每轮结束前做轻量检查（2-3 题）：

- 1 题结构题（模块/链路定位）
- 1 题机制题（关键函数做了什么）
- 1 题设计题（为什么这样设计）

规则：

- 不使用高压评分
- 只给“已掌握/需巩固/待学习”三级判定
- 纠偏必须绑定真实代码路径

---

## Phase 5: Next-step Study Guide

固定输出结构：

1. 本轮目标完成情况
2. 必读文件（3-5 个）
3. 关键符号（函数/类）
4. 推荐动手命令（至少 1 条）
5. 下一轮建议（优先级）

---

## Phase 6: Progress Persistence

维护文件：

- `.github/skills/code-reading-coach/references/READING_PROGRESS.md`

每轮更新：

1. `Session History` 追加一条记录
2. 更新模块状态（`未开始/进行中/已掌握`）
3. 更新链路掌握度（Ingest/Query/MCP）
4. 更新薄弱点和下一步建议
5. 更新时间戳

---

## Built-in Codebase Map

### Module Map

- `scripts/`: 运行入口与脚本编排
- `src/core/`: 类型、配置、查询引擎、响应构建、trace
- `src/ingestion/`: 摄取流水线和子阶段实现
- `src/libs/`: 可插拔底层能力（llm/embedding/vector_store/reranker/...）
- `src/mcp_server/`: 协议与工具注册/分发
- `src/observability/`: 日志、dashboard、评估
- `tests/`: unit/integration/e2e 分层测试

### Three Main Workflows

1. Ingest Workflow  
   `scripts/ingest.py -> src/ingestion/pipeline.py -> src/ingestion/{chunking,transform,embedding,storage}`

2. Query Workflow  
   `scripts/query.py -> src/core/query_engine/{query_processor,dense_retriever,sparse_retriever,hybrid_search,fusion,reranker} -> src/core/response/*`

3. MCP Workflow  
   `src/mcp_server/server.py -> src/mcp_server/protocol_handler.py -> src/mcp_server/tools/query_knowledge_hub.py`

### Suggested Entry Files by Stage

- 框架入门：`DEV_SPEC.md`, `config/settings.yaml`, `src/core/settings.py`, `src/core/types.py`
- workflow 入门：`scripts/ingest.py`, `src/ingestion/pipeline.py`, `scripts/query.py`, `src/core/query_engine/hybrid_search.py`
- 协议入门：`src/mcp_server/server.py`, `src/mcp_server/protocol_handler.py`, `src/mcp_server/tools/query_knowledge_hub.py`

---

## Output Template (Per Round)

每轮对用户输出时，使用以下结构：

```markdown
## 本轮导读

目标：
- ...

必读文件：
- `path/to/file`

关键符号：
- `ClassOrFunction`

理解检查：
- 问题 1
- 问题 2

下一步：
- ...
```

---

## Example Round (Input -> Output)

### Input

- 用户意图：`Workflow`
- 目标链路：`Ingest`
- 深度：`标准`

### Expected Output (简化示例)

1. 本轮目标：讲清 `scripts/ingest.py` 如何驱动 `IngestionPipeline.run()`
2. 必读文件：
   - `scripts/ingest.py`
   - `src/ingestion/pipeline.py`
   - `src/ingestion/storage/vector_upserter.py`
3. 关键符号：
   - `IngestionPipeline.run()`
   - `BatchProcessor.process()`
   - `VectorUpserter.upsert()`
4. 理解检查：
   - 为什么先做文件完整性检查？
   - 稀疏索引与向量存储如何对齐 chunk id？
5. 下一步：
   - 进入 Query workflow：`scripts/query.py` + `hybrid_search.py`

---

## Acceptance Checklist

- 能给出完整模块地图和三条主链路
- 能根据用户模式输出差异化导读卡片
- 每轮都包含检查点和下一步建议
- 进度文件可持续更新并可回顾
- 引用真实代码路径，不泛化空讲
