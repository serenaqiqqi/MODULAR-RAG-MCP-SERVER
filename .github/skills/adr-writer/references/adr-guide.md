# ADR 写作参考（Modular RAG MCP Server）

## 何时写 ADR

适合记录的情况：

- 在多种架构/技术路线间做**长期有效**的选择（存储、检索融合、Provider 抽象边界等）
- 团队需要**可追溯**的「为什么当时这样定」
- 变更会影响多个模块或对外 MCP 契约

不必写 ADR：

- 单行 bug 修复、命名调整、纯文档错别字
- 已在 `DEV_SPEC.md` 中定死且无争议的实现细节（除非决策偏离 spec）

## 状态

| 状态 | 含义 |
|------|------|
| Proposed | 草案，讨论中 |
| Accepted | 已采纳，可按此实施 |
| Deprecated | 不再推荐，但可能仍在代码中 |
| Superseded | 被新 ADR 取代，文中应链接新 ADR |

状态变更时保留原文，在文首更新 Status，必要时加「修订说明」小节。

## 与本项目对齐

起草时对照：

- **三层架构**：`core/`、`ingestion/`、`libs/` 职责是否被破坏
- **可插拔 Provider**：新决策是否需扩展 factory / settings
- **可观测性**：是否影响 trace、Dashboard 指标
- **MCP 工具契约**：是否改变工具输入输出或语义

## 质量检查（落盘前自检）

- [ ] Context 写清「不决定会怎样」
- [ ] Decision 可验证、无模糊词堆砌
- [ ] Consequences 含负面后果与缓解措施
- [ ] 链接到具体路径（如 `src/ingestion/storage/vector_upserter.py`）
- [ ] 与 `DEV_SPEC.md` 一致或明确说明「spec 将随后更新」

## 编号与命名

- 四位序号：`0001`、`0002` … 勿跳号重用已删除编号
- 文件名英文 kebab-case，与标题含义一致
- 一篇 ADR 只记录**一个**核心决策；强相关但可独立的决策可拆篇并在文中互链
