# Code Reading Progress

> Last updated: 2026-03-26
> Overall status: Not started

## Learning Profile

- Current mode: Map / Workflow / Module / Review
- Current focus: N/A
- Current depth: overview / standard / deep-dive
- Total sessions: 0

## Module Progress

| Module | Scope | Status | Confidence (1-5) | Last session | Notes |
|--------|-------|--------|------------------|--------------|-------|
| scripts | Entry scripts and orchestration | Not started | 0 | - | - |
| core | Types/settings/query/response/trace | Not started | 0 | - | - |
| ingestion | End-to-end ingestion pipeline | Not started | 0 | - | - |
| libs | Pluggable providers and factories | Not started | 0 | - | - |
| mcp_server | Protocol and tools | Not started | 0 | - | - |
| observability | Logging/dashboard/evaluation | Not started | 0 | - | - |
| tests | Unit/integration/e2e strategy | Not started | 0 | - | - |

## Workflow Mastery

| Workflow | Key path | Status | Confidence (1-5) | Weak points |
|----------|----------|--------|------------------|-------------|
| Ingest | `scripts/ingest.py -> src/ingestion/pipeline.py` | Not started | 0 | - |
| Query | `scripts/query.py -> src/core/query_engine/* -> src/core/response/*` | Not started | 0 | - |
| MCP | `src/mcp_server/server.py -> protocol_handler.py -> tools/query_knowledge_hub.py` | Not started | 0 | - |

## Session History

| # | Date | Mode | Focus | Files covered | Checkpoint result | Next step |
|---|------|------|-------|---------------|-------------------|-----------|

## Recurrent Weak Points

- None

## Next Suggested Session

- Mode: Map
- Focus: `core + ingestion`
- Goal:
  1. Understand module boundaries
  2. Trace ingest workflow from script to storage

## Update Rules

1. Append one row to `Session History` after each guided reading round.
2. Update the corresponding `Module Progress` rows:
   - `Status`: Not started / In progress / Mastered
   - `Confidence`: integer 1-5
3. Update `Workflow Mastery` confidence and weak points if workflow-related learning occurred.
4. Keep `Recurrent Weak Points` as a short, deduplicated list.
5. Refresh `Last updated` and `Total sessions`.
