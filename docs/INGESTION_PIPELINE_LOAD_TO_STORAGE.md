# Ingestion Pipeline：从 Load 到 Storage

本文说明 `IngestionPipeline.run` 在 **Stage 2（Load）** 至 **Stage 6（Storage）** 各步的**产出类型**与**持久化位置**，并与 `src/ingestion/pipeline.py` 实现一致。Stage 1（Integrity）仅在开头简要交代，便于理解「为何有时不会进入 Load」。

---

## 实例设定

- **文件**：`handbook.pdf`（可读路径下的 PDF）
- **命令**：`python scripts/ingest.py --path handbook.pdf --collection handbook`
- **向量库**：以 `config/settings.yaml` 为准；默认常见为 **Chroma**，`vector_store.persist_directory` 例如 `./data/db/chroma`

---

## 0. 进入 Load 之前：Stage 1 Integrity

- **行为**：对文件计算 **SHA256（`file_hash`）**，查询 `data/db/ingestion_history.db`（`SQLiteIntegrityChecker`）。
- **若 `force=False` 且 `should_skip(file_hash)` 为真**：表示该哈希曾 **`mark_success`**，**直接 `return PipelineResult`**，**不会执行 Load / Chunking / Storage**。
- **跳过时的结果**：`success=True`，`stages["integrity"]` 含 `skipped: True`、`reason: "already_processed"`。

实现见：`src/ingestion/pipeline.py`（Stage 1）、`src/libs/loader/file_integrity.py`（`should_skip` / `mark_success`）。

以下假设 **不跳过**，流水线继续。

---

## Stage 2：Load（文档加载）

### 产出

| 产出 | 说明 |
|------|------|
| **`Document`** | `PdfLoader.load(path)`：`id`（如 `doc_` + 文件 SHA256 前 16 位）、`text`（Markdown；图用 `[IMAGE: {image_id}]`）、`metadata`（**必含 `source_path`**，另有 `doc_type`、`doc_hash`、`title`、可选 **`images`**）。 |

### 持久化

| 位置 | 内容 |
|------|------|
| **图片文件** | `IngestionPipeline` 将 `PdfLoader` 的 `image_storage_dir` 设为 `resolve_path(f"data/images/{collection}")`；Loader 内为 **`image_storage_dir / doc_hash`**，即 **`data/images/handbook/<doc_hash>/`** 下的图片文件（`doc_hash` 为整文件 SHA256 十六进制串）。 |
| **`ingestion_history`** | 此阶段**不写**成功标记；仅在整条流水线成功结束调用 **`mark_success`**。 |

实现见：`src/libs/loader/pdf_loader.py`（`load`、`metadata`、`image_dir`）。

---

## Stage 3：Chunking（分块）

### 产出

| 产出 | 说明 |
|------|------|
| **`List[Chunk]`** | `DocumentChunker.split_document(document)`：每块含 `id`、`text`、`metadata`（**必含 `source_path`**），及 `chunk_index` / 偏移等可选字段。 |

### 持久化

- **无**（仅内存中的 `Chunk` 列表）。

实现见：`src/ingestion/chunking/document_chunker.py`（由 `pipeline` 调用）。

---

## Stage 4：Transform（变换链）

### 产出

对同一批 **`Chunk`** 顺序执行：

1. **`ChunkRefiner.transform`**：精炼 `text` / 元数据（规则或 LLM）。
2. **`MetadataEnricher.transform`**：如 `title`、`tags`、`summary` 等。
3. **`ImageCaptioner.transform`**：如 `metadata["image_captions"]`（Vision 未配置时可退化）。

### 持久化

- **无**（仍为内存中的 `Chunk` 列表）。

实现见：`src/ingestion/pipeline.py`（Stage 4 子步骤 4a–4c）。

---

## Stage 5：Encoding（编码）

### 产出

| 产出 | 说明 |
|------|------|
| **`dense_vectors`** | 与 `chunks` **同序** 的稠密向量列表（`BatchProcessor` → `DenseEncoder` + 配置中的 Embedding）。 |
| **`sparse_stats`** | 与 chunks **同序** 的 BM25 用统计（词频、文档长度等，`SparseEncoder`）。 |

### 持久化

- **无**（列表与 dict 仅在内存中，供 Stage 6 使用）。

实现见：`src/ingestion/embedding/batch_processor.py`、`pipeline.py` Stage 5。

---

## Stage 6：Storage（存储）

以下用 **3 个 chunk** 的延续例子说明。

### 6a：向量库 Upsert

- **调用**：`vector_ids = vector_upserter.upsert(chunks, dense_vectors, trace)`。
- **行为**：`VectorUpserter` 为每个 chunk 生成**稳定** `chunk_id`，组装 `id`、`vector`、`metadata`（含全文 `text` 等），调用 **`VectorStoreFactory`** 创建的向量库 **upsert**。
- **返回值 `vector_ids`**：与输入 **同序** 的向量库记录 id（用于下一步与 BM25 对齐）。

### 持久化

| 位置 | 内容 |
|------|------|
| **向量库** | 如 Chroma：**`settings.yaml` 中 `vector_store.persist_directory`**（常见 `./data/db/chroma`），逻辑集合名为运行时 **`collection`**（本例 `handbook`）。 |

实现见：`src/ingestion/storage/vector_upserter.py`。

### 6b：BM25 索引

- **对齐**：`for stat, vid in zip(sparse_stats, vector_ids): stat["chunk_id"] = vid`，使每条稀疏统计带上**与向量库一致**的 id，便于稀疏命中后回查同一 chunk。
- **调用**：`bm25_indexer.add_documents(sparse_stats, collection=..., doc_id=document.id, ...)`。

### 持久化

| 位置 | 内容 |
|------|------|
| **BM25 索引目录** | **`data/db/bm25/{collection}/`**，本例 **`data/db/bm25/handbook/`**。 |

实现见：`src/ingestion/pipeline.py`（Stage 6 注释说明与 `SparseRetriever` 对齐意图）。

### 6c：图片索引

- **行为**：遍历 `document.metadata["images"]`，对存在的路径调用 **`ImageStorage.register_image`**（图片文件已在 Load 阶段落盘）。

### 持久化

| 位置 | 内容 |
|------|------|
| **`data/db/image_index.db`** | 图片 id、路径、collection、`doc_hash`、页码等索引。 |

实现见：`src/ingestion/storage/image_storage.py`、`pipeline.py` Stage 6c。

### 6d：标记成功

- **调用**：`integrity_checker.mark_success(file_hash, file_path, collection)`。

### 持久化

| 位置 | 内容 |
|------|------|
| **`data/db/ingestion_history.db`** | 该 `file_hash` 记录为 **`success`**，下次同内容文件默认跳过整条流水线（除非 `force=True`）。 |

---

## 端到端速览表

| 阶段 | 主要产出 | 常见持久化位置 |
|------|----------|----------------|
| 1 Integrity | 是否跳过 / `file_hash` | `ingestion_history.db`（查询）；成功标记在末尾写入 |
| 2 Load | `Document` | 图片 → **`data/images/{collection}/{doc_hash}/`** |
| 3 Chunking | `List[Chunk]` | — |
| 4 Transform | 更新后的 `List[Chunk]` | — |
| 5 Encoding | `dense_vectors`、`sparse_stats` | — |
| 6a Vector | `vector_ids` | **`vector_store.persist_directory`**（如 Chroma） |
| 6b BM25 | 索引条目 | **`data/db/bm25/{collection}/`** |
| 6c Image index | 登记记录 | **`data/db/image_index.db`** |
| 收尾 | — | **`ingestion_history.db`** → `success` |

---

## 设计要点（与代码注释一致）

- **配置驱动**：各阶段依赖 `Settings`（`config/settings.yaml`）。
- **幂等**：完整性成功记录 + `force`；向量写入侧 `VectorUpserter` 使用稳定 chunk id（见 `vector_upserter.py` 文档字符串）。
- **双路对齐**：BM25 统计中的 `chunk_id` 与向量库返回的 id 一致，便于 Hybrid 检索后统一回溯。

---

## 相关源码入口

| 文件 | 作用 |
|------|------|
| `src/ingestion/pipeline.py` | `IngestionPipeline.run` 六阶段编排 |
| `src/libs/loader/pdf_loader.py` | PDF → `Document`、图片落盘 |
| `src/libs/loader/file_integrity.py` | 哈希、跳过、成功/失败标记 |
| `src/ingestion/storage/vector_upserter.py` | 向量 upsert |
| `src/ingestion/storage/bm25_indexer.py` | BM25 写入 |
| `src/ingestion/storage/image_storage.py` | 图片索引 |
| `scripts/ingest.py` | CLI 入口，调用 `pipeline.run` |
