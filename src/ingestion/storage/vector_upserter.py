"""Vector Upserter for writing chunks to vector database.
# 这个文件的作用：把 chunk 和它对应的向量写入向量数据库。

This module implements the VectorUpserter component, responsible for:
# 下面说明这个模块负责哪些事：

- Generating deterministic chunk IDs from content
# 1）根据内容生成稳定的 chunk_id

- Transforming chunks and vectors into storage records
# 2）把 chunk 和 vector 组装成可写入存储的 record

- Calling VectorStore for idempotent writes
# 3）调用 VectorStore 执行幂等写入

- Supporting batch operations with consistent ordering
# 4）支持批量写入，并且保证顺序一致

Design Principles:
# 下面是设计原则：

- Idempotent: Same content produces same ID, repeated writes safe
# 幂等：同样内容生成同样 ID，所以重复写入也安全

- Observable: Accepts TraceContext for future integration
# 可观测：支持传 trace，后续可接入追踪体系

- Config-Driven: Uses VectorStoreFactory from settings
# 配置驱动：通过 settings 和工厂来创建 vector store

- Deterministic: Stable hash-based ID generation
# 确定性：基于哈希来生成稳定 ID

- Type-Safe: Full type hints and validation
# 类型安全：有完整类型标注和输入校验
"""

import hashlib
# 导入 hashlib，用来做 SHA256 哈希，后面生成 chunk_id 要用。

from typing import List, Dict, Any, Optional
# 导入类型标注工具：
# List：列表
# Dict：字典
# Any：任意类型
# Optional：可为空

from src.core.types import Chunk
# 导入 Chunk 类型，表示要写入的文本块对象。

from src.core.settings import Settings
# 导入 Settings 类型，表示项目配置对象。

from src.libs.vector_store.vector_store_factory import VectorStoreFactory
# 导入 VectorStoreFactory，用来根据配置创建具体的向量库实例。


class VectorUpserter:
    """Write chunks and vectors to vector database with idempotent guarantees.
    # 这个类的职责：把 chunk 和向量写入向量数据库，并保证幂等。
    
    This upserter receives chunks and their dense vectors from DenseEncoder,
    # 它接收 chunk 和对应的 dense vector，

    generates stable chunk IDs, and writes them to the configured vector store.
    # 然后生成稳定 chunk_id，再写入配置好的向量库。
    
    Chunk ID Format:
        {source_path_hash}_{chunk_index:04d}_{content_hash}
        # chunk_id 的格式是：
        # source_path 的哈希 + chunk_index 的四位补零 + chunk.text 的哈希
        
    Where:
        - source_path_hash = first 8 chars of SHA256(source_path)
        # source_path_hash：source_path 做 SHA256 后取前 8 位

        - chunk_index = zero-padded 4-digit index
        # chunk_index：4 位补零，比如 3 会变成 0003

        - content_hash = first 8 chars of SHA256(chunk.text)
        # content_hash：chunk.text 做 SHA256 后取前 8 位
        
    This ensures:
        - Same content → same ID (idempotent)
        # 同样内容得到同样 ID，保证幂等

        - Content change → different ID (versioning)
        # 内容一变，ID 也变，相当于天然带版本变化

        - Human-readable with source traceability
        # ID 里还保留了一点来源可追溯性，不是完全随机字符串
    
    Example:
        >>> upserter = VectorUpserter(settings)
        >>> 
        >>> chunks = [
        ...     Chunk(id="temp1", text="Hello world", metadata={"source_path": "doc.pdf", "chunk_index": 0}),
        ...     Chunk(id="temp2", text="Python rocks", metadata={"source_path": "doc.pdf", "chunk_index": 1})
        ... ]
        >>> vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> 
        >>> upserter.upsert(chunks, vectors)
        >>> # Chunks written with stable IDs like: "a1b2c3d4_0000_e5f6g7h8"
        # 这段例子演示了如何创建 upserter，再把 chunks 和 vectors 写进去。
    """
    
    def __init__(self, settings: Settings, collection_name: Optional[str] = None):
        """Initialize VectorUpserter with configured vector store.
        # 初始化 VectorUpserter，并创建具体的向量库对象。
        
        Args:
            settings: Application settings containing vector_store configuration.
            # settings：应用配置对象，里面包含 vector_store 的配置

            collection_name: Optional collection name to override settings default.
            # collection_name：可选参数，用来覆盖配置里的默认 collection 名
        
        Raises:
            ValueError: If settings are invalid or vector store cannot be created.
            # 如果配置非法，或者向量库创建失败，可能报错
        """
        self.settings = settings
        # 把 settings 保存到实例上，后面可能还会用。

        kwargs = {}
        # 先准备一个空字典 kwargs，后面如果传了 collection_name，就塞进去。

        if collection_name:
            # 如果调用时显式传了 collection_name
            kwargs['collection_name'] = collection_name
            # 就把它放进 kwargs，交给工厂方法使用
        
        self.vector_store = VectorStoreFactory.create(settings, **kwargs)
        # 调用 VectorStoreFactory.create(...)
        # 根据 settings 和可选 collection_name 创建出真正的 vector_store 实例
        # 比如可能是 ChromaStore、QdrantStore 等
    
    def upsert(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        """Upsert chunks with their vectors to vector store.
        # 主方法：把 chunks 和对应 vectors 一起写进向量库。
        
        Args:
            chunks: List of Chunk objects to store.
            # chunks：要写入的 chunk 列表

            vectors: List of embedding vectors (same order and length as chunks).
            # vectors：每个 chunk 对应的向量列表
            # 顺序必须和 chunks 一一对应，长度也必须一样

            trace: Optional TraceContext for observability (reserved for Stage F).
            # trace：可选的追踪对象，用于可观测性
        
        Returns:
            List of generated chunk IDs (same order as input chunks).
            # 返回生成好的 chunk_id 列表，顺序和输入 chunks 一致
        
        Raises:
            ValueError: If chunks and vectors lengths don't match, or if required
                       metadata fields are missing.
            # 如果 chunks 和 vectors 长度不一致，或者 chunk 缺少必须 metadata，会报 ValueError

            RuntimeError: If vector store upsert operation fails.
            # 如果底层向量库写入失败，会报 RuntimeError
        
        Example:
            >>> chunks = [Chunk(...), Chunk(...)]
            >>> vectors = [[0.1, 0.2], [0.3, 0.4]]
            >>> chunk_ids = upserter.upsert(chunks, vectors)
            >>> len(chunk_ids) == len(chunks)  # True
        """
        # Validate input lengths match
        # 先校验 chunks 和 vectors 的长度是否一致
        if len(chunks) != len(vectors):
            # 如果两个列表长度不一样，说明没法一一对应
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match vector count ({len(vectors)})"
            )
            # 抛出 ValueError，并明确报出两边长度是多少
        
        if not chunks:
            # 如果 chunks 是空列表，也不允许写入
            raise ValueError("Cannot upsert empty chunks list")
            # 直接报错，禁止空写入
        
        # Generate stable chunk IDs and build records
        # 下面开始生成 chunk_id，并构建要写入向量库的 records
        records = []
        # records：保存最终要交给 vector_store.upsert(...) 的记录列表

        chunk_ids = []
        # chunk_ids：保存每个 chunk 对应生成出来的 chunk_id，最后作为返回值
        
        for chunk, vector in zip(chunks, vectors):
            # 把 chunks 和 vectors 按顺序一一配对遍历
            # zip 的意思就是第一个 chunk 配第一个 vector，以此类推

            # Generate deterministic chunk ID
            # 先为当前 chunk 生成确定性的 chunk_id
            chunk_id = self._generate_chunk_id(chunk)
            # 调用内部方法 _generate_chunk_id，根据 metadata 和 text 生成稳定 ID

            chunk_ids.append(chunk_id)
            # 把当前 chunk_id 放到 chunk_ids 列表里
            
            # Build storage record
            # 构建一个可写入向量库的 record
            record = {
                "id": chunk_id,
                # 记录的唯一 ID，就是刚刚生成的 chunk_id

                "vector": vector,
                # 当前 chunk 对应的 dense 向量

                "metadata": {
                    **chunk.metadata,  # Preserve all original metadata
                    # 把 chunk 原本 metadata 里的所有字段先原样保留

                    "text": chunk.text,  # Store text for retrieval
                    # 再额外放一个 text 字段，把 chunk 正文也存进去
                    # 这样后续检索命中后可以直接拿正文

                    "chunk_id": chunk_id,  # Redundant but useful for queries
                    # 再额外存一份 chunk_id 到 metadata 里
                    # 虽然和外层 id 重复，但查询和调试时会更方便
                },
            }
            records.append(record)
            # 把当前 record 放到 records 列表里
        
        # Perform idempotent upsert
        # 下面正式执行幂等 upsert 写入
        try:
            self.vector_store.upsert(records, trace=trace)
            # 调用底层 vector_store 的 upsert 方法，把所有 records 写进去
            # trace 也原样传下去，方便后续记录追踪信息
        except Exception as e:
            # 如果底层向量库报了任何异常，就进这里
            raise RuntimeError(
                f"Vector store upsert failed: {str(e)}"
            ) from e
            # 把底层异常包装成 RuntimeError 抛出去
            # from e 表示保留原始异常链，方便调试
        
        return chunk_ids
        # 如果成功，返回全部生成好的 chunk_id 列表
    
    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """Generate deterministic chunk ID from content.
        # 内部辅助方法：根据 chunk 内容和 metadata 生成确定性的 chunk_id。
        
        Args:
            chunk: Chunk object to generate ID for.
            # 输入：一个 Chunk 对象
        
        Returns:
            Stable chunk ID string.
            # 输出：一个稳定的 chunk_id 字符串
        
        Raises:
            ValueError: If required metadata fields are missing.
            # 如果缺少必须 metadata，就报 ValueError
        """
        # Validate required metadata
        # 先检查生成 chunk_id 所需的 metadata 是否都存在
        if "source_path" not in chunk.metadata:
            # 如果没有 source_path
            raise ValueError("Chunk metadata must contain 'source_path'")
            # 直接报错，因为没有 source_path 就没法生成规范 ID

        if "chunk_index" not in chunk.metadata:
            # 如果没有 chunk_index
            raise ValueError("Chunk metadata must contain 'chunk_index'")
            # 也直接报错，因为 chunk_index 也是 ID 组成部分
        
        source_path = chunk.metadata["source_path"]
        # 取出 source_path，用于做哈希

        chunk_index = chunk.metadata["chunk_index"]
        # 取出 chunk_index，用于拼接到 ID 中
        
        # Compute stable hashes
        # 下面开始计算稳定哈希
        source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
        # source_path 先转成 utf-8 字节，再做 sha256，再转成十六进制字符串，再取前 8 位

        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
        # chunk.text 也同样做 sha256，取前 8 位
        
        # Format: {source_hash}_{index:04d}_{content_hash}
        # 按约定格式拼接 chunk_id
        chunk_id = f"{source_hash}_{chunk_index:04d}_{content_hash}"
        # 这里的 {chunk_index:04d} 表示把 chunk_index 格式化成 4 位，不够前面补 0
        # 比如 7 会变成 0007
        
        return chunk_id
        # 返回生成好的 chunk_id
    
    def upsert_batch(
        self,
        batches: List[tuple[List[Chunk], List[List[float]]]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        """Upsert multiple batches of chunks and vectors.
        # 批量写入方法：一次接收多个 batch，再统一写入。
        
        This is a convenience method for processing outputs from BatchProcessor.
        # 这是一个方便方法，主要给 BatchProcessor 的输出结果使用。

        All batches are flattened and processed in a single upsert operation
        # 它会先把所有 batch 拍平，

        to maintain ordering and reduce vector store round trips.
        # 然后做一次统一 upsert，这样可以保持顺序，也减少和向量库的来回交互次数。
        
        Args:
            batches: List of (chunks, vectors) tuples from batch processing.
            # batches：一个列表，里面每个元素都是 (chunks, vectors) 这样的二元组

            trace: Optional TraceContext for observability.
            # trace：可选追踪对象
        
        Returns:
            List of all generated chunk IDs in order.
            # 返回所有生成的 chunk_id，顺序和拍平后的输入一致
        
        Example:
            >>> batch1 = ([chunk1, chunk2], [[0.1, 0.2], [0.3, 0.4]])
            >>> batch2 = ([chunk3], [[0.5, 0.6]])
            >>> chunk_ids = upserter.upsert_batch([batch1, batch2])
            >>> len(chunk_ids)  # 3
        """
        # Flatten all batches
        # 下面先把所有 batch 展平成两个大列表
        all_chunks = []
        # 用来保存所有 batch 里的 chunks

        all_vectors = []
        # 用来保存所有 batch 里的 vectors
        
        for chunks, vectors in batches:
            # 逐个取出每个 batch 的 chunks 和 vectors
            all_chunks.extend(chunks)
            # 把当前 batch 的 chunks 全部追加到总 all_chunks 里

            all_vectors.extend(vectors)
            # 把当前 batch 的 vectors 全部追加到总 all_vectors 里
        
        # Single upsert operation
        # 最后走一次统一 upsert
        return self.upsert(all_chunks, all_vectors, trace=trace)
        # 直接复用前面的 upsert 方法，把拍平后的 chunks 和 vectors 一次性写入