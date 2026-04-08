"""Dense Encoder for generating embeddings from text chunks.

This module implements the Dense Encoder component of the Ingestion Pipeline,
responsible for converting text chunks into dense vector representations using
configurable embedding providers.

Design Principles:
- Config-Driven: Uses factory pattern to obtain embedding provider from settings
- Batch Processing: Optimizes API calls through batching
- Observable: Accepts TraceContext for future observability integration
- Error Handling: Individual failures shouldn't crash entire batch
- Deterministic: Same inputs produce same outputs
"""

# 从 typing 里导入类型提示工具：
# List 表示列表，Optional 表示“可以是某个类型，也可以是 None”，Any 表示任意类型
from typing import List, Optional, Any

# 导入项目里的 Chunk 类型，后面 encode() 收到的就是一组 Chunk
from src.core.types import Chunk

# 导入 embedding 抽象基类
# DenseEncoder 不关心你底层用 OpenAI、Azure 还是本地模型，只要求它符合 BaseEmbedding 接口
from src.libs.embedding.base_embedding import BaseEmbedding


class DenseEncoder:
    """Encodes text chunks into dense vectors using BaseEmbedding provider.
    
    This encoder acts as a bridge between the ingestion pipeline and the
    pluggable embedding layer. It handles batching, error recovery, and
    maintains alignment between input chunks and output vectors.
    
    Design:
    - Dependency Injection: Receives BaseEmbedding instance (no direct factory call)
    - Batch-First: Processes all chunks in configurable batch sizes
    - Stateless: No internal state between encode() calls
    
    Example:
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> encoder = DenseEncoder(embedding, batch_size=32)
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world", metadata={})]
        >>> vectors = encoder.encode(chunks)
        >>> print(len(vectors))  # 1
        >>> print(len(vectors[0]))  # dimension (e.g., 1536)
    """
    
    def __init__(
        self,
        embedding: BaseEmbedding,
        batch_size: int = 100,
    ):
        """Initialize DenseEncoder.
        
        Args:
            embedding: Embedding provider instance (from EmbeddingFactory)
            batch_size: Number of chunks to process per API call (default: 100)
        
        Raises:
            ValueError: If batch_size <= 0
        """
        # batch_size 必须大于 0
        # 因为每批处理 0 个或负数个 chunk 是没有意义的
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        # 保存 embedding 对象
        # 后面真正做向量化时会调用 self.embedding.embed(...)
        self.embedding = embedding

        # 保存每批处理的大小
        self.batch_size = batch_size
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """Encode chunks into dense vectors.
        
        This method:
        1. Extracts text from each chunk
        2. Batches texts according to batch_size
        3. Calls embedding.embed() for each batch
        4. Concatenates results maintaining chunk order
        
        Args:
            chunks: List of Chunk objects to encode
            trace: Optional TraceContext for observability (reserved for Stage F)
        
        Returns:
            List of dense vectors (one per chunk, in same order).
            Each vector is a list of floats with dimension matching the embedding model.
        
        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If embedding provider fails for all batches
        
        Example:
            >>> chunks = [
            ...     Chunk(id="1", text="First chunk", metadata={}),
            ...     Chunk(id="2", text="Second chunk", metadata={})
            ... ]
            >>> vectors = encoder.encode(chunks)
            >>> len(vectors) == len(chunks)  # True
        """
        # 如果传进来的 chunks 是空列表，直接报错
        # 因为没有任何内容可以编码
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        # 从每个 chunk 里把 text 提出来
        # 最终 embedding 模型真正吃的是这些字符串，不是整个 Chunk 对象
        texts = [chunk.text for chunk in chunks]
        
        # 检查每个 text 都不能是空的
        # 比如 None、""、"   " 这种都不允许
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunks[i].id}) has empty or whitespace-only text"
                )
        
        # 准备一个总列表，用来收集所有 batch 的向量结果
        all_vectors: List[List[float]] = []
        
        # 按 batch_size 分批处理 texts
        # range(0, len(texts), self.batch_size) 的意思是：
        # 从 0 开始，每次跳 batch_size 个位置
        for batch_start in range(0, len(texts), self.batch_size):
            # 计算这一批的结束位置
            # min(...) 是为了防止最后一批越界
            batch_end = min(batch_start + self.batch_size, len(texts))

            # 切出当前这一批的文本
            batch_texts = texts[batch_start:batch_end]
            
            try:
                # 调用 embedding provider 做真正的向量化
                # texts=batch_texts：把这一批文本传进去
                # trace=trace：把 trace 往下透传，供后续可观测性使用
                batch_vectors = self.embedding.embed(
                    texts=batch_texts,
                    trace=trace,
                )
                
                # 检查返回结果的数量对不对
                # 正常情况：传进去几条文本，就应该返回几个向量
                if len(batch_vectors) != len(batch_texts):
                    raise RuntimeError(
                        f"Embedding provider returned {len(batch_vectors)} vectors "
                        f"for {len(batch_texts)} texts in batch {batch_start}-{batch_end}"
                    )
                
                # 这一批没问题，就把向量追加到总结果里
                all_vectors.extend(batch_vectors)
                
            except Exception as e:
                # 如果这一批失败，就重新抛出一个更清楚的错误
                # 顺便把是哪一批失败了也写进去，方便排查
                raise RuntimeError(
                    f"Failed to encode batch {batch_start}-{batch_end}: {str(e)}"
                ) from e
        
        # 所有批次跑完后，再做一次总校验
        # 最终向量总数应该和 chunks 总数一致
        if len(all_vectors) != len(chunks):
            raise RuntimeError(
                f"Vector count mismatch: got {len(all_vectors)} vectors "
                f"for {len(chunks)} chunks"
            )
        
        # 再检查所有向量的维度是否一致
        # 例如第一个向量是 1536 维，那后面每个向量都应该是 1536 维
        if all_vectors:
            # 先拿第一个向量的长度当标准维度
            expected_dim = len(all_vectors[0])

            # 逐个检查每个向量
            for i, vec in enumerate(all_vectors):
                if len(vec) != expected_dim:
                    raise RuntimeError(
                        f"Inconsistent vector dimensions: vector {i} has "
                        f"{len(vec)} dimensions, expected {expected_dim}"
                    )
        
        # 全部通过检查后，返回最终结果
        # 返回值结构是：
        # [
        #   [0.12, 0.98, ...],   # chunk 1 的向量
        #   [0.33, 0.44, ...],   # chunk 2 的向量
        # ]
        return all_vectors
    
    def get_batch_count(self, num_chunks: int) -> int:
        """Calculate number of batches needed for given chunk count.
        
        Utility method for logging/progress tracking.
        
        Args:
            num_chunks: Number of chunks to encode
        
        Returns:
            Number of batches required
        """
        # 如果 chunk 数小于等于 0，就不需要任何 batch
        if num_chunks <= 0:
            return 0

        # 这是“向上取整”的写法
        # 作用：算出 num_chunks 需要分成多少批
        #
        # 举例：
        # batch_size = 100
        # num_chunks = 250
        # 则结果是 3 批
        return (num_chunks + self.batch_size - 1) // self.batch_size