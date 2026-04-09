"""Batch Processor for orchestrating dense and sparse encoding.
# 这个文件的说明：它是一个“批处理调度器”，负责把 chunk 分批，
# 然后统一协调 dense 编码和 sparse 编码两条流程。

This module implements the Batch Processor component of the Ingestion Pipeline,
# 这一句说明它属于 Ingestion Pipeline（数据摄取流水线）中的一个组件。

responsible for coordinating the encoding workflow and managing batch operations.
# 它的职责有两个：
# 1）协调编码流程
# 2）管理批量处理

Design Principles:
# 下面列的是这个模块的设计原则。

- Orchestration: Coordinates DenseEncoder and SparseEncoder in unified workflow
# 编排：把 DenseEncoder 和 SparseEncoder 放进同一个统一流程里调度

- Config-Driven: Batch size from settings, not hardcoded
# 配置驱动：批大小应该来自配置，不应该写死

- Observable: Records batch timing and statistics via TraceContext
# 可观测：如果传了 trace，就记录每批耗时和统计信息

- Error Handling: Individual batch failures don't crash entire pipeline
# 错误处理：某一批失败，不会让整个流程直接崩掉

- Deterministic: Same inputs produce same batching and results
# 确定性：同样的输入，会得到同样的分批结果和输出
"""

from typing import List, Dict, Any, Optional, Tuple
# 从 typing 导入类型标注工具：
# List：列表
# Dict：字典
# Any：任意类型
# Optional：可为空
# Tuple：元组
# 注意：这里的 Tuple 在本文件里其实没用到。

import time
# 导入 time 模块，用来统计开始时间、结束时间、耗时。

from dataclasses import dataclass
# 导入 dataclass 装饰器，用来快速定义“纯数据类”。

from src.core.types import Chunk
# 导入 Chunk 类型，表示要处理的文本块对象。

from src.ingestion.embedding.dense_encoder import DenseEncoder
# 导入 DenseEncoder，负责生成 dense embedding（稠密向量）。

from src.ingestion.embedding.sparse_encoder import SparseEncoder
# 导入 SparseEncoder，负责生成 sparse statistics / sparse representation（稀疏统计/表示）。


@dataclass
# 用 dataclass 装饰这个类，表示它主要是拿来装数据的，不强调复杂行为。
class BatchResult:
    """Result of batch processing operation.
    # 这个类表示“一次批处理”的最终结果。
    
    Attributes:
        dense_vectors: List of dense embeddings (one per chunk)
        # dense_vectors：每个 chunk 对应一个 dense 向量

        sparse_stats: List of term statistics (one per chunk)
        # sparse_stats：每个 chunk 对应一个 sparse 统计结果

        batch_count: Number of batches processed
        # batch_count：一共处理了多少批

        total_time: Total processing time in seconds
        # total_time：总耗时，单位秒

        successful_chunks: Number of successfully processed chunks
        # successful_chunks：成功处理的 chunk 数量

        failed_chunks: Number of chunks that failed processing
        # failed_chunks：失败的 chunk 数量
    """
    dense_vectors: List[List[float]]
    # 保存所有 dense 向量，外层 list 是多个 chunk，内层 list 是单个 chunk 的向量值

    sparse_stats: List[Dict[str, Any]]
    # 保存所有 sparse 统计信息，通常每个 chunk 对应一个字典

    batch_count: int
    # 一共分成了多少个 batch

    total_time: float
    # 整个 process 运行的总耗时

    successful_chunks: int
    # 成功处理的 chunk 数

    failed_chunks: int
    # 失败处理的 chunk 数


class BatchProcessor:
    """Orchestrates batch processing of chunks through encoding pipeline.
    # 这个类是批处理调度器，负责把 chunks 分批并送去编码流程。
    
    This processor manages the workflow of converting chunks into both dense
    # 这句说明：它负责把 chunks 转成 dense 表示，

    and sparse representations. It divides chunks into batches, drives the
    # 同时也转成 sparse 表示。它会先把 chunks 分批，再驱动编码器执行。

    encoders, and collects timing metrics.
    # 最后还会收集时间统计信息。
    
    Design:
    # 下面是这个类的设计特点。

    - Stateless: No state maintained between process() calls
    # 无状态：两次 process() 调用之间不保留处理状态

    - Parallel Encodings: Dense and sparse encoding happen independently
    # 并行含义偏逻辑层面：dense 和 sparse 两类编码彼此独立
    # 注意：这里代码里不是并发执行，只是“分别执行两个独立编码步骤”

    - Metrics Collection: Records batch-level timing for observability
    # 指标收集：记录每一批的耗时，方便观测

    - Order Preservation: Output order matches input chunk order
    # 顺序保持：输出顺序和输入 chunk 的顺序一致
    
    Example:
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> dense_encoder = DenseEncoder(embedding, batch_size=2)
        >>> sparse_encoder = SparseEncoder()
        >>> 
        >>> processor = BatchProcessor(
        ...     dense_encoder=dense_encoder,
        ...     sparse_encoder=sparse_encoder,
        ...     batch_size=2
        ... )
        >>> 
        >>> chunks = [
        ...     Chunk(id="1", text="Hello", metadata={}),
        ...     Chunk(id="2", text="World", metadata={})
        ... ]
        >>> result = processor.process(chunks)
        >>> len(result.dense_vectors) == len(chunks)  # True
        >>> len(result.sparse_stats) == len(chunks)  # True
    # 这段例子展示了怎么创建 processor、传入 chunks、拿到结果。
    """
    
    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = 100,
    ):
        """Initialize BatchProcessor.
        # 初始化 BatchProcessor。
        
        Args:
            dense_encoder: DenseEncoder instance for embedding generation
            # dense_encoder：传入的稠密编码器

            sparse_encoder: SparseEncoder instance for term statistics
            # sparse_encoder：传入的稀疏编码器

            batch_size: Number of chunks to process per batch (default: 100)
            # batch_size：每批处理多少个 chunk，默认 100
        
        Raises:
            ValueError: If batch_size <= 0
            # 如果 batch_size 小于等于 0，直接报错
        """
        if batch_size <= 0:
            # 先检查 batch_size 合不合法
            raise ValueError(f"batch_size must be positive, got {batch_size}")
            # 如果 batch_size 非法，抛出 ValueError，并把实际值写进报错信息
        
        self.dense_encoder = dense_encoder
        # 把传入的 dense_encoder 保存到实例上，后面 process 时会用

        self.sparse_encoder = sparse_encoder
        # 把传入的 sparse_encoder 保存到实例上

        self.batch_size = batch_size
        # 把批大小保存到实例上
    
    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> BatchResult:
        """Process chunks through dense and sparse encoding pipeline.
        # 主处理方法：把 chunks 走完整个 dense + sparse 编码流程。
        
        Workflow:
        1. Validate inputs
        # 第一步：检查输入是否合法

        2. Create batches from chunks
        # 第二步：把 chunks 按 batch_size 分批

        3. Process each batch through both encoders
        # 第三步：每一批分别过 dense_encoder 和 sparse_encoder

        4. Collect results and timing metrics
        # 第四步：收集结果和耗时统计

        5. Record to TraceContext if provided
        # 第五步：如果传了 trace，就把这些过程记录进去
        
        Args:
            chunks: List of Chunk objects to process
            # 要处理的 chunk 列表

            trace: Optional TraceContext for observability
            # 可选的 trace，用来记录监控/观测数据
        
        Returns:
            BatchResult containing vectors, statistics, and metrics
            # 返回一个 BatchResult，里面装着向量、统计信息和指标
        
        Raises:
            ValueError: If chunks list is empty
            # 如果 chunks 是空列表，报 ValueError

            RuntimeError: If both encoders fail completely
            # 这里文档写了可能抛 RuntimeError，
            # 但当前代码里其实没有主动 raise RuntimeError
        
        Example:
            >>> chunks = [Chunk(id=f"{i}", text=f"Text {i}", metadata={}) 
            ...           for i in range(5)]
            >>> result = processor.process(chunks)
            >>> result.batch_count  # 3 (with batch_size=2)
            >>> result.successful_chunks  # 5
        """
        if not chunks:
            # 如果 chunks 为空，直接拒绝处理
            raise ValueError("Cannot process empty chunks list")
            # 抛出异常，告诉调用方不能处理空列表
        
        start_time = time.time()
        # 记录整个 process 开始时间，后面用来算总耗时
        
        # Create batches
        # 下面开始分批
        batches = self._create_batches(chunks)
        # 调用内部方法 _create_batches，把 chunks 切成多个 batch

        batch_count = len(batches)
        # 记录总共有多少批
        
        # Process all batches
        # 下面初始化用于收集总结果的容器和计数器
        dense_vectors: List[List[float]] = []
        # 用来收集所有批次的 dense 向量结果

        sparse_stats: List[Dict[str, Any]] = []
        # 用来收集所有批次的 sparse 统计结果

        successful_chunks = 0
        # 成功处理的 chunk 数，初始为 0

        failed_chunks = 0
        # 失败的 chunk 数，初始为 0
        
        for batch_idx, batch in enumerate(batches):
            # 逐批处理，batch_idx 是批次下标，batch 是当前这批 chunk 列表
            batch_start = time.time()
            # 记录当前批次开始时间，用来算该批耗时
            
            try:
                # 进入 try，表示这一批的 dense/sparse 编码都放在同一个异常保护块里

                # Dense encoding
                # 先做 dense 编码
                batch_dense = self.dense_encoder.encode(batch, trace=trace)
                # 调用 dense_encoder 的 encode 方法，对当前 batch 做稠密向量编码

                dense_vectors.extend(batch_dense)
                # 把当前批的 dense 结果追加到总结果 dense_vectors 里
                
                # Sparse encoding
                # 再做 sparse 编码
                batch_sparse = self.sparse_encoder.encode(batch, trace=trace)
                # 调用 sparse_encoder 的 encode 方法，对当前 batch 做稀疏编码/统计

                sparse_stats.extend(batch_sparse)
                # 把当前批的 sparse 结果追加到总结果 sparse_stats 里
                
                successful_chunks += len(batch)
                # 当前批两个编码都走完了，就把这批的 chunk 数计入成功数
                
            except Exception as e:
                # 只要当前批处理过程中任意一步报错，就会进这里

                # Record failure but continue with remaining batches
                # 记录失败，但不会让整个流程停掉，后续 batch 继续处理
                failed_chunks += len(batch)
                # 把当前批全部视为失败 chunk，累加到 failed_chunks

                if trace:
                    # 如果传了 trace，就把这一批的错误也记录进去
                    trace.record_stage(
                        f"batch_{batch_idx}_error",
                        # stage 名字写成 batch_序号_error，方便后面查
                        {"error": str(e), "batch_size": len(batch)}
                        # 记录报错信息和当前批大小
                    )
            
            batch_duration = time.time() - batch_start
            # 不管成功还是失败，都会计算当前批从开始到现在的耗时
            
            # Record batch timing if trace available
            # 如果有 trace，就记录当前批的耗时和处理信息
            if trace:
                trace.record_stage(
                    f"batch_{batch_idx}",
                    # 当前批次的 stage 名
                    {
                        "batch_size": len(batch),
                        # 当前批大小

                        "duration_seconds": batch_duration,
                        # 当前批耗时（秒）

                        "chunks_processed": len(batch)
                        # 当前批处理了多少个 chunk
                    }
                )
        
        total_time = time.time() - start_time
        # 所有批处理结束后，计算整个 process 的总耗时
        
        # Record overall processing statistics
        # 如果有 trace，再记录一份全局汇总统计
        if trace:
            trace.record_stage(
                "batch_processing",
                # 总体 stage 名叫 batch_processing
                {
                    "total_chunks": len(chunks),
                    # 输入总 chunk 数

                    "batch_count": batch_count,
                    # 总批次数

                    "batch_size": self.batch_size,
                    # 使用的批大小配置

                    "successful_chunks": successful_chunks,
                    # 成功 chunk 数

                    "failed_chunks": failed_chunks,
                    # 失败 chunk 数

                    "total_time_seconds": total_time
                    # 总耗时（秒）
                }
            )
        
        return BatchResult(
            dense_vectors=dense_vectors,
            # 返回所有 dense 向量结果

            sparse_stats=sparse_stats,
            # 返回所有 sparse 统计结果

            batch_count=batch_count,
            # 返回总批次数

            total_time=total_time,
            # 返回总耗时

            successful_chunks=successful_chunks,
            # 返回成功 chunk 数

            failed_chunks=failed_chunks
            # 返回失败 chunk 数
        )
    
    def _create_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """Divide chunks into batches of specified size.
        # 内部辅助方法：把 chunks 按指定 batch_size 切成多批。
        
        Args:
            chunks: List of chunks to batch
            # 输入：要分批的 chunk 列表
        
        Returns:
            List of batches, where each batch is a list of chunks.
            # 返回值：一个二维列表，里面每个元素都是一批 chunks

            Order is preserved: first batch contains chunks[0:batch_size],
            # 顺序是保留的：第一批就是 chunks[0:batch_size]

            second batch contains chunks[batch_size:2*batch_size], etc.
            # 第二批就是下一段，依次类推
        
        Example:
            >>> chunks = [Chunk(id=f"{i}", text="", metadata={}) for i in range(5)]
            >>> batches = processor._create_batches(chunks)
            >>> len(batches)  # 3 (with batch_size=2)
            >>> [len(b) for b in batches]  # [2, 2, 1]
        """
        batches = []
        # 先创建一个空列表，用来装所有 batch

        for i in range(0, len(chunks), self.batch_size):
            # 从 0 开始，每次步长走 self.batch_size
            # 例如 batch_size=2，i 会是 0, 2, 4 ...

            batch = chunks[i:i + self.batch_size]
            # 用切片取出当前这一批
            # 比如 i=2, batch_size=2，就是 chunks[2:4]

            batches.append(batch)
            # 把当前 batch 放进 batches 里

        return batches
        # 返回所有批次
    
    def get_batch_count(self, total_chunks: int) -> int:
        """Calculate number of batches for given chunk count.
        # 根据总 chunk 数，算出会被分成多少批。
        
        Utility method for planning and testing.
        # 这是个工具方法，主要给规划和测试时使用。
        
        Args:
            total_chunks: Total number of chunks to process
            # 输入：总 chunk 数
        
        Returns:
            Number of batches that will be created
            # 输出：最终会有多少个 batch
        
        Example:
            >>> processor.get_batch_count(5)  # 3 (with batch_size=2)
            >>> processor.get_batch_count(4)  # 2
            >>> processor.get_batch_count(0)  # 0
        """
        if total_chunks <= 0:
            # 如果 chunk 总数小于等于 0，直接返回 0 批
            return 0

        return (total_chunks + self.batch_size - 1) // self.batch_size
        # 这里是“向上取整”的经典写法
        # 作用：算 total_chunks 需要多少批才能装完
        # 例如 total_chunks=5, batch_size=2
        # (5 + 2 - 1) // 2 = 6 // 2 = 3
        # 也就是 5 个 chunk 需要 3 批