"""Hybrid Search Engine orchestrating Dense + Sparse retrieval with RRF Fusion.
# 这个文件实现“混合检索引擎”：把 Dense 检索、Sparse 检索、RRF 融合串起来

This module implements the HybridSearch class that combines:
# 这一行说明：这里实现了 HybridSearch 类，它会组合多个组件一起工作
1. QueryProcessor: Preprocess queries and extract keywords/filters
# 组件1：QueryProcessor，负责预处理 query，提取关键词和过滤条件
2. DenseRetriever: Semantic search using embeddings
# 组件2：DenseRetriever，负责语义检索
3. SparseRetriever: Keyword search using BM25
# 组件3：SparseRetriever，负责 BM25 关键词检索
4. RRFFusion: Combine results using Reciprocal Rank Fusion
# 组件4：RRFFusion，负责把两路结果融合起来

Design Principles:
# 下面是在说整体设计原则
- Graceful Degradation: If one retrieval path fails, fall back to the other
# 优雅降级：如果一条检索链路挂了，就退化到另一条，不让整个搜索直接废掉
- Pluggable: All components injected via constructor for testability
# 可插拔：组件都通过构造函数传进来，方便替换和测试
- Observable: TraceContext integration for debugging and monitoring
# 可观测：支持 trace，方便调试和监控
- Config-Driven: Top-k and other parameters read from settings
# 配置驱动：top-k 等参数尽量从 settings 读取
"""

from __future__ import annotations
# 让类型注解延迟解析，避免有些类型在运行时还没定义就报错

import logging
# 导入日志模块

import time
# 导入时间模块，用来统计每个阶段耗时

from concurrent.futures import ThreadPoolExecutor, as_completed
# 导入线程池并发工具：
# ThreadPoolExecutor：线程池
# as_completed：按任务完成顺序迭代 future（这个文件里其实没用到）

from dataclasses import dataclass, field
# 导入 dataclass 工具：
# dataclass：快速定义“主要存数据”的类
# field：给 dataclass 字段设置默认工厂等配置

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
# 导入类型注解工具

from src.core.types import ProcessedQuery, RetrievalResult
# 导入两个核心类型：
# ProcessedQuery：处理后的 query 对象
# RetrievalResult：统一的检索结果对象

if TYPE_CHECKING:
    # 这一段只在类型检查时生效，运行时不会真的导入
    from src.core.query_engine.dense_retriever import DenseRetriever
    # DenseRetriever 类型
    from src.core.query_engine.fusion import RRFFusion
    # RRFFusion 类型
    from src.core.query_engine.query_processor import QueryProcessor
    # QueryProcessor 类型
    from src.core.query_engine.sparse_retriever import SparseRetriever
    # SparseRetriever 类型
    from src.core.settings import Settings
    # Settings 类型

logger = logging.getLogger(__name__)
# 获取当前文件自己的 logger


def _snapshot_results(
    results: Optional[List[RetrievalResult]],
    # 传入检索结果列表，也允许传 None
) -> List[Dict[str, Any]]:
    """Create a serialisable snapshot of retrieval results for trace storage.
    # 把检索结果转成“可序列化”的快照，方便写进 trace / 日志

    Args:
        results: List of RetrievalResult objects.
        # 原始检索结果列表

    Returns:
        List of dicts with chunk_id, score, full text, source.
        # 返回一个字典列表，每条只保留关键字段，方便存日志
    """
    if not results:
        # 如果结果为空或 None，直接返回空列表
        return []
    return [
        {
            "chunk_id": r.chunk_id,
            # 每条结果的 chunk_id
            "score": round(r.score, 4),
            # 分数保留 4 位小数，日志里更清爽
            "text": r.text or "",
            # 文本内容，没有就给空字符串
            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
            # 来源字段优先取 source_path，没有再取 source，还没有就给空字符串
        }
        for r in results
        # 遍历每个 RetrievalResult，转成普通 dict
    ]


@dataclass
# 用 dataclass 定义一个“配置对象”
class HybridSearchConfig:
    """Configuration for HybridSearch.
    # 这个类专门放 HybridSearch 的配置项
    
    Attributes:
        dense_top_k: Number of results from dense retrieval
        # dense 检索返回多少条
        sparse_top_k: Number of results from sparse retrieval
        # sparse 检索返回多少条
        fusion_top_k: Final number of results after fusion
        # 融合后最终保留多少条
        enable_dense: Whether to use dense retrieval
        # 是否开启 dense 检索
        enable_sparse: Whether to use sparse retrieval
        # 是否开启 sparse 检索
        parallel_retrieval: Whether to run retrievals in parallel
        # 是否并行跑两路检索
        metadata_filter_post: Apply metadata filters after fusion (fallback)
        # 是否在融合后再做一层 metadata 过滤兜底
    """
    dense_top_k: int = 20
    # dense 检索默认召回 20 条
    sparse_top_k: int = 20
    # sparse 检索默认召回 20 条
    fusion_top_k: int = 10
    # 最终默认返回 10 条
    enable_dense: bool = True
    # 默认开启 dense
    enable_sparse: bool = True
    # 默认开启 sparse
    parallel_retrieval: bool = True
    # 默认并行跑两路检索
    metadata_filter_post: bool = True
    # 默认在融合后再做一次 metadata 过滤


@dataclass
# 用 dataclass 定义一个“返回详情对象”
class HybridSearchResult:
    """Result of a hybrid search operation.
    # 这个类表示一次混合检索的完整结果，不只是最终结果，还包含调试信息
    
    Attributes:
        results: Final ranked list of RetrievalResults
        # 最终返回给用户的结果列表
        dense_results: Results from dense retrieval (for debugging)
        # dense 路径的原始结果，主要用于调试
        sparse_results: Results from sparse retrieval (for debugging)
        # sparse 路径的原始结果，主要用于调试
        dense_error: Error message if dense retrieval failed
        # dense 路径如果失败了，这里记录错误信息
        sparse_error: Error message if sparse retrieval failed
        # sparse 路径如果失败了，这里记录错误信息
        used_fallback: Whether fallback mode was used
        # 是否触发了降级逻辑
        processed_query: The processed query (for debugging)
        # 预处理后的 query，也留着方便调试
    """
    results: List[RetrievalResult] = field(default_factory=list)
    # 最终结果列表，默认是空列表
    dense_results: Optional[List[RetrievalResult]] = None
    # dense 原始结果，默认 None
    sparse_results: Optional[List[RetrievalResult]] = None
    # sparse 原始结果，默认 None
    dense_error: Optional[str] = None
    # dense 错误信息，默认 None
    sparse_error: Optional[str] = None
    # sparse 错误信息，默认 None
    used_fallback: bool = False
    # 默认没触发 fallback
    processed_query: Optional[ProcessedQuery] = None
    # 默认没有保存 processed_query


class HybridSearch:
    """Hybrid Search Engine combining Dense and Sparse retrieval.
    # 这个类是混合检索引擎本体，负责把 Dense + Sparse + Fusion 串起来
    
    This class orchestrates the complete hybrid search flow:
    # 下面在说明完整执行流程
    1. Query Processing: Extract keywords and filters from raw query
    # 第一步：处理原始 query，提取关键词和过滤条件
    2. Parallel Retrieval: Run Dense and Sparse retrievers concurrently
    # 第二步：并行跑 Dense 和 Sparse 两路检索
    3. Fusion: Combine results using RRF algorithm
    # 第三步：用 RRF 融合两路结果
    4. Post-Filtering: Apply metadata filters if specified
    # 第四步：如果有需要，再做后置 metadata 过滤
    
    Design Principles Applied:
    # 设计原则
    - Graceful Degradation: If one path fails, use results from the other
    # 一条路失败时，就退化到另一条路
    - Pluggable: All components via dependency injection
    # 全部组件都支持依赖注入
    - Observable: TraceContext support for debugging
    # 支持 trace，便于观察各阶段行为
    - Config-Driven: All parameters from settings
    # 参数尽量从配置里取
    
    Example:
        >>> # Initialize components
        # 先初始化各个组件
        >>> query_processor = QueryProcessor()
        # 创建 query 处理器
        >>> dense_retriever = DenseRetriever(settings, embedding_client, vector_store)
        # 创建 dense 检索器
        >>> sparse_retriever = SparseRetriever(settings, bm25_indexer, vector_store)
        # 创建 sparse 检索器
        >>> fusion = RRFFusion(k=60)
        # 创建 RRF 融合器
        >>> 
        >>> # Create HybridSearch
        # 创建 HybridSearch
        >>> hybrid = HybridSearch(
        ...     settings=settings,
        ...     query_processor=query_processor,
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ...     fusion=fusion
        ... )
        # 把所有组件组装起来
        >>> 
        >>> # Search
        # 发起检索
        >>> results = hybrid.search("如何配置 Azure OpenAI？", top_k=10)
        # 搜索一条 query，返回前 10 条结果
    """
    
    def __init__(
        self,
        # 类实例本身
        settings: Optional[Settings] = None,
        # 配置对象，可不传
        query_processor: Optional[QueryProcessor] = None,
        # QueryProcessor，可不传
        dense_retriever: Optional[DenseRetriever] = None,
        # DenseRetriever，可不传
        sparse_retriever: Optional[SparseRetriever] = None,
        # SparseRetriever，可不传
        fusion: Optional[RRFFusion] = None,
        # RRFFusion，可不传
        config: Optional[HybridSearchConfig] = None,
        # HybridSearchConfig，可不传
    ) -> None:
        """Initialize HybridSearch with components.
        # 初始化 HybridSearch，把各个组件装进来
        
        Args:
            settings: Application settings for extracting configuration.
            # settings：配置对象，用来提取默认配置
            query_processor: QueryProcessor for preprocessing queries.
            # query_processor：负责 query 预处理
            dense_retriever: DenseRetriever for semantic search.
            # dense_retriever：负责语义检索
            sparse_retriever: SparseRetriever for keyword search.
            # sparse_retriever：负责关键词检索
            fusion: RRFFusion for combining results.
            # fusion：负责融合结果
            config: Optional HybridSearchConfig. If not provided, extracted from settings.
            # config：也可以直接传配置；不传就从 settings 提取
        
        Note:
            At least one of dense_retriever or sparse_retriever must be provided
            # dense 和 sparse 至少要有一个，不然没法检索
            for search to function. The search will gracefully degrade if one
            # 只要有一个能工作，search 还能继续
            is unavailable or fails.
            # 如果另一个不可用或失败，会自动降级
        """
        self.query_processor = query_processor
        # 保存 query_processor
        self.dense_retriever = dense_retriever
        # 保存 dense_retriever
        self.sparse_retriever = sparse_retriever
        # 保存 sparse_retriever
        self.fusion = fusion
        # 保存 fusion
        
        # Extract config from settings or use provided/default
        # 配置优先用传进来的 config；如果没有，就从 settings 提取
        self.config = config or self._extract_config(settings)
        # 最终拿到 HybridSearchConfig
        
        logger.info(
            f"HybridSearch initialized: dense={self.dense_retriever is not None}, "
            f"sparse={self.sparse_retriever is not None}, "
            f"config={self.config}"
        )
        # 打初始化日志：说明 dense/sparse 有没有配好，以及配置是什么
    
    def _extract_config(self, settings: Optional[Settings]) -> HybridSearchConfig:
        """Extract HybridSearchConfig from Settings.
        # 从 Settings 里提取 HybridSearch 的配置
        
        Args:
            settings: Application settings object.
            # 配置对象
            
        Returns:
            HybridSearchConfig with values from settings or defaults.
            # 返回配置对象；如果 settings 不完整，就用默认值
        """
        if settings is None:
            # 如果根本没传 settings，就直接返回默认配置
            return HybridSearchConfig()
        
        retrieval_config = getattr(settings, 'retrieval', None)
        # 从 settings 上读取 retrieval 配置，没有就给 None
        if retrieval_config is None:
            # 如果 retrieval 配置不存在，也返回默认配置
            return HybridSearchConfig()
        
        return HybridSearchConfig(
            dense_top_k=getattr(retrieval_config, 'dense_top_k', 20),
            # 从 retrieval_config 里取 dense_top_k，没有就用 20
            sparse_top_k=getattr(retrieval_config, 'sparse_top_k', 20),
            # 取 sparse_top_k，没有就用 20
            fusion_top_k=getattr(retrieval_config, 'fusion_top_k', 10),
            # 取 fusion_top_k，没有就用 10
            enable_dense=True,
            # 这里直接默认开启 dense
            enable_sparse=True,
            # 这里直接默认开启 sparse
            parallel_retrieval=True,
            # 默认并行检索
            metadata_filter_post=True,
            # 默认后置 metadata 过滤
        )
    
    def search(
        self,
        # 类实例本身
        query: str,
        # 用户输入的原始 query
        top_k: Optional[int] = None,
        # 这次最多返回多少条，不传就用 config.fusion_top_k
        filters: Optional[Dict[str, Any]] = None,
        # 显式传入的 metadata 过滤条件
        trace: Optional[Any] = None,
        # 可选 trace，用于可观测性
        return_details: bool = False,
        # 是否返回带调试细节的结果对象，而不是只返回最终结果列表
    ) -> List[RetrievalResult] | HybridSearchResult:
        """Perform hybrid search combining Dense and Sparse retrieval.
        # 执行混合检索：Dense + Sparse + Fusion
        
        Args:
            query: The search query string.
            # 原始查询文本
            top_k: Maximum number of results to return. If None, uses config.fusion_top_k.
            # 最多返回多少条
            filters: Optional metadata filters (e.g., {"collection": "docs"}).
            # 可选 metadata 过滤条件
            trace: Optional TraceContext for observability.
            # 可选 trace
            return_details: If True, return HybridSearchResult with debug info.
            # 如果为 True，就返回带调试细节的 HybridSearchResult
        
        Returns:
            If return_details=False: List of RetrievalResult sorted by relevance.
            # 默认只返回最终结果列表
            If return_details=True: HybridSearchResult with full details.
            # 如果 return_details=True，就返回完整细节对象
        
        Raises:
            ValueError: If query is empty or invalid.
            # query 非法时抛 ValueError
            RuntimeError: If both retrievers fail or are unavailable.
            # 两路检索都不可用或都失败时抛 RuntimeError
        
        Example:
            >>> results = hybrid.search("Azure configuration", top_k=5)
            # 发起检索
            >>> for r in results:
            ...     print(f"[{r.score:.4f}] {r.chunk_id}: {r.text[:50]}...")
            # 打印结果
        """
        # Validate query
        # 先校验 query
        if not query or not query.strip():
            # 如果 query 是空，或者全是空白字符
            raise ValueError("Query cannot be empty or whitespace-only")
            # 直接报错
        
        effective_top_k = top_k if top_k is not None else self.config.fusion_top_k
        # 计算这次真正用的 top_k：优先用参数，否则用配置里的 fusion_top_k
        
        logger.debug(f"HybridSearch: query='{query[:50]}...', top_k={effective_top_k}")
        # 打 debug 日志，只展示 query 前 50 个字符
        
        # Step 1: Process query
        # 第一步：先处理 query
        _t0 = time.monotonic()
        # 记录开始时间，用来算耗时
        processed_query = self._process_query(query)
        # 调内部方法处理 query，得到 ProcessedQuery
        _elapsed = (time.monotonic() - _t0) * 1000.0
        # 计算 query processing 耗时，单位毫秒
        if trace is not None:
            # 如果传了 trace，就把这一阶段记录下来
            trace.record_stage("query_processing", {
                "method": "query_processor",
                # 记录这个阶段的方法名
                "original_query": query,
                # 原始 query
                "keywords": processed_query.keywords,
                # 处理后提取出的关键词
            }, elapsed_ms=_elapsed)
            # 把阶段信息和耗时写进 trace
        
        # Merge explicit filters with query-extracted filters
        # 把 query 里提取出的 filters 和调用方显式传入的 filters 合并起来
        merged_filters = self._merge_filters(processed_query.filters, filters)
        # 合并后得到最终 filters
        
        # Step 2: Run retrievals
        # 第二步：跑两路检索
        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
            processed_query=processed_query,
            # 传入处理后的 query
            filters=merged_filters,
            # 传入最终过滤条件
            trace=trace,
            # 传入 trace
        )
        # 返回四个值：dense 结果、sparse 结果、dense 错误、sparse 错误
        
        # Step 3: Handle fallback scenarios
        # 第三步：处理各种降级场景
        used_fallback = False
        # 默认没走 fallback
        if dense_error and sparse_error:
            # 如果 dense 和 sparse 都失败了
            raise RuntimeError(
                f"Both retrieval paths failed. "
                f"Dense error: {dense_error}. Sparse error: {sparse_error}"
            )
            # 直接抛错，因为两条路都不能用
        elif dense_error:
            # 如果 dense 挂了，但 sparse 正常
            logger.warning(f"Dense retrieval failed, using sparse only: {dense_error}")
            # 打 warning，说明改用 sparse 单路结果
            used_fallback = True
            # 标记触发了 fallback
            fused_results = sparse_results or []
            # 直接把 sparse 结果当最终融合结果
        elif sparse_error:
            # 如果 sparse 挂了，但 dense 正常
            logger.warning(f"Sparse retrieval failed, using dense only: {sparse_error}")
            # 打 warning，说明改用 dense 单路结果
            used_fallback = True
            # 标记触发了 fallback
            fused_results = dense_results or []
            # 直接把 dense 结果当最终融合结果
        elif not dense_results and not sparse_results:
            # 如果两路都成功了，但都没查到结果
            fused_results = []
            # 最终结果就是空列表
        else:
            # Step 4: Fuse results
            # 否则，说明至少有一路有结果，并且没有错误，就做融合
            fused_results = self._fuse_results(
                dense_results=dense_results or [],
                # 没有 dense 结果就传空列表
                sparse_results=sparse_results or [],
                # 没有 sparse 结果就传空列表
                top_k=effective_top_k,
                # 融合后最多保留多少条
                trace=trace,
                # 传入 trace
            )
        
        # Step 5: Apply post-fusion metadata filters (if any)
        # 第五步：如果有过滤条件，并且配置允许，就在融合后再做一层 metadata 过滤
        if merged_filters and self.config.metadata_filter_post:
            # 只有 filters 不为空且允许 post filter 时才执行
            fused_results = self._apply_metadata_filters(fused_results, merged_filters)
            # 对融合后的结果做 metadata 过滤
        
        # Step 6: Limit to top_k
        # 第六步：最后再截断成 top_k
        final_results = fused_results[:effective_top_k]
        # 只保留前 effective_top_k 条
        
        logger.debug(f"HybridSearch: returning {len(final_results)} results")
        # 打 debug 日志，说明最终返回多少条
        
        if return_details:
            # 如果调用方想要完整调试信息
            return HybridSearchResult(
                results=final_results,
                # 最终结果
                dense_results=dense_results,
                # dense 原始结果
                sparse_results=sparse_results,
                # sparse 原始结果
                dense_error=dense_error,
                # dense 错误信息
                sparse_error=sparse_error,
                # sparse 错误信息
                used_fallback=used_fallback,
                # 是否用了 fallback
                processed_query=processed_query,
                # 处理后的 query
            )
        
        return final_results
        # 默认只返回最终结果列表
    
    def _process_query(self, query: str) -> ProcessedQuery:
        """Process raw query using QueryProcessor.
        # 用 QueryProcessor 处理原始 query
        
        Args:
            query: Raw query string.
            # 原始 query
            
        Returns:
            ProcessedQuery with keywords and filters.
            # 返回处理后的 query 对象，里面一般有关键词和过滤条件
        """
        if self.query_processor is None:
            # 如果没配置 QueryProcessor
            # Fallback: create basic ProcessedQuery
            # 退化处理：自己构造一个最简单的 ProcessedQuery
            logger.warning("No QueryProcessor configured, using basic tokenization")
            # 打 warning，说明这里只用了最基础的分词方式
            keywords = query.split()
            # 最简单的做法：直接按空格切分成关键词
            return ProcessedQuery(
                original_query=query,
                # 原始 query
                keywords=keywords,
                # 用 split 出来的关键词
                filters={},
                # 没有任何 filters
            )
        
        return self.query_processor.process(query)
        # 如果配置了 QueryProcessor，就调用它的 process 方法
    
    def _merge_filters(
        self,
        query_filters: Dict[str, Any],
        # QueryProcessor 从 query 中提取出来的 filters
        explicit_filters: Optional[Dict[str, Any]],
        # 调用 search() 时显式传入的 filters
    ) -> Dict[str, Any]:
        """Merge query-extracted filters with explicit filters.
        # 合并两类过滤条件
        
        Explicit filters take precedence over query-extracted filters.
        # 规则：显式传入的 filters 优先级更高
        
        Args:
            query_filters: Filters extracted from query by QueryProcessor.
            # query 自带解析出来的 filters
            explicit_filters: Filters passed explicitly to search().
            # 调用方手动传进来的 filters
            
        Returns:
            Merged filter dictionary.
            # 返回合并后的字典
        """
        merged = query_filters.copy() if query_filters else {}
        # 先复制一份 query_filters；如果没有就从空字典开始
        if explicit_filters:
            # 如果显式 filters 存在
            merged.update(explicit_filters)
            # 用 update 覆盖同名键，达到“显式 filters 优先”的效果
        return merged
        # 返回合并结果
    
    def _run_retrievals(
        self,
        processed_query: ProcessedQuery,
        # 处理后的 query
        filters: Optional[Dict[str, Any]],
        # 合并后的过滤条件
        trace: Optional[Any],
        # trace
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """Run Dense and Sparse retrievals.
        # 执行 Dense 和 Sparse 两路检索
        
        Runs in parallel if configured, otherwise sequentially.
        # 如果配置允许就并行，否则顺序执行
        
        Args:
            processed_query: The processed query with keywords.
            # 处理后的 query
            filters: Merged filters to apply.
            # 要应用的过滤条件
            trace: Optional TraceContext.
            # trace
            
        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error).
            # 返回四元组：dense 结果、sparse 结果、dense 错误、sparse 错误
        """
        dense_results: Optional[List[RetrievalResult]] = None
        # 先把 dense_results 初始化为空
        sparse_results: Optional[List[RetrievalResult]] = None
        # 先把 sparse_results 初始化为空
        dense_error: Optional[str] = None
        # 先把 dense_error 初始化为空
        sparse_error: Optional[str] = None
        # 先把 sparse_error 初始化为空
        
        # Determine what to run
        # 先判断这次到底要不要跑 dense / sparse
        run_dense = (
            self.config.enable_dense 
            and self.dense_retriever is not None
        )
        # 只有配置允许 + dense_retriever 已配置，才跑 dense
        
        run_sparse = (
            self.config.enable_sparse 
            and self.sparse_retriever is not None
            and processed_query.keywords  # Need keywords for sparse
        )
        # 只有配置允许 + sparse_retriever 已配置 + 有关键词，才跑 sparse
        
        if not run_dense and not run_sparse:
            # 如果两路都不能跑
            # Nothing to run
            # 根本没东西可跑
            if self.dense_retriever is None and self.sparse_retriever is None:
                # 如果原因是两个 retriever 都没配置
                dense_error = "No retriever configured"
                # 给 dense_error 写上错误信息
                sparse_error = "No retriever configured"
                # 给 sparse_error 写上错误信息
            return dense_results, sparse_results, dense_error, sparse_error
            # 直接返回
        
        if self.config.parallel_retrieval and run_dense and run_sparse:
            # 如果配置允许并行，并且两路都能跑
            # Run in parallel
            # 那就并行执行
            dense_results, sparse_results, dense_error, sparse_error = (
                self._run_parallel_retrievals(processed_query, filters, trace)
            )
            # 调并行执行的方法
        else:
            # Run sequentially
            # 否则就顺序执行
            if run_dense:
                # 如果可以跑 dense
                dense_results, dense_error = self._run_dense_retrieval(
                    processed_query.original_query, filters, trace
                )
                # 执行 dense 检索
            
            if run_sparse:
                # 如果可以跑 sparse
                sparse_results, sparse_error = self._run_sparse_retrieval(
                    processed_query.keywords, filters, trace
                )
                # 执行 sparse 检索
        
        return dense_results, sparse_results, dense_error, sparse_error
        # 返回四元组
    
    def _run_parallel_retrievals(
        self,
        processed_query: ProcessedQuery,
        # 处理后的 query
        filters: Optional[Dict[str, Any]],
        # 过滤条件
        trace: Optional[Any],
        # trace
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """Run Dense and Sparse retrievals in parallel using ThreadPoolExecutor.
        # 用线程池并行执行 Dense 和 Sparse 两路检索
        
        Args:
            processed_query: The processed query.
            # 处理后的 query
            filters: Filters to apply.
            # 过滤条件
            trace: Optional TraceContext.
            # trace
            
        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error).
            # 返回四元组
        """
        dense_results: Optional[List[RetrievalResult]] = None
        # 初始化 dense_results
        sparse_results: Optional[List[RetrievalResult]] = None
        # 初始化 sparse_results
        dense_error: Optional[str] = None
        # 初始化 dense_error
        sparse_error: Optional[str] = None
        # 初始化 sparse_error
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 创建一个线程池，同时最多开 2 个 worker
            futures = {}
            # 准备一个字典，存不同任务对应的 future
            
            # Submit dense retrieval
            # 提交 dense 检索任务
            futures['dense'] = executor.submit(
                self._run_dense_retrieval,
                processed_query.original_query,
                filters,
                trace,
            )
            # 在线程池中异步执行 _run_dense_retrieval
            
            # Submit sparse retrieval
            # 提交 sparse 检索任务
            futures['sparse'] = executor.submit(
                self._run_sparse_retrieval,
                processed_query.keywords,
                filters,
                trace,
            )
            # 在线程池中异步执行 _run_sparse_retrieval
            
            # Collect results
            # 收集两个任务的执行结果
            for name, future in futures.items():
                # 遍历每个 future
                try:
                    results, error = future.result(timeout=30)
                    # 等最多 30 秒拿结果；结果格式是 (results, error)
                    if name == 'dense':
                        # 如果这是 dense 任务
                        dense_results = results
                        # 保存 dense 结果
                        dense_error = error
                        # 保存 dense 错误
                    else:
                        # 否则就是 sparse 任务
                        sparse_results = results
                        # 保存 sparse 结果
                        sparse_error = error
                        # 保存 sparse 错误
                except Exception as e:
                    # 如果 future.result 本身抛异常（比如超时、线程里未捕获异常）
                    error_msg = f"{name} retrieval failed with exception: {e}"
                    # 拼一个清楚的错误信息
                    logger.error(error_msg)
                    # 打 error 日志
                    if name == 'dense':
                        # 如果失败的是 dense
                        dense_error = error_msg
                        # 记录到 dense_error
                    else:
                        # 否则失败的是 sparse
                        sparse_error = error_msg
                        # 记录到 sparse_error
        
        return dense_results, sparse_results, dense_error, sparse_error
        # 返回四元组
    
    def _run_dense_retrieval(
        self,
        query: str,
        # 原始 query 文本
        filters: Optional[Dict[str, Any]],
        # 过滤条件
        trace: Optional[Any],
        # trace
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """Run dense retrieval with error handling.
        # 执行 dense 检索，并做异常处理
        
        Args:
            query: Original query string.
            # 原始 query
            filters: Filters to apply.
            # 过滤条件
            trace: Optional TraceContext.
            # trace
            
        Returns:
            Tuple of (results, error). If successful, error is None.
            # 返回 (结果, 错误信息)，成功时 error 为 None
        """
        if self.dense_retriever is None:
            # 如果根本没配置 dense_retriever
            return None, "Dense retriever not configured"
            # 直接返回错误
        
        try:
            # 用 try 包起来，防止 dense 检索报错
            _t0 = time.monotonic()
            # 记录开始时间
            results = self.dense_retriever.retrieve(
                query=query,
                # 传原始 query 给 dense 检索器
                top_k=self.config.dense_top_k,
                # dense 路径取多少条，用 config 里的 dense_top_k
                filters=filters,
                # 传过滤条件
                trace=trace,
                # 传 trace
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            # 计算耗时，单位毫秒
            if trace is not None:
                # 如果传了 trace，就记录 dense 阶段
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    # 记录方法名
                    "provider": getattr(self.dense_retriever, 'provider_name', 'unknown'),
                    # 记录 provider 名字，如果对象没有 provider_name 就写 unknown
                    "top_k": self.config.dense_top_k,
                    # 记录 top_k
                    "result_count": len(results) if results else 0,
                    # 记录返回条数
                    "chunks": _snapshot_results(results),
                    # 记录结果快照
                }, elapsed_ms=_elapsed)
            return results, None
            # 成功的话返回结果和 None 错误
        except Exception as e:
            # dense 检索过程中出任何异常都进这里
            error_msg = f"Dense retrieval error: {e}"
            # 拼一个错误信息
            logger.error(error_msg)
            # 打 error 日志
            if trace is not None:
                # 如果有 trace，就把失败信息也记录进去
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    # 方法名
                    "error": error_msg,
                    # 错误信息
                    "result_count": 0,
                    # 没结果
                })
            return None, error_msg
            # 返回 None 结果 + 错误信息
    
    def _run_sparse_retrieval(
        self,
        keywords: List[str],
        # 关键词列表
        filters: Optional[Dict[str, Any]],
        # 过滤条件
        trace: Optional[Any],
        # trace
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """Run sparse retrieval with error handling.
        # 执行 sparse 检索，并做异常处理
        
        Args:
            keywords: List of keywords from QueryProcessor.
            # QueryProcessor 处理出来的关键词
            filters: Filters to apply.
            # 过滤条件
            trace: Optional TraceContext.
            # trace
            
        Returns:
            Tuple of (results, error). If successful, error is None.
            # 返回 (结果, 错误信息)，成功时 error 为 None
        """
        if self.sparse_retriever is None:
            # 如果没配置 sparse_retriever
            return None, "Sparse retriever not configured"
            # 直接返回错误
        
        if not keywords:
            # 如果关键词列表为空
            return [], None  # No keywords, return empty (not an error)
            # 这里不算错误，只是返回空结果
        
        try:
            # 用 try 包起来，防止 sparse 检索报错
            # Extract collection from filters if present
            # 如果 filters 里有 collection，就提取出来传给 sparse retriever
            collection = filters.get('collection') if filters else None
            # 没有 filters 就给 None
            
            _t0 = time.monotonic()
            # 记录开始时间
            results = self.sparse_retriever.retrieve(
                keywords=keywords,
                # 传关键词列表
                top_k=self.config.sparse_top_k,
                # sparse 路径取多少条，用 config 里的 sparse_top_k
                collection=collection,
                # 传 collection
                trace=trace,
                # 传 trace
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            # 计算耗时
            if trace is not None:
                # 如果有 trace，就记录 sparse 阶段
                trace.record_stage("sparse_retrieval", {
                    "method": "bm25",
                    # 这里明确方法是 bm25
                    "keyword_count": len(keywords),
                    # 记录关键词个数
                    "top_k": self.config.sparse_top_k,
                    # 记录 top_k
                    "result_count": len(results) if results else 0,
                    # 记录返回条数
                    "chunks": _snapshot_results(results),
                    # 记录结果快照
                }, elapsed_ms=_elapsed)
            return results, None
            # 成功返回结果和 None 错误
        except Exception as e:
            # sparse 检索出错时进这里
            error_msg = f"Sparse retrieval error: {e}"
            # 拼错误信息
            logger.error(error_msg)
            # 打 error 日志
            return None, error_msg
            # 返回 None 结果 + 错误信息
    
    def _fuse_results(
        self,
        dense_results: List[RetrievalResult],
        # dense 路径返回的结果列表
        sparse_results: List[RetrievalResult],
        # sparse 路径返回的结果列表
        top_k: int,
        # 融合后最多保留多少条
        trace: Optional[Any],
        # trace
    ) -> List[RetrievalResult]:
        """Fuse Dense and Sparse results using RRF.
        # 用 RRF 融合 dense 和 sparse 的结果
        
        Args:
            dense_results: Results from dense retrieval.
            # dense 结果
            sparse_results: Results from sparse retrieval.
            # sparse 结果
            top_k: Number of results to return after fusion.
            # 融合后返回多少条
            trace: Optional TraceContext.
            # trace
            
        Returns:
            Fused and ranked list of RetrievalResults.
            # 返回融合并排序后的结果列表
        """
        if self.fusion is None:
            # 如果没配置 fusion
            # Fallback: interleave results (simple round-robin)
            # 降级方案：不用 RRF，改成简单交替插入结果
            logger.warning("No fusion configured, using simple interleave")
            # 打 warning
            return self._interleave_results(dense_results, sparse_results, top_k)
            # 调交替插入的兜底方法
        
        # Build ranking lists for RRF
        # 构造传给 RRF 的 ranking_lists
        ranking_lists = []
        # 先准备一个空列表
        if dense_results:
            # 如果 dense 有结果
            ranking_lists.append(dense_results)
            # 把 dense 结果列表加入 ranking_lists
        if sparse_results:
            # 如果 sparse 有结果
            ranking_lists.append(sparse_results)
            # 把 sparse 结果列表加入 ranking_lists
        
        if not ranking_lists:
            # 如果两边都没有结果
            return []
            # 直接返回空列表
        
        if len(ranking_lists) == 1:
            # 如果只有一路有结果
            # Only one source, no fusion needed
            # 只有一个来源时，根本不需要融合
            return ranking_lists[0][:top_k]
            # 直接截断返回
        
        _t0 = time.monotonic()
        # 记录融合开始时间
        fused = self.fusion.fuse(
            ranking_lists=ranking_lists,
            # 把多个排序列表传进去
            top_k=top_k,
            # 融合后最多取多少条
            trace=trace,
            # trace
        )
        _elapsed = (time.monotonic() - _t0) * 1000.0
        # 计算融合耗时
        if trace is not None:
            # 如果有 trace，就把 fusion 阶段记录进去
            trace.record_stage("fusion", {
                "method": "rrf",
                # 记录融合方法是 rrf
                "input_lists": len(ranking_lists),
                # 记录输入了几路结果列表
                "top_k": top_k,
                # 记录 top_k
                "result_count": len(fused),
                # 记录融合后结果数
                "chunks": _snapshot_results(fused),
                # 记录融合后结果快照
            }, elapsed_ms=_elapsed)
        return fused
        # 返回融合好的结果
    
    def _interleave_results(
        self,
        dense_results: List[RetrievalResult],
        # dense 结果
        sparse_results: List[RetrievalResult],
        # sparse 结果
        top_k: int,
        # 最多返回多少条
    ) -> List[RetrievalResult]:
        """Simple interleave fallback when no fusion is configured.
        # 当没有 fusion 组件时，使用“交替插入”作为简单兜底策略
        
        Args:
            dense_results: Results from dense retrieval.
            # dense 结果
            sparse_results: Results from sparse retrieval.
            # sparse 结果
            top_k: Maximum results to return.
            # 最多返回多少条
            
        Returns:
            Interleaved results, deduped by chunk_id.
            # 返回交替插入后的结果，并按 chunk_id 去重
        """
        seen_ids = set()
        # 用来记录已经加入过的 chunk_id，避免重复
        interleaved = []
        # 最终结果列表
        
        d_idx, s_idx = 0, 0
        # dense 和 sparse 的当前读取下标
        while len(interleaved) < top_k and (d_idx < len(dense_results) or s_idx < len(sparse_results)):
            # 只要还没够 top_k，并且至少还有一边有剩余结果，就继续循环
            # Alternate between dense and sparse
            # 轮流从 dense 和 sparse 各取一条
            if d_idx < len(dense_results):
                # 如果 dense 还有剩余
                r = dense_results[d_idx]
                # 取当前 dense 结果
                d_idx += 1
                # dense 下标后移
                if r.chunk_id not in seen_ids:
                    # 如果这条还没出现过
                    seen_ids.add(r.chunk_id)
                    # 记到 seen_ids 里
                    interleaved.append(r)
                    # 加入最终结果
            
            if len(interleaved) >= top_k:
                # 如果已经够 top_k 了，就提前退出
                break
            
            if s_idx < len(sparse_results):
                # 如果 sparse 还有剩余
                r = sparse_results[s_idx]
                # 取当前 sparse 结果
                s_idx += 1
                # sparse 下标后移
                if r.chunk_id not in seen_ids:
                    # 如果这条还没出现过
                    seen_ids.add(r.chunk_id)
                    # 记到 seen_ids 里
                    interleaved.append(r)
                    # 加入最终结果
        
        return interleaved
        # 返回交替插入后的结果
    
    def _apply_metadata_filters(
        self,
        results: List[RetrievalResult],
        # 要过滤的结果列表
        filters: Dict[str, Any],
        # 过滤条件
    ) -> List[RetrievalResult]:
        """Apply metadata filters to results (post-fusion fallback).
        # 对结果做 metadata 过滤，这里是“融合后兜底过滤”
        
        This is a backup filter mechanism for cases where the underlying
        # 这是备份机制：防止底层存储没完全支持这些 filter 语法

        storage doesn't fully support the filter syntax.
        # 底层没过滤干净时，这里再补一刀
        
        Args:
            results: Results to filter.
            # 要过滤的结果
            filters: Filter conditions to apply.
            # 要应用的过滤条件
            
        Returns:
            Filtered results.
            # 过滤后的结果
        """
        if not filters:
            # 如果没有过滤条件
            return results
            # 原样返回
        
        filtered = []
        # 准备一个空列表装过滤后的结果
        for result in results:
            # 遍历每条结果
            if self._matches_filters(result.metadata, filters):
                # 如果这条结果的 metadata 满足过滤条件
                filtered.append(result)
                # 就加入 filtered
        
        return filtered
        # 返回过滤后的结果
    
    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        # 单条结果的 metadata
        filters: Dict[str, Any],
        # 过滤条件
    ) -> bool:
        """Check if metadata matches all filter conditions.
        # 检查一条 metadata 是否满足所有过滤条件
        
        Args:
            metadata: Result metadata.
            # 结果的 metadata
            filters: Filter conditions.
            # 过滤条件
            
        Returns:
            True if all filters match, False otherwise.
            # 全部满足返回 True，否则返回 False
        """
        for key, value in filters.items():
            # 遍历每一条过滤条件
            if key == "collection":
                # 如果过滤的是 collection
                # Collection might be in different metadata keys
                # collection 可能存在于不同的 metadata 字段名里
                meta_collection = (
                    metadata.get("collection") 
                    or metadata.get("source_collection")
                )
                # 优先取 metadata["collection"]，没有就取 source_collection
                if meta_collection != value:
                    # 如果 collection 不相等
                    return False
                    # 不匹配，直接返回 False
            elif key == "doc_type":
                # 如果过滤的是文档类型
                if metadata.get("doc_type") != value:
                    # doc_type 不相等就不通过
                    return False
            elif key == "tags":
                # 如果过滤的是 tags
                # Tags is a list - check intersection
                # tags 一般是列表，所以这里不是全等，而是看有没有交集
                meta_tags = metadata.get("tags", [])
                # 取结果里的 tags，没有就给空列表
                if not isinstance(value, list):
                    # 如果过滤条件里的 value 不是 list
                    value = [value]
                    # 那就包装成单元素列表，统一处理
                if not set(meta_tags) & set(value):
                    # 如果两边集合没有交集
                    return False
                    # 不通过
            elif key == "source_path":
                # 如果过滤的是 source_path
                # Partial match for path
                # 这里做的是“部分匹配”，不是全等
                source = metadata.get("source_path", "")
                # 取 source_path，没有就给空字符串
                if value not in source:
                    # 只要 value 不是 source 的子串，就不通过
                    return False
            else:
                # Generic exact match
                # 其他字段统一走“精确相等匹配”
                if metadata.get(key) != value:
                    # 不相等就不通过
                    return False
        
        return True
        # 全部过滤条件都通过了，返回 True


def create_hybrid_search(
    settings: Optional[Settings] = None,
    # 配置对象
    query_processor: Optional[QueryProcessor] = None,
    # QueryProcessor，可选
    dense_retriever: Optional[DenseRetriever] = None,
    # DenseRetriever，可选
    sparse_retriever: Optional[SparseRetriever] = None,
    # SparseRetriever，可选
    fusion: Optional[RRFFusion] = None,
    # RRFFusion，可选
) -> HybridSearch:
    """Factory function to create HybridSearch with default components.
    # 一个工厂函数：方便快速创建 HybridSearch
    
    This is a convenience function that creates a HybridSearch with
    # 它是个便捷函数

    default RRFFusion if not provided.
    # 如果没传 fusion，就自动创建默认 RRFFusion
    
    Args:
        settings: Application settings.
        # 配置对象
        query_processor: QueryProcessor instance.
        # QueryProcessor 实例
        dense_retriever: DenseRetriever instance.
        # DenseRetriever 实例
        sparse_retriever: SparseRetriever instance.
        # SparseRetriever 实例
        fusion: RRFFusion instance. If None, creates default with k=60.
        # RRFFusion 实例；如果不传，就默认创建一个 k=60 的
        
    Returns:
        Configured HybridSearch instance.
        # 返回配置好的 HybridSearch
    
    Example:
        >>> hybrid = create_hybrid_search(
        ...     settings=settings,
        ...     query_processor=QueryProcessor(),
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ... )
        # 用工厂函数快速创建 HybridSearch
    """
    # Create default fusion if not provided
    # 如果没传 fusion，就自己创建一个默认的 fusion
    if fusion is None:
        # 只有 fusion 为 None 时才创建
        from src.core.query_engine.fusion import RRFFusion
        # 延迟导入 RRFFusion，避免循环依赖
        rrf_k = 60
        # 默认 RRF 的 k 参数是 60
        if settings is not None:
            # 如果有 settings，就尝试从配置里读取 rrf_k
            retrieval_config = getattr(settings, 'retrieval', None)
            # 读取 retrieval 配置
            if retrieval_config is not None:
                # 如果 retrieval 配置存在
                rrf_k = getattr(retrieval_config, 'rrf_k', 60)
                # 优先用配置里的 rrf_k，没有就仍然是 60
        fusion = RRFFusion(k=rrf_k)
        # 创建 RRFFusion 实例
    
    return HybridSearch(
        settings=settings,
        # 传 settings
        query_processor=query_processor,
        # 传 query_processor
        dense_retriever=dense_retriever,
        # 传 dense_retriever
        sparse_retriever=sparse_retriever,
        # 传 sparse_retriever
        fusion=fusion,
        # 传 fusion
    )
    # 返回创建好的 HybridSearch 实例