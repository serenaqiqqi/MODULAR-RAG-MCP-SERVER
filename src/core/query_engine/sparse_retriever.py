"""Sparse Retriever for keyword-based search using BM25.
# 这个文件是做“稀疏检索”的：用 BM25 这种关键词匹配方法做搜索

This module implements the SparseRetriever component that performs keyword-based
# 这一行说明：这里实现了 SparseRetriever 这个组件，负责关键词检索

search using BM25 inverted indexes. It forms the Sparse route in the Hybrid
# 它底层靠 BM25 倒排索引来查，属于 Hybrid Search 里的 Sparse 这一条路

Search Engine, complementing the DenseRetriever's semantic search.
# 它和 DenseRetriever 互补：Dense 负责语义，Sparse 负责关键词精确匹配
"""

from __future__ import annotations
# 让类型注解延迟解析，避免有些类型在运行时还没定义就报错

import logging
# 导入日志模块，用来打印 info/debug/warning 等日志

from typing import TYPE_CHECKING, Any, Dict, List, Optional
# 导入类型注解工具：
# TYPE_CHECKING：只在类型检查时导入
# Any：任意类型
# Dict/List/Optional：字典、列表、可为空

from src.core.types import RetrievalResult
# 导入统一的检索结果类型，后面会把原始结果整理成这个标准格式

if TYPE_CHECKING:
    # 这一段只在类型检查时生效，运行代码时不会真正导入
    from src.core.settings import Settings
    # Settings：配置对象类型
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    # BM25Indexer：BM25 索引器类型
    from src.libs.vector_store.base_vector_store import BaseVectorStore
    # BaseVectorStore：向量库抽象基类类型

logger = logging.getLogger(__name__)
# 获取当前文件自己的 logger，方便日志里区分来源


class SparseRetriever:
    """Sparse retriever using BM25 keyword-based search.
    # 这个类是“稀疏检索器”，核心是基于 BM25 做关键词检索
    
    This class performs keyword-based retrieval by:
    # 下面在说明这个类整体是怎么工作的
    1. Querying the BM25 index with keywords to get matching chunk IDs and scores
    # 第一步：拿关键词去查 BM25 索引，得到匹配到的 chunk_id 和分数
    2. Fetching text and metadata from the vector store using get_by_ids()
    # 第二步：再去 vector store 里根据这些 chunk_id 把正文和 metadata 取出来
    3. Returning normalized RetrievalResult objects
    # 第三步：把查到的数据整理成统一的 RetrievalResult 对象
    
    Design Principles Applied:
    # 下面是设计原则
    - Pluggable: Accepts bm25_indexer and vector_store via dependency injection.
    # 可插拔：bm25_indexer 和 vector_store 都是外部注入的，方便替换
    - Config-Driven: Default top_k and collection read from settings.
    # 配置驱动：默认 top_k 和 collection 可以从配置里读
    - Observable: Accepts optional TraceContext for observability integration.
    # 可观测：支持 trace 参数，方便做链路追踪
    - Fail-Fast: Validates inputs early with clear error messages.
    # 快速失败：参数不合法尽早报错，不拖到后面
    - Type-Safe: Returns standardized RetrievalResult objects (same as DenseRetriever).
    # 类型统一：返回的也是标准 RetrievalResult，和 DenseRetriever 一样
    
    Attributes:
        bm25_indexer: The BM25 indexer for keyword search.
        # bm25_indexer：BM25 索引器，负责做关键词检索
        vector_store: The vector store for fetching text and metadata.
        # vector_store：向量库，这里不是拿来做相似度检索，而是拿来根据 id 取回正文和 metadata
        default_top_k: Default number of results to return.
        # default_top_k：默认返回多少条结果
        default_collection: Default BM25 index collection to query.
        # default_collection：默认查哪个 BM25 collection
    
    Example:
        >>> from src.ingestion.storage.bm25_indexer import BM25Indexer
        # 导入 BM25Indexer
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        # 导入向量库工厂
        >>> 
        >>> settings = Settings.load('config/settings.yaml')
        # 加载配置
        >>> bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
        # 创建 BM25 索引器，指定索引目录
        >>> bm25_indexer.load("default")
        # 加载名为 default 的索引
        >>> vector_store = VectorStoreFactory.create(settings)
        # 根据配置创建向量库实例
        >>> 
        >>> retriever = SparseRetriever(
        ...     settings=settings,
        ...     bm25_indexer=bm25_indexer,
        ...     vector_store=vector_store
        ... )
        # 创建 SparseRetriever 实例
        >>> results = retriever.retrieve(["RAG", "retrieval"], top_k=5)
        # 传入关键词列表做检索
    """
    
    def __init__(
        self,
        # 类实例本身
        settings: Optional[Settings] = None,
        # 配置对象，可不传
        bm25_indexer: Optional[BM25Indexer] = None,
        # BM25 索引器，可不传
        vector_store: Optional[BaseVectorStore] = None,
        # 向量库实例，可不传
        default_top_k: int = 10,
        # 默认返回 10 条
        default_collection: str = "default",
        # 默认查询的 BM25 collection 名字是 default
    ) -> None:
        """Initialize SparseRetriever with dependencies.
        # 初始化 SparseRetriever，把依赖对象放进来
        
        Args:
            settings: Application settings. Used to extract default_top_k if not provided.
            # settings：配置对象，如果有的话会尝试从里面读 sparse_top_k
            bm25_indexer: BM25 indexer for keyword search.
                          Required for actual retrieval operations.
            # bm25_indexer：真正做 BM25 关键词检索的对象，实际查询时必须有
            vector_store: Vector store for fetching text and metadata.
                          Required for actual retrieval operations.
            # vector_store：真正根据 id 回查正文和 metadata 的对象，实际查询时必须有
            default_top_k: Default number of results to return (default: 10).
                           Can be overridden from settings.retrieval.sparse_top_k.
            # default_top_k：默认返回条数；如果配置里有 sparse_top_k，会覆盖它
            default_collection: Default BM25 index collection name (default: "default").
            # default_collection：默认查询的索引集合名
        
        Note:
            Dependencies can be injected for testing (with mocks) or for
            production use (with real implementations from factories).
            # 测试时可以传 mock，正式环境里可以传真实实现
        """
        self.bm25_indexer = bm25_indexer
        # 把传进来的 bm25_indexer 挂到实例上
        self.vector_store = vector_store
        # 把传进来的 vector_store 挂到实例上
        self.default_collection = default_collection
        # 记录默认 collection 名称
        
        # Extract default_top_k from settings if available
        # 如果有 settings，就尝试从配置里读默认 top_k
        self.default_top_k = default_top_k
        # 先用传进来的 default_top_k 作为默认值
        if settings is not None:
            # 只有 settings 不为空时才继续读取
            retrieval_config = getattr(settings, 'retrieval', None)
            # 从 settings 上取 retrieval 配置，没有就返回 None
            if retrieval_config is not None:
                # 如果 retrieval 配置存在，就继续读取 sparse_top_k
                self.default_top_k = getattr(
                    retrieval_config, 'sparse_top_k', default_top_k
                )
                # 如果 retrieval.sparse_top_k 存在，就用它；否则继续用 default_top_k
        
        logger.info(
            f"SparseRetriever initialized with default_top_k={self.default_top_k}, "
            f"default_collection='{self.default_collection}'"
        )
        # 打一条初始化日志，说明默认 top_k 和默认 collection 是什么
    
    def retrieve(
        self,
        # 类实例本身
        keywords: List[str],
        # 关键词列表，不是整句 query，而是一组 token/关键词
        top_k: Optional[int] = None,
        # 本次最多返回多少条，不传就用默认值
        collection: Optional[str] = None,
        # 本次要查哪个 collection，不传就用默认 collection
        trace: Optional[Any] = None,
        # 可选 trace，给可观测性系统用
    ) -> List[RetrievalResult]:
        """Retrieve chunks matching the given keywords using BM25.
        # 用 BM25 根据关键词找匹配的 chunks
        
        Args:
            keywords: List of keywords to search for (typically from QueryProcessor).
            # keywords：要查的关键词列表，通常是 QueryProcessor 处理后的结果
            top_k: Maximum number of results to return. If None, uses default_top_k.
            # top_k：最多返回多少条，不传就用默认值
            collection: BM25 index collection to query. If None, uses default_collection.
            # collection：要查哪个 BM25 collection，不传就用默认值
            trace: Optional TraceContext for observability (reserved for Stage F).
            # trace：可选追踪对象
        
        Returns:
            List of RetrievalResult objects, sorted by BM25 score (descending).
            Each result contains chunk_id, score, text, and metadata.
            # 返回标准 RetrievalResult 列表，按 BM25 分数从高到低排好
        
        Raises:
            ValueError: If keywords list is empty.
            # keywords 不合法时抛 ValueError
            RuntimeError: If bm25_indexer or vector_store is not configured,
                          or if the retrieval operation fails.
            # bm25_indexer / vector_store 没配好，或中途失败时抛 RuntimeError
        
        Example:
            >>> results = retriever.retrieve(["Azure", "OpenAI", "配置"])
            # 用几个关键词做检索
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
            # 打印每条结果的分数、chunk_id、文本前 50 个字符
        """
        # Validate inputs
        # 先校验输入参数
        self._validate_keywords(keywords)
        # 检查 keywords 是不是合法的列表、是不是空列表
        self._validate_dependencies()
        # 检查 bm25_indexer 和 vector_store 有没有配好
        
        # Use defaults if not specified
        # 如果这次没传 top_k / collection，就使用默认值
        effective_top_k = top_k if top_k is not None else self.default_top_k
        # 这次真正使用的 top_k
        effective_collection = collection if collection is not None else self.default_collection
        # 这次真正使用的 collection
        
        logger.debug(
            f"Retrieving for keywords={keywords[:5]}{'...' if len(keywords) > 5 else ''}, "
            f"top_k={effective_top_k}, collection='{effective_collection}'"
        )
        # 打 debug 日志：
        # 关键词太多的话只展示前 5 个，避免日志太长
        # 同时记下本次 top_k 和 collection
        
        # Step 1: Ensure index is loaded
        # 第一步：先确保对应 collection 的 BM25 索引已经加载好了
        if not self._ensure_index_loaded(effective_collection):
            # 如果索引没加载成功
            logger.warning(
                f"BM25 index for collection '{effective_collection}' not available. "
                "Returning empty results."
            )
            # 打 warning，说明这个 collection 的 BM25 索引不可用
            return []
            # 直接返回空列表，不往下继续
        
        # Step 2: Query BM25 index
        # 第二步：真正去查 BM25 索引
        try:
            # 用 try 包起来，防止 query 过程出错
            bm25_results = self.bm25_indexer.query(
                query_terms=keywords,
                # 把关键词列表作为查询词传给 BM25
                top_k=effective_top_k,
                # 返回多少条
                trace=trace,
                # 可选 trace，往下透传
            )
        except Exception as e:
            # 只要 BM25 查询出错，就进这里
            raise RuntimeError(
                f"Failed to query BM25 index: {e}. "
                "Check index availability and query terms."
            ) from e
            # 包装成更明确的 RuntimeError，提醒检查索引和查询词
        
        # Early return if no matches
        # 如果一个结果都没查到，就直接返回
        if not bm25_results:
            logger.debug("BM25 query returned no results")
            # 打 debug 日志，说明没有匹配项
            return []
            # 返回空列表
        
        # Step 3: Fetch text and metadata from vector store
        # 第三步：根据 BM25 查到的 chunk_id，再去 vector store 里取正文和 metadata
        chunk_ids = [r["chunk_id"] for r in bm25_results]
        # 从 BM25 结果里抽出所有 chunk_id，组成一个列表
        try:
            # 用 try 包起来，防止去向量库回查时失败
            records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
            # 调 vector_store.get_by_ids，一次性把这些 id 对应的记录取回来
        except Exception as e:
            # 回查向量库失败时进这里
            raise RuntimeError(
                f"Failed to fetch records from vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e
            # 包装成更明确的 RuntimeError，提醒检查向量库配置和数据
        
        # Step 4: Merge BM25 scores with text/metadata
        # 第四步：把 BM25 分数和 vector store 里的正文、metadata 合并起来
        results = self._merge_results(bm25_results, records)
        # 调内部方法做合并和标准化
        
        logger.debug(f"Retrieved {len(results)} results for keywords")
        # 打 debug 日志，说明最终拿到了多少条结果
        return results
        # 返回最终整理好的 RetrievalResult 列表
    
    def _validate_keywords(self, keywords: List[str]) -> None:
        """Validate the keywords list.
        # 校验关键词列表是否合法
        
        Args:
            keywords: Keywords list to validate.
            # 要校验的关键词列表
        
        Raises:
            ValueError: If keywords is empty or not a list.
            # 如果不是列表，或者是空列表，就抛 ValueError
        """
        if not isinstance(keywords, list):
            # 如果 keywords 不是 list 类型
            raise ValueError(
                f"Keywords must be a list, got {type(keywords).__name__}"
            )
            # 报错并告诉你实际传进来的类型是什么
        if not keywords:
            # 如果 keywords 是空列表
            raise ValueError("Keywords list cannot be empty")
            # 报错：关键词列表不能为空
        # Filter out empty strings but allow the call to proceed
        # （这里注释在说：即使列表里有空字符串，也不拦截）
        # (empty strings will simply not match anything)
        # 空字符串本身查不到东西，所以放过去也问题不大
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are configured.
        # 校验必要依赖有没有准备好
        
        Raises:
            RuntimeError: If bm25_indexer or vector_store is None.
            # 如果 bm25_indexer 或 vector_store 没配置，就抛 RuntimeError
        """
        if self.bm25_indexer is None:
            # 如果没传 bm25_indexer
            raise RuntimeError(
                "SparseRetriever requires a bm25_indexer. "
                "Provide one during initialization or via setter."
            )
            # 报错：SparseRetriever 必须有 bm25_indexer
        if self.vector_store is None:
            # 如果没传 vector_store
            raise RuntimeError(
                "SparseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )
            # 报错：SparseRetriever 必须有 vector_store
    
    def _ensure_index_loaded(self, collection: str) -> bool:
        """Ensure the BM25 index is loaded for the given collection.
        # 确保指定 collection 的 BM25 索引已经加载好
        
        Always reloads from disk because the index may have been updated
        # 每次都会从磁盘重新加载，因为索引文件可能被别的进程更新过

        by another process (e.g., dashboard ingestion).  The load is
        # 比如 dashboard 在做 ingestion 时就可能改了索引

        fast (a single JSON file read) compared to the overall query.
        # 这里重新加载代价不大，只是读一下文件，相比整个查询过程很快
        
        Args:
            collection: The collection name to load.
            # 要加载的 collection 名字
        
        Returns:
            True if index is loaded and ready, False otherwise.
            # 加载成功返回 True，否则返回 False
        """
        try:
            # 用 try 包起来，防止 load 时报错
            loaded = self.bm25_indexer.load(collection=collection)
            # 调 bm25_indexer.load 去加载这个 collection 的索引
            return loaded
            # 直接把 load 的返回值返回出去
        except Exception as e:
            # 如果加载索引时报错
            logger.warning(f"Failed to load BM25 index for collection '{collection}': {e}")
            # 打 warning，记录哪个 collection 加载失败、为什么失败
            return False
            # 返回 False，告诉上层“索引不可用”
    
    def _merge_results(
        self,
        # 类实例本身
        bm25_results: List[Dict[str, Any]],
        # BM25 查询结果列表，每一项一般有 chunk_id 和 score
        records: List[Dict[str, Any]],
        # vector store 回查到的记录列表，每一项一般有 id、text、metadata
    ) -> List[RetrievalResult]:
        """Merge BM25 scores with text and metadata from vector store.
        # 把 BM25 分数和 vector store 的正文、metadata 合并起来
        
        Args:
            bm25_results: Results from BM25 query, each with 'chunk_id' and 'score'.
            # BM25 查回来的结果，每条通常有 chunk_id 和 score
            records: Records from vector store, each with 'id', 'text', 'metadata'.
            # 向量库查回来的记录，每条通常有 id、text、metadata
        
        Returns:
            List of RetrievalResult objects with complete information.
            # 返回信息完整的 RetrievalResult 列表
        """
        results = []
        # 先准备一个空列表，用来装最终结果
        
        for bm25_result, record in zip(bm25_results, records):
            # 用 zip 把 BM25 结果和回查记录一一配对
            chunk_id = bm25_result["chunk_id"]
            # 取出 BM25 结果里的 chunk_id
            score = bm25_result["score"]
            # 取出 BM25 分数
            
            # Handle case where record was not found
            # 处理“向量库里没查到这条记录”的情况
            if not record:
                # 如果 record 为空
                logger.warning(
                    f"No record found in vector store for chunk_id='{chunk_id}'. "
                    "Skipping this result."
                )
                # 打 warning，说明这个 chunk_id 在 vector store 里没找到
                continue
                # 跳过这条，继续下一条
            
            # Validate record has expected fields
            # 从 record 里取出需要的字段
            text = record.get('text', '')
            # 取正文，没有就默认空字符串
            metadata = record.get('metadata', {})
            # 取 metadata，没有就默认空字典
            
            try:
                # 尝试把这条结果组装成标准 RetrievalResult
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    # 直接用 BM25 那边的 chunk_id
                    score=float(score),
                    # 分数转成 float
                    text=str(text),
                    # 正文转成字符串
                    metadata=metadata,
                    # metadata 直接放进去
                )
                results.append(result)
                # 组装成功就加入结果列表
            except (ValueError, TypeError) as e:
                # 如果某个字段类型不对导致创建失败，就进这里
                logger.warning(
                    f"Failed to create RetrievalResult for chunk_id='{chunk_id}': {e}. "
                    "Skipping this result."
                )
                # 打 warning，说明这条结果组装失败了
                continue
                # 跳过坏数据，继续下一条
        
        return results
        # 返回最终合并好的结果列表


def create_sparse_retriever(
    settings: Settings,
    # 配置对象，这里要求必须传
    bm25_indexer: Optional[BM25Indexer] = None,
    # 可选：外部传现成的 bm25_indexer
    vector_store: Optional[BaseVectorStore] = None,
    # 可选：外部传现成的 vector_store
    index_dir: str = "data/db/bm25",
    # 如果要自己创建 BM25Indexer，就默认用这个索引目录
) -> SparseRetriever:
    """Factory function to create a SparseRetriever with optional dependency injection.
    # 一个工厂函数：方便你快速创建 SparseRetriever
    
    This function simplifies SparseRetriever creation by automatically creating
    dependencies from factories if not provided.
    # 如果你没传依赖，它会自动帮你创建
    
    Args:
        settings: Application settings.
        # 应用配置
        bm25_indexer: Optional pre-configured BM25 indexer.
                      If None, created with default index_dir.
        # 可选：你可以自己传 BM25Indexer；不传就按 index_dir 新建一个
        vector_store: Optional pre-configured vector store.
                      If None, created from VectorStoreFactory.
        # 可选：你可以自己传 vector_store；不传就用工厂创建
        index_dir: Directory for BM25 index files (default: "data/db/bm25").
        # BM25 索引文件所在目录
    
    Returns:
        Configured SparseRetriever instance.
        # 返回一个已经配置好的 SparseRetriever 实例
    
    Example:
        >>> settings = Settings.load('config/settings.yaml')
        # 加载配置
        >>> retriever = create_sparse_retriever(settings)
        # 直接用工厂函数创建 SparseRetriever
    """
    # Lazy import to avoid circular dependencies
    # 延迟导入，避免模块互相 import 造成循环依赖
    if bm25_indexer is None:
        # 如果外面没传 bm25_indexer，就自己创建
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        # 在函数内部再导入 BM25Indexer
        bm25_indexer = BM25Indexer(index_dir=index_dir)
        # 用 index_dir 创建一个 BM25Indexer
    
    if vector_store is None:
        # 如果外面没传 vector_store，就自己创建
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        # 在函数内部再导入向量库工厂
        vector_store = VectorStoreFactory.create(settings)
        # 用配置创建 vector_store
    
    return SparseRetriever(
        settings=settings,
        # 把 settings 传进去
        bm25_indexer=bm25_indexer,
        # 把准备好的 bm25_indexer 传进去
        vector_store=vector_store,
        # 把准备好的 vector_store 传进去
    )
    # 返回创建好的 SparseRetriever 实例