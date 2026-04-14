"""Dense Retriever for semantic search using vector embeddings.
# 这个文件是做“稠密检索”的：把用户问题转成向量，再去向量库里找最相近的内容

This module implements the DenseRetriever component that performs semantic search
# 这一行说明：这里实现了 DenseRetriever 这个组件，专门负责语义检索

by embedding the query and retrieving similar chunks from the vector store.
# 做法是：先把 query 转成 embedding 向量，再去 vector store 查相似 chunk

It forms the Dense route in the Hybrid Search Engine.
# 它是混合检索（Hybrid Search）里“Dense 这一条路”
"""

from __future__ import annotations
# 让类型注解延迟解析，避免有些类型在运行时还没定义就报错

import logging
# 导入日志模块，用来打印运行信息、调试信息、警告信息

from typing import TYPE_CHECKING, Any, Dict, List, Optional
# 导入类型注解工具：
# TYPE_CHECKING：只在类型检查时导入，不在运行时真正导入
# Any：任意类型
# Dict/List/Optional：字典、列表、可为空

from src.core.types import RetrievalResult
# 导入统一的检索结果类型，后面会把原始结果转成这个标准格式

if TYPE_CHECKING:
    # 这里只有在类型检查阶段才会执行，运行代码时不会真正导入
    from src.core.settings import Settings
    # Settings 是配置对象类型
    from src.libs.embedding.base_embedding import BaseEmbedding
    # BaseEmbedding 是 embedding 客户端的抽象基类类型
    from src.libs.vector_store.base_vector_store import BaseVectorStore
    # BaseVectorStore 是向量库的抽象基类类型

logger = logging.getLogger(__name__)
# 拿到一个当前文件专属的 logger，方便打印日志时知道来自哪个模块


class DenseRetriever:
    """Dense retriever using embedding-based semantic search.
    # 这个类是“稠密检索器”，核心思路是基于 embedding 做语义检索
    
    This class performs semantic retrieval by:
    # 下面在解释这个类的整体流程
    1. Embedding the query using the configured embedding client
    # 第一步：把用户 query 用 embedding 模型编码成向量
    2. Querying the vector store for similar vectors
    # 第二步：拿这个向量去向量数据库里搜相似向量
    3. Returning normalized RetrievalResult objects
    # 第三步：把查回来的原始结果整理成统一的 RetrievalResult 格式
    
    Design Principles Applied:
    # 下面在说这个类的设计原则
    - Pluggable: Accepts embedding_client and vector_store via dependency injection.
    # 可插拔：embedding_client 和 vector_store 都是外部传进来的，方便替换实现
    - Config-Driven: Default top_k read from settings.retrieval.dense_top_k.
    # 配置驱动：默认返回多少条结果，可以从配置里读
    - Observable: Accepts optional TraceContext for observability integration.
    # 可观测：支持传 trace，方便打链路追踪
    - Fail-Fast: Validates inputs early with clear error messages.
    # 快速失败：参数不对就早点报错，不要拖到后面
    - Type-Safe: Returns standardized RetrievalResult objects.
    # 类型统一：返回的都是标准 RetrievalResult 对象
    
    Attributes:
        embedding_client: The embedding provider for query vectorization.
        # embedding_client：负责把 query 转成向量的客户端
        vector_store: The vector store for similarity search.
        # vector_store：负责相似度检索的向量库
        default_top_k: Default number of results to return.
        # default_top_k：默认返回多少条结果
    
    Example:
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        # 从工厂里创建 embedding 客户端
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        # 从工厂里创建向量库客户端
        >>> 
        >>> settings = Settings.load('config/settings.yaml')
        # 加载配置
        >>> embedding_client = EmbeddingFactory.create(settings)
        # 根据配置创建 embedding 客户端
        >>> vector_store = VectorStoreFactory.create(settings)
        # 根据配置创建向量库客户端
        >>> 
        >>> retriever = DenseRetriever(
        ...     settings=settings,
        ...     embedding_client=embedding_client,
        ...     vector_store=vector_store
        ... )
        # 创建 DenseRetriever 实例
        >>> results = retriever.retrieve("What is RAG?", top_k=5)
        # 调 retrieve 做一次检索
    """
    
    def __init__(
        self,
        # 类实例本身
        settings: Optional[Settings] = None,
        # 配置对象，可以不传
        embedding_client: Optional[BaseEmbedding] = None,
        # embedding 客户端，可以不传
        vector_store: Optional[BaseVectorStore] = None,
        # 向量库客户端，可以不传
        default_top_k: int = 10,
        # 默认返回 10 条结果
    ) -> None:
        """Initialize DenseRetriever with dependencies.
        # 初始化 DenseRetriever，并把依赖注入进来
        
        Args:
            settings: Application settings. Used to extract default_top_k if not provided.
            # settings：应用配置，如果有的话，会尝试从里面读取 dense_top_k
            embedding_client: Embedding provider for query vectorization.
                              Required for actual retrieval operations.
            # embedding_client：真正做 query 向量化的对象，实际检索时必须有
            vector_store: Vector store for similarity search.
                          Required for actual retrieval operations.
            # vector_store：真正做向量检索的对象，实际检索时必须有
            default_top_k: Default number of results to return (default: 10).
                           Can be overridden from settings.retrieval.dense_top_k.
            # default_top_k：默认返回条数；如果配置里有 dense_top_k，会覆盖这里
        
        Raises:
            ValueError: If embedding_client or vector_store is None when required.
            # 这里文档里写的是 ValueError，但实际上这个 __init__ 里并不会直接报这个错
        
        Note:
            Dependencies can be injected for testing (with mocks) or for
            production use (with real implementations from factories).
            # 这些依赖可以在测试时传 mock，也可以在线上用工厂创建真实实现
        """
        self.embedding_client = embedding_client
        # 把传进来的 embedding_client 挂到实例上，后面 retrieve 会用
        self.vector_store = vector_store
        # 把传进来的 vector_store 挂到实例上，后面 retrieve 会用
        
        # Extract default_top_k from settings if available
        # 如果有 settings，就尝试从配置里读默认 top_k
        self.default_top_k = default_top_k
        # 先把传进来的 default_top_k 作为默认值
        if settings is not None:
            # 只有 settings 不为空时，才继续往下读配置
            retrieval_config = getattr(settings, 'retrieval', None)
            # 从 settings 上取 retrieval 配置；如果没有 retrieval，就返回 None
            if retrieval_config is not None:
                # 如果 retrieval 配置存在，就继续取 dense_top_k
                self.default_top_k = getattr(
                    retrieval_config, 'dense_top_k', default_top_k
                )
                # 如果 retrieval.dense_top_k 存在，就用它；否则还是用传进来的 default_top_k
        
        logger.info(
            f"DenseRetriever initialized with default_top_k={self.default_top_k}"
        )
        # 记录一条初始化日志，方便后面排查当前默认 top_k 是多少
    
    def retrieve(
        self,
        # 类实例本身
        query: str,
        # 用户查询文本
        top_k: Optional[int] = None,
        # 本次想返回多少条，没传就用默认值
        filters: Optional[Dict[str, Any]] = None,
        # 元数据过滤条件，比如只查某个 collection
        trace: Optional[Any] = None,
        # 可选的追踪上下文，用来做 observability
    ) -> List[RetrievalResult]:
        """Retrieve semantically similar chunks for a query.
        # 根据 query 找出语义上最相近的 chunks
        
        Args:
            query: The search query string. Must not be empty.
            # query：查询字符串，不能为空
            top_k: Maximum number of results to return. If None, uses default_top_k.
            # top_k：最多返回多少条，不传就用默认值
            filters: Optional metadata filters (e.g., {"collection": "api-docs"}).
            # filters：可选的元数据过滤条件
            trace: Optional TraceContext for observability (reserved for Stage F).
            # trace：可选追踪对象，给可观测性系统用
        
        Returns:
            List of RetrievalResult objects, sorted by similarity (descending).
            Each result contains chunk_id, score, text, and metadata.
            # 返回值是 RetrievalResult 列表，按相似度从高到低排好
        
        Raises:
            ValueError: If query is empty or invalid.
            # query 不合法时抛 ValueError
            RuntimeError: If embedding_client or vector_store is not configured,
                          or if the retrieval operation fails.
            # embedding_client / vector_store 没配好，或者中途调用失败时抛 RuntimeError
        
        Example:
            >>> results = retriever.retrieve("How to configure Azure OpenAI?")
            # 调用 retrieve 做检索
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
            # 打印每条结果的分数、chunk_id 和前 50 个字符
        """
        # Validate inputs
        # 先校验输入参数是否合法
        self._validate_query(query)
        # 检查 query 是不是合法字符串、是不是空白
        self._validate_dependencies()
        # 检查 embedding_client 和 vector_store 有没有配好
        
        # Use default top_k if not specified
        # 如果这次没传 top_k，就用实例上的默认值
        effective_top_k = top_k if top_k is not None else self.default_top_k
        # 最终真正使用的 top_k
        
        logger.debug(f"Retrieving for query='{query[:50]}...', top_k={effective_top_k}")
        # 打一条 debug 日志：当前在检索什么 query、top_k 是多少
        # query[:50] 只截前 50 个字符，避免日志太长
        
        # Step 1: Embed the query
        # 第一步：先把 query 转成向量
        try:
            # 用 try 包起来，防止 embedding 过程中报错
            query_vectors = self.embedding_client.embed([query], trace=trace)
            # 调 embedding_client.embed，对单条 query 做编码
            # 这里传的是 [query] 列表，说明 embed 接口是批量输入风格
            query_vector = query_vectors[0]
            # 取第一个向量，也就是这条 query 对应的向量
        except Exception as e:
            # 只要 embedding 过程中任何地方出错，都会进这里
            raise RuntimeError(
                f"Failed to embed query: {e}. "
                "Check embedding client configuration and connectivity."
            ) from e
            # 把底层异常包装成更明确的 RuntimeError，提示检查 embedding 配置和连通性
        
        # Step 2: Query the vector store
        # 第二步：拿 query 向量去查向量库
        try:
            # 用 try 包起来，防止查向量库时报错
            raw_results = self.vector_store.query(
                vector=query_vector,
                # 要检索的查询向量
                top_k=effective_top_k,
                # 返回多少条
                filters=filters,
                # 过滤条件
                trace=trace,
                # 可选 trace，往下透传
            )
            # 调 vector_store.query，拿回原始检索结果
        except Exception as e:
            # 只要向量库查询出错，就进这里
            raise RuntimeError(
                f"Failed to query vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e
            # 包装成更清楚的错误，提醒检查向量库配置和数据状态
        
        # Step 3: Transform to RetrievalResult objects
        # 第三步：把向量库返回的原始结果转成统一的 RetrievalResult 对象
        results = self._transform_results(raw_results)
        # 调内部方法做格式转换
        
        logger.debug(f"Retrieved {len(results)} results for query")
        # 打一条 debug 日志，说明最终拿到了多少条结果
        return results
        # 返回标准化后的结果列表
    
    def _validate_query(self, query: str) -> None:
        """Validate the query string.
        # 校验 query 字符串是否合法
        
        Args:
            query: Query string to validate.
            # 要校验的 query
        
        Raises:
            ValueError: If query is empty or not a string.
            # 如果 query 不是字符串，或者是空字符串/全空白字符串，就报错
        """
        if not isinstance(query, str):
            # 如果 query 不是字符串类型
            raise ValueError(
                f"Query must be a string, got {type(query).__name__}"
            )
            # 抛 ValueError，告诉你实际传进来的类型是什么
        if not query.strip():
            # query.strip() 会去掉首尾空白；如果去完后为空，说明 query 只有空格/换行/制表符
            raise ValueError("Query cannot be empty or whitespace-only")
            # 抛 ValueError，明确告诉你 query 不能是空白
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are configured.
        # 校验必需依赖有没有准备好
        
        Raises:
            RuntimeError: If embedding_client or vector_store is None.
            # 如果 embedding_client 或 vector_store 没有配置，就报 RuntimeError
        """
        if self.embedding_client is None:
            # 如果没传 embedding_client
            raise RuntimeError(
                "DenseRetriever requires an embedding_client. "
                "Provide one during initialization or via setter."
            )
            # 报错：DenseRetriever 必须有 embedding_client
        if self.vector_store is None:
            # 如果没传 vector_store
            raise RuntimeError(
                "DenseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )
            # 报错：DenseRetriever 必须有 vector_store
    
    def _transform_results(
        self,
        # 类实例本身
        raw_results: List[Dict[str, Any]],
        # 向量库返回的原始结果列表，每一项通常是字典
    ) -> List[RetrievalResult]:
        """Transform raw vector store results to RetrievalResult objects.
        # 把向量库原始结果，转成统一的 RetrievalResult 对象
        
        Args:
            raw_results: Raw results from vector store query.
                         Each result should have: id, score, text, metadata.
            # 原始结果里每条通常应该包含 id、score、text、metadata
        
        Returns:
            List of RetrievalResult objects.
            # 返回转换好的 RetrievalResult 列表
        """
        results = []
        # 先准备一个空列表，用来装转换后的结果
        for raw in raw_results:
            # 遍历每一条原始结果
            try:
                # 每一条都尝试转成标准 RetrievalResult
                result = RetrievalResult(
                    chunk_id=str(raw.get('id', '')),
                    # 从 raw 里取 id，转成字符串；如果没有 id，就给空字符串
                    score=float(raw.get('score', 0.0)),
                    # 从 raw 里取 score，转成 float；没有就默认 0.0
                    text=str(raw.get('text', '')),
                    # 从 raw 里取 text，转成字符串；没有就默认空字符串
                    metadata=raw.get('metadata', {}),
                    # 从 raw 里取 metadata；没有就默认空字典
                )
                results.append(result)
                # 转换成功就加入结果列表
            except (ValueError, TypeError) as e:
                # 如果某条结果字段类型不对，比如 score 转 float 失败，就进这里
                logger.warning(
                    f"Failed to transform result {raw.get('id', 'unknown')}: {e}. "
                    "Skipping this result."
                )
                # 打 warning 日志，说明哪条结果转换失败了，并且跳过它
                continue
                # 继续处理下一条，不让一条坏数据拖垮整个结果集
        
        return results
        # 返回最终转换好的结果列表


def create_dense_retriever(
    settings: Settings,
    # 配置对象，这里要求必须传
    embedding_client: Optional[BaseEmbedding] = None,
    # 可选传入现成的 embedding_client
    vector_store: Optional[BaseVectorStore] = None,
    # 可选传入现成的 vector_store
) -> DenseRetriever:
    """Factory function to create a DenseRetriever with optional dependency injection.
    # 一个工厂函数：帮你更方便地创建 DenseRetriever
    
    This function simplifies DenseRetriever creation by automatically creating
    dependencies from factories if not provided.
    # 如果你没传 embedding_client / vector_store，它会自动用工厂创建
    
    Args:
        settings: Application settings.
        # 应用配置
        embedding_client: Optional pre-configured embedding client.
                          If None, created from EmbeddingFactory.
        # 可选：你可以自己传 embedding_client；不传就自动创建
        vector_store: Optional pre-configured vector store.
                      If None, created from VectorStoreFactory.
        # 可选：你可以自己传 vector_store；不传就自动创建
    
    Returns:
        Configured DenseRetriever instance.
        # 返回一个已经配置好的 DenseRetriever 实例
    
    Example:
        >>> settings = Settings.load('config/settings.yaml')
        # 先加载配置
        >>> retriever = create_dense_retriever(settings)
        # 直接调用工厂函数创建 retriever
    """
    # Lazy import to avoid circular dependencies
    # 这里用“延迟导入”，避免模块之间互相 import 造成循环依赖
    if embedding_client is None:
        # 如果外面没传 embedding_client，就自己创建一个
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        # 在函数内部再导入工厂类
        embedding_client = EmbeddingFactory.create(settings)
        # 用配置创建 embedding_client
    
    if vector_store is None:
        # 如果外面没传 vector_store，就自己创建一个
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        # 在函数内部再导入向量库工厂
        vector_store = VectorStoreFactory.create(settings)
        # 用配置创建 vector_store
    
    return DenseRetriever(
        settings=settings,
        # 把 settings 传进去
        embedding_client=embedding_client,
        # 把准备好的 embedding_client 传进去
        vector_store=vector_store,
        # 把准备好的 vector_store 传进去
    )
    # 返回创建好的 DenseRetriever 实例