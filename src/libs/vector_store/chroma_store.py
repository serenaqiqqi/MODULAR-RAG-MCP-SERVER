"""ChromaDB VectorStore implementation.

This module provides a concrete implementation of BaseVectorStore using ChromaDB,
a lightweight, open-source embedding database designed for local-first deployment.
"""
# 上面：用 ChromaDB 做本地向量库，实现 BaseVectorStore（写入、检索、按条件删等）。

from __future__ import annotations  # 类型注解可前向引用

import logging  # 日志
from pathlib import Path  # 路径（部分逻辑用）
from typing import TYPE_CHECKING, Any, Dict, List, Optional  # 类型注解

try:
    import chromadb  # Chroma 客户端库
    from chromadb.config import Settings as ChromaSettings  # 客户端行为配置（telemetry 等）
    CHROMADB_AVAILABLE = True  # 标记已安装
except ImportError:
    CHROMADB_AVAILABLE = False  # 没装则 __init__ 会报错

from src.core.settings import resolve_path  # 把配置里的相对路径解析到项目根
from src.libs.vector_store.base_vector_store import BaseVectorStore  # 向量库抽象接口

if TYPE_CHECKING:  # 仅类型检查时 import，避免运行时循环依赖
    from src.core.settings import Settings  # 完整配置类型

logger = logging.getLogger(__name__)  # 本模块 logger


class ChromaStore(BaseVectorStore):  # 具体实现：持久化目录 + 一个 collection
    """ChromaDB implementation of VectorStore.
    
    This class provides local-first, persistent vector storage using ChromaDB.
    It supports upsert, query, and metadata filtering operations.
    
    Design Principles Applied:
    - Pluggable: Implements BaseVectorStore interface, swappable with other providers.
    - Config-Driven: All settings (persist_directory, collection_name) from settings.yaml.
    - Idempotent: upsert operations with same ID overwrite existing records.
    - Observable: Accepts optional TraceContext (reserved for Stage F).
    - Fail-Fast: Validates dependencies and configuration on initialization.
    
    Attributes:
        client: ChromaDB client instance.
        collection: ChromaDB collection for storing vectors.
        collection_name: Name of the collection.
        persist_directory: Directory path for persistent storage.
    
    Example:
        >>> settings = Settings.load('config/settings.yaml')
        >>> store = ChromaStore(settings=settings)
        >>> records = [
        ...     {
        ...         'id': 'doc1_chunk0',
        ...         'vector': [0.1, 0.2, 0.3],
        ...         'metadata': {'source': 'doc1.pdf'}
        ...     }
        ... ]
        >>> store.upsert(records)
        >>> results = store.query([0.1, 0.2, 0.3], top_k=5)
    """
    # 启动时建 PersistentClient、get_or_create_collection；相似度用 cosine

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialize ChromaStore with configuration.
        
        Args:
            settings: Application settings containing vector_store configuration.
            **kwargs: Optional overrides for collection_name or persist_directory.
        
        Raises:
            ImportError: If chromadb package is not installed.
            ValueError: If required configuration is missing.
            RuntimeError: If ChromaDB client initialization fails.
        """
        if not CHROMADB_AVAILABLE:  # 依赖检查
            raise ImportError(
                "chromadb package is required for ChromaStore. "
                "Install it with: pip install chromadb"
            )

        # Extract configuration
        try:
            vector_store_config = settings.vector_store  # yaml 里 vector_store 段
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store. "
                "Please ensure 'vector_store' section exists in settings.yaml"
            ) from e

        # Collection name (allow override)
        self.collection_name = kwargs.get(
            'collection_name',  # 工厂可传入，例如按业务 collection 隔离
            getattr(vector_store_config, 'collection_name', 'knowledge_hub')
        )

        # Persist directory (allow override)
        persist_dir_str = kwargs.get(
            'persist_directory',
            getattr(vector_store_config, 'persist_directory', './data/db/chroma')
        )
        self.persist_directory = resolve_path(persist_dir_str)  # 绝对/项目相对统一路径

        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)  # 磁盘上建好目录

        logger.info(
            f"Initializing ChromaStore: collection='{self.collection_name}', "
            f"persist_directory='{self.persist_directory}'"
        )

        # Initialize ChromaDB client with persistent storage
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),  # 数据落盘位置
                settings=ChromaSettings(
                    anonymized_telemetry=False,  # 不上报匿名统计
                    allow_reset=True,  # 允许 reset（若上层调用）
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{self.persist_directory}': {e}"
            ) from e

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # HNSW 索引：余弦距离（与 query 里 score 换算一致）
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get or create collection '{self.collection_name}': {e}"
            ) from e

        logger.info(
            f"ChromaStore initialized successfully. "
            f"Collection count: {self.collection.count()}"  # 当前已有多少条
        )

    def upsert(
        self,
        records: List[Dict[str, Any]],  # 每条含 id、vector、可选 metadata（可含 text）
        trace: Optional[Any] = None,  # 预留追踪，当前未用
        **kwargs: Any,  # 扩展预留
    ) -> None:
        """Insert or update records in ChromaDB.
        
        Args:
            records: List of records to upsert. Each record must have:
                - 'id': Unique identifier (str)
                - 'vector': Embedding vector (List[float])
                - 'metadata': Optional metadata dict
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Provider-specific parameters (unused for Chroma).
        
        Raises:
            ValueError: If records list is empty or contains invalid entries.
            RuntimeError: If the upsert operation fails.
        """
        # Validate records
        self.validate_records(records)  # 基类检查 id/vector 等

        # Prepare data for ChromaDB
        ids = []  # 每条记录一个字符串 id
        embeddings = []  # 与 ids 同序的向量
        metadatas = []  # 与 ids 同序的元数据
        documents = []  # Chroma 要求每条记录带 document 字符串（这里存 chunk 正文或 id）

        for record in records:
            ids.append(str(record['id']))  # 统一 str
            embeddings.append(record['vector'])  # List[float]

            # Metadata: extract or default to empty dict
            metadata = record.get('metadata', {})
            # Ensure all metadata values are JSON-serializable
            # ChromaDB requires string, int, float, or bool values
            sanitized_metadata = self._sanitize_metadata(metadata)  # 过滤/转成 Chroma 支持的类型

            # ChromaDB requires non-empty metadata dict
            if not sanitized_metadata:  # 全被删掉时给个占位，否则 upsert 可能报错
                sanitized_metadata = {'_placeholder': 'true'}

            metadatas.append(sanitized_metadata)

            # Document: use metadata.text if available, otherwise use id
            document = metadata.get('text', record['id'])  # 优先存 chunk 正文，没有就用 id
            documents.append(str(document))

        # Perform upsert (ChromaDB's add() is idempotent with same IDs)
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,  # 与 ingestion pipeline 写入格式一致
            )
            logger.debug(f"Successfully upserted {len(records)} records to ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert {len(records)} records to ChromaDB: {e}"
            ) from e

    def query(
        self,
        vector: List[float],  # 查询向量（与入库时同模型同维度）
        top_k: int = 10,  # 最多返回条数
        filters: Optional[Dict[str, Any]] = None,  # 元数据过滤，如 source_path
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:  # 每条含 id、score、text、metadata
        """Query ChromaDB for similar vectors.
        
        Args:
            vector: Query vector (embedding) to search for.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (e.g., {'source': 'doc1.pdf'}).
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Provider-specific parameters (unused for Chroma).
        
        Returns:
            List of matching records, sorted by similarity (descending).
            Each record contains:
                - 'id': Record identifier
                - 'score': Similarity score (1.0 = identical, 0.0 = orthogonal)
                - 'metadata': Associated metadata
        
        Raises:
            ValueError: If vector is empty or top_k is invalid.
            RuntimeError: If the query operation fails.
        """
        # Validate query parameters
        self.validate_query_vector(vector, top_k)  # 非空向量、top_k 合法

        # Build ChromaDB where clause from filters
        where_clause = self._build_where_clause(filters) if filters else None  # 无过滤则查全库

        # Perform query
        try:
            results = self.collection.query(
                query_embeddings=[vector],  # 一次查一个查询向量
                n_results=top_k,  # 最多返回几条
                where=where_clause,  # 元数据过滤
                include=["metadatas", "distances", "documents"]  # 需要距离和文档内容
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query ChromaDB with top_k={top_k}: {e}"
            ) from e

        # Transform results to standard format
        # ChromaDB returns nested lists: [[id1, id2, ...]]
        output = []

        if results and results['ids'] and results['ids'][0]:  # 有命中时第一层是 batch 维
            ids = results['ids'][0]  # 本批（单查询）的 id 列表
            distances = results['distances'][0] if 'distances' in results else [0.0] * len(ids)  # 余弦距离
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(ids)
            documents = results['documents'][0] if 'documents' in results else [''] * len(ids)

            for i, record_id in enumerate(ids):
                # Convert distance to similarity score
                # ChromaDB returns cosine distance (0=identical, 2=opposite)
                # Convert to similarity: score = 1 - (distance / 2)
                distance = distances[i]
                score = 1.0 - (distance / 2.0)  # 映射到越大越相似

                output.append({
                    'id': record_id,
                    'score': max(0.0, score),  # Clamp to [0, 1]  # 防止数值误差出负数
                    'text': documents[i] if documents[i] else '',  # Include text from documents
                    'metadata': metadatas[i] if metadatas[i] else {}
                })

        logger.debug(f"Query returned {len(output)} results")
        return output

    def delete(
        self,
        ids: List[str],  # 要删的向量 id 列表
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Delete records from ChromaDB by IDs.
        
        Args:
            ids: List of record IDs to delete.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.
        
        Raises:
            ValueError: If ids list is empty.
            RuntimeError: If the delete operation fails.
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")

        try:
            self.collection.delete(ids=[str(id_) for id_ in ids])  # 按主键删
            logger.debug(f"Successfully deleted {len(ids)} records from ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {len(ids)} records from ChromaDB: {e}"
            ) from e

    def clear(
        self,
        collection_name: Optional[str] = None,  # None 表示清空当前 self.collection 同名集合
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Clear all records from the ChromaDB collection.
        
        Args:
            collection_name: Optional collection name to clear. If None, clears current collection.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.
        
        Raises:
            RuntimeError: If the clear operation fails.
        """
        try:
            target_collection = collection_name or self.collection_name  # 默认同当前 collection

            # Delete and recreate collection (most efficient way to clear in Chroma)
            self.client.delete_collection(name=target_collection)  # 整个 collection 删掉
            self.collection = self.client.get_or_create_collection(
                name=target_collection,
                metadata={"hnsw:space": "cosine"}  # 与初始化一致
            )
            logger.info(f"Successfully cleared collection '{target_collection}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to clear collection '{collection_name or self.collection_name}': {e}"
            ) from e

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],  # 如 doc_hash，与 metadata 键一致
        trace: Optional[Any] = None,
    ) -> int:  # 实际删掉几条
        """Delete records matching a metadata filter.

        Args:
            filter_dict: Metadata key/value pairs to match
                (e.g. ``{"source_hash": "abc123"}``).
            trace: Optional TraceContext for observability.

        Returns:
            Number of records deleted.

        Raises:
            ValueError: If *filter_dict* is empty.
            RuntimeError: If the operation fails.
        """
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")

        try:
            where = self._build_where_clause(filter_dict)  # 转成 Chroma 的 where
            # Query matching IDs first
            results = self.collection.get(where=where, include=[])  # 不要向量/文档，只要 id
            matching_ids = results.get("ids", [])

            if not matching_ids:
                logger.debug(f"delete_by_metadata: no records matched {filter_dict}")
                return 0

            self.collection.delete(ids=matching_ids)  # 再按 id 批量删
            logger.info(
                f"delete_by_metadata: deleted {len(matching_ids)} records "
                f"matching {filter_dict}"
            )
            return len(matching_ids)  # pipeline 里用来打日志
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete by metadata {filter_dict}: {e}"
            ) from e

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure ChromaDB compatibility.
        
        ChromaDB requires metadata values to be str, int, float, or bool.
        This method converts or filters out incompatible types.
        
        Args:
            metadata: Raw metadata dict.
        
        Returns:
            Sanitized metadata dict.
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):  # Chroma 原生支持的标量
                sanitized[key] = value
            elif value is None:
                # Skip None values
                continue  # None 不能存进 Chroma metadata
            elif isinstance(value, (list, tuple)):
                # Convert to comma-separated string
                sanitized[key] = ",".join(str(v) for v in value)  # 列表压成一串
            else:
                # Convert to string as fallback
                sanitized[key] = str(value)  # dict 等复杂类型变字符串

        return sanitized

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters.
        
        Converts standard filter dict to ChromaDB's query format.
        
        Args:
            filters: Standard filter dict (e.g., {'source': 'doc1.pdf'}).
        
        Returns:
            ChromaDB where clause dict.
        
        Note:
            ChromaDB supports operators like $eq, $ne, $gt, $lt, $in, etc.
            For simplicity, we currently support only exact equality matches.
            Future enhancement: support complex filters.
        """
        # Simple implementation: exact equality matches only
        # For complex filters (e.g., {'score': {'$gt': 0.5}}), extend this method
        where = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                # Already in ChromaDB operator format (e.g., {'$eq': 'value'})
                where[key] = value  # 透传 $eq/$in 等
            else:
                # Simple equality
                where[key] = value  # Chroma 简写：值即等值匹配

        return where

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection.
        
        Returns:
            Dict containing collection statistics:
                - count: Number of records in collection
                - name: Collection name
                - metadata: Collection metadata
        """
        return {
            'count': self.collection.count(),  # 条数
            'name': self.collection_name,  # 逻辑名
            'metadata': self.collection.metadata  # 含 hnsw:space 等
        }

    def get_by_ids(
        self,
        ids: List[str],  # 通常是向量库里的 chunk id（与 BM25 对齐后）
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:  # 与 ids 同序；找不到的位置是 {}
        """Retrieve records by their IDs from ChromaDB.
        
        This method is used by SparseRetriever to fetch text and metadata
        for chunks that were matched by BM25 search.
        
        Args:
            ids: List of record IDs to retrieve.
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters (unused for Chroma).
        
        Returns:
            List of records in the same order as input ids.
            Each record contains:
                - 'id': Record identifier
                - 'text': The stored text content
                - 'metadata': Associated metadata
            If an ID is not found, an empty dict is returned for that position.
        
        Raises:
            ValueError: If ids list is empty.
            RuntimeError: If the retrieval operation fails.
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")

        # Ensure all IDs are strings
        str_ids = [str(id_) for id_ in ids]  # 与 upsert 时 id 类型一致

        try:
            # ChromaDB's get method retrieves records by IDs
            results = self.collection.get(
                ids=str_ids,
                include=["metadatas", "documents"]  # 稀疏检索命中后要拼回正文
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get records by IDs from ChromaDB: {e}"
            ) from e

        # Build a mapping from ID to result for O(1) lookup
        id_to_result: Dict[str, Dict[str, Any]] = {}

        if results and results.get('ids'):
            result_ids = results['ids']  # Chroma 返回顺序不一定等于请求顺序
            documents = results.get('documents', [None] * len(result_ids))
            metadatas = results.get('metadatas', [{}] * len(result_ids))

            for i, record_id in enumerate(result_ids):
                id_to_result[record_id] = {
                    'id': record_id,
                    'text': documents[i] if documents and documents[i] else '',
                    'metadata': metadatas[i] if metadatas and metadatas[i] else {}
                }

        # Return results in the same order as input ids
        output = []
        for id_ in str_ids:
            if id_ in id_to_result:
                output.append(id_to_result[id_])
            else:
                # ID not found, return empty dict
                output.append({})  # 调用方按位置判断是否命中

        logger.debug(f"Retrieved {len([r for r in output if r])} of {len(ids)} records by IDs")
        return output
