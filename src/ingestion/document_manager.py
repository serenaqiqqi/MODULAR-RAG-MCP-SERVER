"""Cross-store document lifecycle management.

This module provides a single entry-point for listing, inspecting, and
deleting documents across all four storage backends (ChromaDB, BM25,
ImageStorage, FileIntegrityChecker).

Design Principles:
- Coordinated: one call cascades into all relevant stores.
- Fail-safe: partial failures are reported but do not abort remaining stores.
- Read-only safe: list / stats / detail methods never mutate data.
"""
# 上文：跨四类存储编排文档列表/详情/删除；协调调用、失败可部分成功、只读接口不修改数据。

from __future__ import annotations  # 允许类体内前向引用类型注解

import logging  # 标准库日志
from dataclasses import dataclass, field  # 数据类与可变默认字段工厂
from typing import Any, Dict, List, Optional  # 类型注解

logger = logging.getLogger(__name__)  # 本模块 logger


# ---------------------------------------------------------------------------
# Result data-classes
# ---------------------------------------------------------------------------
# 以下为对外返回/传递用的数据结构定义

@dataclass
class DocumentInfo:
    """Summary information about an ingested document."""
    # 单条已摄取文档的摘要（列表视图用）

    source_path: str  # 原始文件路径
    source_hash: str  # 文件内容 SHA256，与 pipeline 中 doc_id 一致
    collection: Optional[str] = None  # 所属集合，可为空
    chunk_count: int = 0  # Chroma 中该文档对应的 chunk 条数
    image_count: int = 0  # 该文档关联图片数量
    processed_at: Optional[str] = None  # 完整性表中记录的处理时间戳


@dataclass
class DocumentDetail(DocumentInfo):
    """Extended document info including chunk IDs and image IDs."""
    # 在 DocumentInfo 基础上增加 chunk/图片 id 列表（详情视图用）

    chunk_ids: List[str] = field(default_factory=list)  # 向量库中的 chunk/向量 id
    image_ids: List[str] = field(default_factory=list)  # 图片存储中的 image id


@dataclass
class DeleteResult:
    """Outcome of a delete_document operation."""
    # delete_document 的汇总结果与各存储是否成功

    success: bool  # 是否整体视为成功（有 errors 时会被置为 False）
    chunks_deleted: int = 0  # Chroma 删除的条数
    bm25_removed: bool = False  # BM25 是否成功移除该文档 posting
    images_deleted: int = 0  # 实际删除的图片条目数
    integrity_removed: bool = False  # 完整性表记录是否删除
    errors: List[str] = field(default_factory=list)  # 各存储报错信息列表


@dataclass
class CollectionStats:
    """Aggregate statistics for a collection."""
    # 某个集合（或全库）的聚合统计

    collection: Optional[str] = None  # 集合名；None 表示跨集合汇总
    document_count: int = 0  # 文档篇数
    chunk_count: int = 0  # chunk 总数
    image_count: int = 0  # 图片总数


# ---------------------------------------------------------------------------
# DocumentManager
# ---------------------------------------------------------------------------
# 协调四类后端：向量、BM25、图片索引、摄取历史

class DocumentManager:
    """Coordinate document lifecycle across all storage backends.

    Args:
        chroma_store: ChromaStore instance (vector store).
        bm25_indexer: BM25Indexer instance (sparse index).
        image_storage: ImageStorage instance (image files + SQLite index).
        file_integrity: SQLiteIntegrityChecker instance (ingestion history).
    """
    # 构造时注入四个组件；删除时按序清理各存储并收集错误

    def __init__(
        self,
        chroma_store: Any,
        bm25_indexer: Any,
        image_storage: Any,
        file_integrity: Any,
    ) -> None:
        self.chroma = chroma_store  # 向量库封装（需有 delete_by_metadata、collection）
        self.bm25 = bm25_indexer  # BM25 索引器
        self.images = image_storage  # 图片存储与 SQLite 索引
        self.integrity = file_integrity  # 摄取历史 SQLite

    # ------------------------------------------------------------------
    # list_documents：
    #输入：可选的 collection（筛选某个库）
    #输出：DocumentInfo 列表，每个元素包含 source_path、source_hash、collection、chunk_count、image_count、processed_at
    #DocumentInfo 数据结构：
    #source_path: 原始文件路径
    #source_hash: 文件内容 SHA256，与 pipeline 中 doc_id 一致
    #collection: 所属集合，可为空
    #chunk_count: Chroma 中该文档对应的 chunk 条数
    #image_count: 该文档关联图片数量
    #processed_at: 完整性表中记录的处理时间戳
    # ------------------------------------------------------------------

    def list_documents(
        self, collection: Optional[str] = None
    ) -> List[DocumentInfo]:
        """Return a list of ingested documents.

        Combines information from the integrity checker (source_path,
        hash, processed_at) with counts from ChromaDB and ImageStorage.

        Args:
            collection: Optional collection filter.

        Returns:
            List of ``DocumentInfo`` objects.
        """
        records = self.integrity.list_processed(collection)  # 从历史表取已成功处理的记录

        docs: List[DocumentInfo] = []
        for rec in records:  # 每条记录对应一篇逻辑文档
            source_hash = rec["file_hash"]  # 内容哈希作为主键
            source_path = rec["file_path"]  # 入库时的路径
            coll = rec.get("collection")  # 集合名

            # Count chunks in Chroma
            chunk_count = self._count_chunks(source_hash)  # 按 metadata.doc_hash 统计

            # Count images
            image_count = self._count_images(source_hash)  # 按 doc_hash 在图片索引中统计

            docs.append(
                DocumentInfo(
                    source_path=source_path,
                    source_hash=source_hash,
                    collection=coll,
                    chunk_count=chunk_count,
                    image_count=image_count,
                    processed_at=rec.get("processed_at"),
                )
            )

        return docs

    # ------------------------------------------------------------------
    # get_document_detail
    # ------------------------------------------------------------------

    def get_document_detail(self, doc_id: str) -> Optional[DocumentDetail]:
        """Get detailed information about a single document.

        *doc_id* is matched against the ``source_hash`` stored in the
        integrity checker.

        Args:
            doc_id: The document's source_hash.

        Returns:
            ``DocumentDetail`` with chunk/image IDs, or *None* if not found.
        """
        # Look up integrity record
        all_records = self.integrity.list_processed()  # 无集合过滤，遍历匹配 doc_id
        record = None
        for rec in all_records:
            if rec["file_hash"] == doc_id:  # doc_id 即 source_hash
                record = rec
                break

        if record is None:
            return None  # 未找到摄取记录

        source_hash = record["file_hash"]

        # Collect chunk IDs from Chroma
        chunk_ids = self._get_chunk_ids(source_hash)

        # Collect image IDs
        image_ids = self._get_image_ids(source_hash)

        return DocumentDetail(
            source_path=record["file_path"],
            source_hash=source_hash,
            collection=record.get("collection"),
            chunk_count=len(chunk_ids),
            image_count=len(image_ids),
            processed_at=record.get("processed_at"),
            chunk_ids=chunk_ids,
            image_ids=image_ids,
        )

    # ------------------------------------------------------------------
    # delete_document
    # ------------------------------------------------------------------

    def delete_document(
        self,
        source_path: str, #原始文件路径
        collection: str = "default", #所属集合，可为空
        source_hash: Optional[str] = None, #预计算的 SHA256 哈希
    ) -> DeleteResult:
        """Delete a document from all storage backends.

        Coordinates deletion across ChromaDB, BM25, ImageStorage, and
        FileIntegrity.  Partial failures are captured in
        ``DeleteResult.errors`` but do not prevent remaining stores
        from being cleaned.

        The document is identified by its *source_hash*.  When the hash
        is not supplied the method tries to compute it from the file;
        if the file no longer exists it falls back to looking up the
        hash from the integrity records by path.

        Args:
            source_path: Original filesystem path of the document.
            collection: Collection the document belongs to.
            source_hash: Pre-computed SHA-256 hash.  When provided the
                method will not attempt to read the source file.

        Returns:
            ``DeleteResult`` summarising what was cleaned.
        """
        result = DeleteResult(success=True)  # 初始假设成功，有错误时再覆盖
        #如果 source_hash 为空，则尝试从文件计算哈希，如果文件不存在，则从完整性表中根据路径查找 source_hash

        # Resolve hash – prefer caller-supplied, then file, then DB lookup
        if source_hash is None:
            try:
                source_hash = self.integrity.compute_sha256(source_path)  # 文件仍存在则直接算哈希
            except Exception as e:
                source_hash = self._hash_from_path(source_path)  # 文件已删则从库按路径反查
                if source_hash is None:
                    result.success = False
                    result.errors.append(f"Cannot identify document: {e}")
                    return result  # 无法确定文档身份则提前返回

        # 1. ChromaDB – delete chunks matching source_hash
        try:
            count = self.chroma.delete_by_metadata(
                {"doc_hash": source_hash}  # 与入库 metadata 键一致
            )
            result.chunks_deleted = count
        except Exception as e:
            result.errors.append(f"ChromaDB delete failed: {e}")  # 记录但继续后续存储

        # 2. BM25 – remove postings for this document
        try:
            result.bm25_removed = self.bm25.remove_document(
                source_hash, collection
            )
        except Exception as e:
            result.errors.append(f"BM25 remove failed: {e}")

        # 3. ImageStorage – delete images by doc_hash
        try:
            images = self.images.list_images(doc_hash=source_hash)  # 先枚举再逐条删
            deleted_imgs = 0
            for img in images:
                if self.images.delete_image(img["image_id"]):  # 返回是否删除成功
                    deleted_imgs += 1
            result.images_deleted = deleted_imgs
        except Exception as e:
            result.errors.append(f"ImageStorage delete failed: {e}")

        # 4. FileIntegrity – remove the ingestion record
        try:
            result.integrity_removed = self.integrity.remove_record(
                source_hash
            )
        except Exception as e:
            result.errors.append(f"FileIntegrity remove failed: {e}")

        if result.errors:
            result.success = False  # 任一存储报错则整体标为失败（已尽力清理）

        return result

    # ------------------------------------------------------------------
    # get_collection_stats
    # ------------------------------------------------------------------

    def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> CollectionStats:
        """Return aggregate statistics for a collection.

        Args:
            collection: Collection name.  When *None*, stats span
                all collections.

        Returns:
            ``CollectionStats`` dataclass.
        """
        docs = self.list_documents(collection)  # 复用列表逻辑
        chunk_total = sum(d.chunk_count for d in docs)  # 累加 chunk
        image_total = sum(d.image_count for d in docs)  # 累加图片

        return CollectionStats(
            collection=collection,
            document_count=len(docs),
            chunk_count=chunk_total,
            image_count=image_total,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    #去向量库（Chroma）里查这个文档有多少个 chunk
    # _get_chunk_ids: 获取 Chroma 中属于某个 source_hash 的 chunk id 列表
    # _count_images: 统计 ImageStorage 中属于某个 source_hash 的图片数量
    # _get_image_ids: 获取 ImageStorage 中属于某个 source_hash 的图片 id 列表
    # _hash_from_path: 从完整性表中根据路径查找 source_hash
    def _count_chunks(self, source_hash: str) -> int:
        """Count chunks in Chroma that belong to *source_hash*."""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]  # 只要 id，不要向量/文档
            )
            return len(results.get("ids", []))
        except Exception:
            return 0  # 查询失败时保守返回 0

    def _get_chunk_ids(self, source_hash: str) -> List[str]:
        """Return chunk IDs from Chroma matching *source_hash*."""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return results.get("ids", [])
        except Exception:
            return []

    #去图片存储里查这个文档有多少张图片
    #输入：source_hash
    #输出：图片数量
    def _count_images(self, source_hash: str) -> int:
        """Count images belonging to *source_hash*."""
        try:
            return len(self.images.list_images(doc_hash=source_hash))
        except Exception:
            return 0

    #去图片存储里查这个文档有多少张图片
    #输入：source_hash
    #输出：图片 id 列表
    def _get_image_ids(self, source_hash: str) -> List[str]:
        """Return image IDs belonging to *source_hash*."""
        try:
            imgs = self.images.list_images(doc_hash=source_hash)
            return [img["image_id"] for img in imgs]  # 提取 id 列表
        except Exception:
            return []
    #从完整性表中根据路径查找 source_hash
    #输入：source_path
    #输出：source_hash
    def _hash_from_path(self, source_path: str) -> Optional[str]:
        """Try to find a source_hash from integrity records by path."""
        try:
            for rec in self.integrity.list_processed():  # 遍历成功记录匹配路径
                if rec["file_path"] == source_path:
                    return rec["file_hash"]
        except Exception:
            pass  # 忽略遍历异常
        return None  # 未找到
