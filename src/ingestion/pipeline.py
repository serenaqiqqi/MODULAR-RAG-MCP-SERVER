"""Ingestion Pipeline orchestrator for the Modular RAG MCP Server.

This module implements the main pipeline that orchestrates the complete
document ingestion flow:
    1. File Integrity Check (SHA256 skip check)
    2. Document Loading (PDF → Document)
    3. Chunking (Document → Chunks)
    4. Transform (Refine + Enrich + Caption)
    5. Encoding (Dense + Sparse vectors)
    6. Storage (VectorStore + BM25 Index + ImageStorage)

Design Principles:
- Config-Driven: All components configured via settings.yaml
- Observable: Logs progress and stage completion
- Graceful Degradation: LLM failures don't block pipeline
- Idempotent: SHA256-based skip for unchanged files
"""
# 上文：模块级文档字符串，说明本文件职责、六步流水线与设计原则（配置驱动、可观测、降级、幂等）。

from pathlib import Path  # 面向对象的路径操作，便于跨平台与拼接
from typing import Callable, List, Optional, Dict, Any  # 类型注解：回调、列表、可选、字典、任意
import time  # 单调时钟，用于各阶段耗时统计（毫秒）

from src.core.settings import Settings, load_settings, resolve_path  # 配置模型、加载配置、解析项目内相对路径
from src.core.types import Document, Chunk  # 核心数据：整篇文档与文本块
from src.core.trace.trace_context import TraceContext  # 可选：端到端追踪/可观测上下文
from src.observability.logger import get_logger  # 统一日志工厂

# Libs layer imports —— 基础设施层：校验、加载、向量后端工厂等
from src.libs.loader.file_integrity import SQLiteIntegrityChecker  # SQLite 记录文件哈希，支持跳过与成功/失败标记
from src.libs.loader.pdf_loader import PdfLoader  # PDF 解析为 Document，并可落盘图片
from src.libs.embedding.embedding_factory import EmbeddingFactory  # 按 settings 创建稠密嵌入后端
from src.libs.vector_store.vector_store_factory import VectorStoreFactory  # 向量库工厂（本文件未直接调用，保留依赖一致性时可删）

# Ingestion layer imports —— 入库层：分块、变换、编码、存储
from src.ingestion.chunking.document_chunker import DocumentChunker  # 将 Document 切分为 Chunk 列表
from src.ingestion.transform.chunk_refiner import ChunkRefiner  # 块级文本精炼（规则/LLM）
from src.ingestion.transform.metadata_enricher import MetadataEnricher  # 元数据增强（标题、标签等）
from src.ingestion.transform.image_captioner import ImageCaptioner  # 多模态：为图中占位块生成说明
from src.ingestion.embedding.dense_encoder import DenseEncoder  # 稠密向量编码封装
from src.ingestion.embedding.sparse_encoder import SparseEncoder  # 稀疏/BM25 统计特征
from src.ingestion.embedding.batch_processor import BatchProcessor  # 批量稠密+稀疏，统一调度
from src.ingestion.storage.bm25_indexer import BM25Indexer  # 关键词检索用的 BM25 索引写入
from src.ingestion.storage.vector_upserter import VectorUpserter  # 向量库 upsert（如 Chroma）
from src.ingestion.storage.image_storage import ImageStorage  # 图片路径与元数据登记

logger = get_logger(__name__)  # 本模块 logger，日志名称为包路径


class PipelineResult:
    """Result of pipeline execution with detailed statistics.
    
    Attributes:
        success: Whether pipeline completed successfully
        file_path: Path to the processed file
        doc_id: Document ID (SHA256 hash)
        chunk_count: Number of chunks generated
        image_count: Number of images processed
        vector_ids: List of vector IDs stored
        error: Error message if pipeline failed
        stages: Dict of stage names to their individual results
    """
    # 单次流水线执行结果；docstring 列出各字段语义，供 IDE/文档生成使用。

    def __init__(
        self,
        success: bool,
        file_path: str,
        doc_id: Optional[str] = None,
        chunk_count: int = 0,
        image_count: int = 0,
        vector_ids: Optional[List[str]] = None,
        error: Optional[str] = None,
        stages: Optional[Dict[str, Any]] = None
    ):
        self.success = success  # 是否整体成功（含“跳过也算成功”）
        self.file_path = file_path  # 被处理文件的字符串路径
        self.doc_id = doc_id  # 文档标识，此处与文件 SHA256 一致
        self.chunk_count = chunk_count  # 生成的 chunk 数量
        self.image_count = image_count  # 关联图片数量（来自 Document.metadata）
        self.vector_ids = vector_ids or []  # 入库后的向量 id 列表；None 时退化为空列表
        self.error = error  # 失败时的错误信息字符串
        self.stages = stages or {}  # 各阶段摘要统计；None 时退化为空 dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,  # 布尔成功标志
            "file_path": self.file_path,  # 文件路径
            "doc_id": self.doc_id,  # 文档 id
            "chunk_count": self.chunk_count,  # chunk 数
            "image_count": self.image_count,  # 图片数
            "vector_ids_count": len(self.vector_ids),  # 只序列化数量，避免向量 id 列表过大
            "error": self.error,  # 错误信息
            "stages": self.stages  # 分阶段详情
        }


class IngestionPipeline:
    """Main pipeline orchestrator for document ingestion.
    
    This class coordinates all stages of the ingestion process:
    - File integrity checking for incremental processing
    - Document loading (PDF with image extraction)
    - Text chunking with configurable splitter
    - Chunk refinement (rule-based + LLM)
    - Metadata enrichment (rule-based + LLM)
    - Image captioning (Vision LLM)
    - Dense embedding (Azure text-embedding-ada-002)
    - Sparse encoding (BM25 term statistics)
    - Vector storage (ChromaDB)
    - BM25 index building
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> pipeline = IngestionPipeline(settings)
        >>> result = pipeline.run("documents/report.pdf", collection="contracts")
        >>> print(f"Processed {result.chunk_count} chunks")
    """
    # 编排器类：__init__ 装配依赖，run 执行六阶段，close 释放资源。

    def __init__(
        self,
        settings: Settings,
        collection: str = "default",
        force: bool = False
    ):
        """Initialize pipeline with all components.
        
        Args:
            settings: Application settings from settings.yaml
            collection: Collection name for organizing documents
            force: If True, re-process even if file was previously processed
        """
        self.settings = settings  # 全局配置对象
        self.collection = collection  # 集合名：向量库/BM25/图片子目录隔离
        self.force = force  # True 时忽略“已处理跳过”逻辑

        # Initialize all components
        logger.info("Initializing Ingestion Pipeline components...")  # 启动日志

        # Stage 1: File Integrity：给文件算一个唯一ID（hash），看这个文件之前有没有处理过，避免重复入库（向量爆炸）
        self.integrity_checker = SQLiteIntegrityChecker(db_path=str(resolve_path("data/db/ingestion_history.db")))  # 持久化哈希与状态
        logger.info("  ✓ FileIntegrityChecker initialized")  # 组件就绪日志

        # Stage 2: Loader
        self.loader = PdfLoader(
            extract_images=True,  # 从 PDF 抽取图片并落盘
            image_storage_dir=str(resolve_path(f"data/images/{collection}"))  # 按 collection 分目录存图
        )
        logger.info("  ✓ PdfLoader initialized")  # 加载器就绪

        # Stage 3: Chunker
        self.chunker = DocumentChunker(settings)  # 按 settings 中的分块策略切分
        logger.info("  ✓ DocumentChunker initialized")  # 分块器就绪

        # Stage 4: Transforms
        self.chunk_refiner = ChunkRefiner(settings)  # 块文本精炼
        logger.info(f"  ✓ ChunkRefiner initialized (use_llm={self.chunk_refiner.use_llm})")  # 是否启用 LLM

        self.metadata_enricher = MetadataEnricher(settings)  # 元数据增强
        logger.info(f"  ✓ MetadataEnricher initialized (use_llm={self.metadata_enricher.use_llm})")  # LLM 开关

        self.image_captioner = ImageCaptioner(settings)  # 图片描述生成
        has_vision = self.image_captioner.llm is not None  # 是否配置了视觉/多模态模型
        logger.info(f"  ✓ ImageCaptioner initialized (vision_enabled={has_vision})")  # 能力摘要

        # Stage 5: Encoders
        embedding = EmbeddingFactory.create(settings)  # 创建具体 Embedding 客户端（OpenAI/Azure 等）
        batch_size = settings.ingestion.batch_size if settings.ingestion else 100  # 批大小，缺省 100
        self.dense_encoder = DenseEncoder(embedding, batch_size=batch_size)  # 稠密向量批编码
        logger.info(f"  ✓ DenseEncoder initialized (provider={settings.embedding.provider})")  # 提供商名

        self.sparse_encoder = SparseEncoder()  # BM25 用词频等统计
        logger.info("  ✓ SparseEncoder initialized")  # 稀疏编码器就绪

        self.batch_processor = BatchProcessor(
            dense_encoder=self.dense_encoder,  # 注入稠密编码器
            sparse_encoder=self.sparse_encoder,  # 注入稀疏编码器
            batch_size=batch_size  # 与稠密侧批大小一致
        )
        logger.info(f"  ✓ BatchProcessor initialized (batch_size={batch_size})")  # 批处理器就绪

        # Stage 6: Storage
        self.vector_upserter = VectorUpserter(settings, collection_name=collection)  # 向量库写入，绑定 collection
        logger.info(f"  ✓ VectorUpserter initialized (provider={settings.vector_store.provider}, collection={collection})")  # 向量后端

        self.bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))  # BM25 索引目录按 collection
        logger.info("  ✓ BM25Indexer initialized")  # BM25 就绪

        self.image_storage = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),  # 图片索引 SQLite
            images_root=str(resolve_path("data/images"))  # 图片根目录，与 PdfLoader 落盘路径一致
        )
        logger.info("  ✓ ImageStorage initialized")  # 图片索引就绪

        logger.info("Pipeline initialization complete!")  # 全部组件初始化完成

    def run(
        self,
        file_path: str,
        trace: Optional[TraceContext] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> PipelineResult:
        """Execute the full ingestion pipeline on a file.
        
        Args:
            file_path: Path to the file to process (e.g., PDF)
            trace: Optional trace context for observability
            on_progress: Optional callback ``(stage_name, current, total)``
                invoked when each pipeline stage completes.  *current* is
                the 1-based index of the completed stage; *total* is the
                number of stages (currently 6).
        
        Returns:
            PipelineResult with success status and statistics
        """
        file_path = Path(file_path)  # 转为 Path 便于后续统一 str()
        stages: Dict[str, Any] = {}  # 累积各阶段统计，最终放入 PipelineResult
        _total_stages = 6  # 进度回调中的总阶段数（固定六步）

        def _notify(stage_name: str, step: int) -> None:
            if on_progress is not None:  # 调用方未传回调则跳过
                on_progress(stage_name, step, _total_stages)  # 通知当前阶段名与步号

        logger.info(f"=" * 60)  # 分隔线
        logger.info(f"Starting Ingestion Pipeline for: {file_path}")  # 开始处理哪个文件
        logger.info(f"Collection: {self.collection}")  # 当前集合名
        logger.info(f"=" * 60)  # 分隔线

        try:
            # ─────────────────────────────────────────────────────────────
            # Stage 1: File Integrity Check  完整性 → 加载 → 分块 → 变换 → 编码 → 存储 → 标记成功。
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📋 Stage 1: File Integrity Check")  # 阶段标题
            _notify("integrity", 1)  # 进度：阶段 1

            file_hash = self.integrity_checker.compute_sha256(str(file_path))  # 全文 SHA256，作 doc_id 与去重键
            logger.info(f"  File hash: {file_hash[:16]}...")  # 日志只打前 16 位

            if not self.force and self.integrity_checker.should_skip(file_hash):  # 非强制且库中已成功处理过则跳过
                logger.info(f"  ⏭️  File already processed, skipping (use force=True to reprocess)")  # 说明跳过原因
                return PipelineResult(
                    success=True,  # 跳过视为成功
                    file_path=str(file_path),  # 路径字符串
                    doc_id=file_hash,  # 哈希即文档 id
                    stages={"integrity": {"skipped": True, "reason": "already_processed"}}  # 仅记录跳过信息
                )

            stages["integrity"] = {"file_hash": file_hash, "skipped": False}  # 需要完整跑流水线
            logger.info("  ✓ File needs processing")  # 确认继续

            # ─────────────────────────────────────────────────────────────
            # Stage 2: Document Loading
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📄 Stage 2: Document Loading")  # 阶段标题
            _notify("load", 2)  # 进度：阶段 2

            _t0 = time.monotonic()  # 加载开始单调时间
            document = self.loader.load(str(file_path))  # PDF → Document（文本+图片元数据）
            """
            document的类型
              return Document(
                id=doc_id,
                text=text_content,
                metadata=metadata
            )
            """
            _elapsed = (time.monotonic() - _t0) * 1000.0  # 耗时毫秒

            text_preview = document.text[:200].replace('\n', ' ') + "..." if len(document.text) > 200 else document.text  # 预览截断
            image_count = len(document.metadata.get("images", []))  # 抽取的图片条目数

            logger.info(f"  Document ID: {document.id}")  # Loader 赋予的文档 id
            logger.info(f"  Text length: {len(document.text)} chars")  # 正文长度
            logger.info(f"  Images extracted: {image_count}")  # 图片数量
            logger.info(f"  Preview: {text_preview[:100]}...")  # 更短预览

            stages["loading"] = {
                "doc_id": document.id,  # 文档 id
                "text_length": len(document.text),  # 字符数
                "image_count": image_count  # 图片数
            }
            if trace is not None:  # 若需要追踪
                trace.record_stage("load", {
                    "method": "markitdown",  # 与 Loader 实现对应的标记（历史命名）
                    "doc_id": document.id,
                    "text_length": len(document.text),
                    "image_count": image_count,
                    "text_preview": document.text,  # 追踪里保留全文预览
                }, elapsed_ms=_elapsed)  # 记录耗时

            # ─────────────────────────────────────────────────────────────
            # Stage 3: Chunking
            # ─────────────────────────────────────────────────────────────
            logger.info("\n✂️  Stage 3: Document Chunking")  # 阶段标题
            _notify("split", 3)  # 进度：阶段 3

            _t0 = time.monotonic()  # 分块开始时间
            chunks = self.chunker.split_document(document)  # Document → List[Chunk]
            _elapsed = (time.monotonic() - _t0) * 1000.0  # 分块耗时毫秒

            logger.info(f"  Chunks generated: {len(chunks)}")  # chunk 总数
            if chunks:  # 非空时打样例
                logger.info(f"  First chunk ID: {chunks[0].id}")  # 第一个 chunk id
                logger.info(f"  First chunk preview: {chunks[0].text[:100]}...")  # 首块前 100 字

            stages["chunking"] = {
                "chunk_count": len(chunks),  # 块数
                "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0  # 平均字符数
            }
            if trace is not None:
                trace.record_stage("split", {
                    "method": "recursive",  # 分块策略说明
                    "chunk_count": len(chunks),
                    "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text": c.text,
                            "char_len": len(c.text),
                            "chunk_index": c.metadata.get("chunk_index", i),
                        }
                        for i, c in enumerate(chunks)  # 枚举每个 chunk 详情
                    ],
                }, elapsed_ms=_elapsed)

            # ─────────────────────────────────────────────────────────────
            # Stage 4: Transform Pipeline
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔄 Stage 4: Transform Pipeline")  # 阶段标题
            _notify("transform", 4)  # 进度：阶段 4

            # 4a: Chunk Refinement
            logger.info("  4a. Chunk Refinement...")  # 子步骤 4a
            _t0_transform = time.monotonic()  # 整段 transform 起始时间
            # snapshot before refinement
            _pre_refine_texts = {c.id: c.text for c in chunks}  # 精炼前文本快照，供 trace 对比
            chunks = self.chunk_refiner.transform(chunks, trace)  # 原地/替换式更新 chunks
            refined_by_llm = sum(1 for c in chunks if c.metadata.get("refined_by") == "llm")  # LLM 精炼块数
            refined_by_rule = sum(1 for c in chunks if c.metadata.get("refined_by") == "rule")  # 规则精炼块数
            logger.info(f"      LLM refined: {refined_by_llm}, Rule refined: {refined_by_rule}")  # 统计日志

            # 4b: Metadata Enrichment
            logger.info("  4b. Metadata Enrichment...")  # 子步骤 4b
            chunks = self.metadata_enricher.transform(chunks, trace)  # 写入 title/tags/summary 等
            enriched_by_llm = sum(1 for c in chunks if c.metadata.get("enriched_by") == "llm")  # LLM 增强块数
            enriched_by_rule = sum(1 for c in chunks if c.metadata.get("enriched_by") == "rule")  # 规则增强块数
            logger.info(f"      LLM enriched: {enriched_by_llm}, Rule enriched: {enriched_by_rule}")  # 统计日志

            # 4c: Image Captioning
            logger.info("  4c. Image Captioning...")  # 子步骤 4c
            chunks = self.image_captioner.transform(chunks, trace)  # 为含图块写 caption
            captioned = sum(1 for c in chunks if c.metadata.get("image_captions"))  # 含 caption 元数据的块数
            logger.info(f"      Chunks with captions: {captioned}")  # 统计日志

            stages["transform"] = {
                "chunk_refiner": {"llm": refined_by_llm, "rule": refined_by_rule},  # 精炼统计
                "metadata_enricher": {"llm": enriched_by_llm, "rule": enriched_by_rule},  # 增强统计
                "image_captioner": {"captioned_chunks": captioned}  # 配图统计
            }
            _elapsed_transform = (time.monotonic() - _t0_transform) * 1000.0  # transform 总耗时
            if trace is not None:
                trace.record_stage("transform", {
                    "method": "refine+enrich+caption",  # 组合方法名
                    "refined_by_llm": refined_by_llm,
                    "refined_by_rule": refined_by_rule,
                    "enriched_by_llm": enriched_by_llm,
                    "enriched_by_rule": enriched_by_rule,
                    "captioned_chunks": captioned,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text_before": _pre_refine_texts.get(c.id, ""),  # 精炼前
                            "text_after": c.text,  # 精炼后
                            "char_len": len(c.text),
                            "refined_by": c.metadata.get("refined_by", ""),
                            "enriched_by": c.metadata.get("enriched_by", ""),
                            "title": c.metadata.get("title", ""),
                            "tags": c.metadata.get("tags", []),
                            "summary": c.metadata.get("summary", ""),
                        }
                        for c in chunks
                    ],
                }, elapsed_ms=_elapsed_transform)

            # ─────────────────────────────────────────────────────────────
            # Stage 5: Encoding
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔢 Stage 5: Encoding")  # 阶段标题
            _notify("embed", 5)  # 进度：阶段 5

            # Process through BatchProcessor
            _t0 = time.monotonic()  # 编码开始时间
            batch_result = self.batch_processor.process(chunks, trace)  # 稠密向量 + 稀疏统计
            _elapsed = (time.monotonic() - _t0) * 1000.0  # 编码耗时

            dense_vectors = batch_result.dense_vectors  # 与 chunks 顺序对齐的向量列表
            sparse_stats = batch_result.sparse_stats  # 每块 BM25 相关统计

            logger.info(f"  Dense vectors: {len(dense_vectors)} (dim={len(dense_vectors[0]) if dense_vectors else 0})")  # 数量与维度
            logger.info(f"  Sparse stats: {len(sparse_stats)} documents")  # 稀疏文档条数

            stages["encoding"] = {
                "dense_vector_count": len(dense_vectors),  # 稠密条数
                "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,  # 向量维度
                "sparse_doc_count": len(sparse_stats)  # 稀疏条目数
            }
            if trace is not None:
                # Build per-chunk encoding details (both dense & sparse)
                chunk_details = []  # 逐块编码摘要列表
                for idx, c in enumerate(chunks):  # 与 dense/sparse 按下标对齐
                    detail: dict = {
                        "chunk_id": c.id,
                        "char_len": len(c.text),
                    }
                    # Dense: vector dimension (same for all, but confirm per-chunk)
                    if idx < len(dense_vectors):  # 防止越界
                        detail["dense_dim"] = len(dense_vectors[idx])  # 该块向量维度
                    # Sparse: BM25 term stats
                    if idx < len(sparse_stats):  # 防止越界
                        ss = sparse_stats[idx]  # 该块稀疏统计 dict
                        detail["doc_length"] = ss.get("doc_length", 0)  # 文档长度（分词后等）
                        detail["unique_terms"] = ss.get("unique_terms", 0)  # 唯一词数
                        # Top-10 terms by frequency for inspection
                        tf = ss.get("term_frequencies", {})  # 词频表
                        top_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]  # 频率 Top10
                        detail["top_terms"] = [{"term": t, "freq": f} for t, f in top_terms]  # 可读的 top 词
                    chunk_details.append(detail)  # 加入列表

                trace.record_stage("embed", {
                    "method": "batch_processor",
                    "dense_vector_count": len(dense_vectors),
                    "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                    "sparse_doc_count": len(sparse_stats),
                    "chunks": chunk_details,
                }, elapsed_ms=_elapsed)

            # ─────────────────────────────────────────────────────────────
            # Stage 6: Storage
            # ─────────────────────────────────────────────────────────────
            logger.info("\n💾 Stage 6: Storage")  # 阶段标题
            _notify("upsert", 6)  # 进度：阶段 6

            # 6a: Vector Upsert
            logger.info("  6a. Vector Storage (ChromaDB)...")  # 子步骤：向量库
            _t0_storage = time.monotonic()  # 存储阶段起始时间
            vector_ids = self.vector_upserter.upsert(chunks, dense_vectors, trace)  # 返回每条入库 id
            logger.info(f"      Stored {len(vector_ids)} vectors")  # 入库条数

            # Align BM25 chunk_ids with Chroma vector IDs so the SparseRetriever
            # can look up BM25 hits in the vector store after retrieval.
            for stat, vid in zip(sparse_stats, vector_ids):  # 稀疏统计与向量 id 一一对应
                stat["chunk_id"] = vid  # 后续稀疏检索命中可用向量库 id 回查

            # 6b: BM25 Index
            logger.info("  6b. BM25 Index...")  # 子步骤：BM25
            self.bm25_indexer.add_documents(
                sparse_stats,  # 已带上与向量库一致的 chunk_id
                collection=self.collection,  # 集合名
                doc_id=document.id,  # 父文档 id
                trace=trace,
            )
            logger.info(f"      Index built for {len(sparse_stats)} documents")  # BM25 文档数

            # 6c: Register images in image storage index
            # Note: Images are already saved by PdfLoader, we just need to index them
            logger.info("  6c. Image Storage Index...")  # 子步骤：图片索引
            images = document.metadata.get("images", [])  # 加载阶段写入的图片列表
            for img in images:  # 遍历每张图
                img_path = Path(img["path"])  # 磁盘路径
                if img_path.exists():  # 存在才登记，避免脏数据
                    self.image_storage.register_image(
                        image_id=img["id"],  # 图片唯一 id
                        file_path=img_path,  # Path 对象
                        collection=self.collection,  # 集合
                        doc_hash=file_hash,  # 与完整性哈希关联
                        page_num=img.get("page", 0)  # 页码，缺省 0
                    )
            logger.info(f"      Indexed {len(images)} images")  # 元数据中的图片总数

            stages["storage"] = {
                "vector_count": len(vector_ids),  # 向量条数
                "bm25_docs": len(sparse_stats),  # BM25 条数
                "images_indexed": len(images)  # 图片条数（含路径不存在的未登记）
            }
            _elapsed_storage = (time.monotonic() - _t0_storage) * 1000.0  # 存储阶段总耗时
            if trace is not None:
                # Per-chunk storage mapping: chunk_id → vector_id
                chunk_storage = [
                    {
                        "chunk_id": c.id,  # 逻辑 chunk id
                        "vector_id": vector_ids[i] if i < len(vector_ids) else "—",  # 库内 id，缺则占位
                        "collection": self.collection,
                        "store": "ChromaDB",  # 固定标注后端名
                    }
                    for i, c in enumerate(chunks)
                ]
                # Image storage details
                image_storage_details = [
                    {
                        "image_id": img["id"],
                        "file_path": str(img["path"]),
                        "page": img.get("page", 0),
                        "doc_hash": file_hash,
                    }
                    for img in images
                ]
                trace.record_stage("upsert", {
                    "dense_store": {
                        "backend": "ChromaDB",
                        "collection": self.collection,
                        "count": len(vector_ids),
                        "path": "data/db/chroma/",
                    },
                    "sparse_store": {
                        "backend": "BM25",
                        "collection": self.collection,
                        "count": len(sparse_stats),
                        "path": f"data/db/bm25/{self.collection}/",
                    },
                    "image_store": {
                        "backend": "ImageStorage (JSON index)",
                        "count": len(images),
                        "images": image_storage_details,
                    },
                    "chunk_mapping": chunk_storage,
                }, elapsed_ms=_elapsed_storage)

            # ─────────────────────────────────────────────────────────────
            # Mark Success
            # ─────────────────────────────────────────────────────────────
            self.integrity_checker.mark_success(file_hash, str(file_path), self.collection)  # 持久化“已成功处理”

            logger.info("\n" + "=" * 60)  # 结束分隔
            logger.info("✅ Pipeline completed successfully!")  # 成功总览
            logger.info(f"   Chunks: {len(chunks)}")  # chunk 数
            logger.info(f"   Vectors: {len(vector_ids)}")  # 向量数
            logger.info(f"   Images: {len(images)}")  # 图片数
            logger.info("=" * 60)  # 结束分隔

            return PipelineResult(
                success=True,  # 全流程成功
                file_path=str(file_path),
                doc_id=file_hash,
                chunk_count=len(chunks),
                image_count=len(images),
                vector_ids=vector_ids,
                stages=stages
            )

        except Exception as e:  # 任一阶段未捕获的异常
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)  # 打栈
            self.integrity_checker.mark_failed(file_hash, str(file_path), str(e))  # 记录失败原因（若 file_hash 已定义）

            return PipelineResult(
                success=False,  # 失败
                file_path=str(file_path),
                doc_id=file_hash if 'file_hash' in locals() else None,  # 哈希未算出时无 doc_id
                error=str(e),  # 错误信息
                stages=stages  # 失败前已写入的阶段仍可查看
            )

    def close(self) -> None:
        """Clean up resources."""
        self.image_storage.close()  # 关闭图片索引 DB 连接等资源


def run_pipeline(
    file_path: str,
    settings_path: Optional[str] = None,
    collection: str = "default",
    force: bool = False
) -> PipelineResult:
    """Convenience function to run the pipeline.
    
    Args:
        file_path: Path to file to process
        settings_path: Path to settings.yaml (default: <repo>/config/settings.yaml)
        collection: Collection name
        force: Force reprocessing
    
    Returns:
        PipelineResult with execution details
    """
    settings = load_settings(settings_path)  # 加载 yaml；None 时用项目默认路径
    pipeline = IngestionPipeline(settings, collection=collection, force=force)  # 构造编排器

    try:
        return pipeline.run(file_path)  # 执行主流程并返回结果
    finally:
        pipeline.close()  # 无论成功失败都释放资源
