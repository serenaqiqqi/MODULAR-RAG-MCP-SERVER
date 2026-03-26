"""Core data types and contracts for the entire pipeline.

This module defines the fundamental data structures used across all pipeline stages:
- ingestion (loaders, transforms, embedding, storage)
- retrieval (query engine, search, reranking)
- mcp_server (tools, response formatting)

Design Principles:
- Centralized contracts: All stages use these types to avoid coupling
- Serializable: All types support dict/JSON conversion
- Extensible metadata: Minimum required fields with flexible extension
- Type-safe: Full type hints for static analysis
"""

# dataclass: 声明结构化对象；field: 声明默认工厂；asdict: 对象转字典。
from dataclasses import dataclass, field, asdict
# 类型注解：字典、列表、可选类型等。
from typing import Dict, Any, List, Optional


# 定义文档对象，表示 Loader 的输出。
@dataclass
class Document:
    """Represents a raw document loaded from source.

    This is the output of Loaders (e.g., PDF Loader) before splitting.

    Attributes:
        id: Unique identifier for the document (e.g., file hash or path-based ID)
        text: Document content in standardized Markdown format.
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Document-level metadata including:
            - source_path (required): Original file path
            - doc_type: Document type (e.g., 'pdf', 'markdown')
            - title: Document title extracted or inferred
            - page_count: Total pages (if applicable)
            - images: List of image references (see Images Field Specification below)
            - Any other custom metadata

    Images Field Specification (metadata.images):
        Structure: List[{"id": str, "path": str, "page": int, "text_offset": int,
                        "text_length": int, "position": dict}]
        Fields:
            - id: Unique image identifier (format: {doc_hash}_{page}_{seq})
            - path: Image file storage path (convention: data/images/{collection}/{image_id}.png)
            - page: Page number in original document (optional, for paginated docs like PDF)
            - text_offset: Starting character position of placeholder in Document.text (0-based)
            - text_length: Length of placeholder string (typically len("[IMAGE: {image_id}]"))
            - position: Physical position info in original doc (optional, e.g., PDF coords, pixel position)
        Note: text_offset and text_length enable precise placeholder location,
              supporting scenarios where the same image appears multiple times

    Example:
        >>> doc = Document(
        ...     id="doc_abc123",
        ...     text="# Title\\n\\nContent...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "doc_type": "pdf",
        ...         "title": "Annual Report 2025"
        ...     }
        ... )
    """

    # 文档唯一标识。
    id: str
    # 文档正文文本。
    text: str
    # 文档元数据，默认空 dict。
    metadata: Dict[str, Any] = field(default_factory=dict)

    # dataclass 初始化后自动执行字段校验。
    def __post_init__(self):
        """Validate required metadata fields."""
        # 保证后续链路总能追溯源文件路径。
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")

    # 序列化：对象 -> dict。
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # asdict 会递归转换 dataclass 字段。
        return asdict(self)

    # 反序列化：dict -> 对象。
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        # 直接展开参数构造。
        return cls(**data)


# 定义分块对象，表示 Splitter 输出。
@dataclass
class Chunk:
    """Represents a text chunk after splitting a Document.

    This is the output of Splitters and input to Transform pipeline.
    Each chunk maintains traceability to its source document.

    Attributes:
        id: Unique chunk identifier (e.g., hash-based or sequential)
        text: Chunk content (subset of original document text).
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Chunk-level metadata inherited and extended from Document:
            - source_path (required): Original file path
            - chunk_index: Sequential position in document (0-based)
            - start_offset: Character offset in original document (optional)
            - end_offset: Character offset in original document (optional)
            - source_ref: Reference to parent document ID (optional)
            - images: Subset of Document.images that fall within this chunk (optional)
            - Any document-level metadata propagated from Document
        start_offset: Starting character position in original document (optional)
        end_offset: Ending character position in original document (optional)
        source_ref: Reference to parent Document.id (optional)

    Note: If chunk contains image placeholders, metadata.images should contain
          only the image references relevant to this chunk's text range.

    Example:
        >>> chunk = Chunk(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "page": 1
        ...     },
        ...     start_offset=0,
        ...     end_offset=150
        ... )
    """

    # 分块唯一标识。
    id: str
    # 分块文本内容。
    text: str
    # 分块元数据，默认空 dict。
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 在原文中的起始字符偏移（可选）。
    start_offset: Optional[int] = None
    # 在原文中的结束字符偏移（可选）。
    end_offset: Optional[int] = None
    # 父文档引用 ID（可选）。
    source_ref: Optional[str] = None

    # 初始化后校验。
    def __post_init__(self):
        """Validate required metadata fields."""
        # source_path 是链路追踪关键字段。
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")

    # 序列化为字典。
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # 统一使用 asdict。
        return asdict(self)

    # 从字典恢复对象。
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        # kwargs 构造。
        return cls(**data)


# 定义可入库的最终分块记录对象。
@dataclass
class ChunkRecord:
    """Represents a fully processed chunk ready for storage and retrieval.

    This is the output of the embedding pipeline and the data structure
    stored in vector databases. It extends Chunk with vector representations.

    Attributes:
        id: Unique chunk identifier (must be stable for idempotent upsert)
        text: Chunk content (same as Chunk.text).
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Extended metadata including:
            - source_path (required): Original file path
            - chunk_index: Sequential position
            - All metadata from Chunk
            - images: Image references from Chunk (see Document.images specification)
            - Any enrichment from Transform pipeline (title, summary, tags)
            - image_captions: Dict[image_id, caption_text] if multimodal enrichment applied
        dense_vector: Dense embedding vector (e.g., from OpenAI, BGE)
        sparse_vector: Sparse vector for BM25/keyword matching (optional)

    Note: Image captions generated by ImageCaptioner are stored in metadata.image_captions
          as a dictionary mapping image_id to generated caption text.

    Example:
        >>> record = ChunkRecord(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "title": "Introduction",
        ...         "summary": "Overview of project goals"
        ...     },
        ...     dense_vector=[0.1, 0.2, ..., 0.3],
        ...     sparse_vector={"word1": 0.5, "word2": 0.3}
        ... )
    """

    # 记录唯一 ID。
    id: str
    # 记录文本内容。
    text: str
    # 扩展元数据。
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 稠密向量（可选）。
    dense_vector: Optional[List[float]] = None
    # 稀疏向量（可选）。
    sparse_vector: Optional[Dict[str, float]] = None

    # 初始化后校验。
    def __post_init__(self):
        """Validate required metadata fields."""
        # source_path 必需，便于回溯。
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")

    # 序列化为 dict。
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # 使用 asdict 统一转换。 asdict 来自 dataclasses，作用是把 dataclass 实例转成普通 dict。
        return asdict(self)

    # 从 dict 构造 ChunkRecord。
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """Create ChunkRecord from dictionary."""
        # kwargs 构造。
        return cls(**data)

    # 从 Chunk 快速构造 ChunkRecord（附带向量）。
    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        dense_vector: Optional[List[float]] = None,
        sparse_vector: Optional[Dict[str, float]] = None
    ) -> "ChunkRecord":
        """Create ChunkRecord from a Chunk with vectors.

        Args:
            chunk: Source Chunk object
            dense_vector: Dense embedding vector
            sparse_vector: Sparse vector representation

        Returns:
            ChunkRecord with all fields populated from chunk
        """
        # 拷贝 metadata，避免和原 chunk 共享同一可变对象。
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )


# 方便使用的元数据类型别名。
Metadata = Dict[str, Any]
# 稠密向量类型别名。
Vector = List[float]
# 稀疏向量类型别名。
SparseVector = Dict[str, float]


# 定义处理后查询对象。 查询预处理结果
@dataclass
class ProcessedQuery:
    """Represents a processed query ready for retrieval.

    This is the output of QueryProcessor, containing extracted keywords
    and parsed filters for downstream Dense/Sparse retrievers.

    Attributes:
        original_query: The raw user query string
        keywords: List of extracted keywords after stopword removal
        filters: Dictionary of filter conditions (e.g., {"collection": "api-docs"})
        expanded_terms: Optional list of synonyms/expanded terms (for future use)

    Example:
        >>> pq = ProcessedQuery(
        ...     original_query="如何配置 Azure OpenAI？",
        ...     keywords=["配置", "Azure", "OpenAI"],
        ...     filters={"collection": "docs"}
        ... )
    """

    # 原始查询字符串。
    original_query: str
    # 分词/抽取后的关键词。
    keywords: List[str] = field(default_factory=list)
    # 结构化过滤条件。
    filters: Dict[str, Any] = field(default_factory=dict)
    # 查询扩展词（预留）。
    expanded_terms: List[str] = field(default_factory=list)

    # 序列化方法。
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # 使用 asdict 输出标准 dict。
        return asdict(self)

    # 反序列化方法。
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedQuery":
        """Create ProcessedQuery from dictionary."""
        # kwargs 构造对象。
        return cls(**data)


# 定义检索结果对象，统一 Dense/Sparse/Hybrid 输出协议。
@dataclass
class RetrievalResult:
    """Represents a single retrieval result from Dense/Sparse retrievers.

    This is the output of DenseRetriever, SparseRetriever, and HybridSearch,
    providing a unified contract for retrieval results across all search methods.

    Attributes:
        chunk_id: Unique identifier for the retrieved chunk
        score: Relevance score (higher = more relevant, normalized to [0, 1])
        text: The actual text content of the retrieved chunk
        metadata: Associated metadata (source_path, chunk_index, title, etc.)

    Example:
        >>> result = RetrievalResult(
        ...     chunk_id="doc1_chunk_003",
        ...     score=0.85,
        ...     text="Azure OpenAI 配置步骤如下...",
        ...     metadata={
        ...         "source_path": "docs/azure-guide.pdf",
        ...         "chunk_index": 3,
        ...         "title": "Azure Configuration"
        ...     }
        ... )
    """

    # 命中分块 ID。
    chunk_id: str
    # 相关性分数。
    score: float
    # 命中的文本内容。
    text: str
    # 附带元数据。
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 初始化后校验。
    def __post_init__(self):
        """Validate fields after initialization."""
        # chunk_id 不能为空字符串。
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        # score 必须是数值类型。
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"score must be numeric, got {type(self.score).__name__}")

    # 序列化输出。
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # 统一输出为 dict。
        return asdict(self)

    # 从 dict 恢复对象。
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Create RetrievalResult from dictionary."""
        # kwargs 构造。
        return cls(**data)
