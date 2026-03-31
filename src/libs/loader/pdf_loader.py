"""PDF Loader implementation using MarkItDown.

This module implements PDF parsing with image extraction support,
converting PDFs to standardized Markdown format with image placeholders.

Features:
- Text extraction and Markdown conversion via MarkItDown
- Image extraction and storage
- Image placeholder insertion with metadata tracking
- Graceful degradation if image extraction fails
"""
# 上面：用 MarkItDown 把 PDF 变成 Markdown 正文；能抽图就抽图并插占位符，抽失败就只保留文字。

from __future__ import annotations  # 类型注解里可以用还没定义完的类名

import hashlib  # 算文件 SHA256，给文档/图片目录当 ID
import logging  # 打日志
from pathlib import Path  # 路径处理
from typing import Any, Dict, List, Optional  # 类型注解

try:
    from markitdown import MarkItDown  # 微软系：PDF → 文本/Markdown
    MARKITDOWN_AVAILABLE = True  # 标记：主解析库已安装
except ImportError:
    MARKITDOWN_AVAILABLE = False  # 没装 markitdown 就不能用这个 Loader

try:
    import fitz  # PyMuPDF，用来从 PDF 里抠图片（可选依赖）
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False  # 没装就只解析文字，不抽图

from PIL import Image  # 读图片尺寸
import io  # 内存里的字节当文件读

from src.core.types import Document  # 项目统一文档类型
from src.libs.loader.base_loader import BaseLoader  # 加载器抽象基类

logger = logging.getLogger(__name__)  # 本模块日志名


class PdfLoader(BaseLoader):  # 继承 BaseLoader，对外主要是 load(path) -> Document
    """PDF Loader using MarkItDown for text extraction and Markdown conversion.
    
    This loader:
    1. Extracts text from PDF and converts to Markdown
    2. Extracts images and saves to data/images/{doc_hash}/
    3. Inserts image placeholders in the format [IMAGE: {image_id}]
    4. Records image metadata in Document.metadata.images
    
    Configuration:
        extract_images: Enable/disable image extraction (default: True)
        image_storage_dir: Base directory for image storage (default: data/images)
    
    Graceful Degradation:
        If image extraction fails, logs warning and continues with text-only parsing.
    """
    # 流程：MarkItDown 出正文 → 可选 PyMuPDF 抽图 → 正文末尾插 [IMAGE: id] → metadata 里记图片信息

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images"
    ):
        """Initialize PDF Loader.
        
        Args:
            extract_images: Whether to extract images from PDFs.
            image_storage_dir: Base directory for storing extracted images.
        """
        if not MARKITDOWN_AVAILABLE:  # 没有 MarkItDown 整个 Loader 没法工作
            raise ImportError(
                "MarkItDown is required for PdfLoader. "
                "Install with: pip install markitdown"
            )

        self.extract_images = extract_images  # 开关：要不要尝试抽图
        self.image_storage_dir = Path(image_storage_dir)  # 图片落盘根目录（下面再按 doc_hash 分子目录）
        self._markitdown = MarkItDown()  # 解析器实例，后面 convert 用

    def load(self, file_path: str | Path) -> Document:
        """Load and parse a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Document with Markdown text and metadata.
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If parsing fails critically.
        """
        # Validate file
        path = self._validate_file(file_path)  # 基类：存在、是文件等
        if path.suffix.lower() != '.pdf':  # 后缀必须是 pdf
            raise ValueError(f"File is not a PDF: {path}")

        # Compute document hash for unique ID and image directory
        doc_hash = self._compute_file_hash(path)  # 整文件 SHA256 十六进制字符串
        doc_id = f"doc_{doc_hash[:16]}"  # Document.id：doc_ + 哈希前 16 位

        # Parse PDF with MarkItDown
        try:
            result = self._markitdown.convert(str(path))  # 转成 MarkItDown 结果对象
            text_content = result.text_content if hasattr(result, 'text_content') else str(result)  # 取出正文，兼容不同返回形态
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e  # 正文都解析不了就失败

        # Initialize metadata
        metadata: Dict[str, Any] = {
            "source_path": str(path),  # Document 要求必须有，供下游追踪来源
            "doc_type": "pdf",
            "doc_hash": doc_hash,  # 与 pipeline 完整性检查用的哈希一致
        }

        # Extract title from first heading if available
        title = self._extract_title(text_content)  # 从 Markdown 里猜标题
        if title:
            metadata["title"] = title

        # Handle image extraction (with graceful degradation)
        if self.extract_images:  # 用户允许抽图
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    path, text_content, doc_hash  # 可能改写正文（追加占位符），并返回图片列表
                )
                if images_metadata:
                    metadata["images"] = images_metadata  # 给 Chunk/多模态用
            except Exception as e:
                logger.warning(
                    f"Image extraction failed for {path}, continuing with text-only: {e}"
                )  # 抽图失败不抛，只警告，仍返回纯文本 Document

        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Hex string of SHA256 hash.
        """
        sha256 = hashlib.sha256()  # 新建哈希对象
        with open(file_path, 'rb') as f:  # 二进制读，避免编码问题
            for chunk in iter(lambda: f.read(8192), b''):  # 每次最多 8KB，读完为止
                sha256.update(chunk)  # 增量更新哈希
        return sha256.hexdigest()  # 64 位十六进制字符串

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from first Markdown heading or first non-empty line.
        
        Args:
            text: Markdown text content.
            
        Returns:
            Title string if found, None otherwise.
        """
        lines = text.split('\n')  # 按行拆

        # First try to find a markdown heading
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()  # 去掉首尾空白
            if line.startswith('# '):  # 一级标题 # 标题
                return line[2:].strip()  # 去掉 "# "

        # Fallback: use first non-empty line as title
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 0:  # 第一个非空行当标题
                return line

        return None  # 实在没有就 None

    def _extract_and_process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Extract images from PDF and insert placeholders.
        
        Uses PyMuPDF to extract images, save them to disk, and insert
        placeholders in the text content.
        
        Args:
            pdf_path: Path to PDF file.
            text_content: Extracted text content.
            doc_hash: Document hash for image directory.
            
        Returns:
            Tuple of (modified_text, images_metadata_list)
        """
        if not self.extract_images:  # 双重保险：关了就立刻返回
            logger.debug(f"Image extraction disabled for {pdf_path}")
            return text_content, []

        if not PYMUPDF_AVAILABLE:  # 没 PyMuPDF 无法枚举 PDF 内图片
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []

        images_metadata = []  # 收集每张图的元数据，最后写入 metadata["images"]
        modified_text = text_content  # 会在文末追加占位符

        try:
            # Create image storage directory
            image_dir = self.image_storage_dir / doc_hash  # 按文档哈希分子目录，避免混文件
            image_dir.mkdir(parents=True, exist_ok=True)  # 没有就创建

            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)  # 打开 PDF 文档对象

            # Track text offset for placeholder insertion
            text_offset = 0  # 当前未在循环里用，预留；占位符实际插在文末

            for page_num in range(len(doc)):  # 0 开始页码
                page = doc[page_num]  # 当前页
                image_list = page.get_images(full=True)  # 这一页上所有图片引用

                for img_index, img_info in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img_info[0]  # 图片在 PDF 内部的交叉引用号
                        base_image = doc.extract_image(xref)  # 解出原始字节和扩展名
                        image_bytes = base_image["image"]  # 图片二进制
                        image_ext = base_image["ext"]  # 如 png、jpeg

                        # Generate image ID and filename
                        image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)  # 全局唯一 id
                        image_filename = f"{image_id}.{image_ext}"  # 磁盘文件名
                        image_path = image_dir / image_filename  # 完整路径

                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)  # 写入磁盘

                        # Get image dimensions
                        try:
                            img = Image.open(io.BytesIO(image_bytes))  # 不落地再读一遍内存
                            width, height = img.size  # 宽高像素
                        except Exception:
                            width, height = 0, 0  # 读不了尺寸就填 0

                        # Create placeholder
                        placeholder = f"[IMAGE: {image_id}]"  # 正文里用这个标记图的位置

                        # Insert placeholder at end of current page's content
                        # (simplified - in production, you'd parse page boundaries)
                        insert_position = len(modified_text)  # 占位符插入点（当前实现：总在当前正文末尾）
                        modified_text += f"\n{placeholder}\n"  # 追加一行占位符

                        # Convert path to be relative to project root or absolute
                        try:
                            relative_path = image_path.relative_to(Path.cwd())  # 相对当前工作目录，路径短一点
                        except ValueError:
                            # If not in cwd, use absolute path
                            relative_path = image_path.absolute()  # 不在同一棵目录树下就用绝对路径

                        # Record metadata
                        image_metadata = {
                            "id": image_id,
                            "path": str(relative_path),  # 下游 ImageStorage 用
                            "page": page_num + 1,  # 人类可读页码从 1 开始
                            "text_offset": insert_position + 1,  # +1 for newline
                            "text_length": len(placeholder),  # 占位符在正文里占多长
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num + 1,
                                "index": img_index  # 该页第几张（从 0）
                            }
                        }
                        images_metadata.append(image_metadata)

                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")

                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue  # 单张失败跳过，继续下一张

            doc.close()  # 释放 PDF 句柄

            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images from {pdf_path}")
            else:
                logger.debug(f"No images found in {pdf_path}")

            return modified_text, images_metadata

        except Exception as e:
            logger.warning(f"Image extraction failed for {pdf_path}: {e}")
            # Graceful degradation: return original text without images
            return text_content, []  # 整体失败：原文不动，图片列表空

    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """Generate unique image ID.
        
        Args:
            doc_hash: Document hash.
            page: Page number (0-based).
            sequence: Image sequence on page (0-based).
            
        Returns:
            Unique image ID string.
        """
        return f"{doc_hash[:8]}_{page}_{sequence}"  # 文档哈希前8位_页码_页内序号，和 types.py 约定一致
