"""Image Captioner transform for enriching chunks with image descriptions.

Performance Optimizations:
1. Only processes images that are actually referenced in chunk text (via [IMAGE: id] placeholder)
2. Uses caption cache to avoid redundant Vision API calls for the same image
3. Skips chunks without image references entirely
4. Parallel processing of unique images with thread-safe caching
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

# 当前文件的日志对象
logger = get_logger(__name__)

# 用正则匹配 chunk 里的图片占位符，比如 [IMAGE: img_001]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

# Vision API 并发时默认最多开几个线程
# 这里比普通文本 LLM 小，因为图片模型更贵、更慢
DEFAULT_MAX_WORKERS = 3  # Lower than text LLM due to higher cost/latency


class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks using Vision LLM.
    
    This transform identifies chunks containing image references, uses a Vision LLM
    to generate descriptive captions, and enriches the chunk text/metadata with
    these captions to improve retrieval for visual content.
    
    Key Features:
    - Only processes images actually referenced in chunk text (not all images in metadata)
    - Caches captions to avoid redundant Vision API calls
    - Thread-safe caption cache for potential future parallelization
    """
    
    def __init__(
        self, 
        settings: Settings, 
        llm: Optional[BaseVisionLLM] = None
    ):
        # 保存全局配置
        self.settings = settings

        # 先把 llm 置空，后面按配置决定是否初始化
        self.llm = None

        # caption 缓存：image_id -> caption
        # 作用：同一张图片不要重复调用 Vision API
        self._caption_cache: Dict[str, str] = {}

        # 线程锁，保证并发读写缓存时安全
        self._cache_lock = threading.Lock()
        
        # 检查 settings 里有没有启用 vision_llm
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 # 如果外部没传 llm，就通过工厂创建一个 vision llm
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
                 # 这里故意不 raise
                 # 因为即使图片描述功能挂了，也不想让整个 pipeline 中断
                 # 相当于这个 transform 退化成“不做事”
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")
        
        # 加载图片描述 prompt
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """Load the image captioning prompt from configuration."""
        # 这里是读取图片描述用的 prompt 文件
        # 如果 prompt 文件存在，就读文件
        # 不存在就用一个默认 prompt
        from src.core.settings import resolve_path
        prompt_path = resolve_path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """Extract image IDs actually referenced in the chunk text.
        
        Args:
            text: Chunk text content
            
        Returns:
            List of image IDs found in [IMAGE: id] placeholders
        """
        # 从 chunk 文本里找所有 [IMAGE: xxx]
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)

        # 去掉首尾空格后返回
        return [m.strip() for m in matches]

    def _get_caption(
        self, 
        img_id: str, 
        img_path: str, 
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """Get caption for an image, using cache if available. Thread-safe.
        
        Args:
            img_id: Image identifier
            img_path: Path to image file
            trace: Optional trace context
            
        Returns:
            Caption string or None if failed
        """
        # 先查缓存，避免重复调模型
        # 这里加锁是为了线程安全
        with self._cache_lock:
            if img_id in self._caption_cache:
                logger.debug(f"Caption cache hit for image {img_id}")
                return self._caption_cache[img_id]
        
        # 检查图片路径是否合法
        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None
        
        try:
            # 把本地图片路径封装成 ImageInput
            image_input = ImageInput(path=img_path)

            # 调 vision llm，让它基于 prompt 给图片生成描述
            response = self.llm.chat_with_image(
                text=self.prompt,
                image=image_input,
                trace=trace
            )

            # 取模型返回内容
            caption = response.content
            
            # 把结果写入缓存
            with self._cache_lock:
                self._caption_cache[img_id] = caption
            logger.debug(f"Generated and cached caption for image {img_id}")
            
            return caption
            
        except Exception as e:
            # 单张图片 caption 失败，返回 None，不中断整体流程
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks and add captions for referenced images.
        
        Only processes images that are actually referenced in chunk text
        via [IMAGE: id] placeholders. Uses caching to avoid redundant API calls.
        Parallel processing for unique images.
        """
        # 如果 vision llm 根本不可用，直接返回原 chunks
        if not self.llm:
            return chunks
        
        # 第一步：从所有 chunk 的 metadata 里把图片信息收集起来
        # 建一个 image_lookup，方便后面通过 image_id 找到对应 path 等信息
        image_lookup: Dict[str, dict] = {}
        for chunk in chunks:
            if chunk.metadata and "images" in chunk.metadata:
                for img_meta in chunk.metadata.get("images", []):
                    img_id = img_meta.get("id")
                    if img_id and img_id not in image_lookup:
                        image_lookup[img_id] = img_meta
        
        logger.info(f"Found {len(image_lookup)} unique images in document")
        
        # 每次处理新文档前，把旧缓存清空
        with self._cache_lock:
            self._caption_cache.clear()
        
        # 第二步：先扫描所有 chunk 文本
        # 只收集“真正被引用到”的图片，而不是 metadata 里所有图片都处理
        images_to_caption: Dict[str, str] = {}  # img_id -> img_path
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            for img_id in referenced_ids:
                if img_id not in images_to_caption:
                    img_meta = image_lookup.get(img_id)
                    if img_meta and img_meta.get("path"):
                        images_to_caption[img_id] = img_meta.get("path")
        
        # 第三步：把要处理的图片并发生成 caption
        # 注意这里只是先把 caption 生成并放进缓存
        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)
        
        # 第四步：再遍历 chunk，把缓存里的 caption 回填到 chunk 里
        processed_chunks = []
        total_captions_added = 0
        
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            
            # 当前 chunk 没引用图片，原样返回
            if not referenced_ids:
                processed_chunks.append(chunk)
                continue
            
            # new_text 是待修改的新文本
            new_text = chunk.text

            # captions 用来收集这个 chunk 对应的图片描述，后面写入 metadata
            captions = []
            
            for img_id in referenced_ids:
                img_id_stripped = img_id.strip()
                
                # 从缓存拿 caption
                # 此时 caption 理论上已经被并发阶段提前生成好了
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_stripped)
                
                if caption:
                    # 记录到 metadata 要写入的数据里
                    captions.append({"id": img_id_stripped, "caption": caption})
                    
                    # 把原占位符替换成：占位符 + 描述文本
                    placeholder = f"[IMAGE: {img_id}]"
                    replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                    new_text = new_text.replace(placeholder, replacement)
                    total_captions_added += 1
                    
            # 把新文本写回当前 chunk
            chunk.text = new_text
            
            # 如果这个 chunk 有生成到 caption，就写进 metadata["image_captions"]
            if captions:
                if "image_captions" not in chunk.metadata:
                    chunk.metadata["image_captions"] = []
                chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
        
        # 统计这次一共缓存了多少张图的 caption
        with self._cache_lock:
            api_calls = len(self._caption_cache)
        logger.info(f"Added {total_captions_added} captions, API calls: {api_calls}")
            
        return processed_chunks
    
    def _generate_captions_parallel(
        self, 
        images_to_caption: Dict[str, str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """Generate captions for multiple images in parallel.
        
        Args:
            images_to_caption: Dict of img_id -> img_path
            trace: Optional trace context
        """
        # 没有图片要处理就直接返回
        if not images_to_caption:
            return
        
        # 线程数不超过默认上限，也不超过图片数量
        max_workers = min(DEFAULT_MAX_WORKERS, len(images_to_caption))
        logger.debug(f"Generating captions for {len(images_to_caption)} images (max_workers={max_workers})")
        
        # 用线程池并发给多张图片生成 caption
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_caption, img_id, img_path, trace): img_id
                for img_id, img_path in images_to_caption.items()
            }
            
            # 哪个任务先完成，就先处理哪个结果
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    caption = future.result()
                    if caption:
                        logger.debug(f"Caption generated for {img_id}")
                except Exception as e:
                    # 即使某一张图失败，也只记日志，不影响其他图片
                    logger.error(f"Failed to generate caption for {img_id}: {e}")