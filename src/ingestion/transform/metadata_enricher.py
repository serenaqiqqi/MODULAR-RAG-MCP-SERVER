"""Metadata enrichment transform: rule-based + optional LLM enhancement."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

# 当前文件的日志对象，后面用来记录运行信息、警告和报错
logger = get_logger(__name__)

# LLM 并发调用时默认最多开几个线程
DEFAULT_MAX_WORKERS = 5


class MetadataEnricher(BaseTransform):
    """Enriches chunk metadata with title, summary, and tags.
    
    # 这个类的作用：
    # 给每个 chunk 补充 metadata，主要补：
    # - title
    # - summary
    # - tags
    #
    # 处理顺序：
    # 1. 先做 rule-based enrichment（规则增强）
    # 2. 如果配置允许，再做 LLM enrichment（模型增强）
    # 3. 如果 LLM 失败，就退回 rule-based 结果
    #
    # 输出里还会标记 enriched_by，说明这次结果是 rule 还是 llm
    
    Processing Pipeline:
        1. Rule-based enrichment: Extract basic metadata from content
        2. (Optional) LLM enrichment: Generate semantic-rich metadata
        3. On LLM failure: Gracefully fallback to rule-based metadata
    
    Output Metadata:
        - title: Brief title/heading for the chunk
        - summary: Concise summary of the content
        - tags: List of relevant keywords/topics
        - enriched_by: "rule" or "llm"
    
    Configuration (via settings.yaml):
        - ingestion.metadata_enricher.use_llm: bool - Enable LLM enhancement
        - ingestion.metadata_enricher.prompt_path: str - Custom prompt file path
    
    Design Principles:
        - Graceful Degradation: LLM errors don't block ingestion
        - Atomic Processing: Each chunk processed independently
        - Observable: Records enriched_by in metadata
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """Initialize MetadataEnricher.
        
        Args:
            settings: Application settings
            llm: Optional LLM instance (for testing; auto-created if None)
            prompt_path: Optional custom prompt file path
        """
        # 保存全局配置
        self.settings = settings

        # 外部如果传入 llm，就先存起来；没传的话后面再懒加载创建
        self._llm = llm

        # prompt 模板缓存，第一次读文件后会存在这里，避免反复读磁盘
        self._prompt_template: Optional[str] = None

        # 如果外部没传 prompt_path，就用默认 prompt 文件
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/metadata_enrichment.txt"))
        
        # Determine if LLM should be used
        # 这里开始从 settings 里读取 metadata_enricher 配置，判断是否启用 LLM
        enricher_config = {}
        if hasattr(settings, 'ingestion') and settings.ingestion is not None:
            ingestion_config = settings.ingestion
            # Check if ingestion has metadata_enricher attribute (dataclass) or dict
            # 这里兼容两种情况：
            # 1. ingestion_config 是对象，有 metadata_enricher 属性
            # 2. ingestion_config 是 dict
            if hasattr(ingestion_config, 'metadata_enricher') and ingestion_config.metadata_enricher:
                enricher_config = ingestion_config.metadata_enricher
            elif isinstance(ingestion_config, dict):
                enricher_config = ingestion_config.get('metadata_enricher', {})
        
        # 最终拿到 use_llm 配置；如果没有配置，默认 False
        self.use_llm = enricher_config.get('use_llm', False) if enricher_config else False
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """Lazy-load LLM instance."""
        # 这是懒加载：
        # 只有真的要用 llm 时，才去创建
        if self.use_llm and self._llm is None:
            try:
                # 根据 settings 通过工厂创建对应 LLM
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for metadata enrichment")
            except Exception as e:
                # 如果 LLM 初始化失败，不让整个流程挂掉
                # 直接关闭 LLM，后面只走 rule-based
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Transform chunks by enriching their metadata.
        
        Args:
            chunks: List of chunks to enrich
            trace: Optional trace context
            
        Returns:
            List of enriched chunks (same length as input)
        """
        # 如果没有 chunk，直接返回空列表
        if not chunks:
            return []
        
        # 如果启用了 LLM，并且 llm 成功可用，就走并行处理
        if self.use_llm and self.llm:
            return self._transform_parallel(chunks, trace)
        else:
            # 否则走顺序处理
            return self._transform_sequential(chunks, trace)
    
    def _enrich_single_chunk(
        self, 
        chunk: Chunk, 
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """Enrich a single chunk. Thread-safe.
        
        Args:
            chunk: Chunk to enrich
            trace: Optional trace context
            
        Returns:
            Tuple of (enriched_chunk, enriched_by, error_message)
        """
        try:
            # Step 1: Rule-based enrichment
            # 第一步永远先做规则增强，先拿到一个最基础、稳定的 metadata
            rule_metadata = self._rule_based_enrich(chunk.text)
            
            # Step 2: LLM enhancement
            # 第二步如果开了 LLM，就尝试用 LLM 做更语义化的增强
            if self.use_llm and self.llm:
                llm_metadata = self._llm_enrich(chunk.text, trace)
                
                if llm_metadata:
                    # LLM 成功，优先使用 LLM 结果
                    enriched_metadata = llm_metadata
                    enriched_by = "llm"
                else:
                    # LLM 失败，退回 rule-based 结果
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                    enriched_metadata['enrich_fallback_reason'] = "llm_failed"
            else:
                # 没启用 LLM，直接使用规则增强结果
                enriched_metadata = rule_metadata
                enriched_by = "rule"
            
            # 把原有 metadata 和新 metadata 合并
            # 后面的同名字段会覆盖前面的字段
            final_metadata = {
                **(chunk.metadata or {}),
                **enriched_metadata,
                'enriched_by': enriched_by
            }
            
            # 生成一个新的 Chunk 对象返回
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=final_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, enriched_by, None)
            
        except Exception as e:
            # 如果单个 chunk 增强失败，这里兜底，保证整批处理不会因为一个 chunk 中断
            logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
            text_preview = ""
            if chunk.text:
                # 取前 100 个字符作为兜底 summary
                text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
            minimal_metadata = {
                **(chunk.metadata or {}),
                'title': 'Untitled',
                'summary': text_preview,
                'tags': [],
                'enriched_by': 'error',
                'enrich_error': str(e)
            }
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text or "",
                metadata=minimal_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, "error", str(e))
    
    def _transform_parallel(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        # 线程数不能超过 chunk 数量，也不能超过默认上限
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))

        # 先占位，后面按原顺序把结果放回去
        enriched_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0
        
        logger.debug(f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # future_to_idx 的作用：
            # 记录“某个异步任务 future 对应原 chunks 里的哪个下标”
            future_to_idx = {
                executor.submit(self._enrich_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            
            # 谁先处理完，就先拿谁的结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    enriched_chunk, enriched_by, error = future.result()
                    # 放回原始位置，保证输出顺序和输入顺序一致
                    enriched_chunks[idx] = enriched_chunk
                    
                    # 统计有多少是 llm 成功增强的
                    if enriched_by == "llm":
                        llm_enhanced_count += 1
                    # 统计有多少是 fallback 到 rule 的
                    elif enriched_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    # 理论上 _enrich_single_chunk 已经自己做过兜底
                    # 这里算第二层兜底
                    logger.error(f"Unexpected error in parallel enrichment: {e}")
                    enriched_chunks[idx] = chunks[idx]
        
        # 成功数 = 不为 None 的结果个数
        success_count = sum(1 for c in enriched_chunks if c is not None)
        
        # 如果有 trace，就把这一步的统计信息记进去
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _transform_sequential(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks sequentially (fallback when LLM disabled)."""
        enriched_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        # 一个一个 chunk 顺序处理
        for chunk in chunks:
            try:
                # Step 1: Rule-based enrichment (always performed)
                # 先做规则增强
                rule_metadata = self._rule_based_enrich(chunk.text)
                
                # Step 2: Optional LLM enhancement
                # 如果允许，就再尝试 LLM 增强
                if self.use_llm and self.llm:
                    llm_metadata = self._llm_enrich(chunk.text, trace)
                    
                    if llm_metadata:
                        # LLM success
                        enriched_metadata = llm_metadata
                        enriched_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM failed, fallback to rule-based
                        enriched_metadata = rule_metadata
                        enriched_by = "rule"
                        fallback_count += 1
                        enriched_metadata['enrich_fallback_reason'] = "llm_failed"
                else:
                    # LLM disabled, use rule-based
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                
                # Merge enriched metadata with existing metadata
                # 合并原 metadata 和增强后的 metadata
                final_metadata = {
                    **(chunk.metadata or {}),
                    **enriched_metadata,
                    'enriched_by': enriched_by
                }
                
                # Create enriched chunk
                # 构造新的 chunk 返回
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=final_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
                success_count += 1
                
            except Exception as e:
                # Atomic failure: log and preserve original with minimal metadata
                # 单个 chunk 失败时，给它补最小 metadata，避免整批中断
                logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
                # Handle None text case
                text_preview = ""
                if chunk.text:
                    text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
                minimal_metadata = {
                    **(chunk.metadata or {}),
                    'title': 'Untitled',
                    'summary': text_preview,
                    'tags': [],
                    'enriched_by': 'error',
                    'enrich_error': str(e)
                }
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text or "",  # Ensure text is not None
                    metadata=minimal_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
        
        # Record trace
        # 把顺序处理这一步的统计信息记到 trace
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """Extract metadata using rule-based heuristics.
        
        Args:
            text: Chunk text content
            
        Returns:
            Dictionary with title, summary, tags
            
        Raises:
            TypeError: If text is None
        """
        # text 不能为空，否则规则提取没法做
        if text is None:
            raise TypeError("Chunk text cannot be None")
        
        # Extract title from first heading or first line
        # 提取标题
        title = self._extract_title(text)
        
        # Generate summary from first sentences
        # 提取摘要
        summary = self._extract_summary(text)
        
        # Extract tags from common patterns
        # 提取标签
        tags = self._extract_tags(text)
        
        return {
            'title': title,
            'summary': summary,
            'tags': tags
        }
    
    def _extract_title(self, text: str) -> str:
        """Extract title from text using heuristics.
        
        Priority:
            1. Markdown heading (# Title)
            2. First line if short enough
            3. First sentence
            4. First N characters
        """
        if not text:
            return "Untitled"
        
        # Check for markdown heading
        # 先找 markdown 标题，比如 # xxx
        heading_match = re.match(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
        
        # Use first line if it's short and looks like a title
        # 如果第一行不长，而且不像一句完整句子，就把它当标题
        first_line = text.split('\n')[0].strip()
        if first_line and len(first_line) <= 100 and not first_line.endswith(('.', ',', ';')):
            return first_line
        
        # Use first sentence (without trailing punctuation)
        # 再不行就取第一句话
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and sentences[0]:
            title = sentences[0].strip()
            # Remove trailing punctuation if present
            title = re.sub(r'[.!?]+$', '', title)
            if len(title) <= 150:
                return title
            return title[:147] + "..."
        
        # Fallback: first 100 chars
        # 最后兜底：直接取前 100 个字符
        return text[:100].strip() + ("..." if len(text) > 100 else "")
    
    def _extract_summary(self, text: str, max_sentences: int = 3) -> str:
        """Extract summary from text using first N sentences.
        
        Args:
            text: Source text
            max_sentences: Maximum number of sentences to include
            
        Returns:
            Summary text
        """
        if not text:
            return ""
        
        # Split into sentences
        # 按句子切开
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Take first N sentences
        # 取前几句作为摘要
        summary_sentences = sentences[:max_sentences]
        summary = ' '.join(summary_sentences).strip()
        
        # Limit length
        # 太长就截断
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        return summary
    
    def _extract_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """Extract tags using keyword extraction heuristics.
        
        Args:
            text: Source text
            max_tags: Maximum number of tags to extract
            
        Returns:
            List of tag strings
        """
        if not text:
            return []
        
        tags = set()
        
        # Extract capitalized words (potential proper nouns)
        # 提取首字母大写的词，可能是专有名词
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        tags.update(capitalized[:5])
        
        # Extract code identifiers (camelCase, snake_case)
        # 提取代码变量名风格的词，比如 camelCase / snake_case
        identifiers = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b', text)
        tags.update(identifiers[:5])
        
        # Extract markdown bold/italic terms (potential keywords)
        # 提取 markdown 加粗/斜体里的词，可能是重点关键词
        markdown_keywords = re.findall(r'\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_', text)
        for match in markdown_keywords[:5]:
            for group in match:
                if group:
                    tags.add(group.strip())
        
        # Convert to list and limit
        # 转成有序列表，并限制最大数量
        tag_list = sorted(list(tags))[:max_tags]
        
        return tag_list
    
    def _llm_enrich(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[Dict[str, Any]]:
        """Enrich metadata using LLM.
        
        Args:
            text: Chunk text content
            trace: Optional trace context
            
        Returns:
            Dictionary with title, summary, tags, or None on failure
        """
        # 如果 llm 不可用，直接返回 None
        if not self.llm:
            return None
        
        try:
            # Load prompt template
            # 先读取 prompt 模板
            prompt = self._load_prompt()
            
            # Build prompt with text
            # 把 chunk 文本塞进 prompt；这里只取前 2000 字符，避免 prompt 过长
            formatted_prompt = prompt.replace("{chunk_text}", text[:2000])  # Limit text length
            
            # Call LLM
            # 组织成消息格式发给模型
            messages = [Message(role="user", content=formatted_prompt)]
            response = self.llm.chat(messages)
            
            if not response:
                # 模型没返回内容，直接视为失败
                logger.warning("LLM returned empty response for metadata enrichment")
                return None
            
            # Extract text from response (handle both string and ChatResponse object)
            # 兼容不同模型返回格式：
            # - 可能直接是字符串
            # - 也可能是对象，对象里有 content 或 text
            response_text = response
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif not isinstance(response, str):
                response_text = str(response)
            
            # Parse LLM response
            # 把模型输出解析成 title / summary / tags
            metadata = self._parse_llm_response(response_text)
            
            # 如果 trace 存在，就记录这一阶段的调用结果
            if trace:
                trace.record_stage("metadata_enricher_llm", {
                    "success": True,
                    "response_length": len(response_text)
                })
            
            return metadata
            
        except Exception as e:
            # LLM 出错不抛到外层，直接记日志并返回 None，交给上层 fallback
            logger.warning(f"LLM enrichment failed: {e}")
            if trace:
                trace.record_stage("metadata_enricher_llm", {
                    "success": False,
                    "error": str(e)
                })
            return None
    
    def _load_prompt(self) -> str:
        """Load prompt template from file with caching."""
        # 如果 prompt 已经加载过，直接走缓存
        if self._prompt_template is not None:
            return self._prompt_template
        
        prompt_file = Path(self._prompt_path)
        
        # prompt 文件不存在就报错
        if not prompt_file.exists():
            raise FileNotFoundError(f"Metadata enrichment prompt file not found: {self._prompt_path}")
        
        # 读取 prompt 文件内容并缓存起来
        self._prompt_template = prompt_file.read_text(encoding='utf-8').strip()
        return self._prompt_template
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into metadata dict.
        
        Expected format:
            TITLE: ...
            SUMMARY: ...
            TAGS: tag1, tag2, tag3
        """
        if not response:
            return None
        
        # 初始化结果结构
        title = ""
        summary = ""
        tags = []
        
        # 按行切开，逐行解析
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 解析 TITLE
            if line.upper().startswith('TITLE:'):
                title = line[6:].strip()
            # 解析 SUMMARY
            elif line.upper().startswith('SUMMARY:'):
                summary = line[8:].strip()
            # 解析 TAGS
            elif line.upper().startswith('TAGS:'):
                tags_str = line[5:].strip()
                tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        
        # 如果关键字段一个都没解析出来，说明格式不对，返回 None
        if not title and not summary and not tags:
            logger.warning(f"Failed to parse LLM response format: {response[:200]}")
            return None
        
        # 如果缺字段，就尽量兜底
        return {
            'title': title or 'Untitled',
            'summary': summary or '',
            'tags': tags
        }