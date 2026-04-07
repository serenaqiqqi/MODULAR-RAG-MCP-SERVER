"""Chunk refinement transform: rule-based cleaning + optional LLM enhancement."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

# 创建当前文件专用的 logger，后面所有运行信息、warning、error 都走它
logger = get_logger(__name__)

# 当开启 LLM 精修时，最多并发处理多少个 chunk
# 这里限制为 5，避免一下子开太多线程去打模型接口
DEFAULT_MAX_WORKERS = 5


class ChunkRefiner(BaseTransform):
    """Refines chunks through rule-based cleaning and optional LLM enhancement.
    
    Processing Pipeline:
        1. Rule-based refine: Remove noise (whitespace, headers/footers, HTML)
        2. (Optional) LLM refine: Intelligent content improvement
        3. On LLM failure: Gracefully fallback to rule-based result
    
    Configuration (via settings.yaml):
        - ingestion.chunk_refiner.use_llm: bool - Enable LLM enhancement
        - ingestion.chunk_refiner.prompt_path: str - Custom prompt file path
    
    Design Principles:
        - Graceful Degradation: LLM errors don't block ingestion
        - Atomic Processing: Each chunk processed independently
        - Observable: Records refined_by in metadata
    """

    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """Initialize ChunkRefiner.
        
        Args:
            settings: Application settings
            llm: Optional LLM instance (for testing; auto-created if None)
            prompt_path: Optional custom prompt file path
        """
        # 保存全局配置对象，后面读取 LLM 配置、路径配置都要用
        self.settings = settings

        # 允许外部直接传入 llm，通常用于测试
        # 如果没传，后面会在真正需要时懒加载创建
        self._llm = llm

        # prompt 模板缓存
        # 第一次从文件读出来后，会放到这里，避免每个 chunk 都重新读文件
        self._prompt_template: Optional[str] = None

        # prompt 文件路径：
        # 如果调用者自己传了 prompt_path，就优先用它；
        # 否则默认去读 config/prompts/chunk_refinement.txt
        self._prompt_path = prompt_path or str(
            resolve_path("config/prompts/chunk_refinement.txt")
        )

        # 从 settings 里读取是否启用 LLM 精修
        # 写法比较“防御式”：
        # - settings 可能没有 ingestion 属性
        # - ingestion 里可能没有 chunk_refiner
        # - chunk_refiner 里可能没有 use_llm
        # 所以层层兜底，最终默认 False
        self.use_llm = getattr(
            getattr(settings, 'ingestion', None),
            'chunk_refiner',
            {}
        ).get('use_llm', False) if hasattr(settings, 'ingestion') else False

    @property
    def llm(self) -> Optional[BaseLLM]:
        """Lazy-load LLM instance."""
        # 这里是懒加载逻辑：
        # 只有真的启用了 LLM，并且当前还没创建过实例，才去创建
        if self.use_llm and self._llm is None:
            try:
                # 根据 settings 中的 provider / model 等配置，动态创建对应 LLM
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for chunk refinement")
            except Exception as e:
                # 如果 LLM 初始化失败，不让整个 ingestion 挂掉
                # 而是记录 warning，并退回到纯规则清洗模式
                logger.warning(
                    f"Failed to initialize LLM: {e}. Falling back to rule-based only."
                )
                self.use_llm = False
        return self._llm

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Transform chunks through refinement pipeline.
        
        Args:
            chunks: List of chunks to refine
            trace: Optional trace context
            
        Returns:
            List of refined chunks (same length as input)
        """
        # 如果输入为空，直接返回空列表
        if not chunks:
            return []

        # 总入口分流逻辑：
        # - 如果启用了 LLM，且 llm 能正常拿到实例，就走并行版本
        # - 否则走串行版本
        # 这样可以在启用 LLM 时提升吞吐，在关闭 LLM 时保持实现简单稳妥
        if self.use_llm and self.llm:
            return self._transform_parallel(chunks, trace)
        else:
            return self._transform_sequential(chunks, trace)

    def _refine_single_chunk(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """Refine a single chunk. Thread-safe.
        
        Args:
            chunk: Chunk to refine
            trace: Optional trace context
            
        Returns:
            Tuple of (refined_chunk, refined_by, error_message)
        """
        try:
            # 第一步：先做规则清洗
            # 这是“保底步骤”，不依赖 LLM，负责去掉页眉页脚、HTML 噪声、脏空格等
            rule_refined_text = self._rule_based_refine(chunk.text)

            # 第二步：如果允许使用 LLM，则在规则清洗结果上继续做智能增强
            if self.use_llm and self.llm:
                llm_refined_text = self._llm_refine(rule_refined_text, trace)

                # 如果 LLM 成功返回了有效文本，就用 LLM 结果
                if llm_refined_text:
                    refined_text = llm_refined_text
                    refined_by = "llm"
                else:
                    # 如果 LLM 没返回内容或失败，则优雅降级回规则清洗结果
                    refined_text = rule_refined_text
                    refined_by = "rule"
            else:
                # 没开启 LLM，就直接使用规则清洗结果
                refined_text = rule_refined_text
                refined_by = "rule"

            # 构造一个新的 Chunk 返回，而不是直接修改原对象
            # 这样更清晰，也更符合“原子化处理”的思路
            refined_chunk = Chunk(
                id=chunk.id,
                text=refined_text,
                metadata={
                    # 保留原 metadata
                    **(chunk.metadata or {}),
                    # 补充一个 refined_by 字段，标记最终结果是 rule 还是 llm 产出的
                    'refined_by': refined_by
                },
                # source_ref 原样保留，用于和父文档建立溯源关系
                source_ref=chunk.source_ref
            )
            return (refined_chunk, refined_by, None)

        except Exception as e:
            # 单个 chunk 处理失败时，不让整个批次中断
            # 记录错误，然后把原 chunk 原样返回
            logger.error(f"Failed to refine chunk {chunk.id}: {e}")
            return (chunk, "error", str(e))

    def _transform_parallel(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        # 并发线程数最多不超过 DEFAULT_MAX_WORKERS，
        # 同时也不超过实际 chunk 数量
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))

        # 先创建一个和输入等长的列表，后面按原始索引把结果放回去
        # 这样即便 futures 完成顺序乱了，输出顺序仍能和输入一致
        refined_chunks = [None] * len(chunks)

        # 统计信息：多少个 chunk 真正用了 LLM，多少个退回 rule
        llm_enhanced_count = 0
        fallback_count = 0

        logger.debug(
            f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})"
        )

        # 用线程池并行处理多个 chunk
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，并记录 future 对应的是原列表哪个下标
            future_to_idx = {
                executor.submit(self._refine_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }

            # as_completed 会按“谁先做完谁先返回”的顺序迭代
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    refined_chunk, refined_by, error = future.result()
                    refined_chunks[idx] = refined_chunk

                    # 统计：真正用了 LLM 的数量
                    if refined_by == "llm":
                        llm_enhanced_count += 1
                    # 统计：返回 rule 且没有报错，视为 fallback / rule-based success
                    elif refined_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    # 理论上 _refine_single_chunk 已经 try/except 过了
                    # 这里再兜一层，防止 future 层面出现意外
                    logger.error(f"Unexpected error in parallel refinement: {e}")
                    refined_chunks[idx] = chunks[idx]

        # 成功数：最终结果里非 None 的个数
        success_count = sum(1 for c in refined_chunks if c is not None)

        # 如果启用了 trace，就把本阶段关键指标打到 trace 里
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })

        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )

        return refined_chunks

    def _transform_sequential(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks sequentially (fallback when LLM disabled)."""
        # 串行模式下，按顺序一个个处理 chunk
        refined_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0

        for chunk in chunks:
            try:
                # 第一步：规则清洗
                # 这一步总会执行
                rule_refined_text = self._rule_based_refine(chunk.text)

                # 第二步：可选 LLM 精修
                if self.use_llm and self.llm:
                    llm_refined_text = self._llm_refine(rule_refined_text, trace)

                    if llm_refined_text:
                        # LLM 成功：用 LLM 返回结果
                        refined_text = llm_refined_text
                        refined_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM 失败：退回规则清洗结果
                        refined_text = rule_refined_text
                        refined_by = "rule"
                        fallback_count += 1

                        # 如果原 chunk 有 metadata，顺手补一个回退原因
                        if chunk.metadata:
                            chunk.metadata['refine_fallback_reason'] = "llm_failed"
                else:
                    # 没启用 LLM，直接使用规则清洗结果
                    refined_text = rule_refined_text
                    refined_by = "rule"

                # 重新创建一个 refined chunk
                refined_chunk = Chunk(
                    id=chunk.id,
                    text=refined_text,
                    metadata={
                        **(chunk.metadata or {}),
                        'refined_by': refined_by
                    },
                    source_ref=chunk.source_ref
                )
                refined_chunks.append(refined_chunk)
                success_count += 1

            except Exception as e:
                # 串行模式下也遵守同样原则：
                # 单个 chunk 出错，只保留原始 chunk，不中断整个列表处理
                logger.error(f"Failed to refine chunk {chunk.id}: {e}")
                refined_chunks.append(chunk)

        # 记录 trace
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })

        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )

        return refined_chunks

    def _rule_based_refine(self, text: str) -> str:
        """Apply rule-based text cleaning.
        
        Cleaning operations:
            1. Remove page headers/footers (separator lines + metadata)
            2. Remove HTML comments
            3. Remove HTML tags (preserve content)
            4. Normalize excessive whitespace
            5. Preserve code blocks and Markdown formatting
        
        Args:
            text: Raw chunk text
            
        Returns:
            Cleaned text
        """
        # None 或空字符串，原样返回
        if not text:
            return text

        # 如果全是空白字符，直接返回空字符串
        if not text.strip():
            return ""

        # 先把 Markdown 代码块提取出来，换成占位符
        # 这样后面的正则清洗不会误伤代码内容
        code_blocks = []
        code_block_pattern = r'```[\s\S]*?```'

        def extract_code_block(match):
            # 把完整代码块保存到列表中
            code_blocks.append(match.group(0))
            # 在原文本中留一个占位符，后面再还原
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"

        text = re.sub(code_block_pattern, extract_code_block, text)

        # 1. 清理由大量横线组成的页眉页脚分隔区域
        # 比如：
        # ───────────
        # Page 3 / Footer / Confidential
        # ───────────
        text = re.sub(
            r'─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        # 如果还有单独残留的长横线，也一并删掉
        text = re.sub(r'─{10,}', '', text)

        # 2. 删除 HTML 注释，比如 <!-- something -->
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # 3. 删除 HTML 标签，但保留标签里的正文内容
        # 例如 <p>Hello</p> -> Hello
        text = re.sub(r'<[^>]+>', '', text)

        # 4. 规范空白字符

        # 连续多个空格压成一个空格
        text = re.sub(r' {2,}', ' ', text)

        # 连续 3 个及以上换行压成 2 个换行
        # 这样既能清理过多空行，又能保留段落边界
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 5. 去掉每一行右侧多余空白
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # 6. 把之前保护起来的代码块放回原位
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", code_block)

        # 最后清掉整体首尾空白
        text = text.strip()

        return text

    def _llm_refine(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """Apply LLM-based intelligent refinement.
        
        Args:
            text: Rule-refined text
            trace: Optional trace context
            
        Returns:
            LLM-refined text, or None if refinement failed
        """
        # 没内容就没必要再调 LLM
        if not text or not text.strip():
            return text

        try:
            # 先加载 prompt 模板
            prompt_template = self._load_prompt()
            if not prompt_template:
                logger.warning("Prompt template not found, skipping LLM refinement")
                return None

            # prompt 模板里必须包含 {text} 占位符
            # 否则不知道把当前 chunk 文本插到哪里
            if '{text}' not in prompt_template:
                logger.error("Prompt template missing {text} placeholder")
                return None

            # 把规则清洗后的文本塞进 prompt 模板中
            prompt = prompt_template.replace('{text}', text)

            # 构造成统一 Message 格式，再发给 llm.chat()
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace)

            # 兼容两种返回形式：
            # - 有些实现直接返回 str
            # - 有些实现返回 ChatResponse 对象，需要取 .content
            if isinstance(response, str):
                refined_text = response
            else:
                refined_text = response.content

            # 返回前再做一次非空判断和 strip
            if refined_text and refined_text.strip():
                return refined_text.strip()
            else:
                logger.warning("LLM returned empty result")
                return None

        except Exception as e:
            # LLM 失败时，只记 warning，不抛异常
            # 上层会自动 fallback 到 rule-based 结果
            logger.warning(f"LLM refinement failed: {e}")
            return None

    def _load_prompt(self) -> Optional[str]:
        """Load prompt template from file.
        
        Returns:
            Prompt template string, or None if file not found
        """
        # 如果之前已经加载过，直接用缓存
        if self._prompt_template is not None:
            return self._prompt_template

        try:
            prompt_path = Path(self._prompt_path)

            # prompt 文件不存在，直接返回 None
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {self._prompt_path}")
                return None

            # 读取 prompt 文件内容并缓存
            self._prompt_template = prompt_path.read_text(encoding='utf-8')
            logger.debug(f"Loaded prompt template from {self._prompt_path}")
            return self._prompt_template

        except Exception as e:
            # 读取 prompt 失败时也不抛异常，交给上层做降级处理
            logger.error(f"Failed to load prompt template: {e}")
            return None