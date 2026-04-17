"""Core layer Reranker orchestrating libs.reranker backends with fallback support.
# 这个文件是“核心层”的 rerank 编排器
# 它不自己实现具体的重排算法
# 它的工作是：调用 libs.reranker 里的后端，并在失败时做 fallback 兜底

This module implements the CoreReranker class that:
# 这个模块主要实现了 CoreReranker 这个类，它负责下面几件事

1. Integrates with libs.reranker (LLM, CrossEncoder, None) via RerankerFactory
# 1）通过 RerankerFactory 接入具体后端，比如 LLM、CrossEncoder、NoneReranker

2. Provides graceful fallback when backend fails or times out
# 2）如果后端报错或者超时，不会直接把整个流程搞崩，而是优雅回退

3. Converts RetrievalResult to/from reranker input/output format
# 3）把 core 层的 RetrievalResult 转成 reranker 后端能吃的格式
#    rerank 完之后，再转回 RetrievalResult

4. Supports TraceContext for observability
# 4）支持 trace，把 rerank 阶段的信息记下来，方便后面排查

Design Principles:
# 下面是这个模块遵守的设计原则

- Pluggable: Uses RerankerFactory to instantiate configured backend
# 可插拔：到底用哪个重排器，不写死，交给工厂按配置创建

- Config-Driven: Reads rerank settings from settings.yaml
# 配置驱动：参数都从 settings.yaml 里来

- Graceful Fallback: Returns original order on backend failure
# 优雅回退：后端失败了就返回原始顺序，不让查询整个失败

- Observable: TraceContext integration for debugging
# 可观测：支持 trace，方便调试
"""

from __future__ import annotations
# 让类型标注可以延迟解析
# 这样在类里引用还没真正定义的类型时更稳一些

import logging
# 日志模块，用来打印 warning / info / debug

import time
# 时间模块，这里主要用来计算 rerank 耗时

from dataclasses import dataclass, field
# dataclass 用来快速定义“数据对象类”
# field(default_factory=...) 用来给列表这种可变对象安全设默认值

from typing import TYPE_CHECKING, Any, Dict, List, Optional
# 一些类型标注用到的工具
# TYPE_CHECKING 表示“只在类型检查阶段导入，不在运行时真的导入”

from src.core.types import RetrievalResult
# 引入核心层统一的检索结果类型 RetrievalResult

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
# BaseReranker：所有 reranker 后端的统一抽象基类
# NoneReranker：一个“什么都不做”的重排器，常用来兜底

from src.libs.reranker.reranker_factory import RerankerFactory
# reranker 工厂，根据配置创建具体后端

if TYPE_CHECKING:
    from src.core.settings import Settings
    # 这里只给类型检查器看，运行时不会真的 import
    # 防止循环导入或者多余开销

logger = logging.getLogger(__name__)
# 以当前模块名创建 logger，后面打印日志都走它


class RerankError(RuntimeError):
    """Raised when reranking fails."""
    # 自定义异常：当 rerank 真失败且不允许 fallback 时抛这个


@dataclass
class RerankConfig:
    """Configuration for CoreReranker.
    # 这个 dataclass 专门保存 CoreReranker 用到的配置
    
    Attributes:
        enabled: Whether reranking is enabled
        # enabled：是否开启 rerank
        
        top_k: Number of results to return after reranking
        # top_k：rerank 后最多保留多少条
        
        timeout: Timeout for reranker backend (seconds)
        # timeout：后端超时时间（秒）
        
        fallback_on_error: Whether to return original order on error
        # fallback_on_error：出错时要不要返回原始顺序兜底
    """
    enabled: bool = True
    # 默认开启 rerank

    top_k: int = 5
    # 默认返回前 5 条

    timeout: float = 30.0
    # 默认超时 30 秒

    fallback_on_error: bool = True
    # 默认开启出错兜底


@dataclass
class RerankResult:
    """Result of a rerank operation.
    # 这个 dataclass 是“rerank 完整返回结果”的包装对象
    # 不只是最终 results，还会带一些调试和状态信息
    
    Attributes:
        results: Reranked list of RetrievalResults
        # results：重排后的 RetrievalResult 列表
        
        used_fallback: Whether fallback was used due to backend failure
        # used_fallback：这次有没有走 fallback
        
        fallback_reason: Reason for fallback (if applicable)
        # fallback_reason：如果走了 fallback，原因是什么
        
        reranker_type: Type of reranker used ('llm', 'cross_encoder', 'none')
        # reranker_type：这次用的是哪种 reranker
        
        original_order: Original results before reranking (for debugging)
        # original_order：重排前的原始顺序，方便调试对比
    """
    results: List[RetrievalResult] = field(default_factory=list)
    # 真正的重排结果列表，默认空列表

    used_fallback: bool = False
    # 默认认为没有走 fallback

    fallback_reason: Optional[str] = None
    # 默认没有 fallback 原因

    reranker_type: str = "none"
    # 默认类型先写成 none

    original_order: Optional[List[RetrievalResult]] = None
    # 默认不保存原始顺序；有需要时再带上


class CoreReranker:
    """Core layer Reranker with fallback support.
    # 这是核心层的 rerank 编排类
    # 它把 libs 层的 reranker 封装起来，并统一处理 fallback
    
    This class wraps libs.reranker implementations and provides:
    # 这个类主要提供下面几件事
    
    1. Type conversion between RetrievalResult and reranker dict format
    # 1）在 core 层结果对象 和 libs 层 dict 格式之间做转换
    
    2. Graceful fallback when backend fails
    # 2）后端失败时优雅回退
    
    3. Configuration-driven backend selection
    # 3）按配置选择具体 backend
    
    4. TraceContext integration
    # 4）支持 trace 打点
    
    Design Principles Applied:
    # 这里再强调一次设计原则
    
    - Pluggable: Backend via RerankerFactory
    # 可插拔：后端通过工厂创建
    
    - Config-Driven: All parameters from settings
    # 配置驱动：所有参数都从 settings 里取
    
    - Fallback: Returns original order on failure
    # fallback：失败时返回原顺序
    
    - Observable: TraceContext support
    # 可观测：支持 trace
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> reranker = CoreReranker(settings)
        >>> results = [RetrievalResult(chunk_id="1", score=0.8, text="...", metadata={})]
        >>> reranked = reranker.rerank("query", results)
        >>> print(reranked.results)
    """
    
    def __init__(
        self,
        settings: Settings,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankConfig] = None,
    ) -> None:
        """Initialize CoreReranker.
        # 初始化 CoreReranker
        
        Args:
            settings: Application settings containing rerank configuration.
            # settings：项目总配置，里面会有 rerank 相关设置
            
            reranker: Optional reranker backend. If None, creates via RerankerFactory.
            # reranker：如果外部已经传了具体 backend，就直接用它
            #           否则就自己从工厂里创建
            
            config: Optional RerankConfig. If None, extracts from settings.
            # config：如果外部直接传了配置对象，就直接用
            #         否则从 settings 里提取
        """
        self.settings = settings
        # 把总配置先存起来，后面别的方法也可能要用
        
        # Extract config from settings or use provided
        # 先决定这次到底用哪份 rerank 配置
        if config is not None:
            self.config = config
            # 如果外部直接传了 config，就优先用外部传进来的
        else:
            self.config = self._extract_config(settings)
            # 否则从 settings 里提取一份 RerankConfig
        
        # Initialize reranker backend
        # 接下来初始化真正的 reranker 后端
        if reranker is not None:
            self._reranker = reranker
            # 如果外部已经传了具体 backend，就直接拿来用
        elif not self.config.enabled:
            self._reranker = NoneReranker(settings=settings)
            # 如果配置里根本没开 rerank，那就直接用 NoneReranker
            # 也就是“什么都不做”的后端
        else:
            try:
                self._reranker = RerankerFactory.create(settings)
                # 这里是真正走工厂创建后端
                # 可能创建出 LLMReranker / CrossEncoderReranker 等
            except Exception as e:
                logger.warning(f"Failed to create reranker, using NoneReranker: {e}")
                # 如果工厂创建失败，打 warning 日志
                self._reranker = NoneReranker(settings=settings)
                # 然后降级为 NoneReranker，不让整个系统初始化失败
        
        # Determine reranker type for result reporting
        # 最后根据 backend 类名，算出一个人类可读的 reranker 类型字符串
        self._reranker_type = self._get_reranker_type()
    
    def _extract_config(self, settings: Settings) -> RerankConfig:
        """Extract RerankConfig from settings.
        # 从总配置 settings 里，抽出 rerank 相关配置
        
        Args:
            settings: Application settings.
            # settings：项目总配置
            
        Returns:
            RerankConfig with values from settings.
            # 返回整理好的 RerankConfig 对象
        """
        try:
            rerank_settings = settings.rerank
            # 先取出 settings 里 rerank 这一段
            
            return RerankConfig(
                enabled=bool(rerank_settings.enabled) if rerank_settings else False,
                # 是否开启 rerank；如果根本没有 rerank 配置，就默认 False
                
                top_k=int(rerank_settings.top_k) if rerank_settings and hasattr(rerank_settings, 'top_k') else 5,
                # top_k：优先用配置里的值；没有就默认 5
                
                timeout=float(getattr(rerank_settings, 'timeout', 30.0)) if rerank_settings else 30.0,
                # timeout：有就取，没有就默认 30 秒
                
                fallback_on_error=True,
                # 这里直接写死成 True
                # 也就是默认 backend 报错时走回退
            )
        except AttributeError:
            logger.warning("Missing rerank configuration, using defaults (disabled)")
            # 如果 settings 里连 rerank 这一段都没有，打 warning
            return RerankConfig(enabled=False)
            # 然后返回一个默认配置：关闭 rerank
    
    def _get_reranker_type(self) -> str:
        """Get the type name of the current reranker backend.
        # 根据当前 backend 的类名，推断一个简短类型名
        
        Returns:
            String identifier for the reranker type.
            # 返回类似 llm / cross_encoder / none 这种字符串
        """
        class_name = self._reranker.__class__.__name__
        # 拿到当前 backend 的类名，比如 LLMReranker / CrossEncoderReranker
        
        if "LLM" in class_name:
            return "llm"
            # 类名里带 LLM，就认定是 llm 后端
        elif "CrossEncoder" in class_name:
            return "cross_encoder"
            # 类名里带 CrossEncoder，就认定是 cross_encoder 后端
        elif "None" in class_name:
            return "none"
            # 类名里带 None，就认定是 none
        else:
            return class_name.lower()
            # 其他情况就把类名转小写直接返回
    
    def _results_to_candidates(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Convert RetrievalResults to reranker candidate format.
        # 把 core 层的 RetrievalResult 列表
        # 转成 reranker backend 习惯吃的 dict 列表
        
        Args:
            results: List of RetrievalResult objects.
            # 原始检索结果列表
            
        Returns:
            List of dicts suitable for reranker input.
            # 返回给 reranker backend 的候选列表
        """
        candidates = []
        # 先准备一个空列表
        
        for result in results:
            # 遍历每一条 RetrievalResult
            candidates.append({
                "id": result.chunk_id,
                # reranker backend 用 id 识别候选
                
                "text": result.text,
                # 候选文本正文
                
                "score": result.score,
                # 原始检索分数，也一起传过去
                
                "metadata": result.metadata.copy(),
                # metadata 复制一份，避免后面原地改坏原对象
            })
        return candidates
        # 返回转换后的候选列表
    
    def _candidates_to_results(
        self,
        candidates: List[Dict[str, Any]],
        original_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Convert reranked candidates back to RetrievalResults.
        # 把 reranker backend 返回的候选 dict
        # 再转回 core 层统一使用的 RetrievalResult
        
        Args:
            candidates: Reranked candidates from reranker.
            # backend 返回的重排后 candidates
            
            original_results: Original results for reference.
            # 原始 RetrievalResult 列表，用来做回查
            
        Returns:
            List of RetrievalResult in reranked order.
            # 返回按重排后顺序排好的 RetrievalResult 列表
        """
        # Build lookup from original results
        # 先建一个字典：chunk_id -> 原始 RetrievalResult
        id_to_original = {r.chunk_id: r for r in original_results}
        
        results = []
        # 准备接最终转回去的 RetrievalResult 列表
        
        for candidate in candidates:
            # 遍历 backend 返回的每个候选
            chunk_id = candidate["id"]
            # 取出候选的 id
            
            # Get original result or build new one
            # 优先从原始结果里找同 id 的对象，保持 text / metadata 连贯
            if chunk_id in id_to_original:
                original = id_to_original[chunk_id]
                # 拿到这条候选原始对应的 RetrievalResult
                
                # Create new result with updated score
                # 这里要用 rerank 后的新分数覆盖原始分数
                rerank_score = candidate.get("rerank_score", candidate.get("score", 0.0))
                # 优先取 rerank_score
                # 如果没有，就退一步用 score
                # 再没有就给 0.0
                
                results.append(RetrievalResult(
                    chunk_id=original.chunk_id,
                    # id 继续沿用原始的
                    
                    score=rerank_score,
                    # score 更新成 rerank 分数
                    
                    text=original.text,
                    # text 继续沿用原始正文
                    
                    metadata={
                        **original.metadata,
                        # 先把原 metadata 展开保留
                        
                        "original_score": original.score,
                        # 额外存一下重排前原始分数
                        
                        "rerank_score": rerank_score,
                        # 再存一下 rerank 后的新分数
                        
                        "reranked": True,
                        # 明确标记这条结果已经被 rerank 过
                    },
                ))
            else:
                # Candidate not in original - build from candidate data
                # 理论上一般不会走到这
                # 只有 backend 返回了一个原始列表里根本没有的 id 才会进这里
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    # id 直接用 candidate 的
                    
                    score=candidate.get("rerank_score", candidate.get("score", 0.0)),
                    # 分数照样优先 rerank_score
                    
                    text=candidate.get("text", ""),
                    # 如果 candidate 自己带了 text 就用它，否则空串
                    
                    metadata=candidate.get("metadata", {}),
                    # metadata 也是直接从 candidate 里拿；没有就空 dict
                ))
        
        return results
        # 返回转回来的 RetrievalResult 列表
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> RerankResult:
        """Rerank retrieval results using configured backend.
        # 用当前配置好的 backend 对检索结果做重排
        
        Args:
            query: The user query string.
            # query：用户原始问题
            
            results: List of RetrievalResult objects to rerank.
            # results：前面检索阶段返回的候选结果
            
            top_k: Number of results to return. If None, uses config.top_k.
            # top_k：本次最多返回多少条；不传就用配置里的
            
            trace: Optional TraceContext for observability.
            # trace：可选的 trace，用来记录 rerank 阶段信息
            
            **kwargs: Additional parameters passed to reranker backend.
            # kwargs：额外参数，原样传给具体 backend
            
        Returns:
            RerankResult containing reranked results and metadata.
            # 返回完整的 RerankResult，包括结果和状态信息
        """
        effective_top_k = top_k if top_k is not None else self.config.top_k
        # 先算这次实际生效的 top_k
        # 本次调用传了就用本次的；否则用配置默认值
        
        # Early return for empty or single results
        # 先处理几种根本不需要 rerank 的场景
        if not results:
            return RerankResult(
                results=[],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
            # 如果压根没有结果，直接返回空
        
        if len(results) == 1:
            return RerankResult(
                results=results[:],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
            # 如果只有 1 条结果，也没必要重排，直接原样返回
        
        # If reranking disabled, return top_k results in original order
        # 如果配置里没开 rerank，或者实际 backend 就是 NoneReranker
        # 那也不真正做重排，直接按原顺序截断返回
        if not self.config.enabled or isinstance(self._reranker, NoneReranker):
            return RerankResult(
                results=results[:effective_top_k],
                # 保留原顺序，只截到 top_k
                
                used_fallback=False,
                # 这不算 fallback，因为这是“正常关闭 rerank”
                
                reranker_type="none",
                # 明确标记类型是 none
                
                original_order=results[:],
                # 带上原始顺序，方便调试
            )
        
        # Convert to reranker input format
        # 把 RetrievalResult 转成 reranker backend 能吃的 candidates 格式
        candidates = self._results_to_candidates(results)
        
        # Attempt reranking
        # 下面正式尝试调用 backend 做重排
        try:
            logger.debug(f"Reranking {len(candidates)} candidates with {self._reranker_type}")
            # 打一条 debug 日志：本次要重排多少条，用的是什么 backend
            
            _t0 = time.monotonic()
            # 记录开始时间
            
            reranked_candidates = self._reranker.rerank(
                query=query,
                # 把原始 query 传给 backend
                
                candidates=candidates,
                # 把候选列表传给 backend
                
                trace=trace,
                # trace 也往下传，后端如果支持也可以继续打点
                
                **kwargs,
                # 额外参数也一并透传
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            # 算出 rerank 本次耗时，单位毫秒
            
            # Convert back to RetrievalResult
            # 把 backend 返回的 candidates 再转回 RetrievalResult
            reranked_results = self._candidates_to_results(reranked_candidates, results)
            
            # Apply top_k limit
            # 再做一次 top_k 截断，只保留前 effective_top_k 条
            final_results = reranked_results[:effective_top_k]
            
            logger.info(f"Reranking complete: {len(final_results)} results returned")
            # 打 info 日志：本次 rerank 成功，返回了多少条
            
            if trace is not None:
                trace.record_stage("rerank", {
                    # 如果有 trace，就把 rerank 阶段信息记进去
                    
                    "method": self._reranker_type,
                    # 这次 rerank 用的方式，比如 llm / cross_encoder
                    
                    "provider": self._reranker_type,
                    # provider 这里也直接写成同样的类型名
                    
                    "input_count": len(candidates),
                    # 输入了多少条候选
                    
                    "output_count": len(final_results),
                    # 最终输出多少条
                    
                    "chunks": [
                        {
                            "chunk_id": r.chunk_id,
                            # 记录每条结果的 chunk_id
                            
                            "score": round(r.score, 4),
                            # 分数保留 4 位小数，方便看
                            
                            "text": r.text or "",
                            # 记录正文；如果没有就空串
                            
                            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                            # 优先从 metadata.source_path 取来源
                            # 取不到再看 source
                        }
                        for r in final_results
                    ],
                }, elapsed_ms=_elapsed)
                # 把上面这堆信息连同耗时一起记进 rerank 阶段
            
            return RerankResult(
                results=final_results,
                # 最终重排后的结果
                
                used_fallback=False,
                # 正常成功，不是 fallback
                
                reranker_type=self._reranker_type,
                # 记录这次用的是哪种 backend
                
                original_order=results[:],
                # 同时带上重排前原始顺序，方便前后对比
            )
            
        except Exception as e:
            logger.warning(f"Reranking failed, using fallback: {e}")
            # 只要 backend 调用过程中有任何异常，就打 warning
            # 然后尝试走 fallback
            
            if self.config.fallback_on_error:
                # 如果配置允许“出错时兜底”
                
                # Return original order as fallback
                # 那就返回原始顺序，至少保证 query 流程还能继续
                fallback_results = []
                # 先准备 fallback 结果列表
                
                for result in results[:effective_top_k]:
                    # 只对前 effective_top_k 条原始结果做 fallback
                    fallback_results.append(RetrievalResult(
                        chunk_id=result.chunk_id,
                        # id 保持不变
                        
                        score=result.score,
                        # 分数也保持原始检索分数
                        
                        text=result.text,
                        # 正文保持不变
                        
                        metadata={
                            **result.metadata,
                            # 先保留原 metadata
                            
                            "reranked": False,
                            # 标记这次其实没 rerank 成功
                            
                            "rerank_fallback": True,
                            # 标记这条结果是 fallback 返回的
                        },
                    ))
                
                return RerankResult(
                    results=fallback_results,
                    # 返回 fallback 结果
                    
                    used_fallback=True,
                    # 显式告诉上层：这次用到了 fallback
                    
                    fallback_reason=str(e),
                    # 把具体报错原因也带出去
                    
                    reranker_type=self._reranker_type,
                    # 记录原本想用的 backend 类型
                    
                    original_order=results[:],
                    # 同样保留原始顺序
                )
            else:
                raise RerankError(f"Reranking failed and fallback disabled: {e}") from e
                # 如果配置明确不允许 fallback
                # 那就真正抛出 RerankError，让上层自己处理
    
    @property
    def reranker_type(self) -> str:
        """Get the type of the current reranker backend."""
        # 只读属性：给外部看当前 backend 类型
        return self._reranker_type
    
    @property
    def is_enabled(self) -> bool:
        """Check if reranking is enabled."""
        # 只读属性：判断当前 rerank 是否真正启用
        # 条件有两个：
        # 1）配置上 enabled=True
        # 2）实际 backend 不是 NoneReranker
        return self.config.enabled and not isinstance(self._reranker, NoneReranker)


def create_core_reranker(
    settings: Settings,
    reranker: Optional[BaseReranker] = None,
) -> CoreReranker:
    """Factory function to create a CoreReranker instance.
    # 一个便捷工厂函数
    # 让外部可以不用直接写 CoreReranker(...)，而是统一走这个函数创建
    
    Args:
        settings: Application settings.
        # settings：项目配置
        
        reranker: Optional reranker backend override.
        # reranker：可选，外部传具体 backend 覆盖默认创建逻辑
        
    Returns:
        Configured CoreReranker instance.
        # 返回配置好的 CoreReranker 实例
    """
    return CoreReranker(settings=settings, reranker=reranker)
    # 本质上就是包了一层，最后还是返回 CoreReranker 实例