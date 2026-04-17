"""Cross-Encoder based Reranker implementation.
# 这个文件实现的是“基于 Cross-Encoder 的重排器”

This module implements reranking using Cross-Encoder models that directly score
# 它的核心思想是：把 query 和 passage 两个文本直接成对送进模型打分

(query, passage) pairs. Supports both local models via sentence-transformers
# 这种模型直接看 (query, passage) 这对输入，给一个相关性分数
# 当前支持通过 sentence-transformers 加载本地模型

and API-based endpoints.
# 设计上也兼容 API 型端点，不过当前这份代码主要是本地 CrossEncoder
"""

from __future__ import annotations
# 让类型标注延迟解析，减少前向引用时的麻烦

import logging
# 日志模块，用来打印 info / debug / error

from typing import Any, Dict, List, Optional
# 类型标注相关：任意类型、字典、列表、可选值

from src.libs.reranker.base_reranker import BaseReranker
# 引入 reranker 抽象基类
# 这个类会给当前实现提供统一接口和一些基础校验方法

logger = logging.getLogger(__name__)
# 以当前模块名创建 logger，后面打印日志都走它


class CrossEncoderRerankError(RuntimeError):
    """Raised when Cross-Encoder reranking fails."""
    # 自定义异常：只要 Cross-Encoder 重排失败，就抛这个


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder based reranker for scoring query-passage pairs.
    # 这是一个基于 Cross-Encoder 的重排器类
    # 它会对 (query, passage) 对直接打分
    
    This implementation uses Cross-Encoder models (e.g., ms-marco-MiniLM)
    # 它用的是 Cross-Encoder 模型，比如 ms-marco-MiniLM
    
    that directly encode and score (query, passage) pairs, providing more
    # 这种模型会直接把 query 和 passage 一起编码并打相关性分
    
    accurate relevance scores than bi-encoder approaches at the cost of
    # 相比 bi-encoder（双塔模型），它通常更准
    
    higher computational requirements.
    # 但代价也更高，算得更慢、更吃资源
    
    Design Principles Applied:
    # 下面是这个类遵守的设计原则
    
    - Pluggable: Can be swapped with other reranker implementations via factory.
    # 可插拔：可以通过工厂和别的 reranker 实现互换
    
    - Config-Driven: Model name and parameters come from settings.yaml.
    # 配置驱动：模型名和参数从 settings.yaml 里来
    
    - Observable: Supports TraceContext for monitoring (Stage F integration).
    # 可观测：支持 trace，方便后面监控
    
    - Fallback-Aware: Provides timeout/failure signals for upstream fallback.
    # 可感知 fallback：自己不做 fallback，但会把失败信号抛给上层
    
    - Deterministic Testing: Supports mock scorer injection for testing.
    # 方便测试：支持注入 mock 模型，测试时不一定真的加载大模型
    """
    
    def __init__(
        self,
        settings: Any,
        model: Optional[Any] = None,
        timeout: float = 10.0,
        **kwargs: Any
    ) -> None:
        """Initialize the Cross-Encoder Reranker.
        # 初始化 CrossEncoderReranker
        
        Args:
            settings: Application settings containing rerank configuration.
            # settings：项目总配置，里面会有 rerank 相关设置
            
            model: Optional pre-initialized CrossEncoder model. If None, creates
                from settings.rerank.model. Used for testing to inject mock models.
            # model：可选，外部直接传一个已经准备好的模型对象
            #        如果不传，就从 settings.rerank.model 里读模型名自己加载
            #        测试时很常用，因为可以直接塞 mock model
            
            timeout: Maximum time (seconds) to wait for reranking. Default 10s.
                Used to enable fallback strategies when reranking takes too long.
            # timeout：重排最大等待时间，默认 10 秒
            #          当前这份代码里只是保存下来，真正超时控制没在这里实现
            
            **kwargs: Additional provider-specific parameters.
            # kwargs：其他额外参数，先统一收着
        """
        self.settings = settings
        # 保存总配置
        
        self.timeout = timeout
        # 保存超时时间
        
        self.kwargs = kwargs
        # 保存其他附加参数
        
        # Initialize or inject model
        # 下面开始准备真正要用的 CrossEncoder 模型
        if model is not None:
            self.model = model
            # 如果外部直接传了 model，就直接用
            # 常见于单元测试：注入一个假的 mock model
        else:
            try:
                model_name = self._get_model_name_from_settings(settings)
                # 先从配置里拿模型名
                
                self.model = self._load_cross_encoder_model(model_name)
                # 再根据模型名把真正的 CrossEncoder 模型加载出来
            except Exception as e:
                raise CrossEncoderRerankError(
                    f"Failed to initialize Cross-Encoder model: {e}"
                ) from e
                # 只要初始化模型这一步失败，就统一包装成 CrossEncoderRerankError 抛出去
    
    def _get_model_name_from_settings(self, settings: Any) -> str:
        """Extract model name from settings.
        # 从 settings 里提取 rerank 用的模型名
        
        Args:
            settings: Application settings.
            # settings：项目总配置
        
        Returns:
            Model name string.
            # 返回模型名字字符串
        
        Raises:
            AttributeError: If rerank.model is not configured.
            # 如果没配 rerank.model，就报 AttributeError
        """
        try:
            model_name = settings.rerank.model
            # 取出配置里的 rerank.model
            
            if not model_name or not isinstance(model_name, str):
                raise ValueError("Model name must be a non-empty string")
                # 模型名必须是非空字符串，否则直接报错
                
            return model_name
            # 合法就直接返回模型名
        except AttributeError as e:
            raise AttributeError(
                "Missing configuration: settings.rerank.model. "
                "Please specify 'rerank.model' in settings.yaml"
            ) from e
            # 如果 settings 里压根没有 rerank.model 这个字段
            # 就抛一个更可读的错误，提醒你去 settings.yaml 里配
    
    def _load_cross_encoder_model(self, model_name: str) -> Any:
        """Load the Cross-Encoder model.
        # 根据模型名加载真正的 Cross-Encoder 模型
        
        Args:
            model_name: Name or path of the Cross-Encoder model.
            # model_name：模型名或者本地模型路径
        
        Returns:
            Initialized CrossEncoder instance.
            # 返回初始化好的 CrossEncoder 模型对象
        
        Raises:
            ImportError: If sentence-transformers is not installed.
            # 如果没装 sentence-transformers，会报 ImportError
            
            RuntimeError: If model loading fails.
            # 如果模型加载失败，会报 RuntimeError
        """
        try:
            from sentence_transformers import CrossEncoder
            # 动态导入 CrossEncoder
            # 这样只有真正用到这个 reranker 时才要求装 sentence-transformers
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Cross-Encoder reranking. "
                "Install it with: pip install sentence-transformers"
            ) from e
            # 如果没装依赖，就抛出一个带安装提示的错误
        
        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}")
            # 打日志：开始加载模型
            
            model = CrossEncoder(model_name)
            # 按模型名实例化 CrossEncoder 模型
            
            logger.info(f"Cross-Encoder model loaded successfully: {model_name}")
            # 打日志：模型加载成功
            
            return model
            # 返回加载好的模型对象
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Cross-Encoder model '{model_name}': {e}"
            ) from e
            # 只要模型加载过程有异常，就包装成 RuntimeError 抛出去
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using Cross-Encoder scoring.
        # 用 Cross-Encoder 对候选列表做重排
        
        Args:
            query: The user query string.
            # query：用户的问题
            
            candidates: List of candidate records to rerank. Each must contain
                either 'text' or 'content' field for scoring.
            # candidates：候选列表
            # 每个候选至少要能拿到正文，支持 text 或 content 字段
            
            trace: Optional TraceContext for observability (Stage F integration).
            # trace：可选 trace，用来做可观测记录
            
            **kwargs: Additional parameters (top_k to limit output, etc.).
            # kwargs：额外参数，比如 top_k
        
        Returns:
            Reranked list of candidates ordered by relevance score (descending).
            Each candidate includes a 'rerank_score' field with the model's score.
            # 返回按相关性从高到低排好的候选列表
            # 每个候选都会多一个 rerank_score 字段
        
        Raises:
            ValueError: If query or candidates are invalid.
            # 如果输入 query 或 candidates 非法，会报 ValueError
            
            CrossEncoderRerankError: If scoring fails or times out.
            # 如果模型打分失败，会报 CrossEncoderRerankError
        """
        # Validate inputs
        # 先校验输入是否合法
        self.validate_query(query)
        # 检查 query 合不合法，比如不能为空
        
        self.validate_candidates(candidates)
        # 检查 candidates 合不合法，比如是不是列表、有没有内容等
        
        # Extract top_k parameter
        # 读取 top_k 参数
        top_k = kwargs.get("top_k", len(candidates))
        # 如果调用时传了 top_k 就用它
        # 没传就默认返回全部候选
        
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
            # top_k 必须是正整数，否则直接报错
        
        try:
            # Prepare (query, passage) pairs for scoring
            # 第一步：把 query 和每个 candidate 的正文配成 (query, passage) 对
            pairs = self._prepare_pairs(query, candidates)
            
            # Score pairs using the model
            # 第二步：用 Cross-Encoder 模型给每一对打分
            scores = self._score_pairs(pairs, trace=trace)
            
            # Attach scores to candidates and sort
            # 第三步：把分数贴回候选上，再按分数排序
            reranked = self._attach_scores_and_sort(candidates, scores, top_k)
            
            if trace:
                self._log_trace(trace, query, len(candidates), len(reranked))
                # 如果传了 trace，就记录一下这次 rerank 的基本信息
            
            return reranked
            # 返回已经重排好的候选列表
            
        except Exception as e:
            logger.error(f"Cross-Encoder reranking failed: {e}", exc_info=True)
            # 只要中间任何一步失败，就记 error 日志，附带完整异常堆栈
            
            # Signal failure for upstream fallback logic
            # 这里不自己做 fallback，而是把失败信号抛给上层
            raise CrossEncoderRerankError(
                f"Cross-Encoder reranking failed: {e}"
            ) from e
    
    def _prepare_pairs(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """Prepare (query, passage) pairs for scoring.
        # 把 query 和 candidates 处理成模型能吃的 (query, passage) 对
        
        Args:
            query: The user query.
            # query：用户问题
            
            candidates: List of candidate records.
            # candidates：候选列表
        
        Returns:
            List of (query, passage_text) tuples.
            # 返回一个列表，里面每项都是 (query, passage_text)
        """
        pairs = []
        # 先准备一个空列表来装所有 pair
        
        for candidate in candidates:
            # 遍历每个候选
            
            # Extract text from candidate (support both 'text' and 'content' keys)
            # 从候选里取正文，兼容 text 和 content 两种字段名
            text = candidate.get("text") or candidate.get("content", "")
            # 优先用 text
            # 没有 text 就用 content
            # 再没有就给空字符串
            
            if not isinstance(text, str):
                text = str(text)
                # 如果正文不是字符串，就强制转成字符串
            
            pairs.append((query, text))
            # 组成一个 (query, text) 二元组塞进去
        
        return pairs
        # 返回最终拼好的所有 query-passage 对
    
    def _score_pairs(
        self,
        pairs: List[tuple[str, str]],
        trace: Optional[Any] = None
    ) -> List[float]:
        """Score (query, passage) pairs using the Cross-Encoder model.
        # 用 Cross-Encoder 模型给每个 (query, passage) 对打分
        
        Args:
            pairs: List of (query, passage) tuples.
            # pairs：一组 query-passage 对
            
            trace: Optional TraceContext for observability.
            # trace：可选 trace，目前这个函数里没真正用到
        
        Returns:
            List of relevance scores (one per pair).
            # 返回一组分数，顺序和 pairs 一一对应
        
        Raises:
            CrossEncoderRerankError: If scoring fails or times out.
            # 打分失败时抛 CrossEncoderRerankError
        """
        try:
            # Use model.predict() to score all pairs in batch
            # 直接调用模型的 predict() 批量打分
            scores = self.model.predict(pairs)
            
            # Convert numpy array to list if needed
            # 如果返回的是 numpy 数组，就转成普通 Python list
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            return scores
            # 返回分数列表
            
        except Exception as e:
            raise CrossEncoderRerankError(
                f"Failed to score pairs with Cross-Encoder: {e}"
            ) from e
            # 只要模型打分出错，就包装成 CrossEncoderRerankError 抛出去
    
    def _attach_scores_and_sort(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Attach scores to candidates and sort by relevance.
        # 把模型打出来的分数贴回 candidates，再按分数从高到低排序
        
        Args:
            candidates: Original candidate list.
            # 原始候选列表
            
            scores: Relevance scores from the model.
            # 模型打出来的相关性分数
            
            top_k: Number of top candidates to return.
            # 最后只保留前 top_k 条
        
        Returns:
            Sorted list of top_k candidates with 'rerank_score' field added.
            # 返回已经排序好并且带 rerank_score 的候选列表
        """
        # Attach scores to candidates
        # 先把分数贴回每个 candidate
        scored_candidates = []
        # 准备一个新列表，避免修改原始输入
        
        for candidate, score in zip(candidates, scores):
            # candidates 和 scores 一一对应同时遍历
            
            # Create a copy to avoid modifying original
            # 复制一份 candidate，避免改到原始对象
            candidate_copy = candidate.copy()
            
            candidate_copy["rerank_score"] = float(score)
            # 给复制后的 candidate 加上 rerank_score 字段
            
            scored_candidates.append(candidate_copy)
            # 放进新列表里
        
        # Sort by score (descending) and take top_k
        # 按 rerank_score 从高到低排序，再截前 top_k 条
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x["rerank_score"],
            # 排序键：rerank_score
            
            reverse=True
            # reverse=True 表示降序，高分在前
        )
        
        return sorted_candidates[:top_k]
        # 只返回前 top_k 条
    
    def _log_trace(
        self,
        trace: Any,
        query: str,
        input_count: int,
        output_count: int
    ) -> None:
        """Log reranking operation to trace context.
        # 记录 rerank 相关的 trace 信息
        
        Args:
            trace: TraceContext instance.
            # trace：TraceContext 实例
            
            query: The query string.
            # query：用户问题
            
            input_count: Number of input candidates.
            # input_count：输入候选数
            
            output_count: Number of output candidates.
            # output_count：输出候选数
        """
        # Placeholder for Stage F integration
        # 这里目前只是一个占位实现，真正的 trace 集成以后再补
        
        # Future: trace.log_rerank_step(...)
        # 未来可能会在这里调用更正式的 trace 记录方法
        
        logger.debug(
            f"Cross-Encoder rerank: query='{query[:50]}...', "
            f"input={input_count}, output={output_count}"
        )
        # 现在只是先打一条 debug 日志：
        # query 截前 50 个字符，外加输入输出数量