"""OpenAI Embedding implementation.

This module provides the OpenAI Embedding implementation that works with
the standard OpenAI Embeddings API.
"""
# 上面：用 OpenAI 官方 Embeddings HTTP API，把一批文本变成向量（也可兼容 Azure 部署 URL）。

from __future__ import annotations  # 类型注解里可用前向引用

import os  # 读环境变量里的 OPENAI_API_KEY
from typing import Any, List, Optional  # 类型注解

from src.libs.embedding.base_embedding import BaseEmbedding  # 统一接口：embed / get_dimension


class OpenAIEmbeddingError(RuntimeError):  # 调用失败时抛这个，方便上层 catch
    """Raised when OpenAI Embeddings API call fails."""


class OpenAIEmbedding(BaseEmbedding):  # 实现 BaseEmbedding，供工厂创建
    """OpenAI Embedding provider implementation.
    
    This class implements the BaseEmbedding interface for OpenAI's Embeddings API.
    It supports text-embedding-3-small, text-embedding-3-large, and older models
    like text-embedding-ada-002.
    
    Attributes:
        api_key: The API key for authentication.
        model: The model identifier to use.
        dimensions: Optional dimension reduction (only for text-embedding-3-*).
        base_url: The base URL for the API (default: OpenAI's endpoint).
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OpenAIEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    # 支持官方 OpenAI；若配置里有 azure_endpoint 则走「Azure 兼容 URL + api-version」模式

    DEFAULT_BASE_URL = "https://api.openai.com/v1"  # 默认直连 OpenAI 的 v1 根路径

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI Embedding provider.
        
        Args:
            settings: Application settings containing Embedding configuration.
            api_key: Optional API key override (falls back to settings.embedding.api_key or env var).
            base_url: Optional base URL override.
            **kwargs: Additional configuration overrides.
        
        Raises:
            ValueError: If API key is not provided and not found in environment.
        
        Note:
            When azure_endpoint is present in settings, the provider automatically
            constructs the Azure-compatible OpenAI URL and uses api-key auth.
        """
        self.model = settings.embedding.model  # 模型名，如 text-embedding-3-small

        # Extract optional dimensions setting
        self.dimensions = getattr(settings.embedding, 'dimensions', None)  # 仅 3 系模型可降维

        # API key: explicit > settings > env var
        self.api_key = (
            api_key  # 调用方直接传入
            or getattr(settings.embedding, 'api_key', None)  # settings.yaml
            or os.environ.get("OPENAI_API_KEY")  # 环境变量兜底
        )
        if not self.api_key:  # 三处都没有就没法鉴权
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (embedding.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )

        # Azure-compatible mode detection
        azure_endpoint = getattr(settings.embedding, 'azure_endpoint', None)  # 有则走 Azure 部署地址
        self.api_version = getattr(settings.embedding, 'api_version', None)  # Azure OpenAI 要 api-version
        self._use_azure_auth = False  # True 时客户端会带 api-key 头等

        if base_url:
            self.base_url = base_url  # 显式覆盖最优先
        elif azure_endpoint:
            # Azure-compatible mode: construct deployment-based URL
            deployment = getattr(settings.embedding, 'deployment_name', None) or self.model  # 部署名缺省用 model
            self.base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}"  # 标准 Azure OpenAI 路径
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"  # 给个常用默认版本
        else:
            settings_base_url = getattr(settings.embedding, 'base_url', None)  # 自建兼容网关可填
            self.base_url = settings_base_url if settings_base_url else self.DEFAULT_BASE_URL  # 否则官方默认

        # Store any additional kwargs for future use
        self._extra_config = kwargs  # 预留扩展，当前 embed 里未用

    def embed(
        self,
        texts: List[str],  # 一批待编码文本，顺序与返回向量一致
        trace: Optional[Any] = None,  # 预留观测/追踪，当前未使用
        **kwargs: Any,  # 可覆盖 dimensions 等
    ) -> List[List[float]]:  # 每条文本对应一个 float 向量
        """Generate embeddings for a batch of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed. Must not be empty.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Override parameters (dimensions, etc.).
        
        Returns:
            List of embedding vectors, where each vector is a list of floats.
            The length of the outer list matches len(texts).
        
        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            OpenAIEmbeddingError: If API call fails.
        """
        # Validate input
        self.validate_texts(texts)  # 基类：非空、类型等

        # Import OpenAI client (lazy import to avoid dependency at module level)
        try:
            from openai import OpenAI  # 没装 openai 包则在这里报错
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package not installed. "
                "Install with: pip install openai"
            ) from e

        # Initialize OpenAI client
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,  # 官方或 Azure 部署根 URL
        }
        # Azure-compatible mode: add api-version query param and api-key header
        if self._use_azure_auth and self.api_version:  # Azure 要求 query 里带 api-version
            client_kwargs["default_query"] = {"api-version": self.api_version}
            client_kwargs["default_headers"] = {"api-key": self.api_key}  # Azure 常用 api-key 头（与 OpenAI Bearer 不同）

        client = OpenAI(**client_kwargs)  # 官方 SDK，同一套 client 调 embeddings

        # Prepare API call parameters
        api_params = {
            "input": texts,  # 一批字符串
            "model": self.model,  # Azure 模式下有时与 deployment 同名
        }

        # Add dimensions if specified (only for text-embedding-3-* models)
        # text-embedding-ada-002 does NOT support the dimensions parameter
        dimensions = kwargs.get("dimensions", self.dimensions)  # 调用可覆盖实例上的 dimensions
        if dimensions is not None and self.model.startswith("text-embedding-3"):  # 只对 3 系传 dimensions
            api_params["dimensions"] = dimensions

        # Call OpenAI API
        try:
            response = client.embeddings.create(**api_params)  # 发 HTTP 请求
        except Exception as e:
            raise OpenAIEmbeddingError(
                f"OpenAI Embeddings API call failed: {e}"
            ) from e

        # Extract embeddings from response
        # Response format: response.data is a list of objects with .embedding attribute
        try:
            embeddings = [item.embedding for item in response.data]  # 按顺序取出每条向量
        except (AttributeError, KeyError) as e:
            raise OpenAIEmbeddingError(
                f"Failed to parse OpenAI Embeddings API response: {e}"
            ) from e

        # Verify output matches input length
        if len(embeddings) != len(texts):  # 防止 API 异常返回条数不对
            raise OpenAIEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        return embeddings

    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension for the configured model.
        
        Returns:
            The embedding dimension, or None if not deterministic.
        
        Note:
            For text-embedding-3-* models with custom dimensions, returns
            the configured dimension. For other models, returns their default.
        """
        # If dimensions explicitly configured, return it
        if self.dimensions is not None:  # 用户配了降维后的维度，就以配置为准
            return self.dimensions

        # Model-specific defaults
        model_dimensions = {
            "text-embedding-3-small": 1536,  # 默认满维
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        return model_dimensions.get(self.model)  # 未知模型返回 None，表示无法静态确定
