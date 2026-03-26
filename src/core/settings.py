"""Configuration loading and validation for the Modular RAG MCP Server."""

# 启用延迟类型注解解析，避免前向引用在运行时立即求值。
from __future__ import annotations

# dataclass 用来声明不可变配置对象。
from dataclasses import dataclass
# Path 用于路径拼接与标准化。
from pathlib import Path
# 类型注解工具。
from typing import Any, Dict, List, Optional, Union

# 读取 YAML 配置文件。
import yaml

# ---------------------------------------------------------------------------
# Repo root & path resolution
# ---------------------------------------------------------------------------
# 锚定当前文件位置：<repo>/src/core/settings.py -> parents[2] 即仓库根目录。
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# 默认配置文件绝对路径：<repo>/config/settings.yaml。
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """Resolve a repo-relative path to an absolute path.

    If *relative* is already absolute it is returned as-is. Otherwise
    it is resolved against :data:`REPO_ROOT`.

    >>> resolve_path("config/settings.yaml")  # doctest: +SKIP
    PosixPath('/home/user/Modular-RAG-MCP-Server/config/settings.yaml')
    """
    # 将输入统一包装成 Path，便于后续判断与运算。
    p = Path(relative)
    # 如果本来就是绝对路径，直接返回，避免二次拼接。
    if p.is_absolute():
        return p
    # 否则按仓库根目录拼接并 resolve，得到规范绝对路径。
    return (REPO_ROOT / p).resolve()


# 定义配置错误类型，统一抛出点便于上层捕获。
class SettingsError(ValueError):
    """Raised when settings validation fails."""


def _require_mapping(data: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    # 尝试从当前层读取指定字段。
    value = data.get(key)
    # 字段不存在或为 None，视为缺失。
    if value is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    # 字段存在但类型不是 dict（YAML mapping）则报错。
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {path}.{key}")
    # 校验通过后返回该子映射。
    return value


def _require_value(data: Dict[str, Any], key: str, path: str) -> Any:
    # 检查 key 是否存在且值不为 None。
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    # 返回原值（不做类型约束）。
    return data[key]


def _require_str(data: Dict[str, Any], key: str, path: str) -> str:
    # 先保证字段存在。
    value = _require_value(data, key, path)
    # 再保证类型是 str 且去空白后非空。
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    # 返回校验后的字符串。
    return value


def _require_int(data: Dict[str, Any], key: str, path: str) -> int:
    # 先保证字段存在。
    value = _require_value(data, key, path)
    # 要求类型为 int。
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    # 返回整型值。
    return value


def _require_number(data: Dict[str, Any], key: str, path: str) -> float:
    # 先保证字段存在。
    value = _require_value(data, key, path)
    # 允许 int 或 float。
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    # 统一转成 float，便于后续处理（如 temperature）。
    return float(value)


def _require_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    # 先保证字段存在。
    value = _require_value(data, key, path)
    # 要求类型为 bool。
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    # 返回布尔值。
    return value


def _require_list(data: Dict[str, Any], key: str, path: str) -> List[Any]:
    # 先保证字段存在。
    value = _require_value(data, key, path)
    # 要求类型为 list。
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    # 返回列表。
    return value


# 冻结的 LLM 配置结构，实例创建后不可变。
@dataclass(frozen=True)
class LLMSettings:
    # LLM 提供商标识，如 openai/azure/ollama/deepseek。
    provider: str
    # 模型名（或默认模型标识）。
    model: str
    # 生成温度。
    temperature: float
    # 最大输出 token 数。
    max_tokens: int
    # 以下为 Azure/OpenAI 相关可选字段：API Key。
    api_key: Optional[str] = None
    # 以下为 Azure/OpenAI 相关可选字段：API 版本。
    api_version: Optional[str] = None
    # 以下为 Azure/OpenAI 相关可选字段：Azure 端点。
    azure_endpoint: Optional[str] = None
    # 以下为 Azure/OpenAI 相关可选字段：部署名。
    deployment_name: Optional[str] = None
    # Ollama/OpenAI-compatible 场景可用的自定义 base_url。
    base_url: Optional[str] = None


# 冻结的 Embedding 配置结构。
@dataclass(frozen=True)
class EmbeddingSettings:
    # Embedding 提供商。
    provider: str
    # Embedding 模型名。
    model: str
    # 向量维度（用于校验/约束）。
    dimensions: int
    # API Key（可选）。
    api_key: Optional[str] = None
    # API 版本（可选）。
    api_version: Optional[str] = None
    # Azure 端点（可选）。
    azure_endpoint: Optional[str] = None
    # Azure 部署名（可选）。
    deployment_name: Optional[str] = None
    # 自定义 base_url（可选）。
    base_url: Optional[str] = None


# 冻结的向量库配置结构。
@dataclass(frozen=True)
class VectorStoreSettings:
    # 向量库类型，如 chroma/qdrant/pinecone。
    provider: str
    # 向量库持久化路径。
    persist_directory: str
    # 集合名称。
    collection_name: str


# 冻结的检索参数结构。
@dataclass(frozen=True)
class RetrievalSettings:
    # Dense 检索召回数。
    dense_top_k: int
    # Sparse 检索召回数。
    sparse_top_k: int
    # 融合后保留数量。
    fusion_top_k: int
    # RRF 常数。
    rrf_k: int


# 冻结的重排配置结构。
@dataclass(frozen=True)
class RerankSettings:
    # 是否启用重排。
    enabled: bool
    # 重排提供商。
    provider: str
    # 重排模型。
    model: str
    # 重排后保留数量。
    top_k: int


# 冻结的评估配置结构。
@dataclass(frozen=True)
class EvaluationSettings:
    # 是否启用评估。
    enabled: bool
    # 评估提供商。
    provider: str
    # 评估指标列表。
    metrics: List[str]


# 冻结的可观测性配置结构。
@dataclass(frozen=True)
class ObservabilitySettings:
    # 日志级别。
    log_level: str
    # 是否启用 tracing。
    trace_enabled: bool
    # trace 输出文件路径。
    trace_file: str
    # 是否启用结构化日志。
    structured_logging: bool


# 冻结的视觉 LLM 配置结构。
@dataclass(frozen=True)
class VisionLLMSettings:
    # 是否启用视觉能力。
    enabled: bool
    # 视觉模型提供商。
    provider: str
    # 视觉模型名。
    model: str
    # 图像最大边长限制。
    max_image_size: int
    # 视觉 API Key（可选）。
    api_key: Optional[str] = None
    # 视觉 API 版本（可选）。
    api_version: Optional[str] = None
    # 视觉 Azure 端点（可选）。
    azure_endpoint: Optional[str] = None
    # 视觉部署名（可选）。
    deployment_name: Optional[str] = None
    # 视觉 base_url（可选）。
    base_url: Optional[str] = None


# 冻结的入库配置结构。
@dataclass(frozen=True)
class IngestionSettings:
    # 分块大小。
    chunk_size: int
    # 分块重叠。
    chunk_overlap: int
    # 分块策略。
    splitter: str
    # 批处理大小。
    batch_size: int
    # 分块精修子配置（结构动态，故用 Dict[str, Any]）。
    chunk_refiner: Optional[Dict[str, Any]] = None
    # 元数据增强子配置（结构动态，故用 Dict[str, Any]）。
    metadata_enricher: Optional[Dict[str, Any]] = None


# 顶层配置对象：聚合所有子模块配置。
@dataclass(frozen=True)
class Settings:
    # 主 LLM 配置。
    llm: LLMSettings
    # Embedding 配置。
    embedding: EmbeddingSettings
    # 向量库配置。
    vector_store: VectorStoreSettings
    # 检索配置。
    retrieval: RetrievalSettings
    # 重排配置。
    rerank: RerankSettings
    # 评估配置。
    evaluation: EvaluationSettings
    # 可观测性配置。
    observability: ObservabilitySettings
    # 入库配置（可选）。
    ingestion: Optional[IngestionSettings] = None
    # 视觉 LLM 配置（可选）。
    vision_llm: Optional[VisionLLMSettings] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        # 根对象必须是 dict（YAML mapping）。
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        # 读取并校验必需模块：llm。
        llm = _require_mapping(data, "llm", "settings")
        # 读取并校验必需模块：embedding。
        embedding = _require_mapping(data, "embedding", "settings")
        # 读取并校验必需模块：vector_store。
        vector_store = _require_mapping(data, "vector_store", "settings")
        # 读取并校验必需模块：retrieval。
        retrieval = _require_mapping(data, "retrieval", "settings")
        # 读取并校验必需模块：rerank。
        rerank = _require_mapping(data, "rerank", "settings")
        # 读取并校验必需模块：evaluation。
        evaluation = _require_mapping(data, "evaluation", "settings")
        # 读取并校验必需模块：observability。
        observability = _require_mapping(data, "observability", "settings")

        # 默认入库配置为空（表示配置文件中未提供 ingestion）。
        ingestion_settings = None
        # 若存在 ingestion 块，则解析该可选模块。
        if "ingestion" in data:
            # 读取 ingestion 子映射并做 mapping 校验。
            ingestion = _require_mapping(data, "ingestion", "settings")
            # 将 ingestion 映射为强类型对象。
            ingestion_settings = IngestionSettings(
                # 读取 chunk_size（int）。
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                # 读取 chunk_overlap（int）。
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                # 读取 splitter（non-empty str）。
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                # 读取 batch_size（int）。
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
                # 读取可选子配置 chunk_refiner（可缺省）。
                chunk_refiner=ingestion.get("chunk_refiner"),
                # 读取可选子配置 metadata_enricher（可缺省）。
                metadata_enricher=ingestion.get("metadata_enricher"),
            )

        # 默认视觉配置为空（表示配置文件中未提供 vision_llm）。
        vision_llm_settings = None
        # 若存在 vision_llm 块，则解析该可选模块。
        if "vision_llm" in data:
            # 读取 vision_llm 子映射并做 mapping 校验。
            vision_llm = _require_mapping(data, "vision_llm", "settings")
            # 将 vision_llm 映射为强类型对象。
            vision_llm_settings = VisionLLMSettings(
                # 读取 enabled（bool）。
                enabled=_require_bool(vision_llm, "enabled", "vision_llm"),
                # 读取 provider（non-empty str）。
                provider=_require_str(vision_llm, "provider", "vision_llm"),
                # 读取 model（non-empty str）。
                model=_require_str(vision_llm, "model", "vision_llm"),
                # 读取 max_image_size（int）。
                max_image_size=_require_int(vision_llm, "max_image_size", "vision_llm"),
                # 以下字段都是可选透传，留给 provider 层解释。
                api_key=vision_llm.get("api_key"),
                api_version=vision_llm.get("api_version"),
                azure_endpoint=vision_llm.get("azure_endpoint"),
                deployment_name=vision_llm.get("deployment_name"),
                base_url=vision_llm.get("base_url"),
            )

        # 组装顶层 Settings 对象。
        settings = cls(
            # 组装 LLMSettings。
            llm=LLMSettings(
                # provider 必填非空字符串。
                provider=_require_str(llm, "provider", "llm"),
                # model 必填非空字符串。
                model=_require_str(llm, "model", "llm"),
                # temperature 必填数值，统一转 float。
                temperature=_require_number(llm, "temperature", "llm"),
                # max_tokens 必填 int。
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                # 以下参数可选（provider 再做语义校验）。
                api_key=llm.get("api_key"),
                api_version=llm.get("api_version"),
                azure_endpoint=llm.get("azure_endpoint"),
                deployment_name=llm.get("deployment_name"),
                base_url=llm.get("base_url"),
            ),
            # 组装 EmbeddingSettings。
            embedding=EmbeddingSettings(
                # provider 必填非空字符串。
                provider=_require_str(embedding, "provider", "embedding"),
                # model 必填非空字符串。
                model=_require_str(embedding, "model", "embedding"),
                # dimensions 必填 int。
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                # 以下参数可选透传。
                api_key=embedding.get("api_key"),
                api_version=embedding.get("api_version"),
                azure_endpoint=embedding.get("azure_endpoint"),
                deployment_name=embedding.get("deployment_name"),
                base_url=embedding.get("base_url"),
            ),
            # 组装 VectorStoreSettings。
            vector_store=VectorStoreSettings(
                # provider 必填非空字符串。
                provider=_require_str(vector_store, "provider", "vector_store"),
                # persist_directory 必填非空字符串。
                persist_directory=_require_str(vector_store, "persist_directory", "vector_store"),
                # collection_name 必填非空字符串。
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            # 组装 RetrievalSettings。
            retrieval=RetrievalSettings(
                # dense_top_k 必填 int。
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                # sparse_top_k 必填 int。
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                # fusion_top_k 必填 int。
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                # rrf_k 必填 int。
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
            ),
            # 组装 RerankSettings。
            rerank=RerankSettings(
                # enabled 必填 bool。
                enabled=_require_bool(rerank, "enabled", "rerank"),
                # provider 必填非空字符串。
                provider=_require_str(rerank, "provider", "rerank"),
                # model 必填非空字符串。
                model=_require_str(rerank, "model", "rerank"),
                # top_k 必填 int。
                top_k=_require_int(rerank, "top_k", "rerank"),
            ),
            # 组装 EvaluationSettings。
            evaluation=EvaluationSettings(
                # enabled 必填 bool。
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                # provider 必填非空字符串。
                provider=_require_str(evaluation, "provider", "evaluation"),
                # metrics 必填 list；并将每项转成 str，保证类型稳定。
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
            ),
            # 组装 ObservabilitySettings。
            observability=ObservabilitySettings(
                # log_level 必填非空字符串。
                log_level=_require_str(observability, "log_level", "observability"),
                # trace_enabled 必填 bool。
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                # trace_file 必填非空字符串。
                trace_file=_require_str(observability, "trace_file", "observability"),
                # structured_logging 必填 bool。
                structured_logging=_require_bool(observability, "structured_logging", "observability"),
            ),
            # 注入可选 ingestion 模块（可能为 None）。
            ingestion=ingestion_settings,
            # 注入可选 vision_llm 模块（可能为 None）。
            vision_llm=vision_llm_settings,
        )

        # 返回解析完成的 Settings 对象。
        return settings


def validate_settings(settings: Settings) -> None:
    """Validate settings and raise SettingsError if invalid."""

    # llm.provider 必须存在且非空。
    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    # embedding.provider 必须存在且非空。
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    # vector_store.provider 必须存在且非空。
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    # retrieval.rrf_k 必须存在且为真值（0 会被视为非法）。
    if not settings.retrieval.rrf_k:
        raise SettingsError("Missing required field: retrieval.rrf_k")
    # rerank.provider 必须存在且非空。
    if not settings.rerank.provider:
        raise SettingsError("Missing required field: rerank.provider")
    # evaluation.provider 必须存在且非空。
    if not settings.evaluation.provider:
        raise SettingsError("Missing required field: evaluation.provider")
    # observability.log_level 必须存在且非空。
    if not settings.observability.log_level:
        raise SettingsError("Missing required field: observability.log_level")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file and validate required fields.

    Args:
        path: Path to settings YAML. Defaults to
            ``<repo>/config/settings.yaml`` (absolute, CWD-independent).
    """
    # 有传 path 则优先使用；否则使用默认配置路径。
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    # 如果是相对路径，转成仓库绝对路径。
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    # 配置文件不存在则快速失败。
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    # 以 UTF-8 打开配置文件。
    with settings_path.open("r", encoding="utf-8") as handle:
        # 安全加载 YAML 到 Python dict/list 结构。
        data = yaml.safe_load(handle)

    # 解析 dict 为强类型 Settings（空文件时 data 可能为 None）。
    settings = Settings.from_dict(data or {})
    # 做二次基础校验。
    validate_settings(settings)
    # 返回最终可用配置对象。
    return settings