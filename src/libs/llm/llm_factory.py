"""Factory for creating LLM provider instances.

This module implements the Factory Pattern to instantiate the appropriate
LLM provider based on configuration, enabling configuration-driven selection
of different backends without code changes.

Supports both text-only LLMs and Vision LLMs (multimodal).
"""
# 上面这段：用配置选后端，不用改代码；支持纯文本模型和多模态视觉模型。

from __future__ import annotations  # 允许类型里写尚未定义的类名

from typing import TYPE_CHECKING, Any  # TYPE_CHECKING：只给类型检查器看的导入

from src.libs.llm.base_llm import BaseLLM  # 文本对话的抽象基类
from src.libs.llm.base_vision_llm import BaseVisionLLM  # 看图+对话的抽象基类

if TYPE_CHECKING:  # 运行时不真的 import，避免循环依赖
    from src.core.settings import Settings  # 应用配置类型，仅用于注解


# Import and register Vision LLM providers at module load time
def _register_vision_providers() -> None:  # 在文件最末尾会被调用一次
    """Register all Vision LLM provider implementations.
    
    This function is called at module import time to populate the
    Vision LLM provider registry. Add new providers here as they
    are implemented.
    """
    # 上面 docstring：说明「谁实现了就在这儿注册」；下面分厂商 try，避免一家挂了全家不注册
    try:  # 能 import 就说明实现了，就注册进工厂
        from src.libs.llm.azure_vision_llm import AzureVisionLLM  # Azure 多模态实现
        from src.libs.llm.llm_factory import LLMFactory  # 延迟 import 避免循环
        LLMFactory.register_vision_provider("azure", AzureVisionLLM)  # 名字 "azure" 对应这个类
    except ImportError:
        # Provider not yet implemented, skip registration
        pass  # 没装依赖或没实现就跳过，不挡程序启动

    try:  # 另一家厂商，逻辑同上
        from src.libs.llm.openai_vision_llm import OpenAIVisionLLM  # Open AI 多模态实现
        from src.libs.llm.llm_factory import LLMFactory  # 同样延迟 import 工厂
        LLMFactory.register_vision_provider("openai", OpenAIVisionLLM)  # 配置里写 openai 时用这类
    except ImportError:
        pass  # 同上，可选实现


class LLMFactory:  # 对外只用这个类的方法创建 LLM，不要直接 new 具体厂商类（除非测试）
    """Factory for creating LLM provider instances.
    
    This factory reads the provider configuration from settings and instantiates
    the corresponding LLM implementation. Supports both text-only LLMs and
    Vision LLMs (multimodal).
    
    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    - Separation: Text and Vision LLM registries are separate.
    """
    # 根据 settings 里写的 provider 名字，new 出对应的实现类（文本一套、视觉另一套）

    # Registry of supported text-only LLM providers (to be populated in B7.x tasks)
    _PROVIDERS: dict[str, type[BaseLLM]] = {}  # 名字小写 -> 文本 LLM 类（未注册时为空）

    # Registry of supported Vision LLM providers (to be populated in B9+ tasks)
    _VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}  # 名字小写 -> 视觉 LLM 类

    @classmethod  # 类方法：往「文本」注册表加一项，一般由各厂商模块在加载时调用
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """Register a new LLM provider implementation.
        
        This method allows provider implementations to register themselves
        with the factory, supporting extensibility.
        
        Args:
            name: The provider identifier (e.g., 'openai', 'azure', 'ollama').
            provider_class: The BaseLLM subclass implementing the provider.
        
        Raises:
            ValueError: If provider_class doesn't inherit from BaseLLM.
        """
        if not issubclass(provider_class, BaseLLM):  # 必须继承文本基类，否则接口不统一
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLLM"
            )
        cls._PROVIDERS[name.lower()] = provider_class  # 统一用小写当 key，避免大小写混用

    @classmethod  # 按 settings 创建「纯文本」LLM 实例
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLLM:
        """Create an LLM instance based on configuration.
        
        Args:
            settings: The application settings containing LLM configuration.
            **override_kwargs: Optional parameters to override config values.
        
        Returns:
            An instance of the configured LLM provider.
        
        Raises:
            ValueError: If the configured provider is not supported.
            AttributeError: If required configuration fields are missing.
        
        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> llm = LLMFactory.create(settings)
            >>> response = llm.chat([Message(role='user', content='Hello')])
        """
        # Extract provider name from settings
        try:
            provider_name = settings.llm.provider.lower()  # 配置里写的 provider，转小写
        except AttributeError as e:  # 没有 llm 或没有 provider 字段
            raise ValueError(
                "Missing required configuration: settings.llm.provider. "
                "Please ensure 'llm.provider' is specified in settings.yaml"
            ) from e

        # Look up provider class in registry
        provider_class = cls._PROVIDERS.get(provider_name)  # 从注册表取类，没有就是 None

        if provider_class is None:  # 配置写了但没人 register
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"  # 列出当前已注册的
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in tasks B7.1-B7.2."
            )

        # Instantiate the provider
        # Provider classes should accept settings and optional kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)  # 把配置和调用方覆盖参数传进去
        except Exception as e:  # 构造失败（缺 key、网络等）包一层，带上 provider 名
            raise RuntimeError(
                f"Failed to instantiate LLM provider '{provider_name}': {e}"
            ) from e

    @classmethod  # 查看当前有哪些文本 provider 已注册（调试用）
    def list_providers(cls) -> list[str]:
        """List all registered provider names.
        
        Returns:
            Sorted list of available provider identifiers.
        """
        return sorted(cls._PROVIDERS.keys())  # 排序后返回，方便展示和测试

    @classmethod  # 往「视觉」注册表加一项，与 register_provider 是两本账
    def register_vision_provider(
        cls,
        name: str,
        provider_class: type[BaseVisionLLM]
    ) -> None:
        """Register a new Vision LLM provider implementation.
        
        This method allows Vision LLM provider implementations to register
        themselves with the factory, supporting extensibility.
        
        Args:
            name: The provider identifier (e.g., 'azure', 'ollama').
            provider_class: The BaseVisionLLM subclass implementing the provider.
        
        Raises:
            ValueError: If provider_class doesn't inherit from BaseVisionLLM.
        """
        if not issubclass(provider_class, BaseVisionLLM):  # 必须继承视觉基类
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVisionLLM"
            )
        cls._VISION_PROVIDERS[name.lower()] = provider_class  # 视觉注册表，与文本表分开

    @classmethod  # 按 settings 创建「能看图」的 LLM 实例
    def create_vision_llm(
        cls,
        settings: Settings,
        **override_kwargs: Any
    ) -> BaseVisionLLM:
        """Create a Vision LLM instance based on configuration.
        
        Vision LLMs support multimodal input (text + image) and are used for
        tasks like image captioning, visual question answering, and document
        understanding with embedded images.
        
        Args:
            settings: The application settings containing Vision LLM configuration.
            **override_kwargs: Optional parameters to override config values.
        
        Returns:
            An instance of the configured Vision LLM provider.
        
        Raises:
            ValueError: If the configured provider is not supported or configuration is missing.
            RuntimeError: If provider instantiation fails.
        
        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> vision_llm = LLMFactory.create_vision_llm(settings)
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image("Describe this", image)
        """
        # Extract provider name from settings
        # Vision LLM config may be nested under settings.vision_llm or settings.llm
        try:
            # Try vision_llm section first
            if hasattr(settings, 'vision_llm') and hasattr(settings.vision_llm, 'provider'):
                provider_name = settings.vision_llm.provider.lower()  # 优先用专用视觉配置
            # Fallback to llm.provider (some providers support both text and vision)
            elif hasattr(settings, 'llm') and hasattr(settings.llm, 'provider'):
                provider_name = settings.llm.provider.lower()  # 没有 vision_llm 就复用文本的 provider 名
            else:
                raise AttributeError("No vision_llm or llm provider configuration found")
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vision_llm.provider or settings.llm.provider. "
                "Please ensure 'vision_llm.provider' or 'llm.provider' is specified in settings.yaml"
            ) from e

        # Look up provider class in vision registry
        provider_class = cls._VISION_PROVIDERS.get(provider_name)  # 注意查的是 _VISION_PROVIDERS，不是文本表

        if provider_class is None:
            available = ", ".join(sorted(cls._VISION_PROVIDERS.keys())) if cls._VISION_PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Vision LLM provider: '{provider_name}'. "
                f"Available Vision LLM providers: {available}. "
                f"Vision LLM implementations will be added in tasks B9+."
            )

        # Instantiate the provider
        try:
            return provider_class(settings=settings, **override_kwargs)  # 与 create() 一样传参方式
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Vision LLM provider '{provider_name}': {e}"
            ) from e

    @classmethod  # 查看当前有哪些视觉 provider 已注册
    def list_vision_providers(cls) -> list[str]:
        """List all registered Vision LLM provider names.
        
        Returns:
            Sorted list of available Vision LLM provider identifiers.
        """
        return sorted(cls._VISION_PROVIDERS.keys())  # 当前已注册的视觉后端名字列表


# Register Vision LLM providers at module load time
_register_vision_providers()  # import 本模块时执行，把 azure/openai 等挂到注册表上
