"""
先定义协议处理器和工具结构 → 把工具注册进 ProtocolHandler → 
再创建底层 Server → 给 Server 绑定 tools/list 和 tools/call 两个处理函数 → 
外部 MCP Client 请求时，最终会转到 ProtocolHandler 去列工具或执行工具
"""

"""MCP Protocol Handler for JSON-RPC 2.0 message handling.
# 这个文件的作用：处理 MCP 协议里和 JSON-RPC 2.0 相关的消息。

This module provides the ProtocolHandler class that encapsulates:
# 这句说明：这个模块主要提供一个 ProtocolHandler 类，它把下面这些事情包起来了：

- Tool registration and schema management
# 1）工具注册和工具 schema 管理

- JSON-RPC error code handling
# 2）JSON-RPC 错误码处理

- Capability negotiation during initialize
# 3）在 initialize 阶段返回 server 能力信息
"""

from __future__ import annotations
# 这个导入让类型注解支持“延迟解析”
# 好处是：类还没定义完时，也能在类型标注里先写类名，避免一些前向引用问题。

from dataclasses import dataclass, field
# dataclass：快速定义“主要用于装数据”的类
# field：给 dataclass 字段设置默认工厂等高级配置

from typing import Any, Callable, Dict, List, Optional
# 导入类型标注工具：
# Any：任意类型
# Callable：可调用对象，通常是函数
# Dict：字典
# List：列表
# Optional：可为空

from mcp import types
# 导入 MCP SDK 里的 types 模块，里面有 Tool、CallToolResult、TextContent 等类型。

from mcp.server.lowlevel import Server
# 导入 MCP 底层 Server 类，用来真正创建一个可运行的 MCP server。

from src.observability.logger import get_logger
# 导入项目里的日志工厂函数，用来拿 logger。


# JSON-RPC 2.0 Error Codes
# 下面这个类专门放 JSON-RPC 2.0 的标准错误码。
class JSONRPCErrorCodes:
    """Standard JSON-RPC 2.0 error codes."""
    # 这个类只是一个常量容器，专门收纳标准错误码。

    PARSE_ERROR = -32700
    # 解析错误：收到的 JSON 根本解析不了

    INVALID_REQUEST = -32600
    # 非法请求：请求结构不符合 JSON-RPC 规范

    METHOD_NOT_FOUND = -32601
    # 方法不存在：请求的方法名不存在

    INVALID_PARAMS = -32602
    # 参数错误：方法有，但参数不对

    INTERNAL_ERROR = -32603
    # 服务器内部错误


@dataclass
# 把这个类声明成 dataclass，说明它主要是装工具定义数据的。
class ToolDefinition:
    """Definition of an MCP tool."""
    # 这个类表示“一个 MCP 工具的完整定义”。

    name: str
    # 工具名，比如 query_knowledge_hub

    description: str
    # 工具说明，给客户端展示“这个工具是干啥的”

    input_schema: Dict[str, Any]
    # 输入参数的 JSON Schema，告诉客户端这个工具需要什么参数

    handler: Callable[..., Any]
    # 真正执行工具逻辑的函数
    # ... 的意思是参数个数不固定
    # Any 的意思是返回值类型先不限制


@dataclass
# 这个类也是 dataclass，用来作为整个 MCP 协议处理器。
class ProtocolHandler:
    """Handles MCP protocol operations including tool registration and execution.

    This class encapsulates:
    # 这句说明它包了哪几类事情：

    - Tool registration with schema validation
    # 1）注册工具，并保存 schema

    - Tool execution with error handling
    # 2）执行工具，并统一做错误处理

    - Capability declaration for initialize response
    # 3）返回 server 的能力声明，给 initialize 响应用

    Attributes:
        server_name: Name of the MCP server.
        # server_name：server 名称

        server_version: Version string of the server.
        # server_version：server 版本号

        tools: Registry of available tools.
        # tools：一个注册表，保存当前可用的所有工具
    """

    server_name: str
    # MCP server 的名字

    server_version: str
    # MCP server 的版本号

    tools: Dict[str, ToolDefinition] = field(default_factory=dict)
    # tools 是一个字典：
    # key 是工具名
    # value 是 ToolDefinition
    # default_factory=dict 表示默认给一个新的空字典，避免多个实例共用同一个字典

    def __post_init__(self) -> None:
        """Initialize logger after dataclass initialization."""
        # dataclass 自动 __init__ 跑完后，会自动再调用 __post_init__
        self._logger = get_logger(log_level="INFO")
        # 给当前 ProtocolHandler 挂一个 logger，日志级别设为 INFO

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register a tool with the protocol handler.

        Args:
            name: Unique name for the tool.
            # 工具唯一名字

            description: Human-readable description of what the tool does.
            # 给人看的工具说明

            input_schema: JSON Schema for the tool's input parameters.
            # 工具输入参数的 JSON Schema

            handler: Async function that executes the tool logic.
            # 真正执行工具逻辑的函数，通常是 async 函数

        Raises:
            ValueError: If a tool with the same name is already registered.
            # 如果同名工具已经注册过了，就报错
        """
        if name in self.tools:
            # 如果当前工具名已经在注册表里了，说明重复注册
            raise ValueError(f"Tool '{name}' is already registered")
            # 直接报错，防止覆盖已有工具

        self.tools[name] = ToolDefinition(
            name=name,
            # 保存工具名

            description=description,
            # 保存工具说明

            input_schema=input_schema,
            # 保存输入 schema

            handler=handler,
            # 保存执行函数
        )
        self._logger.info("Registered tool: %s", name)
        # 记录一条日志：某个工具已经注册完成

    def get_tool_schemas(self) -> List[types.Tool]:
        """Get list of tool schemas for tools/list response.

        Returns:
            List of Tool objects with name, description, and inputSchema.
            # 返回 MCP SDK 里的 Tool 对象列表，给 tools/list 请求使用
        """
        return [
            types.Tool(
                name=tool.name,
                # MCP Tool 对象里的 name 字段

                description=tool.description,
                # MCP Tool 对象里的 description 字段

                inputSchema=tool.input_schema,
                # MCP Tool 对象里的 inputSchema 字段
            )
            for tool in self.tools.values()
            # 遍历当前所有注册过的工具定义，一个个转成 MCP SDK 的 Tool 对象
        ]

    async def execute_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """Execute a registered tool by name.

        Args:
            name: Name of the tool to execute.
            # 要执行的工具名

            arguments: Arguments to pass to the tool handler.
            # 要传给工具 handler 的参数字典

        Returns:
            CallToolResult with content blocks or error indication.
            # 返回 MCP SDK 的 CallToolResult 对象，里面装执行结果或者报错信息

        Raises:
            ValueError: If tool is not found.
            # 这里文档写的是可能 ValueError，但当前代码里实际上没有 raise，而是返回错误结果
        """
        if name not in self.tools:
            # 如果请求的工具名不在注册表里，说明工具不存在
            self._logger.warning("Tool not found: %s", name)
            # 记一条 warning 日志

            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        # 这是一个文本类型的 content block

                        text=f"Error: Tool '{name}' not found",
                        # 返回给客户端的错误文本
                    )
                ],
                isError=True,
                # 明确告诉客户端：这是错误结果
            )

        tool = self.tools[name]
        # 从注册表里拿出对应的 ToolDefinition

        try:
            self._logger.info("Executing tool: %s", name)
            # 记日志：开始执行哪个工具

            result = await tool.handler(**arguments)
            # 真正执行工具
            # **arguments 表示把参数字典拆开传进去
            # 比如 {"query": "abc", "top_k": 3} 会变成 handler(query="abc", top_k=3)

            # Handle different return types
            # 下面开始处理“工具 handler 返回了不同类型结果”的情况
            if isinstance(result, types.CallToolResult):
                # 如果工具本身已经直接返回了标准的 CallToolResult
                return result
                # 那就原样返回，不再包装

            if isinstance(result, str):
                # 如果工具返回的是字符串
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=result)],
                    # 把这个字符串包装成一个 text content
                    isError=False,
                    # 说明这是正常结果，不是错误
                )

            if isinstance(result, list):
                # 如果工具返回的是 list
                return types.CallToolResult(content=result, isError=False)
                # 直接把这个 list 当成 content 列表塞进去返回

            # Default: convert to string
            # 如果既不是 CallToolResult，也不是 str，也不是 list
            # 那就兜底转字符串
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(result))],
                # 把任意对象转成字符串后作为文本返回
                isError=False,
            )

        except TypeError as e:
            # Invalid parameters
            # 这里一般是参数不匹配，比如少传、多传、字段名不对
            self._logger.error("Invalid params for tool %s: %s", name, e)
            # 记一条 error 日志

            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error: Invalid parameters - {e}",
                        # 返回参数错误详情
                    )
                ],
                isError=True,
                # 标记这是错误
            )
        except Exception as e:
            # Internal error - don't leak stack trace
            # 其他所有异常都走这里，属于服务器内部错误
            # 注意：不会把堆栈详情直接返回给客户端，避免暴露内部实现
            self._logger.exception("Internal error executing tool %s", name)
            # logger.exception 会自动带 traceback 记日志

            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error: Internal server error while executing '{name}'",
                        # 返回一个通用内部错误提示，不把具体堆栈暴露出去
                    )
                ],
                isError=True,
                # 标记这是错误
            )

    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities for initialize response.

        Returns:
            Dictionary of server capabilities.
            # 返回一个字典，表示 server 支持哪些能力
        """
        return {
            "tools": {} if self.tools else {},
            # 这里不管有没有工具，最后其实都返回 {}
            # 当前写法等价于："tools": {}
            # 表示 server 声明支持 tools 能力
        }


def _register_default_tools(protocol_handler: ProtocolHandler) -> None:
    """Register all default MCP tools with the protocol handler.

    Args:
        protocol_handler: ProtocolHandler instance to register tools with.
        # 要把默认工具注册到哪个 ProtocolHandler 里
    """
    # Import and register query_knowledge_hub tool
    # 导入并注册 query_knowledge_hub 工具
    from src.mcp_server.tools.query_knowledge_hub import register_tool as register_query_tool
    # 从对应模块里导入 register_tool 函数，并改名成 register_query_tool

    register_query_tool(protocol_handler)
    # 把 query_knowledge_hub 注册进 protocol_handler
    
    # Import and register list_collections tool
    # 导入并注册 list_collections 工具
    from src.mcp_server.tools.list_collections import register_tool as register_list_tool
    # 导入 list_collections 模块里的注册函数

    register_list_tool(protocol_handler)
    # 把 list_collections 注册进 protocol_handler
    
    # Import and register get_document_summary tool
    # 导入并注册 get_document_summary 工具
    from src.mcp_server.tools.get_document_summary import register_tool as register_summary_tool
    # 导入 get_document_summary 模块里的注册函数

    register_summary_tool(protocol_handler)
    # 把 get_document_summary 注册进 protocol_handler


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: Optional[ProtocolHandler] = None,
    register_tools: bool = True,
) -> Server:
    """Create and configure an MCP server with the protocol handler.

    This factory function creates a low-level MCP Server instance and
    # 这个工厂函数会创建一个底层 MCP Server 实例，

    registers the necessary handlers for tools/list and tools/call.
    # 并给它注册 tools/list 和 tools/call 这两个核心处理器。

    Args:
        server_name: Name of the server.
        # server 名字

        server_version: Version string.
        # server 版本号

        protocol_handler: Optional pre-configured protocol handler.
            If None, a new one will be created.
        # 可选传入一个已经配置好的 ProtocolHandler
        # 如果不传，就在函数里新建一个

        register_tools: Whether to register default tools (default: True).
        # 是否自动注册默认工具，默认是 True

    Returns:
        Configured Server instance ready to run.
        # 返回一个已经配置好的 Server，可以直接运行
    """
    if protocol_handler is None:
        # 如果调用方没有传 protocol_handler，就自己创建一个新的
        protocol_handler = ProtocolHandler(
            server_name=server_name,
            # 把 server_name 填进去

            server_version=server_version,
            # 把 server_version 填进去
        )

    # Register default tools if requested
    # 如果设置了要注册默认工具
    if register_tools:
        _register_default_tools(protocol_handler)
        # 就调用上面的辅助函数，把默认工具都注册进去

    # Create low-level server
    # 创建真正的底层 MCP Server 实例
    server = Server(server_name)
    # 这里底层 Server 只需要名字

    # Register tools/list handler
    # 给 server 注册 tools/list 的处理函数
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """Handle tools/list request."""
        # 当客户端发 tools/list 请求时，会走到这里
        return protocol_handler.get_tool_schemas()
        # 调 protocol_handler，把当前所有工具 schema 返回给客户端

    # Register tools/call handler
    # 给 server 注册 tools/call 的处理函数
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """Handle tools/call request."""
        # 当客户端发 tools/call 请求时，会走到这里
        return await protocol_handler.execute_tool(name, arguments)
        # 再把真正的执行工作交给 protocol_handler

    # Store protocol handler on server for access
    # 把 protocol_handler 挂到 server 对象上，后面方便从 server 反查回来
    server._protocol_handler = protocol_handler  # type: ignore[attr-defined]
    # 这里是“动态加属性”
    # type: ignore[attr-defined] 的意思是告诉类型检查器：我知道这个属性不是 Server 原本定义的，别报错

    return server
    # 返回已经创建并配置完成的 server


def get_protocol_handler(server: Server) -> ProtocolHandler:
    """Get the protocol handler from a server instance.

    Args:
        server: Server instance created by create_mcp_server.
        # 传入一个通过 create_mcp_server 创建出来的 server

    Returns:
        The ProtocolHandler associated with the server.
        # 返回挂在这个 server 上的 protocol_handler

    Raises:
        AttributeError: If server was not created with create_mcp_server.
        # 如果这个 server 上根本没有 _protocol_handler 属性，就会报 AttributeError
    """
    return server._protocol_handler  # type: ignore[attr-defined]
    # 直接从 server 身上取出前面挂上的 protocol_handler