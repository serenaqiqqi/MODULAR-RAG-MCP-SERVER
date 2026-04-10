"""MCP Server entry point using official MCP SDK.
# 这个文件的作用：作为 MCP Server 的启动入口。

This module implements the MCP server using the official Python MCP SDK
# 这句说明：这个模块是基于官方 Python MCP SDK 来实现 MCP server 的。

with stdio transport. It ensures stdout only contains protocol messages
# 它使用的是 stdio 传输方式。
# 并且特别强调：stdout 里只能放协议消息，

while all logs go to stderr.
# 所有日志都必须打到 stderr，不能污染 stdout。
"""

from __future__ import annotations
# 让类型注解支持延迟解析，避免某些前向引用问题。

import asyncio
# 导入 asyncio，用来跑异步 server。

import sys
# 导入 sys，后面用 sys.exit(main()) 退出进程。

from typing import TYPE_CHECKING
# TYPE_CHECKING 是一个静态类型检查时为 True、运行时为 False 的标记。
# 常用于只给类型检查器看，不在运行时真正 import 一些东西。

from src.mcp_server.protocol_handler import create_mcp_server
# 导入前面 protocol_handler.py 里封装好的 create_mcp_server 工厂函数。
# 它负责创建并配置 MCP Server。

from src.observability.logger import get_logger
# 导入项目自己的日志工厂，用来拿 logger。

if TYPE_CHECKING:
    pass
    # 这里目前没有放任何“仅供类型检查”的导入内容。
    # 相当于预留位置，现在实际什么都没做。


SERVER_NAME = "modular-rag-mcp-server"
# 定义 MCP server 的名字。

SERVER_VERSION = "0.1.0"
# 定义 MCP server 的版本号。


def _redirect_all_loggers_to_stderr() -> None:
    """Redirect all root logger handlers to stderr.

    MCP stdio transport reserves stdout for JSON-RPC messages.
    # MCP 的 stdio 传输模式下，stdout 是专门留给 JSON-RPC 协议消息的。

    Any logging to stdout corrupts the protocol stream.
    # 如果日志也打到 stdout，就会把协议流弄脏，客户端就没法正常解析了。
    """
    import logging as _logging
    # 在函数内部再导入 logging，并起别名 _logging。
    # 这里这么写主要是把这个依赖局部化。

    root = _logging.getLogger()
    # 拿到 root logger，也就是日志系统最顶层那个 logger。

    stderr_handler = _logging.StreamHandler(sys.stderr)
    # 创建一个新的日志处理器，让它把日志输出到 stderr。

    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    # 给这个 handler 设置日志格式：
    # 时间 + 日志级别 + logger 名字 + 日志消息

    # Replace any existing stream handlers that might point to stdout
    # 下面开始把现有可能指向 stdout 的 StreamHandler 都移除掉。
    for handler in root.handlers[:]:
        # 遍历 root logger 当前所有 handler
        # 这里用 root.handlers[:] 是为了拷贝一份列表，避免边遍历边修改原列表出问题

        if isinstance(handler, _logging.StreamHandler) and not isinstance(
            handler, _logging.FileHandler
        ):
            # 如果这个 handler 是一个“流处理器”
            # 并且它不是“文件处理器”
            # 那它大概率就是 stdout / stderr 这种输出流 handler

            root.removeHandler(handler)
            # 就把它从 root logger 上移除掉

    root.addHandler(stderr_handler)
    # 把刚才新建的 stderr handler 加到 root logger 上
    # 这样后续日志就统一走 stderr 了


def _preload_heavy_imports() -> None:
    """Eagerly import heavy third-party modules in the **main thread**.

    MCP SDK uses anyio + background threads for stdin/stdout I/O.
    # MCP SDK 底层会用 anyio 和后台线程来处理 stdin / stdout 的 I/O。

    When a tool handler runs ``asyncio.to_thread(fn)``, *fn* executes in
    # 当某个 tool handler 后面通过 asyncio.to_thread(fn) 把任务丢到线程池里时，

    a new worker thread.  If it tries to ``import chromadb`` (which
    # 这个 fn 会在新的 worker 线程里执行。
    # 如果它在那个线程里临时 import chromadb，

    transitively pulls in onnxruntime, numpy, sqlite3 C extensions …),
    # 而 chromadb 又会连带 import 一堆很重的库，
    # 比如 onnxruntime、numpy、sqlite3 的 C 扩展等，

    that import can deadlock with the stdin-reader thread because both
    # 那这个 import 过程就可能和读取 stdin 的线程抢 import lock，

    compete for Python's global *import lock*.
    # Python 的 import 有全局锁，多个线程同时 import 重模块时可能卡死。

    Pre-importing here – before anyio spins up its I/O threads – avoids
    # 所以这里选择：在主线程里、在 anyio 启动 I/O 线程之前，
    # 先把重模块 import 好。

    the deadlock entirely: subsequent ``import`` statements in worker
    # 这样可以彻底规避死锁问题。
    # 后面 worker 线程再 import 时，

    threads simply hit ``sys.modules`` and return immediately.
    # 只会直接命中 sys.modules 缓存，很快返回，不会再真正触发重 import。
    """
    # chromadb is the heaviest culprit (onnxruntime, numpy, …)
    # 先预加载最容易引发问题的重依赖 chromadb
    try:
        import chromadb  # noqa: F401
        # 这里 import chromadb，本身不直接使用，只是提前加载
        # noqa: F401 的意思是告诉 lint 工具：虽然没用到，也别报“未使用导入”警告

        import chromadb.config  # noqa: F401
        # 顺便把它的 config 子模块也提前加载
    except ImportError:
        pass  # optional at install time
        # 如果环境里没装 chromadb，就忽略
        # 因为这个依赖可能是可选安装的

    # Internal modules that tools lazy-import inside asyncio.to_thread
    # 下面预加载一些项目内部模块
    # 因为后面工具执行时，可能会在 asyncio.to_thread 里懒加载它们
    try:
        import src.core.query_engine.query_processor  # noqa: F401
        # 预加载 query_processor

        import src.core.query_engine.hybrid_search  # noqa: F401
        # 预加载 hybrid_search

        import src.core.query_engine.dense_retriever  # noqa: F401
        # 预加载 dense_retriever

        import src.core.query_engine.sparse_retriever  # noqa: F401
        # 预加载 sparse_retriever

        import src.core.query_engine.reranker  # noqa: F401
        # 预加载 reranker

        import src.ingestion.storage.bm25_indexer  # noqa: F401
        # 预加载 bm25_indexer

        import src.libs.embedding.embedding_factory  # noqa: F401
        # 预加载 embedding_factory

        import src.libs.vector_store.vector_store_factory  # noqa: F401
        # 预加载 vector_store_factory
    except ImportError:
        pass
        # 如果这些内部模块里某些 import 失败，也先忽略
        # 这样不会因为可选模块缺失直接把整个 server 启动打断


async def run_stdio_server_async() -> int:
    """Run MCP server over stdio asynchronously.

    Returns:
        Exit code.
        # 返回进程退出码
    """
    # Import here to avoid import errors if mcp not installed
    # 这里把 mcp.server.stdio 放到函数内部再 import
    # 这样做是为了避免：只是 import 这个文件时，如果环境没装 mcp 就立刻报错
    import mcp.server.stdio

    # Ensure ALL logging goes to stderr (stdout is reserved for JSON-RPC)
    # 先确保所有日志都只打到 stderr
    _redirect_all_loggers_to_stderr()

    # Pre-load heavy deps in main thread to prevent import-lock deadlocks
    # when tool handlers later call asyncio.to_thread().
    # 再在主线程里预加载重模块，避免后面 tool handler 丢到线程池时 import 死锁
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    # 创建一个 logger，日志级别设为 INFO

    logger.info("Starting MCP server (stdio transport) with official SDK.")
    # 记一条启动日志

    # Create server with protocol handler
    # 创建并配置 MCP Server
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    # 调用前面封装好的工厂函数：
    # 用固定的 server 名称和版本号，创建一个带 protocol_handler 的 MCP server

    # Run with stdio transport
    # 下面开始用 stdio transport 跑这个 server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        # 打开一个 stdio server 上下文，拿到：
        # read_stream：从 stdin 读客户端请求
        # write_stream：往 stdout 写协议响应

        await server.run(
            read_stream,
            # 读请求的流

            write_stream,
            # 写响应的流

            server.create_initialization_options(),
            # 创建初始化选项，告诉客户端这个 server 的初始化能力配置
        )

    logger.info("MCP server shutting down.")
    # 当 stdio server 退出后，记一条关闭日志

    return 0
    # 正常退出，返回 0


def run_stdio_server() -> int:
    """Run MCP server over stdio (synchronous wrapper).

    Returns:
        Exit code.
        # 返回退出码
    """
    return asyncio.run(run_stdio_server_async())
    # 这是一个同步包装器
    # 它用 asyncio.run(...) 去真正执行上面的异步启动函数


def main() -> int:
    """Entry point for stdio MCP server."""
    return run_stdio_server()
    # main 函数本身很简单
    # 直接调用 run_stdio_server，再把退出码返回出去


if __name__ == "__main__":
    # 如果这个文件是被直接运行，而不是被 import
    sys.exit(main())
    # 就执行 main()
    # 并把 main() 返回的退出码交给操作系统