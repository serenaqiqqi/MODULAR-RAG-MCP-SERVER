#!/usr/bin/env python  # 使用当前环境的 python 解释器执行此脚本
#是一个批量/单文件入库 CLI 编排器，核心职责是：解析参数 -> 找文件 -> 调用 IngestionPipeline -> 汇总结果并返回退出码。
"""Ingestion script for the Modular RAG MCP Server.  # 文件级说明：该脚本用于数据摄取

This script provides a command-line interface for ingesting documents into  # 描述脚本提供 CLI 接口
the knowledge hub. It supports processing single files or entire directories.  # 说明支持单文件与目录处理

Usage:  # 用法示例开始
    # Process a single PDF file  # 示例1：处理单个 PDF
    python scripts/ingest.py --path documents/report.pdf --collection contracts  # 示例1命令

    # Process all PDFs in a directory  # 示例2：处理目录内全部 PDF
    python scripts/ingest.py --path documents/ --collection technical_docs  # 示例2命令

    # Force re-processing (ignore previous ingestion)  # 示例3：强制重处理
    python scripts/ingest.py --path documents/report.pdf --collection contracts --force  # 示例3命令

    # Use custom configuration file  # 示例4：使用自定义配置文件
    python scripts/ingest.py --path documents/ --collection contracts --config custom_settings.yaml  # 示例4命令

Exit codes:  # 退出码说明开始
    0 - Success (all files processed)  # 0 表示全部成功
    1 - Partial failure (some files failed)  # 1 表示部分失败
    2 - Complete failure (all files failed or configuration error)  # 2 表示完全失败或配置错误
"""  # 文件说明结束

import argparse  # 命令行参数解析
import os  # 操作系统相关能力（当前脚本中暂未直接使用）
import sys  # Python 运行时与解释器交互
from pathlib import Path  # 跨平台路径对象
from typing import List, Optional  # 类型标注（Optional 当前未直接使用）

# Ensure project root is on sys.path  # 保证仓库根目录可被 Python 导入机制找到
_SCRIPT_DIR = Path(__file__).resolve().parent  # 当前脚本所在目录（绝对路径）
_REPO_ROOT = _SCRIPT_DIR.parent  # 仓库根目录（脚本目录的上级）
sys.path.insert(0, str(_REPO_ROOT))  # 将仓库根目录插入导入搜索路径最前面

# Set UTF-8 encoding for Windows console  # 在 Windows 控制台下显式设置 UTF-8 输出
if sys.platform == "win32":  # 仅在 Windows 平台启用该逻辑
    import io  # 文本流包装能力
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")  # 标准输出设为 UTF-8
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")  # 标准错误设为 UTF-8

# Ensure project root is in path for imports  # 再次确保根目录在 sys.path（与上面逻辑重复但无害）
project_root = Path(__file__).parent.parent  # 通过相对路径计算项目根目录
sys.path.insert(0, str(project_root))  # 再插入一次用于兜底导入

from src.core.settings import load_settings, Settings  # 加载配置与配置类型
from src.core.trace import TraceContext, TraceCollector  # 链路追踪上下文与收集器
from src.ingestion.pipeline import IngestionPipeline, PipelineResult  # 摄取流水线与结果类型
from src.observability.logger import get_logger  # 项目统一日志获取方法

logger = get_logger(__name__)  # 以当前模块名初始化 logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.  # 解析命令行参数

    Returns:  # 返回值说明
        Parsed arguments namespace  # argparse 的命名空间对象
    """
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description="Ingest documents into the Modular RAG knowledge hub.",  # CLI 描述
        formatter_class=argparse.RawDescriptionHelpFormatter,  # 保留 epilog 原格式
        epilog=__doc__,  # 将模块文档作为帮助尾注
    )

    parser.add_argument(  # 注册路径参数
        "--path",
        "-p",
        required=True,  # 路径参数必填
        help="Path to file or directory to ingest. "  # 帮助文本（第一段）
        "If directory, processes all PDF files recursively.",  # 帮助文本（第二段）
    )

    parser.add_argument(  # 注册集合名称参数
        "--collection",
        "-c",
        default="default",  # 默认集合名为 default
        help="Collection name for organizing documents (default: 'default')",  # 帮助文本
    )

    parser.add_argument(  # 注册强制处理开关
        "--force",
        "-f",
        action="store_true",  # 出现该参数则值为 True
        help="Force re-processing even if file was previously ingested",  # 帮助文本
    )

    parser.add_argument(  # 注册配置文件路径参数
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),  # 默认配置路径
        help="Path to configuration file (default: config/settings.yaml)",  # 帮助文本
    )

    parser.add_argument(  # 注册详细输出开关
        "--verbose",
        "-v",
        action="store_true",  # 出现该参数则开启详细日志
        help="Enable verbose output",  # 帮助文本
    )

    parser.add_argument(  # 注册 dry-run 开关
        "--dry-run",
        action="store_true",  # 出现该参数仅演练不执行
        help="List files that would be processed without actually processing",  # 帮助文本
    )

    return parser.parse_args()  # 返回解析后的参数对象


def discover_files(path: str, extensions: List[str] = None) -> List[Path]:
    """Discover files to process from path.  # 从输入路径发现待处理文件

    Args:  # 参数说明开始
        path: File or directory path  # 文件或目录路径
        extensions: List of file extensions to include (default: ['.pdf'])  # 允许扩展名列表

    Returns:  # 返回值说明
        List of file paths to process  # 返回待处理文件路径列表
    """
    if extensions is None:  # 如果未传扩展名列表
        extensions = [".pdf"]  # 默认只处理 PDF

    path = Path(path)  # 将字符串路径转为 Path 对象

    if not path.exists():  # 如果路径不存在
        raise FileNotFoundError(f"Path does not exist: {path}")  # 抛出文件不存在异常

    if path.is_file():  # 如果是单个文件
        if path.suffix.lower() in extensions:  # 扩展名在允许列表中
            return [path]  # 直接返回单文件列表
        else:  # 扩展名不在允许列表中
            raise ValueError(f"Unsupported file type: {path.suffix}. Supported: {extensions}")  # 抛出类型错误

    # Directory: recursively find all matching files  # 目录模式：递归搜索匹配文件
    files = []  # 初始化结果列表
    for ext in extensions:  # 遍历每个允许扩展名
        files.extend(path.rglob(f"*{ext}"))  # 匹配小写扩展名文件
        files.extend(path.rglob(f"*{ext.upper()}"))  # 匹配大写扩展名文件

    # Remove duplicates and sort  # 去重并排序，保证结果稳定
    files = sorted(set(files))  # 先 set 去重，再 sorted 排序

    return files  # 返回发现的文件列表


def print_summary(results: List[PipelineResult], verbose: bool = False) -> None:
    """Print processing summary.  # 打印处理汇总信息

    Args:  # 参数说明开始
        results: List of pipeline results  # 每个文件对应的流水线执行结果
        verbose: Whether to print detailed information  # 是否输出详细列表
    """
    total = len(results)  # 总文件数
    successful = sum(1 for r in results if r.success)  # 成功文件数
    failed = total - successful  # 失败文件数

    total_chunks = sum(r.chunk_count for r in results if r.success)  # 成功文件的 chunk 总数
    total_images = sum(r.image_count for r in results if r.success)  # 成功文件的图片总数

    print("\n" + "=" * 60)  # 打印分割线（上）
    print("INGESTION SUMMARY")  # 打印标题
    print("=" * 60)  # 打印分割线（下）
    print(f"Total files processed: {total}")  # 输出总文件数
    print(f"  [OK] Successful: {successful}")  # 输出成功数
    print(f"  [FAIL] Failed: {failed}")  # 输出失败数
    print(f"\nTotal chunks generated: {total_chunks}")  # 输出 chunk 总数
    print(f"Total images processed: {total_images}")  # 输出图片总数

    if verbose and failed > 0:  # 详细模式下如果有失败项
        print("\nFailed files:")  # 输出失败列表标题
        for r in results:  # 遍历全部结果
            if not r.success:  # 只打印失败项
                print(f"  [FAIL] {r.file_path}: {r.error}")  # 输出失败文件及错误

    if verbose and successful > 0:  # 详细模式下如果有成功项
        print("\nSuccessful files:")  # 输出成功列表标题
        for r in results:  # 遍历全部结果
            if r.success:  # 只处理成功项
                skipped = r.stages.get("integrity", {}).get("skipped", False)  # 判断是否被完整性阶段跳过
                status = "[SKIP] skipped" if skipped else f"[OK] {r.chunk_count} chunks"  # 生成展示状态
                print(f"  {status}: {r.file_path}")  # 输出成功文件状态

    print("=" * 60)  # 打印总结尾部分割线


def main() -> int:
    """Main entry point for the ingestion script.  # 脚本主入口

    Returns:  # 返回值说明
        Exit code (0=success, 1=partial failure, 2=complete failure)  # 进程退出码定义
    """
    args = parse_args()  # 解析命令行参数

    # Setup logging level  # 根据参数配置日志级别
    if args.verbose:  # 如果开启 verbose
        import logging  # 按需导入 logging
        logging.getLogger().setLevel(logging.DEBUG)  # 将根日志器级别设为 DEBUG

    print("[*] Modular RAG Ingestion Script")  # 打印脚本启动标识
    print("=" * 60)  # 打印启动分割线

    # Load configuration  # 加载配置文件
    try:  # 捕获配置加载中的异常
        config_path = Path(args.config)  # 将配置路径参数转为 Path
        if not config_path.exists():  # 配置文件不存在
            print(f"[FAIL] Configuration file not found: {config_path}")  # 输出错误信息
            return 2  # 以配置错误退出

        settings = load_settings(str(config_path))  # 读取并解析配置
        print(f"[OK] Configuration loaded from: {config_path}")  # 输出配置加载成功
    except Exception as e:  # 任意配置相关异常
        print(f"[FAIL] Failed to load configuration: {e}")  # 输出错误详情
        return 2  # 以配置错误退出

    # Discover files  # 发现待处理文件
    try:  # 捕获文件发现阶段异常
        files = discover_files(args.path)  # 根据 path 参数收集文件
        print(f"[INFO] Found {len(files)} file(s) to process")  # 输出发现文件数量

        if len(files) == 0:  # 没有可处理文件
            print("[WARN] No files found to process")  # 输出警告
            return 0  # 无文件不算失败，正常退出

        for f in files:  # 逐个打印待处理文件路径
            print(f"   - {f}")  # 文件列表展示
    except FileNotFoundError as e:  # 输入路径不存在
        print(f"[FAIL] {e}")  # 输出失败原因
        return 2  # 作为严重输入错误退出
    except ValueError as e:  # 输入文件类型不支持
        print(f"[FAIL] {e}")  # 输出失败原因
        return 2  # 作为输入错误退出

    # Dry run mode  # 演练模式：只展示不执行
    if args.dry_run:  # 如果开启 dry-run
        print("\n[INFO] Dry run mode - no files were processed")  # 提示未执行实际处理
        return 0  # 正常退出

    # Initialize pipeline  # 初始化摄取流水线
    print("\n[INFO] Initializing pipeline...")  # 输出初始化提示
    print(f"   Collection: {args.collection}")  # 输出集合名
    print(f"   Force: {args.force}")  # 输出是否强制重处理

    try:  # 捕获流水线初始化异常
        pipeline = IngestionPipeline(  # 创建流水线实例
            settings=settings,  # 注入配置对象
            collection=args.collection,  # 指定目标集合
            force=args.force,  # 指定是否强制模式
        )
    except Exception as e:  # 初始化失败
        print(f"[FAIL] Failed to initialize pipeline: {e}")  # 输出初始化错误
        logger.exception("Pipeline initialization failed")  # 记录异常堆栈
        return 2  # 以完全失败退出

    # Process files  # 开始批量处理文件
    print("\n[INFO] Processing files...")  # 输出处理开始提示
    results: List[PipelineResult] = []  # 存放每个文件的执行结果

    collector = TraceCollector()  # 创建 trace 收集器

    for i, file_path in enumerate(files, 1):  # 带序号遍历所有文件
        print(f"\n[{i}/{len(files)}] Processing: {file_path}")  # 输出当前进度

        try:  # 捕获单文件处理异常，保证批处理不中断
            trace = TraceContext(trace_type="ingestion")  # 为当前文件创建摄取链路上下文
            trace.metadata["source_path"] = str(file_path)  # 记录源文件路径到 trace 元数据
            result = pipeline.run(str(file_path), trace=trace)  # 执行流水线处理
            collector.collect(trace)  # 收集当前 trace
            results.append(result)  # 保存处理结果

            if result.success:  # 如果处理成功
                skipped = result.stages.get("integrity", {}).get("skipped", False)  # 判断是否因去重/完整性检查被跳过
                if skipped:  # 已处理过且被跳过
                    print("   [SKIP] Skipped (already processed)")  # 输出跳过提示
                else:  # 真正执行并成功
                    print(f"   [OK] Success: {result.chunk_count} chunks, {result.image_count} images")  # 输出产物统计
            else:  # 流水线返回失败
                print(f"   [FAIL] Failed: {result.error}")  # 输出失败错误

        except Exception as e:  # 捕获运行期未预期异常
            logger.exception(f"Unexpected error processing {file_path}")  # 记录异常堆栈
            results.append(  # 手工构造失败结果，保持汇总逻辑一致
                PipelineResult(
                    success=False,  # 标记为失败
                    file_path=str(file_path),  # 记录出错文件
                    error=str(e),  # 记录错误文本
                )
            )
            print(f"   [FAIL] Error: {e}")  # 向终端输出异常简述

    # Print summary  # 打印最终汇总
    print_summary(results, args.verbose)  # 根据 verbose 决定摘要粒度

    # Determine exit code  # 按结果计算退出码
    successful = sum(1 for r in results if r.success)  # 统计成功项
    if successful == len(results):  # 全部成功
        return 0  # 返回成功码
    elif successful > 0:  # 部分成功
        return 1  # 返回部分失败码
    else:  # 全部失败
        return 2  # 返回完全失败码


if __name__ == "__main__":  # 当脚本被直接执行时
    sys.exit(main())  # 运行 main 并将返回值作为进程退出码
