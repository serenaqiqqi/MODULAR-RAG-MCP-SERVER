#!/usr/bin/env python
# 这一行叫 shebang，意思是：把这个文件当成 Python 脚本来执行

"""Evaluation script for Modular RAG MCP Server.
# 这个文件是“评估脚本”，专门用来跑 RAG 评测

Runs batch evaluation against a golden test set and outputs a metrics report.
# 它会拿 golden test set（标准测试集）批量评估系统，然后输出评测报告

Usage:
    # Run with default settings (custom evaluator)
    python scripts/evaluate.py
    # 默认运行方式：直接跑评估脚本

    # Specify a custom golden test set
    python scripts/evaluate.py --test-set path/to/golden.json
    # 指定一个自定义测试集文件

    # Use a specific collection
    python scripts/evaluate.py --collection technical_docs
    # 指定评估时搜索哪个 collection

    # JSON output
    python scripts/evaluate.py --json
    # 指定输出 JSON 格式结果，而不是打印漂亮文本

Exit codes:
    0 - Success
    # 返回码 0：表示成功
    1 - Evaluation failure
    # 返回码 1：表示评估过程失败
    2 - Configuration error
    # 返回码 2：表示配置有问题
"""

from __future__ import annotations
# 让类型注解延迟解析，避免有些类型在运行时还没定义就报错

import argparse
# 导入命令行参数解析模块，用来处理 --test-set / --json 这些参数

import json
# 导入 json 模块，用来输出 JSON 格式结果

import sys
# 导入 sys 模块，用来处理退出码、平台信息、路径等

from pathlib import Path
# 导入 Path，方便处理文件路径

# Set UTF-8 encoding for Windows console
# 如果是在 Windows 终端里跑，强制把标准输出和标准错误设成 UTF-8
if sys.platform == "win32":
    # 只有平台是 Windows 才进这里
    import io
    # 导入 io 模块，用来重新包装 stdout / stderr
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    # 把标准输出改成 UTF-8，避免中文乱码
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    # 把标准错误也改成 UTF-8，避免报错信息乱码

# Add project root to path
# 把项目根目录加到 Python 搜索路径里，保证后面能 import src.xxx
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# __file__ 是当前文件路径
# resolve() 得到绝对路径
# parent.parent 表示回到上两级目录，也就是项目根目录
sys.path.insert(0, str(PROJECT_ROOT))
# 把项目根目录插到 sys.path 最前面，优先从项目里找模块


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    # 这个函数专门负责解析命令行参数
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation against a golden test set."
    )
    # 创建一个参数解析器，并写一段脚本用途说明

    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to golden test set JSON file (default: tests/fixtures/golden_test_set.json)",
    )
    # 添加 --test-set 参数
    # 作用：指定评估用的测试集 JSON 文件路径
    # 默认值：tests/fixtures/golden_test_set.json

    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to search within.",
    )
    # 添加 --collection 参数
    # 作用：指定检索时要查哪个 collection
    # 默认不传就是 None

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10).",
    )
    # 添加 --top-k 参数
    # 作用：每个 query 最多召回多少个 chunk
    # 类型是 int
    # 默认值是 10

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text.",
    )
    # 添加 --json 参数
    # 作用：如果传了这个参数，就输出 JSON 格式
    # action="store_true" 的意思是：只要命令里写了 --json，这个值就变成 True

    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip retrieval (evaluate with mock chunks for testing).",
    )
    # 添加 --no-search 参数
    # 作用：跳过真实检索，直接用 mock chunks 做测试
    # 一般用于测试评估流程本身，而不是测试检索能力

    return parser.parse_args()
    # 真正解析命令行参数，并返回 Namespace 对象


def main() -> int:
    """Main entry point."""
    # 主入口函数，整个脚本的主要流程都在这里
    args = parse_args()
    # 先解析命令行参数，拿到 args

    try:
        from src.core.settings import load_settings
        # 导入配置加载函数
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        # 导入评估器工厂
        from src.observability.evaluation.eval_runner import EvalRunner
        # 导入 EvalRunner，后面真正负责跑评估

        settings = load_settings()
        # 加载项目配置
    except Exception as exc:
        # 如果导入模块失败，或者加载配置失败，就进这里
        print(f"❌ Configuration error: {exc}", file=sys.stderr)
        # 把配置错误打印到标准错误
        return 2
        # 返回退出码 2，表示配置错误

    # Create evaluator from config
    # 根据配置创建评估器
    try:
        evaluator = EvaluatorFactory.create(settings)
        # 用工厂根据 settings 创建评估器
        evaluator_name = type(evaluator).__name__
        # 顺便拿到评估器类名，后面打印报告时会用到
    except Exception as exc:
        # 如果评估器创建失败，就进这里
        print(f"❌ Failed to create evaluator: {exc}", file=sys.stderr)
        # 打印错误信息
        return 2
        # 这里也当成配置问题处理，返回 2

    # Create HybridSearch (unless --no-search)
    # 创建 HybridSearch，除非用户明确传了 --no-search
    hybrid_search = None
    # 先默认 hybrid_search 为 None
    if not args.no_search:
        # 如果没有传 --no-search，说明这次评估要走真实检索
        try:
            from src.core.query_engine.query_processor import QueryProcessor
            # 导入 QueryProcessor
            from src.core.query_engine.hybrid_search import create_hybrid_search
            # 导入创建 HybridSearch 的工厂函数
            from src.core.query_engine.dense_retriever import create_dense_retriever
            # 导入创建 DenseRetriever 的工厂函数
            from src.core.query_engine.sparse_retriever import create_sparse_retriever
            # 导入创建 SparseRetriever 的工厂函数
            from src.ingestion.storage.bm25_indexer import BM25Indexer
            # 导入 BM25Indexer
            from src.libs.embedding.embedding_factory import EmbeddingFactory
            # 导入 Embedding 工厂
            from src.libs.vector_store.vector_store_factory import VectorStoreFactory
            # 导入向量库工厂

            collection = args.collection or "default"
            # 如果用户传了 --collection，就用用户的
            # 否则默认用 "default"

            vector_store = VectorStoreFactory.create(
                settings, collection_name=collection,
            )
            # 根据配置创建向量库实例，并指定当前 collection

            embedding_client = EmbeddingFactory.create(settings)
            # 根据配置创建 embedding 客户端

            dense_retriever = create_dense_retriever(
                settings=settings,
                embedding_client=embedding_client,
                vector_store=vector_store,
            )
            # 创建 DenseRetriever，把 embedding_client 和 vector_store 注进去

            bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
            # 创建 BM25Indexer
            # 索引目录是 data/db/bm25/collection名

            sparse_retriever = create_sparse_retriever(
                settings=settings,
                bm25_indexer=bm25_indexer,
                vector_store=vector_store,
            )
            # 创建 SparseRetriever，把 bm25_indexer 和 vector_store 注进去

            sparse_retriever.default_collection = collection
            # 再手动把 sparse_retriever 的默认 collection 设成当前 collection

            query_processor = QueryProcessor()
            # 创建 QueryProcessor

            hybrid_search = create_hybrid_search(
                settings=settings,
                query_processor=query_processor,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
            )
            # 创建完整的 HybridSearch，把 query_processor / dense / sparse 都组装进去

            print(f"✅ HybridSearch initialized for collection: {collection}")
            # 打印一条成功提示，说明 HybridSearch 初始化好了
        except Exception as exc:
            # 如果 HybridSearch 初始化失败，就进这里
            print(f"⚠️  Failed to initialize search (running without retrieval): {exc}")
            # 这里只是警告，不直接退出
            # 意思是：检索初始化失败，那就退化成“不带真实检索”的评估

    # Create and run EvalRunner
    # 创建并运行 EvalRunner
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
    )
    # 创建 EvalRunner，把配置、检索器、评估器都传进去
    # 如果 hybrid_search 是 None，EvalRunner 就会以“无真实检索”的模式运行

    try:
        print(f"\n🔍 Running evaluation with {evaluator_name}...")
        # 打印当前用哪个评估器在跑
        print(f"📄 Test set: {args.test_set}")
        # 打印测试集路径
        print(f"🔢 Top-K: {args.top_k}\n")
        # 打印 top_k

        report = runner.run(
            test_set_path=args.test_set,
            top_k=args.top_k,
            collection=args.collection,
        )
        # 真正执行评估
        # 输入：测试集路径、top_k、collection
        # 输出：report 报告对象
    except Exception as exc:
        # 如果评估过程中报错，就进这里
        print(f"❌ Evaluation failed: {exc}", file=sys.stderr)
        # 打印错误信息
        return 1
        # 返回退出码 1，表示评估失败

    # Output results
    # 下面开始输出评估结果
    if args.json:
        # 如果用户传了 --json
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        # 就把 report 转成 dict，再转成格式化 JSON 输出
        # ensure_ascii=False 的作用是：中文不要转义，直接正常显示
    else:
        _print_report(report)
        # 否则就用自定义的漂亮文本格式打印

    return 0
    # 全部成功执行完，返回退出码 0


def _print_report(report) -> None:
    """Print formatted evaluation report."""
    # 这个函数负责把评估报告以“好读一点”的文本格式打印出来
    print("=" * 60)
    # 打印分隔线
    print("  EVALUATION REPORT")
    # 打印标题
    print("=" * 60)
    # 再打印分隔线
    print(f"  Evaluator: {report.evaluator_name}")
    # 打印评估器名字
    print(f"  Test Set:  {report.test_set_path}")
    # 打印测试集路径
    print(f"  Queries:   {len(report.query_results)}")
    # 打印一共评了多少条 query
    print(f"  Time:      {report.total_elapsed_ms:.0f} ms")
    # 打印总耗时，四舍五入到整数毫秒
    print()
    # 打一个空行

    # Aggregate metrics
    # 下面打印整体聚合指标
    print("─" * 60)
    # 分隔线
    print("  AGGREGATE METRICS")
    # 小标题：整体指标
    print("─" * 60)
    # 分隔线
    if report.aggregate_metrics:
        # 如果有整体指标
        for metric, value in sorted(report.aggregate_metrics.items()):
            # 按指标名字排序后逐个打印
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            # 把 0~1 的分数粗略画成一个文本进度条
            # 例如 0.8 就大概是 16 个 █ 加 4 个 ░
            print(f"  {metric:<25s} {bar} {value:.4f}")
            # 打印：指标名 + 文本条 + 分数
    else:
        print("  (no metrics computed)")
        # 如果一个聚合指标都没有，就打印提示
    print()
    # 空行

    # Per-query details
    # 下面打印每条 query 的详细结果
    print("─" * 60)
    # 分隔线
    print("  PER-QUERY RESULTS")
    # 小标题：逐条 query 结果
    print("─" * 60)
    # 分隔线
    for i, qr in enumerate(report.query_results, 1):
        # 遍历每一条 query 的结果，i 从 1 开始编号
        print(f"\n  [{i}] {qr.query}")
        # 打印第几条 query 以及 query 内容
        print(f"      Retrieved: {len(qr.retrieved_chunk_ids)} chunks")
        # 打印这条 query 一共召回了多少个 chunk
        if qr.metrics:
            # 如果这条 query 有指标
            for metric, value in sorted(qr.metrics.items()):
                # 把每个指标按名字排序后打印
                print(f"      {metric}: {value:.4f}")
                # 打印指标名和值
        else:
            print("      (no metrics)")
            # 如果没有指标，就打印提示
        print(f"      Time: {qr.elapsed_ms:.0f} ms")
        # 打印这一条 query 的耗时

    print()
    # 空行
    print("=" * 60)
    # 最后打印收尾分隔线


if __name__ == "__main__":
    # 只有直接运行这个脚本时，才会进这里
    sys.exit(main())
    # 调 main()，并把返回值作为脚本退出码