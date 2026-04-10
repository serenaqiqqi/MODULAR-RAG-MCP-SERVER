"""Sparse Encoder for generating BM25 term statistics from text chunks.
# 这个文件的作用：从文本 chunk 里提取 BM25 需要的词项统计信息。

This module implements the Sparse Encoder component of the Ingestion Pipeline,
# 这句说明：它是 Ingestion Pipeline 里负责 sparse 编码的组件。

responsible for extracting term statistics needed for BM25 indexing.
# 它的职责是提取 BM25 建索引时需要的统计数据。

Design Principles:
# 下面是设计原则说明：

- Stateless Processing: No internal state between encode() calls
# 无状态：两次 encode() 调用之间不保留内部处理状态

- Observable: Accepts TraceContext for future observability integration
# 可观测：支持传 trace，方便以后接可观测系统

- Deterministic: Same inputs produce same term statistics
# 确定性：相同输入一定得到相同统计结果

- Clear Contracts: Well-defined output structure for downstream BM25Indexer
# 契约清晰：输出格式固定，方便下游 BM25Indexer 使用
"""

from typing import List, Dict, Optional, Any
# 导入类型标注工具：
# List：列表
# Dict：字典
# Optional：可为空
# Any：任意类型

from collections import Counter
# 导入 Counter，用来统计每个词出现了多少次。

import re
# 导入 re，后面用正则判断和过滤标点、空白等内容。

import jieba
# 导入 jieba，用来做中文分词，也能对中英混合文本做切分。

from src.core.types import Chunk
# 导入 Chunk 类型，表示输入是一组文本块对象。


class SparseEncoder:
    """Encodes text chunks into BM25 term statistics.
    # 这个类的职责：把文本 chunk 转成 BM25 所需的统计信息。
    
    This encoder prepares term-level statistics needed for BM25 indexing.
    # 它只负责准备“词项级别统计”，

    The actual index construction is handled by BM25Indexer (C12).
    # 真正的 BM25 索引构建，不在这里做，而是在 BM25Indexer 里做。
    
    Output Structure:
        For each chunk, produces:
        {
            "chunk_id": str,
            # 当前 chunk 的 id

            "term_frequencies": Dict[str, int],  # term -> count in this chunk
            # 当前 chunk 里每个 term 出现了多少次

            "doc_length": int,                    # number of terms in chunk
            # 当前 chunk 分词后的总词数

            "unique_terms": int                   # vocabulary size in chunk
            # 当前 chunk 中不重复词的数量
        }
    
    Design:
    - Tokenization: Simple whitespace + lowercasing (can be enhanced later)
    # 这里文档里写的是简单 tokenization，但当前真实实现其实用了 jieba + 清洗 + lowercase

    - Stop Words: None by default (can add in future iterations)
    # 默认不做停用词过滤，后面可以扩展

    - Deterministic: Same chunk text always produces same statistics
    # 同样 chunk 文本总能得到同样统计
    
    Example:
        >>> from src.core.types import Chunk
        >>> encoder = SparseEncoder()
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world hello", metadata={})]
        >>> stats = encoder.encode(chunks)
        >>> stats[0]["term_frequencies"]["hello"]  # 2
        >>> stats[0]["doc_length"]  # 3
        # 这个例子说明：同一个词重复出现会被统计出来，文档总词数也会被记录。
    """
    
    def __init__(
        self,
        min_term_length: int = 2,
        lowercase: bool = True,
    ):
        """Initialize SparseEncoder.
        # 初始化 SparseEncoder。
        
        Args:
            min_term_length: Minimum character length for a term (default: 2)
            # min_term_length：term 最少要有多少个字符才保留，默认 2

            lowercase: Whether to convert terms to lowercase (default: True)
            # lowercase：是否把 term 转成小写，默认 True
        
        Raises:
            ValueError: If min_term_length < 1
            # 如果最小 term 长度小于 1，直接报错
        """
        if min_term_length < 1:
            # 先检查 min_term_length 合不合法
            raise ValueError(f"min_term_length must be >= 1, got {min_term_length}")
            # 如果传入值非法，抛出 ValueError
        
        self.min_term_length = min_term_length
        # 把最小 term 长度保存到实例上，后面 _tokenize 时会用

        self.lowercase = lowercase
        # 把是否转小写这个配置保存到实例上
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Encode chunks into BM25 term statistics.
        # 主方法：把一组 chunks 编码成 BM25 所需统计结果。
        
        For each chunk, extracts:
        - Term frequencies (term -> count)
        # 每个词出现几次

        - Document length (total terms)
        # 这个 chunk 一共有多少个词

        - Unique terms count
        # 这个 chunk 一共有多少个不同词
        
        Args:
            chunks: List of Chunk objects to encode
            # 输入：要处理的 chunk 列表

            trace: Optional TraceContext for observability (reserved for Stage F)
            # 可选 trace，目前这个函数里其实没有真正使用 trace
        
        Returns:
            List of statistics dictionaries (one per chunk, in same order).
            # 返回一个列表，每个 chunk 对应一个统计字典，顺序和输入一致

            Each dict contains: chunk_id, term_frequencies, doc_length, unique_terms
            # 每个字典都包含这 4 个字段
        
        Raises:
            ValueError: If chunks list is empty
            # 如果 chunks 是空列表，报错

            ValueError: If any chunk has empty text
            # 如果某个 chunk 的 text 为空或全是空白，也报错
        
        Example:
            >>> chunks = [
            ...     Chunk(id="1", text="machine learning", metadata={}),
            ...     Chunk(id="2", text="deep learning networks", metadata={})
            ... ]
            >>> stats = encoder.encode(chunks)
            >>> len(stats) == len(chunks)  # True
            >>> stats[0]["term_frequencies"]["machine"]  # 1
            >>> stats[1]["doc_length"]  # 3
        """
        if not chunks:
            # 如果 chunks 列表为空，直接拒绝处理
            raise ValueError("Cannot encode empty chunks list")
            # 抛出异常
        
        results = []
        # 准备一个空列表，用来存每个 chunk 编码后的统计结果
        
        for i, chunk in enumerate(chunks):
            # 逐个遍历 chunks
            # i 是下标，chunk 是当前块对象

            # Validate chunk text
            # 先校验当前 chunk 的 text 是否有效
            if not chunk.text or not chunk.text.strip():
                # 如果 text 是空字符串，或者去掉空白后什么都没有，就视为非法
                raise ValueError(
                    f"Chunk at index {i} (id={chunk.id}) has empty or whitespace-only text"
                )
                # 抛出异常，并把下标和 chunk.id 写进报错信息，方便定位
            
            # Tokenize and count terms
            # 对当前 chunk 文本做分词，并统计词频
            terms = self._tokenize(chunk.text)
            # 调用内部方法 _tokenize，把文本切成 term 列表

            term_frequencies = Counter(terms)
            # 用 Counter 统计每个 term 出现次数
            # 例如 ["hello", "world", "hello"] -> {"hello": 2, "world": 1}
            
            # Build statistics dict
            # 构建当前 chunk 的统计字典
            stat_dict = {
                "chunk_id": chunk.id,
                # 记录当前 chunk 的 id

                "term_frequencies": dict(term_frequencies),  # Convert Counter to dict
                # 把 Counter 转成普通 dict，便于后续序列化和下游处理

                "doc_length": len(terms),
                # 当前 chunk 分词后的总 term 数

                "unique_terms": len(term_frequencies),
                # 不重复 term 的数量，也就是词表大小
            }
            
            results.append(stat_dict)
            # 把当前 chunk 的统计结果加入 results
        
        return results
        # 返回所有 chunk 的统计结果列表
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms.
        # 内部辅助方法：把一段文本切成 terms。
        
        Uses jieba for Chinese text segmentation and regex for English.
        # 文档说明里说：中文靠 jieba，英文靠 regex。
        # 但当前代码真实逻辑是：先用 jieba 统一切，再用 regex 过滤脏 token。

        This ensures consistent tokenization with the query-side
        # 这样做的目的是：和 query 侧的分词逻辑尽量保持一致，

        (QueryProcessor), which is required for BM25 matching.
        # 因为 BM25 两边分词不一致的话，匹配就会出问题。
        
        Args:
            text: Input text to tokenize
            # 输入：待分词文本
        
        Returns:
            List of valid terms
            # 输出：过滤后的有效词列表
        """
        tokens: List[str] = []
        # 先准备一个空列表，用来存清洗后的 token

        # Use jieba to segment the text (handles both Chinese and English)
        # 先用 jieba 对整段文本做切分，能处理中英文混合
        raw_tokens = jieba.lcut(text)
        # lcut 会直接返回一个 list
        
        # Clean tokens: keep only alphanumeric and Chinese characters
        # 下面开始清洗 token，只保留有效内容
        for token in raw_tokens:
            # 遍历 jieba 切出来的每个原始 token
            token = token.strip()
            # 去掉 token 两端空白，比如空格、换行
            
            if not token:
                # 如果去完空白后 token 为空，就直接跳过
                continue

            # Skip pure punctuation / whitespace
            # 跳过纯标点或纯空白 token
            if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
                # 如果整个 token 都只由空白或非单词字符组成，就说明它没有实际词义
                continue

            tokens.append(token)
            # 合法 token 加入 tokens 列表
        
        # Apply lowercase if configured
        # 如果配置要求转小写，就统一转成小写
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
            # 例如 Hello -> hello
        
        # Filter by minimum length
        # 根据最小长度配置，过滤太短的 term
        terms = [t for t in tokens if len(t) >= self.min_term_length]
        # 例如 min_term_length=2 时，长度小于 2 的 token 会被过滤掉
        
        return terms
        # 返回最终有效的 term 列表
    
    def get_corpus_stats(
        self,
        encoded_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate corpus-level statistics from encoded chunks.
        # 工具方法：根据 encode() 的结果，进一步算整个语料级别的统计信息。
        
        Utility method for BM25Indexer to compute:
        # 这个方法主要给 BM25Indexer 使用，用来算：

        - Average document length
        # 平均文档长度

        - Document frequency (how many docs contain each term)
        # 每个词出现于多少个文档里，也就是 DF

        - Total number of documents
        # 总文档数
        
        Args:
            encoded_chunks: List of statistics dicts from encode()
            # 输入：encode() 返回的统计结果列表
        
        Returns:
            Dictionary with corpus-level statistics:
            # 返回一个语料级统计字典
            {
                "num_docs": int,
                # 文档总数

                "avg_doc_length": float,
                # 平均文档长度

                "document_frequency": Dict[str, int]  # term -> # docs containing it
                # 每个 term 出现于多少个文档
            }
        """
        if not encoded_chunks:
            # 如果传进来的 encoded_chunks 为空
            return {
                "num_docs": 0,
                # 文档数为 0

                "avg_doc_length": 0.0,
                # 平均长度为 0

                "document_frequency": {}
                # DF 为空字典
            }
        
        num_docs = len(encoded_chunks)
        # 总文档数 = encoded_chunks 的长度

        total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)
        # 把所有 chunk 的 doc_length 加起来，得到总长度

        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        # 平均文档长度 = 总长度 / 文档数
        # 这里虽然前面已经保证非空了，但还是做了保护性判断
        
        # Calculate document frequency (DF) for each term
        # 下面开始计算每个 term 的 document frequency（DF）
        doc_freq: Dict[str, int] = {}
        # 准备一个空字典，term -> 出现于多少个文档

        for chunk_stats in encoded_chunks:
            # 遍历每个 chunk 的统计结果
            # Each unique term in this chunk contributes 1 to DF
            # 注意：DF 统计的是“出现在多少个文档中”，不是“总共出现多少次”
            for term in chunk_stats["term_frequencies"].keys():
                # 只遍历当前 chunk 中出现过的唯一 term
                doc_freq[term] = doc_freq.get(term, 0) + 1
                # 当前 term 在一个新文档里出现过，所以 DF +1
        
        return {
            "num_docs": num_docs,
            # 返回总文档数

            "avg_doc_length": avg_doc_length,
            # 返回平均文档长度

            "document_frequency": doc_freq,
            # 返回每个 term 的 DF 字典
        }