"""Query Processor for preprocessing user queries.
# 这个模块是专门用来先处理用户输入的 query 的

This module provides query preprocessing functionality including:
# 下面列的是这个模块目前支持的处理能力

- Keyword extraction using rule-based tokenization
# 用规则方式先把 query 切成词，再提取关键词

- Stopword filtering for Chinese and English
# 支持中英文停用词过滤，比如“的、是、how、the”这种没啥检索价值的词会去掉

- Filter parsing from query syntax (e.g., "collection:docs")
# 支持从 query 里解析过滤条件，比如 collection:docs 这种写法

- Query normalization and cleaning
# 还会顺手把 query 做基础清洗和格式统一

Design Principles:
# 下面是这个模块的设计原则

- Rule-based first: Use simple, deterministic rules for reliability
# 优先用简单可控的规则，不先上复杂模型，这样更稳更容易排查

- Language-aware: Support both Chinese and English queries
# 要考虑中英文 query，不是只处理一种语言

- Extensible: Easy to add synonym expansion or LLM-based processing later
# 设计上留了扩展口，后面想加同义词扩展或者 LLM 改写也方便

- Configuration-driven: Stopwords and patterns configurable via settings
# 一些规则尽量走配置，后面调参数更方便
"""

import re
# 导入正则模块，后面用它做过滤条件匹配、符号判断等

from dataclasses import dataclass, field
# 导入 dataclass 和 field，用来写配置类，少写很多样板代码

from typing import Any, Dict, List, Optional, Pattern, Set
# 导入类型标注，方便把代码写清楚：这里的变量、参数、返回值都是什么类型

import jieba
# 导入 jieba 分词，用来切中文；英文也能保留下来

from src.core.types import ProcessedQuery
# 导入项目里定义好的 ProcessedQuery 类型，最后处理结果会装进这个对象里


# Default stopwords for Chinese
# 下面这块是默认的中文停用词
CHINESE_STOPWORDS: Set[str] = {
    # 疑问词
    # 这类词经常出现在提问里，但通常不适合当检索关键词
    "如何", "怎么", "怎样", "什么", "哪个", "哪些", "为什么", "为何",
    "谁", "多少", "几", "是否", "能否", "可否",
    # 助词
    # 这类词主要是语法作用，一般没有实际检索价值
    "的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀",
    # 介词/连词
    # 这些词连接句子用，通常不是重点关键词
    "在", "于", "和", "与", "或", "及", "并", "而", "但", "但是",
    "因为", "所以", "如果", "那么", "虽然", "然而",
    # 代词
    # 这些代词太泛，不适合当检索词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "这些", "那些", "这里", "那里",
    # 副词
    # 这些词更多是语气和程度，不是内容核心
    "很", "非常", "特别", "更", "最", "都", "也", "还", "又", "再",
    "已", "已经", "正在", "将", "会", "能", "可以", "应该", "必须",
    # 动词(通用)
    # 这些动词太常见，单独拿来检索意义不大
    "是", "有", "做", "进行", "使用", "通过",
    # 量词
    # 量词通常也不重要
    "个", "种", "类",
    # 标点等
    # 这些就是标点符号，肯定不当关键词
    "？", "。", "！", "，", "、",
}

# Default stopwords for English
# 下面这块是默认的英文停用词
ENGLISH_STOPWORDS: Set[str] = {
    # Articles
    # 冠词，一般没检索价值
    "a", "an", "the",
    # Prepositions
    # 介词，主要是连接作用
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "through", "between", "after", "before",
    # Conjunctions
    # 连词，也不是重点内容
    "and", "or", "but", "if", "then", "because", "while", "although",
    # Pronouns
    # 代词，太泛
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose",
    # Auxiliary verbs
    # 系动词、助动词这种一般都不拿来做关键词
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can",
    # Common verbs
    # 太泛的常见动词
    "get", "use", "make",
    # Question words
    # 疑问词，和中文那边一个思路
    "how", "why", "when", "where",
    # Others
    # 其他常见但信息量低的词
    "not", "no", "yes", "so", "very", "just", "also", "too",
}

# Combined default stopwords
# 把中文和英文停用词集合并成一个总集合，后面统一用它过滤
DEFAULT_STOPWORDS: Set[str] = CHINESE_STOPWORDS | ENGLISH_STOPWORDS

# Pattern for filter syntax: key:value
# 这个正则专门用来匹配类似 key:value 的过滤条件
FILTER_PATTERN: Pattern = re.compile(r'(\w+):([^\s]+)')
# (\w+) 表示前面的 key，比如 collection
# : 表示中间那个冒号
# ([^\s]+) 表示后面的 value，抓到下一个空格前为止，比如 docs


@dataclass
# 这里用 dataclass 来写配置类，省得自己手写 __init__
class QueryProcessorConfig:
    """Configuration for QueryProcessor.
    # 这个类是 QueryProcessor 的配置对象
    
    Attributes:
    # 下面列的是配置里有哪些字段
    
        stopwords: Set of words to filter out
        # 停用词集合，命中这些词就过滤掉
        
        min_keyword_length: Minimum length for a keyword to be included
        # 关键词最短长度，太短的词不要
        
        max_keywords: Maximum number of keywords to extract
        # 最多提取多少个关键词，避免太多
        
        enable_filter_parsing: Whether to parse filter syntax from query
        # 要不要解析 query 里的 key:value 过滤条件
    """
    stopwords: Set[str] = field(default_factory=lambda: DEFAULT_STOPWORDS.copy())
    # 停用词默认值：拷贝一份默认停用词集合
    # 这里用 copy() 是为了避免多个实例共用同一个集合，后面互相污染

    min_keyword_length: int = 1
    # 默认关键词最短长度是 1，也就是长度小于 1 的才过滤

    max_keywords: int = 20
    # 默认最多保留 20 个关键词

    enable_filter_parsing: bool = True
    # 默认开启过滤条件解析


class QueryProcessor:
    """Preprocesses user queries for retrieval.
    # 这个类专门负责把用户原始 query 预处理成检索系统更好用的结构
    
    Extracts keywords, filters stopwords, and parses filter syntax
    # 它会做几件事：提关键词、去停用词、解析过滤条件

    to prepare queries for Dense and Sparse retrievers.
    # 处理完之后，给 Dense 检索和 Sparse 检索一起用
    
    Example:
    # 下面是一个使用例子
        >>> processor = QueryProcessor()
        # 创建一个 QueryProcessor 实例

        >>> result = processor.process("如何配置 Azure OpenAI？")
        # 处理一条 query

        >>> print(result.keywords)
        # 看最终抽出来的关键词

        ['配置', 'Azure', 'OpenAI']
        # 结果里就把“如何”这种没啥用的词去掉了，只留下重点词
    """
    
    def __init__(self, config: Optional[QueryProcessorConfig] = None):
        """Initialize QueryProcessor.
        # 初始化 QueryProcessor 对象
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
            # 可以传自定义配置；不传就用默认配置
        """
        self.config = config or QueryProcessorConfig()
        # 如果外面传了配置，就用外面的
        # 如果没传，就自己创建一个默认配置
    
    def process(self, query: str) -> ProcessedQuery:
        """Process a user query into structured format.
        # 这是最核心的方法：把原始 query 处理成结构化结果
        
        Args:
            query: Raw user query string
            # 用户原始输入的 query 字符串
            
        Returns:
            ProcessedQuery with extracted keywords and filters
            # 返回一个 ProcessedQuery，里面有原始 query、关键词、过滤条件
        """
        if not query or not query.strip():
            # 如果 query 是空，或者全是空格，那就直接返回空结果
            return ProcessedQuery(
                original_query=query or "",
                # 原始 query 为空时，保险起见转成空字符串

                keywords=[],
                # 没有关键词

                filters={}
                # 也没有过滤条件
            )
        
        # Normalize query
        # 第一步：先把 query 做基础清洗，比如多余空格处理掉
        normalized = self._normalize(query)
        
        # Extract filters from query (e.g., "collection:docs")
        # 第二步：从 query 里把 collection:docs 这种过滤条件拆出来
        filters, query_without_filters = self._extract_filters(normalized)
        # filters 是解析出来的过滤条件字典
        # query_without_filters 是把 key:value 去掉后的剩余 query 文本
        
        # Tokenize and extract keywords
        # 第三步：对剩余 query 做分词
        tokens = self._tokenize(query_without_filters)
        
        # Filter stopwords and apply constraints
        # 第四步：把分词结果做进一步筛选，留下真正关键词
        keywords = self._filter_keywords(tokens)
        
        return ProcessedQuery(
            original_query=query,
            # 原始 query 保留用户最开始输入的版本，不用清洗后的版本

            keywords=keywords,
            # 放入处理后的关键词列表

            filters=filters
            # 放入解析出的过滤条件
        )
    
    def _normalize(self, query: str) -> str:
        """Normalize query string.
        # 这个私有方法负责做 query 的基础标准化
        
        - Strip whitespace
        # 去掉前后空白、压缩多余空格
        
        - Normalize unicode
        # 文档里说还可以做 unicode 规范化
        
        - Convert to consistent format
        # 总之目标就是把输入变整齐一点
        
        Args:
            query: Raw query string
            # 原始 query
            
        Returns:
            Normalized query string
            # 处理后的规范 query
        """
        # Strip and normalize whitespace
        # 先按任意空白切开，再用单个空格重新拼起来
        normalized = " ".join(query.split())
        # 这样就能把多个空格、换行、tab 都统一成普通单空格

        return normalized
        # 返回标准化后的 query
    
    def _extract_filters(self, query: str) -> tuple[Dict[str, Any], str]:
        """Extract filter syntax from query.
        # 这个私有方法专门从 query 里提取过滤条件
        
        Supports syntax like: "collection:api-docs keyword1 keyword2"
        # 支持这种写法：前面是过滤条件，后面是正常关键词
        
        Args:
            query: Normalized query string
            # 传入的是已经清洗过的 query
            
        Returns:
            Tuple of (filters dict, query without filter syntax)
            # 返回两个东西：
            # 1. filters 字典
            # 2. 去掉过滤条件后的 query
        """
        if not self.config.enable_filter_parsing:
            # 如果配置里关闭了解析过滤条件，那就直接返回空过滤条件，query 原样返回
            return {}, query
        
        filters: Dict[str, Any] = {}
        # 先准备一个空字典，用来装解析结果
        
        # Find all filter patterns
        # 用前面定义好的正则，把所有 key:value 都找出来
        matches = FILTER_PATTERN.findall(query)
        # findall 会返回所有匹配到的 (key, value) 对

        for key, value in matches:
            # 遍历每一组匹配到的过滤条件
            # 比如 collection:docs 会拿到 key='collection', value='docs'

            # Support common filter keys
            # 这里先把 key 转小写，避免大小写写法不一致
            key_lower = key.lower()

            if key_lower in ("collection", "col", "c"):
                # 如果 key 是 collection 或它的简写
                filters["collection"] = value
                # 统一塞到 filters["collection"] 里

            elif key_lower in ("type", "doc_type", "t"):
                # 如果 key 表示文档类型
                filters["doc_type"] = value
                # 统一映射成 doc_type

            elif key_lower in ("source", "src", "s"):
                # 如果 key 表示来源路径
                filters["source_path"] = value
                # 统一映射成 source_path

            elif key_lower in ("tag", "tags"):
                # 如果 key 是 tag/tags，说明这是标签过滤
                # Tags can be comma-separated
                # 支持多个标签逗号分隔，比如 tags:a,b,c
                if "tags" not in filters:
                    # 如果还没建过 tags 列表，就先建一个空列表
                    filters["tags"] = []
                filters["tags"].extend(value.split(","))
                # 把逗号拆开的标签都加进去

            else:
                # Generic filter
                # 如果不是上面这些常见 key，就直接按原样存
                filters[key] = value
        
        # Remove filter patterns from query
        # 下面开始把 query 里的 key:value 这些过滤条件删掉，只留下正常检索文本
        query_without_filters = FILTER_PATTERN.sub("", query).strip()
        # 用正则替换成空字符串，再去掉首尾空格

        query_without_filters = " ".join(query_without_filters.split())
        # 再顺手把中间可能残留的多余空格整理一下
        
        return filters, query_without_filters
        # 把过滤条件和“去掉过滤条件后的 query”一起返回
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/terms.
        # 这个私有方法负责把文本切成词
        
        Uses jieba for Chinese text segmentation, consistent with the
        # 中文分词用 jieba，这样和索引阶段的分词方式保持一致

        index-side tokenizer (SparseEncoder) so BM25 matching works.
        # 这样后面 BM25 检索时，query 侧和文档侧分词不会打架

        English text is handled natively by jieba (preserved as-is).
        # 英文 jieba 也会保留下来，不会乱切坏
        
        Args:
            text: Text to tokenize
            # 需要分词的文本
            
        Returns:
            List of tokens
            # 返回切好的 token 列表
        """
        tokens: List[str] = []
        # 先准备一个空列表，用来装最终 token

        # Use jieba to segment (handles Chinese + keeps English intact)
        # 用 jieba 做分词，中文切词，英文基本保持原样
        raw_tokens = jieba.lcut(text)
        # lcut 会直接返回一个 list，而不是生成器

        for token in raw_tokens:
            # 遍历 jieba 切出来的每个 token
            token = token.strip()
            # 先去掉 token 两边的空白

            if not token:
                # 如果去完空白后变成空字符串，就跳过
                continue

            # Skip pure punctuation / whitespace
            # 如果这个 token 纯粹只是标点或空白，也不要
            if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
                # 这里用正则判断 token 是不是全由空白/非单词字符组成
                continue

            tokens.append(token)
            # 真正有效的 token 才加入结果列表
        
        return tokens
        # 返回清洗后的 token 列表
    
    def _filter_keywords(self, tokens: List[str]) -> List[str]:
        """Filter tokens to get meaningful keywords.
        # 这个私有方法负责把 token 列表进一步筛成“真正有用的关键词”
        
        - Remove stopwords
        # 去掉停用词
        
        - Apply minimum length constraint
        # 去掉太短的词
        
        - Deduplicate while preserving order
        # 去重，但保留原来出现顺序
        
        - Apply maximum count limit
        # 最后还会限制最多保留多少个关键词
        
        Args:
            tokens: List of tokens
            # 分词得到的 token 列表
            
        Returns:
            List of filtered keywords
            # 返回过滤后的关键词列表
        """
        seen: Set[str] = set()
        # 这个集合用来记录已经见过的词，做去重用

        keywords: List[str] = []
        # 这个列表用来存最终关键词
        
        for token in tokens:
            # 遍历每一个 token
            # Normalize for comparison
            # 先转成小写，后面比较时更统一
            token_lower = token.lower()
            
            # Skip if already seen (case-insensitive dedup)
            # 如果这个词已经出现过了，就直接跳过
            # 这里去重是不区分大小写的，比如 OpenAI 和 openai 算同一个
            if token_lower in seen:
                continue
            
            # Skip stopwords (check both original and lowercase)
            # 如果这个词是停用词，也跳过
            # 同时检查原词和小写形式，避免英文大小写漏掉
            if token in self.config.stopwords or token_lower in self.config.stopwords:
                continue
            
            # Skip if too short
            # 如果词太短，也不要
            if len(token) < self.config.min_keyword_length:
                continue
            
            # Add keyword (preserve original case)
            # 走到这里说明这个 token 合格，加入关键词列表
            seen.add(token_lower)
            # 先把小写版本记到 seen 里，后面做去重

            keywords.append(token)
            # 关键词里保留原始写法，比如保留 Azure、OpenAI 这种大小写
            
            # Stop if we have enough
            # 如果已经够最大关键词数量了，就不用再往下处理了
            if len(keywords) >= self.config.max_keywords:
                break
        
        return keywords
        # 返回最终关键词列表
    
    def add_stopwords(self, words: Set[str]) -> None:
        """Add words to stopword set.
        # 这个方法用来动态往停用词集合里新增词
        
        Args:
            words: Set of words to add
            # 传进来一批要新增的停用词
        """
        self.config.stopwords.update(words)
        # 直接把这批词并进当前配置的停用词集合
    
    def remove_stopwords(self, words: Set[str]) -> None:
        """Remove words from stopword set.
        # 这个方法用来从停用词集合里删掉一些词
        
        Args:
            words: Set of words to remove
            # 传进来一批要删除的停用词
        """
        self.config.stopwords -= words
        # 把这些词从停用词集合里减掉


def create_query_processor(
    # 下面这个工厂函数是为了更方便创建 QueryProcessor
    stopwords: Optional[Set[str]] = None,
    # 可选：自定义停用词；不传就走默认停用词

    min_keyword_length: int = 1,
    # 可选：自定义关键词最短长度

    max_keywords: int = 20,
    # 可选：自定义最多提取多少关键词

    enable_filter_parsing: bool = True
    # 可选：是否开启 key:value 过滤条件解析
) -> QueryProcessor:
    """Factory function to create QueryProcessor.
    # 这是一个工厂函数，作用就是帮你快速构造一个配置好的 QueryProcessor
    
    Args:
        stopwords: Custom stopwords set. Uses default if None.
        # 自定义停用词；如果没传，就用默认的
        
        min_keyword_length: Minimum keyword length
        # 关键词最短长度
        
        max_keywords: Maximum keywords to extract
        # 最多提多少个关键词
        
        enable_filter_parsing: Whether to parse filter syntax
        # 要不要解析过滤条件
        
    Returns:
        Configured QueryProcessor instance
        # 返回一个配置好的 QueryProcessor 实例
    """
    config = QueryProcessorConfig(
        # 先根据传入参数组一个配置对象
        stopwords=stopwords if stopwords is not None else DEFAULT_STOPWORDS.copy(),
        # 如果外面传了 stopwords，就用外面的
        # 否则复制一份默认停用词

        min_keyword_length=min_keyword_length,
        # 把最短长度配置塞进去

        max_keywords=max_keywords,
        # 把最大关键词数塞进去

        enable_filter_parsing=enable_filter_parsing
        # 把是否解析过滤条件塞进去
    )
    return QueryProcessor(config)
    # 用这个配置创建并返回一个 QueryProcessor