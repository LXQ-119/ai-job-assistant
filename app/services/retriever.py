"""检索层：BM25 关键词检索 + 可选向量检索 + 混合融合。

为什么默认不是向量检索？
因为 **DeepSeek 官方 API 不提供 embeddings 接口**。默认走 BM25：
纯 Python 实现、零额外依赖、中文用 jieba 分词。想开向量就把
EMBEDDING_BACKEND 设成 api（用别的服务商）或 local（本地模型）。

这正好是面试里"混合检索"的最佳落点：
- BM25 擅长精确术语："function calling"、"P95"、"BM25" 这类词，语义模型反而容易糊
- 向量擅长语义近似："响应太慢" ↔ "延迟高"、"怎么防止胡编" ↔ "抑制幻觉"
两者融合通常比单用任一个都好，而且**融合权重 alpha 是可以拿评测集调出来的**——
这就是简历上那个"命中率 62% → 89%"的来处。
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..llm import LLMError, embed_texts
from .documents import Chunk

# BM25 的标准参数：k1 控制词频饱和，b 控制文档长度归一化。
# 这两个值是信息检索里的经典默认值，一般不需要动。
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    bm25_score: float
    vector_score: float

    def to_citation(self) -> dict:
        return {
            "source": self.chunk.source,
            "heading": self.chunk.heading,
            "chunk_index": self.chunk.index,
            "score": round(self.score, 4),
            "text": self.chunk.text[:500],
        }


# 停用词表很小但足够：中文虚词 + 英文虚词。
# 注意不要把技术词加进来，否则会把有用信号过滤掉。
_STOPWORDS = {
    # 中文
    "的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "什么", "怎么", "为什么", "可以", "我们", "他们", "它",
    # 注意：jieba 会把"这个""那个"切成一个整词，只收单字"这""那"是不够的，
    # 必须把常见的多字虚词也列进来，否则它们会混进倒排统计里稀释信号。
    "这个", "那个", "这些", "那些", "这样", "那样", "这里", "那里",
    "一下", "一些", "一点", "多少", "哪个", "哪些", "怎样", "如何",
    "以及", "或者", "但是", "因为", "所以", "如果", "对于", "关于", "通过", "进行",
    # 英文
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "but", "with", "as", "at", "by", "be", "it", "this", "that",
    "how", "what", "why", "which", "you", "we", "they", "i",
}

_LATIN_RE = re.compile(r"[a-z0-9_+#.\-]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

_jieba_module = None
_jieba_probed = False


def _is_noise(token: str) -> bool:
    """是不是纯标点/符号/空白。

    标点必须挡在分词外面。踩过的坑：问号被当成一个"词"参与打分，
    而它只在少数几张卡里出现，IDF 特别高 —— 一个「？」就贡献了
    负例 89.5% 的分数，把完全无关的卡片顶到第 1 名。
    标点没有任何检索价值，只会制造假信号。
    """
    return bool(token) and all(
        unicodedata.category(ch)[0] in ("P", "S", "Z") for ch in token
    )


def _bigram_segment(text: str) -> list[str]:
    """中文按相邻两字切，英文保留整词。

    为什么默认用它？因为分词器在**查询**和**文档**里的切法可能不一致。

    真实踩到的例子：
        查询"我从小米那次面试里学到了什么？"
            jieba 把"小米"切成了 "从小" + "米"
        而文档里"小米"是一个完整的词
        → 两边对不上，最关键的词直接作废
        → 正确答案从第 1 名掉到第 5 名，被挤出 top 4
        → 系统回答"资料里没有"，可资料里明明有

    二元组不看词，只看相邻两个字：
        "我从小米" → 我从 / 从小 / 小米
    "小米" 这个二元组在查询和文档里都会出现，**永远对得上**。

    代价：索引变大、会多出一些噪音匹配，所以必须配合 min_bm25_score 用。
    """
    lowered = text.lower()
    tokens: list[str] = []

    for word in _LATIN_RE.findall(lowered):
        if word not in _STOPWORDS and not _is_noise(word):
            tokens.append(word)

    for run in _CJK_RE.findall(lowered):
        # 单字也留着，否则只问一个字时会一条都匹配不上
        for ch in run:
            if ch not in _STOPWORDS:
                tokens.append(ch)
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram not in _STOPWORDS:
                tokens.append(bigram)

    return tokens


def _segment(text: str) -> list[str]:
    """分词入口。模式由 SEGMENT_MODE 决定：bigram（默认）或 jieba。

    两种模式都会剔除标点符号。
    """
    global _jieba_module, _jieba_probed

    if get_settings().segment_mode == "bigram":
        return _bigram_segment(text)

    if not _jieba_probed:
        _jieba_probed = True
        try:
            import jieba  # type: ignore

            _jieba_module = jieba
        except ImportError:
            _jieba_module = None

    if _jieba_module is not None:
        raw_tokens = _jieba_module.lcut(text)
    else:
        raw_tokens = _fallback_segment(text)

    return [
        t
        for t in (tok.strip().lower() for tok in raw_tokens)
        if t and t not in _STOPWORDS and not _is_noise(t)
    ]


def _fallback_segment(text: str) -> list[str]:
    """没有 jieba 时的降级分词：英文保留整词，中文切成二元组。"""
    lowered = text.lower()
    tokens = _LATIN_RE.findall(lowered)
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class KnowledgeBase:
    """内存索引 + JSON 持久化。

    没用向量数据库，是刻意的选择：数据量在几千块以内时，
    纯 Python 暴力检索完全够用，而引入 Chroma/Qdrant 会让新人把时间
    花在部署上而不是理解检索本身。**等数据量或并发真的成为瓶颈再换**——
    这句话在面试里比"我用了向量数据库"值钱得多。
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_length: float = 0.0
        self._vectors: list[list[float]] | None = None
        self.warnings: list[str] = []

    # -- 基本属性 -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def vector_enabled(self) -> bool:
        return self._vectors is not None

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for chunk in self._chunks:
            if chunk.source not in seen:
                seen.append(chunk.source)
        return seen

    # -- 建索引 -------------------------------------------------------------

    def clear(self) -> None:
        self._chunks = []
        self._doc_tokens = []
        self._doc_freq = {}
        self._avg_length = 0.0
        self._vectors = None
        self.warnings = []

    def add_chunks(self, chunks: list[Chunk], *, rebuild_vectors: bool = True) -> None:
        """追加并重算索引统计量。

        rebuild_vectors=False 用于"从磁盘恢复索引"的场景：
        向量已经存在磁盘上了，没必要再花钱重新算一遍。
        """
        if not chunks:
            return

        self._chunks.extend(chunks)
        self._doc_tokens = [_segment(c.retrieval_text) for c in self._chunks]

        # 文档频率：某个词在多少个块里出现过（BM25 的 IDF 需要）
        self._doc_freq = {}
        for tokens in self._doc_tokens:
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

        total = sum(len(tokens) for tokens in self._doc_tokens)
        self._avg_length = total / len(self._doc_tokens) if self._doc_tokens else 0.0

        if rebuild_vectors:
            self._rebuild_vectors()

    def _rebuild_vectors(self) -> None:
        settings = get_settings()
        if settings.embedding_backend == "none" or settings.hybrid_alpha <= 0:
            self._vectors = None
            if settings.embedding_backend != "none" and settings.hybrid_alpha <= 0:
                self.warnings = [
                    "EMBEDDING_BACKEND 已配置，但 HYBRID_ALPHA=0，当前仍为纯 BM25 检索。"
                ]
            return

        try:
            self._vectors = embed_texts([c.retrieval_text for c in self._chunks])
            self.warnings = []
        except LLMError as exc:
            # 降级：向量建不起来就退回纯 BM25，而不是让整个服务不可用。
            # 降级必须**可见**，否则会变成"效果莫名变差但查不出原因"。
            self._vectors = None
            self.warnings = [f"向量索引构建失败，已降级为纯 BM25 检索：{exc}"]

    # -- 检索 ---------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[ScoredChunk]:
        """两阶段检索：先用 BM25 圈出候选，再（如果启用）算向量相似度融合。

        先收缩候选再算向量，是省算力的常规做法——没必要给全库算一遍相似度。
        """
        settings = get_settings()
        if not self._chunks:
            return []

        k = top_k or settings.top_k
        pool = max(candidate_k or settings.candidate_k, k)

        bm25_by_index = self._bm25_scores(query)
        candidate_indices = sorted(
            range(len(self._chunks)), key=lambda i: bm25_by_index[i], reverse=True
        )[:pool]

        vector_by_index: dict[int, float] = {}
        if self._vectors is not None and settings.hybrid_alpha > 0:
            try:
                query_vector = embed_texts([query])[0]
                for i in candidate_indices:
                    vector_by_index[i] = _cosine(query_vector, self._vectors[i])
            except LLMError as exc:
                self.warnings = [f"查询向量化失败，本次降级为纯 BM25：{exc}"]

        # 两路分数各自归一化到 0-1 再融合。
        # 不归一化直接相加是错的：BM25 分数是无上界的，会直接压死余弦相似度。
        bm25_values = [bm25_by_index[i] for i in candidate_indices]
        vector_values = [vector_by_index.get(i, 0.0) for i in candidate_indices]
        bm25_norm = _min_max(bm25_values)
        vector_norm = _min_max(vector_values)

        use_vector = bool(vector_by_index) and settings.hybrid_alpha > 0
        alpha = settings.hybrid_alpha if use_vector else 0.0

        # 查询词一个都没命中，且没有向量兜底时，**返回空**比返回随机文档更好。
        # 返回无关内容会诱导模型硬编一个答案，这比老实说"资料里没有"糟糕得多。
        if not use_vector and (not bm25_values or max(bm25_values) <= 0.0):
            return []

        # 最低分阈值：分数最高的那张卡如果都没过线，说明整个库里没有相关内容。
        #
        # 为什么必须有这一道？
        #   BM25 一定会给出一个"最高分"，哪怕全是噪声 —— 因为总有一张卡
        #   恰好撞上一两个常见字。不设门槛的话，系统永远能"找到"东西，
        #   于是永远不说"没找到"，也就永远在骗用户。
        #
        # 阈值怎么定的（不是拍脑袋）：
        #   把评测集里所有正例的最低分、所有负例的最高分都扫出来，
        #   在"零误杀正例"的前提下取窗口中间值，留出安全余量。
        #   当前配置下：正例最低 13.592，负例最高 9.416 → 取 11.5。
        #   注意：**换分词方式或往知识库里加大量新文件后，必须重新扫一遍。**
        if not use_vector and settings.min_bm25_score > 0:
            if max(bm25_values) < settings.min_bm25_score:
                return []

        scored: list[ScoredChunk] = []
        for position, index in enumerate(candidate_indices):
            blended = alpha * vector_norm[position] + (1 - alpha) * bm25_norm[position]
            scored.append(
                ScoredChunk(
                    chunk=self._chunks[index],
                    score=blended,
                    bm25_score=bm25_by_index[index],
                    vector_score=vector_by_index.get(index, 0.0),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]

    def _bm25_scores(self, query: str) -> list[float]:
        total_docs = len(self._chunks)
        query_tokens = _segment(query)
        scores = [0.0] * total_docs

        for token in query_tokens:
            doc_freq = self._doc_freq.get(token, 0)
            if doc_freq == 0:
                continue
            # BM25 的 IDF：出现越少的词权重越高。
            # 加 1 是为了避免高频词算出负分。
            idf = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            for i, tokens in enumerate(self._doc_tokens):
                term_freq = tokens.count(token)
                if term_freq == 0:
                    continue
                doc_length = len(tokens) or 1
                denominator = term_freq + BM25_K1 * (
                    1 - BM25_B + BM25_B * doc_length / (self._avg_length or 1)
                )
                scores[i] += idf * term_freq * (BM25_K1 + 1) / denominator

        return scores

    # -- 持久化 -------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.to_dict() for c in self._chunks],
            "vectors": self._vectors,
            "vector_enabled": self._vectors is not None,
            "warnings": self.warnings,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> bool:
        """从磁盘恢复索引。返回是否成功。"""
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        if not chunks:
            return False

        self.clear()
        # 从磁盘恢复时不重新做向量化：向量已经存过了，
        # 每次启动都重算一遍等于白花钱——这是真实会踩的坑。
        self.add_chunks(chunks, rebuild_vectors=False)

        stored_vectors = payload.get("vectors")
        self.warnings = list(payload.get("warnings", []))
        if stored_vectors and len(stored_vectors) == len(chunks):
            # 只有当维度一致时才用回磁盘上的向量，否则宁可重算
            self._vectors = [[float(x) for x in vec] for vec in stored_vectors]
        return True


def _min_max(values: list[float]) -> list[float]:
    """把一组分数线性拉到 0-1。全部相等时返回全 1（表示"无法区分"）。"""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


# 全局单例：进程内共享一份索引。
KB = KnowledgeBase()
