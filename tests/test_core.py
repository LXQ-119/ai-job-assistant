"""单元测试。

这些测试**不需要联网、不需要 API Key**——因为检索、切块、成本计算
这些纯逻辑部分本来就不该依赖外部服务。这就是把 LLM 调用抽成单独一层的回报：
能测的部分变多了。

跑法：
    pytest -v
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.config import estimate_cost_usd, is_peak_now
from app.services import retriever
from app.services.documents import chunk_text
from app.services.resume import _load_profile
from app.services.retriever import KnowledgeBase, _min_max, _segment


# ---------------------------------------------------------------------------
# 切块
# ---------------------------------------------------------------------------


def test_chunk_keeps_heading_breadcrumb():
    """标题要作为面包屑带到块里——否则单看一个块不知道它属于哪一节。"""
    text = "# 顶层\n\n## 子节\n\n这里是一段正文内容，用来验证标题面包屑。"
    chunks = chunk_text(text, source="t.md", size=500, overlap=50)

    assert len(chunks) == 1
    assert chunks[0].heading == "顶层 > 子节"
    assert chunks[0].retrieval_text.startswith("顶层 > 子节")


def test_chunk_splits_long_text_with_overlap():
    """超长文本要切成多块，且相邻块之间必须有重叠。"""
    paragraph = "这是一句用于测试的正文。" * 60  # 约 720 字
    chunks = chunk_text(paragraph, source="t.md", size=200, overlap=40)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        tail = previous.text[-40:]
        assert tail[:10] in current.text


def test_chunk_ignores_headings_inside_code_block():
    """代码块里的 # 注释不是标题，不能被当成分节。"""
    text = "# 真标题\n\n```python\n# 这只是注释\nprint(1)\n```\n\n正文内容。"
    chunks = chunk_text(text, source="t.md", size=500, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].heading == "真标题"


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------


def test_bm25_ranks_relevant_chunk_first(monkeypatch):
    """BM25 应该把含有关键词的块排在前面。

    这里显式把 MIN_BM25_SCORE 关掉，原因很重要：

        最低分阈值是**按语料标定**出来的绝对分数（当前真实语料取 11.5）。
        而这个测试的语料只有 2 个块，BM25 分数天然只有个位数 ——
        两者根本不可比。

    BM25 的原始分依赖语料规模（IDF 公式里就含 N），
    所以**绝对阈值不能跨语料复用**，换语料必须重新标定。
    这个测试改成"只测排序"，阈值行为由
    test_min_score_threshold_rejects_weak_matches 单独覆盖。
    """
    from app.config import get_settings

    monkeypatch.setenv("MIN_BM25_SCORE", "0")
    get_settings.cache_clear()
    try:
        kb = KnowledgeBase()
        chunks = chunk_text(
            "# 苹果\n\n苹果是一种水果，富含维生素。\n\n# 汽车\n\n汽车是交通工具，需要汽油。",
            source="fruits.md",
            size=200,
            overlap=0,
        )
        kb.add_chunks(chunks)

        results = kb.search("汽车需要什么燃料", top_k=2)

        assert results, "检索不应为空"
        assert "汽车" in results[0].chunk.heading
    finally:
        get_settings.cache_clear()


def test_min_score_threshold_rejects_weak_matches(monkeypatch):
    """分数低于阈值的检索结果，必须被判定为"没找到"。

    为什么必须有这道门槛：
        BM25 对任何问题都会给出一个"最高分"，哪怕全是噪声 ——
        因为总有一张卡恰好撞上一两个常见字。没有门槛的话，
        系统永远能"找到"东西，于是永远不说"没找到"，也就永远在骗用户。
    """
    from app.config import get_settings

    kb = KnowledgeBase()
    kb.add_chunks(
        chunk_text("# 苹果\n\n苹果是一种水果。", source="a.md", size=200, overlap=0)
    )

    # 阈值高得离谱 → 什么都过不了 → 返回空
    monkeypatch.setenv("MIN_BM25_SCORE", "999")
    get_settings.cache_clear()
    try:
        assert kb.search("苹果", top_k=2) == []
    finally:
        get_settings.cache_clear()

    # 阈值关掉 → 同样的查询正常返回
    monkeypatch.setenv("MIN_BM25_SCORE", "0")
    get_settings.cache_clear()
    try:
        assert kb.search("苹果", top_k=2)
    finally:
        get_settings.cache_clear()


def test_no_match_returns_empty_instead_of_random_chunks():
    """查询词一个都没命中时必须返回空，而不是随便给几个块。

    这是质量关键点：返回无关内容会诱导模型硬编答案，
    比老实说"资料里没有提到"糟糕得多。
    """
    kb = KnowledgeBase()
    kb.add_chunks(chunk_text("# 苹果\n\n苹果很甜。", source="a.md", size=200, overlap=0))

    assert kb.search("量子纠缠退相干", top_k=4) == []


def test_segmentation_drops_stopwords():
    """停用词不该进入倒排统计，否则会稀释有效信号。"""
    # 单字停用词在"jieba"和"字符二元组"两种模式下都应被过滤
    assert "的" not in _segment("这是的测试")

    # 多字停用词的过滤效果依赖 jieba；未安装时分词行为不同，所以按条件断言，
    # 避免测试因为环境差异而误报失败。
    if retriever._jieba_module is not None:
        tokens = _segment("这个是可以的")
        assert "这个" not in tokens
        assert "可以" not in tokens


def test_min_max_normalizes_to_unit_range():
    assert _min_max([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]
    # 全部相等时无法区分，返回全 1 而不是除零
    assert _min_max([2.0, 2.0]) == [1.0, 1.0]
    assert _min_max([]) == []


def test_hybrid_alpha_zero_does_not_enable_vectors(monkeypatch):
    """alpha=0 时不应启用向量，避免无谓的向量化开销。

    顺便演示一个真实的坑：get_settings() 用了 lru_cache，
    改了环境变量必须 cache_clear() 才会生效，否则新值根本读不到。
    """
    from app.config import get_settings

    monkeypatch.setenv("HYBRID_ALPHA", "0")
    get_settings.cache_clear()
    try:
        kb = KnowledgeBase()
        kb.add_chunks(chunk_text("# A\n\n内容甲", source="a.md", size=200, overlap=0))
        assert kb.vector_enabled is False
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 成本计算
# ---------------------------------------------------------------------------


def test_cost_differs_between_peak_and_off_peak():
    """高峰单价是非高峰的两倍——这个差异必须体现出来。"""
    off_peak = estimate_cost_usd(
        "deepseek-flash", prompt_tokens=1_000_000, completion_tokens=0, peak=False
    )
    peak = estimate_cost_usd(
        "deepseek-flash", prompt_tokens=1_000_000, completion_tokens=0, peak=True
    )

    # 必须用 approx：二进制浮点下 1000000 * 0.15 / 1000000 并不精确等于 0.15
    assert off_peak == pytest.approx(0.15)
    assert peak == pytest.approx(0.30)


def test_cached_tokens_are_cheaper_than_missed_tokens():
    """缓存命中的输入 token 便宜得多，不能和未命中混为一谈。"""
    no_cache = estimate_cost_usd(
        "deepseek-flash", prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=0, peak=False
    )
    full_cache = estimate_cost_usd(
        "deepseek-flash",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cached_tokens=1_000_000,
        peak=False,
    )

    assert full_cache < no_cache
    assert full_cache == pytest.approx(0.003)


def test_unknown_model_falls_back_instead_of_crashing():
    """遇到没登记的模型要能兜底，不能把成本统计搞崩。"""
    cost = estimate_cost_usd(
        "some-new-model", prompt_tokens=1000, completion_tokens=1000, peak=False
    )
    assert cost > 0


def test_peak_window_matches_documented_hours():
    """高峰时段：UTC 周一至周五 01:00-04:00 与 06:00-10:00。"""
    # 2026-03-02 是周一
    assert is_peak_now(datetime(2026, 3, 2, 2, 0, tzinfo=timezone.utc)) is True
    assert is_peak_now(datetime(2026, 3, 2, 7, 30, tzinfo=timezone.utc)) is True
    assert is_peak_now(datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)) is False
    assert is_peak_now(datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)) is False
    # 2026-03-07 是周六，全天非高峰
    assert is_peak_now(datetime(2026, 3, 7, 2, 0, tzinfo=timezone.utc)) is False


# ---------------------------------------------------------------------------
# 简历解析的容错
# ---------------------------------------------------------------------------


def test_profile_parsing_strips_markdown_fence():
    """模型常把 JSON 包在代码块里，必须能剥掉。"""
    raw = '```json\n{"name": "张明", "skills": ["Python"], "years_of_experience": 0}\n```'
    profile = _load_profile(raw)

    assert profile.name == "张明"
    assert profile.skills == ["Python"]


def test_profile_parsing_unwraps_nested_object():
    """模型有时会套一层外壳，比如 {"resume": {...}}。"""
    raw = '{"resume": {"name": "李雷", "skills": ["Go"]}}'
    profile = _load_profile(raw)

    assert profile.name == "李雷"


def test_profile_parsing_rejects_non_object():
    """返回数组之类的东西时要明确报错，而不是悄悄产出空数据。"""
    with pytest.raises(ValueError):
        _load_profile("[1, 2, 3]")
