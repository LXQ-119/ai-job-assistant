"""修复方案对比实验 —— 三个方案，同一批数据，看谁赢。

方案 A：现状（jieba 分词，标点参与打分）
方案 B：jieba 分词 + 剔除标点
方案 C：中文二元组分词 + 剔除标点

为什么考虑"二元组"（bigram）：
    "小米" 在查询里被 jieba 切成了 "从小" + "米"，
    但在资料里又被切成了完整的 "小米" —— 两边对不上。
    二元组不看词，只看相邻两个字：
        "我从小米那次面试" → 我从 / 从小 / 小米 / 米那 / 那次 / 次面 / 面试 …
    "小米" 这个二元组在两边都会出现，**永远对得上**。
    代价是索引变大、会有一些噪音匹配。

对每个方案都测三件事：
    1. "小米那次面试"这个问题，目标卡排第几
    2. 评测集 23 条正例的 Hit@k 和 MRR
    3. 评测集 7 条负例的拒答率

跑法：
    python -m scripts.compare_fix
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.services.retriever as R  # noqa: E402
from app.services import rag  # noqa: E402

TARGET_QUESTION = "我从小米那次面试里学到了什么？"
LINE = "=" * 78

_CJK_RE = R._CJK_RE  # noqa: SLF001
_LATIN_RE = R._LATIN_RE  # noqa: SLF001
_STOPWORDS = R._STOPWORDS  # noqa: SLF001


def is_noise(token: str) -> bool:
    return bool(token) and all(
        unicodedata.category(ch)[0] in ("P", "S", "Z") for ch in token
    )


def jieba_clean(text: str) -> list[str]:
    """方案 B：jieba 分词 + 去掉标点。"""
    return [t for t in R._original_segment(text) if not is_noise(t)]


def bigram_clean(text: str) -> list[str]:
    """方案 C：中文切成二元组 + 英文整词 + 去掉标点。"""
    lowered = text.lower()
    tokens: list[str] = []

    for word in _LATIN_RE.findall(lowered):
        if word not in _STOPWORDS and not is_noise(word):
            tokens.append(word)

    for run in _CJK_RE.findall(lowered):
        for ch in run:  # 单字也保留，防止只问一个字时落空
            if ch not in _STOPWORDS:
                tokens.append(ch)
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram not in _STOPWORDS:
                tokens.append(bigram)

    return tokens


def run_case(segment_fn, cases: list[dict]) -> dict:
    """换分词函数 → 重建索引 → 跑评测集 + 那个小米问题。"""
    R._segment = segment_fn  # type: ignore[assignment]
    rag.index_paths([], rebuild=True)

    KB = R.KB
    pos_hits = 0
    pos_rr: list[float] = []
    pos_total = 0
    neg_refused = 0
    neg_total = 0

    for case in cases:
        hits = KB.search(case["question"], top_k=4)
        if case["type"] == "negative":
            neg_total += 1
            if not hits:
                neg_refused += 1
            continue
        pos_total += 1
        rank = None
        for position, hit in enumerate(hits, start=1):
            if hit.chunk.source in case["expected_sources"]:
                rank = position
                break
        if rank:
            pos_hits += 1
        pos_rr.append(1.0 / rank if rank else 0.0)

    # 小米那个问题单独看
    target_scores = KB._bm25_scores(TARGET_QUESTION)  # noqa: SLF001
    order = sorted(
        range(len(KB)), key=lambda i: target_scores[i], reverse=True
    )
    target_rank = next(
        (
            r
            for r, i in enumerate(order, start=1)
            if "小米" in (KB._chunks[i].heading or "")
        ),
        None,
    )
    top1 = KB._chunks[order[0]].heading if order else "—"

    return {
        "Hit@k": pos_hits / pos_total if pos_total else 0.0,
        "MRR": sum(pos_rr) / len(pos_rr) if pos_rr else 0.0,
        "拒答率": neg_refused / neg_total if neg_total else 0.0,
        "小米卡排名": target_rank,
        "小米问题第1名": top1 or "—",
        "卡片数": len(KB),
    }


def main() -> None:
    cases = [
        json.loads(line)
        for line in (ROOT / "data/eval/rag_eval_set.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    R._original_segment = R._segment  # type: ignore[attr-defined]  # noqa: SLF001

    plans = [
        ("A 现状：jieba + 标点", R._original_segment),  # noqa: SLF001
        ("B 修标点：jieba + 去标点", jieba_clean),
        ("C 修标点 + 中文二元组", bigram_clean),
    ]

    results = []
    for label, fn in plans:
        print(f"\n正在跑：{label} ……")
        results.append((label, run_case(fn, cases)))

    R._segment = R._original_segment  # type: ignore[assignment]  # noqa: SLF001

    print(f"\n{LINE}")
    print("结果对比")
    print(LINE)
    header = f"{'方案':<26}{'Hit@k':>9}{'MRR':>8}{'拒答率':>9}{'小米卡排名':>11}{'卡片数':>8}"
    print(header)
    print("-" * len(header))
    for label, r in results:
        rank = r["小米卡排名"]
        print(
            f"{label:<26}{r['Hit@k']:>9.2%}{r['MRR']:>8.3f}"
            f"{r['拒答率']:>9.2%}{(str(rank) if rank else '没进榜'):>11}"
            f"{r['卡片数']:>8}"
        )

    print(f"\n小米那个问题，各方案排第 1 的是谁：")
    for label, r in results:
        print(f"  {label:<26} → {r['小米问题第1名']}")


if __name__ == "__main__":
    main()
