"""诊断：为什么明明有的东西，检索说"没有"？

用法：
    python -m scripts.why_missed "我从小米那次面试里学到了什么？"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import index_paths, load_index  # noqa: E402
from app.services.retriever import BM25_B, BM25_K1, KB, _segment  # noqa: E402

LINE = "=" * 78


def main() -> None:
    if not load_index():
        index_paths(rebuild=True)

    question = sys.argv[1] if len(sys.argv) > 1 else "我从小米那次面试里学到了什么？"
    tokens = _segment(question)
    scores = KB._bm25_scores(question)
    n = len(KB._chunks)

    print(f"\n{LINE}")
    print(f"问题：{question}")
    print(f"切出来的词：{tokens}")
    print(LINE)

    print(f"\n每个词在全库的分布：")
    for t in tokens:
        df = KB._doc_freq.get(t, 0)
        where = [
            i for i in range(n) if t in KB._doc_tokens[i]
        ]
        print(f"  「{t}」 df={df:>2}  出现在卡片 {where}")

    print(f"\n{LINE}")
    print("全部 19 张卡的得分（从高到低）：")
    print(LINE)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    for rank, i in enumerate(order, start=1):
        chunk = KB._chunks[i]
        mark = "   ★★ 这是你想要的答案 ★★" if "小米" in (chunk.heading or "") else ""
        print(f"  {rank:>2}. {scores[i]:>8.3f}  {chunk.source} / {chunk.heading}{mark}")

    # 找到目标卡
    target = next(
        (i for i in range(n) if "小米" in (KB._chunks[i].heading or "")), None
    )
    if target is None:
        print("\n找不到含'小米'的卡片。")
        return

    print(f"\n{LINE}")
    print(f"目标卡片的账（第 {order.index(target) + 1} 名，总分 {scores[target]:.3f}）")
    print(LINE)
    dl = len(KB._doc_tokens[target])
    print(f"  {'词':<10}{'tf':>4}{'df':>5}{'IDF':>9}{'贡献':>10}")
    print("  " + "-" * 42)
    for t in tokens:
        df = KB._doc_freq.get(t, 0)
        tf = KB._doc_tokens[target].count(t)
        if df == 0 or tf == 0:
            continue
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / (KB._avg_length or 1))
        print(
            f"  {t:<10}{tf:>4}{df:>5}{idf:>9.3f}"
            f"{idf * tf * (BM25_K1 + 1) / denom:>10.3f}"
        )

    print(f"\n  目标卡里到底有没有'小米'这个词？")
    print(f"    '小米' in KB._doc_freq ? {'小米' in KB._doc_freq}")
    print(f"    目标卡的词列表里有没有含'小'或'米'的词？")
    hits = [t for t in KB._doc_tokens[target] if "小" in t or "米" in t]
    print(f"      {hits[:20]}")


if __name__ == "__main__":
    main()
