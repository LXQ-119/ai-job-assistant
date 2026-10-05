"""顺手查一件事：为什么问"切块"的问题，排第一的却是「常见面试追问」？

第 5 层输出里，正确的那张卡（切块策略）排在第 2，排第 1 的是一张
跟问题无关的卡。看看分数是怎么来的。
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

QUESTION = "切块的时候为什么要保留重复内容？"
LINE = "=" * 78


def main() -> None:
    if not load_index():
        index_paths(rebuild=True)

    tokens = _segment(QUESTION)
    scores = KB._bm25_scores(QUESTION)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    print(f"\n问题：{QUESTION}")
    print(f"切出来的词：{tokens}")
    print(f"\n排前 3 的卡片：")
    for rank, i in enumerate(order, start=1):
        print(f"  {rank}. {KB._chunks[i].heading}   总分 {scores[i]:.3f}")

    print(f"\n{LINE}")
    n = len(KB._chunks)
    for rank, i in enumerate(order, start=1):
        chunk = KB._chunks[i]
        dl = len(KB._doc_tokens[i])
        print(f"\n【第 {rank} 名】{chunk.heading}　（{len(chunk.text)} 字）")
        print(f"  {'词':<10}{'tf':>4}{'df':>5}{'IDF':>9}{'贡献':>10}")
        print("  " + "-" * 42)
        total = 0.0
        for token in tokens:
            df = KB._doc_freq.get(token, 0)
            tf = KB._doc_tokens[i].count(token)
            if df == 0 or tf == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / (KB._avg_length or 1))
            c = idf * tf * (BM25_K1 + 1) / denom
            total += c
            print(f"  {token:<10}{tf:>4}{df:>5}{idf:>9.3f}{c:>10.3f}")
        print("  " + "-" * 42)
        print(f"  {'合计':<10}{'':>4}{'':>5}{'':>9}{total:>10.3f}")


if __name__ == "__main__":
    main()
