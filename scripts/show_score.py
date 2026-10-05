"""把分数拆开：这一分到底是怎么加出来的，4.5 又是怎么定下来的。

两部分：
    第一部分  拿"红烧肉"那张卡，逐词打印加法过程   ← 回答"分数是怎么来的"
    第二部分  把 30 道题的分数全排出来，看线该划哪 ← 回答"为什么是 4.5"

跑法：
    python -m scripts.show_score
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import index_paths, load_index  # noqa: E402
from app.services.retriever import BM25_B, BM25_K1, KB, _segment  # noqa: E402

QUESTION = "红烧肉怎么做才好吃？"
LINE = "=" * 78


def part1() -> None:
    """逐词打印 BM25 加法。"""
    print(f"\n{LINE}")
    print("第一部分：红烧肉这一分，是怎么加出来的")
    print(LINE)

    tokens = _segment(QUESTION)
    scores = KB._bm25_scores(QUESTION)  # noqa: SLF001
    top_index = max(range(len(scores)), key=lambda i: scores[i])
    top_chunk = KB._chunks[top_index]  # noqa: SLF001

    n = len(KB._chunks)  # noqa: SLF001
    dl = len(KB._doc_tokens[top_index])  # noqa: SLF001
    avgdl = KB._avg_length  # noqa: SLF001

    print(f"\n它翻出来的那张卡：{top_chunk.heading or '(无标题)'}")
    print(f"  全库卡片数 N = {n}")
    print(f"  这张卡的词数 dl = {dl}　全库平均词数 avgdl = {avgdl:.1f}")
    print(f"  公式里的固定参数 k1 = {BM25_K1}　b = {BM25_B}")

    print(f"\n问题被切成 {len(tokens)} 个词：{tokens}")
    print(f"\n{'词':<10}{'tf':>4}{'df':>5}{'IDF':>9}{'本词贡献':>12}{'累计':>10}")
    print("-" * 78)

    total = 0.0
    for token in tokens:
        df = KB._doc_freq.get(token, 0)  # noqa: SLF001
        tf = KB._doc_tokens[top_index].count(token)  # noqa: SLF001

        if df == 0 or tf == 0:
            print(f"{token:<10}{tf:>4}{df:>5}{'—':>9}{'0.000':>12}{total:>10.3f}   一个都没撞上")
            continue

        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / (avgdl or 1))
        contribution = idf * tf * (BM25_K1 + 1) / denom
        total += contribution

        print(
            f"{token:<10}{tf:>4}{df:>5}{idf:>9.3f}{contribution:>12.3f}{total:>10.3f}"
        )

    print("-" * 78)
    print(f"{'合计':<10}{'':>4}{'':>5}{'':>9}{'':>12}{total:>10.3f}")
    print(f"\n脚本算出：{total:.3f}　　BM25 实际给出：{scores[top_index]:.3f}")

    print("\n怎么读这张表：")
    print("  tf = 这个词在这张卡里出现几次")
    print("  df = 这个词在全库几张卡里出现过；df 越小 → IDF 越大 → 这个词越'值钱'")
    print("  本词贡献 = IDF × 词频修正（出现越多加得越少，防止刷词）")


def part2() -> None:
    """把 30 道题的分数排出来，看 4.5 划在哪。"""
    print(f"\n{LINE}")
    print("第二部分：4.5 这条线，是怎么定下来的")
    print(LINE)

    eval_path = ROOT / "data/eval/rag_eval_set.jsonl"
    cases = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    pos: list[tuple[float, str]] = []
    neg: list[tuple[float, str]] = []

    for case in cases:
        raw = KB._bm25_scores(case["question"])  # noqa: SLF001
        best = max(raw) if raw else 0.0
        (neg if case["type"] == "negative" else pos).append((best, case["question"]))

    pos.sort()
    neg.sort(reverse=True)

    print(f"\n【正例】知识库里确实有答案的题，共 {len(pos)} 条，按分数从低到高：")
    for score, q in pos[:5]:
        print(f"    {score:>7.3f}   {q}")
    print("    ……（中间省略）")
    for score, q in pos[-2:]:
        print(f"    {score:>7.3f}   {q}")
    print(f"\n    ★ 正例里最低的那个 = {pos[0][0]:.3f}")

    print(f"\n【负例】知识库里根本没有答案的题，共 {len(neg)} 条，按分数从高到低：")
    for score, q in neg:
        flag = "  ← 高于 4.5，挡不住" if score > 4.5 else ""
        print(f"    {score:>7.3f}   {q}{flag}")
    print(f"\n    ★ 负例里最高的那个 = {neg[0][0]:.3f}")

    print(f"\n{LINE}")
    print("看图：")
    print(LINE)
    print(f"""
    正例最低 {pos[0][0]:.3f} ────────────────────────────┐
                                                      │  ← 这一段全是正例，绝不能划到它们头上
    负例最高 {neg[0][0]:.3f} ────────────────────┐          │
                                        │          │
                                        └──────────┴── 两个区间是重叠的
                                        重叠区：既躲不开负例，又碰不得正例
""")

    for line_at in (4.0, 4.5, 4.592, 5.0, 6.0, 7.0):
        killed = sum(1 for s, _ in pos if s < line_at)
        blocked = sum(1 for s, _ in neg if s < line_at)
        print(
            f"    线划在 {line_at:>6.3f} → "
            f"误杀正例 {killed:>2}/{len(pos)}　"
            f"挡住负例 {blocked}/{len(neg)}"
        )

    print("""
怎么选：
  线往高划 → 挡住更多负例，但会误杀正例（用户问得出来的问题，系统说"没找到"）
  线往低划 → 一个正例都不误杀，但负例也挡不住

  正例最低分是 4.592。所以：
    · 4.5 是"贴着正例最低分、往下让一点点"的位置 → 零误杀
    · 让的那一点是安全余量：评测集只有 23 条正例，
      真实用户会问出没测过的问题，留点余量防止把没测过的正常问题误杀

  结论：4.5 不是我算出来的，是拿你自己的数据"称"出来的一个妥协点。
""")


def main() -> None:
    if not load_index():
        index_paths(rebuild=True)
    part1()
    part2()


if __name__ == "__main__":
    main()
