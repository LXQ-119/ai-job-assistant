"""拆开"资料柜"，看它里面到底有没有 AI。

结论先放这：没有。检索不是"理解"，是**数数 + 算分**。

这个脚本把"红烧肉怎么做才好吃？"这句话在检索器里经历的全过程打印出来：
    1. 它被切成了哪些词
    2. 每个词在你的笔记里出现过几次
    3. 最后那 1.0 分是怎么算出来的

跑法：
    python -m scripts.show_tokens
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import index_paths, load_index  # noqa: E402
from app.services.retriever import KB, _segment  # noqa: E402

QUESTION = "红烧肉怎么做才好吃？"
LINE = "=" * 72


def main() -> None:
    if not load_index():
        index_paths(rebuild=True)

    print(f"\n{LINE}")
    print(f"你的问题：{QUESTION}")
    print(LINE)

    tokens = _segment(QUESTION)
    print(f"\n第 1 步：切词。它把这句话切成了 {len(tokens)} 个词：")
    for t in tokens:
        print(f"    「{t}」")

    print(f"\n第 2 步：数数。每个词在 {len(KB)} 张卡片里出现过几次？")
    matched: list[str] = []
    for t in tokens:
        df = KB._doc_freq.get(t, 0)  # noqa: SLF001  教学脚本，故意直接看内部统计
        mark = "  ← 出现过" if df else "  （一次都没有）"
        print(f"    「{t}」出现在 {df} 张卡片里{mark}")
        if df:
            matched.append(t)

    print(f"\n第 3 步：结论。")
    if not matched:
        print("    一个词都没撞上 → 应该返回'没找到'。")
    else:
        print(f"    只有 {matched} 撞上了，其余的词你的笔记里根本没有。")

    print(f"\n{LINE}")
    print("第 4 步：那它翻出来的卡片，为什么相关度显示 1.0？")
    print(LINE)

    hits = KB.search(QUESTION)
    for h in hits:
        print(
            f"    · {h.chunk.heading or '(无标题)'}\n"
            f"        原始 BM25 分 = {h.bm25_score:.3f}   "
            f"显示相关度 = {h.score}"
        )

    if hits:
        top = max(h.bm25_score for h in hits)
        print(f"\n    最高那张卡片的原始分是 {top:.3f} —— 其实很低。")
        print("    但归一化算法把'这批里最高的那个'强行映射成 1.0，")
        print("    所以一张 0.3 分的垃圾卡片，也会被显示成满分。")
        print("\n    这就是那个 1.0 的来历：它不是'很相关'，它只是'这批里最不烂的'。")


if __name__ == "__main__":
    main()
