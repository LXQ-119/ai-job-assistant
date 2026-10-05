"""验证猜想：分数是不是被标点符号污染的？

猜想来自 show_score.py 的输出：
    "红烧肉怎么做才好吃？" 得分 4.357，其中 3.899 分来自那个「？」
    而另外 5 个负例的分数恰好都是 3.899 —— 就是"光一个问号"的分。

如果猜想成立，那么把标点从分词里剔除之后：
    所有负例的分数会大幅下降，甚至归零
    正例的分数也会一起下降（因为正例问题也带问号）

这个实验不改项目代码，只在内存里临时替换分词函数，跑完就恢复。

跑法：
    python -m scripts.scan_punctuation
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
from app.services.retriever import KB  # noqa: E402

LINE = "=" * 78


def is_noise(token: str) -> bool:
    """是不是纯标点/符号（P=标点 S=符号 Z=空白）。"""
    return bool(token) and all(
        unicodedata.category(ch)[0] in ("P", "S", "Z") for ch in token
    )


def main() -> None:
    eval_path = ROOT / "data/eval/rag_eval_set.jsonl"
    cases = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    original_segment = R._segment  # noqa: SLF001

    # ---- 先看看现在有哪些词是纯标点 ----
    sample = "红烧肉怎么做才好吃？BM25 和向量的区别[1]！"
    print(f"\n样例：{sample}")
    print(f"  现在切出来的词：{original_segment(sample)}")
    noise = [t for t in original_segment(sample) if is_noise(t)]
    print(f"  其中纯标点：{noise}")

    def collect() -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
        pos, neg = [], []
        for case in cases:
            raw = R.KB._bm25_scores(case["question"])  # noqa: SLF001
            best = max(raw) if raw else 0.0
            (neg if case["type"] == "negative" else pos).append((best, case["question"]))
        return pos, neg

    # ---- A 组：现在的样子 ----
    rag.index_paths([], rebuild=True)
    pos_a, neg_a = collect()

    # ---- B 组：把标点从分词里剔掉 ----
    def clean_segment(text: str) -> list[str]:
        return [t for t in original_segment(text) if not is_noise(t)]

    R._segment = clean_segment  # type: ignore[assignment]
    rag.index_paths([], rebuild=True)
    pos_b, neg_b = collect()

    R._segment = original_segment  # type: ignore[assignment]

    # ---- 对比 ----
    for label, pos, neg in (("A 组：现在（标点参与打分）", pos_a, neg_a),
                            ("B 组：剔掉标点", pos_b, neg_b)):
        pos_sorted = sorted(pos)
        neg_sorted = sorted(neg, reverse=True)
        print(f"\n{LINE}")
        print(label)
        print(LINE)
        print(f"  正例最低分 = {pos_sorted[0][0]:.3f}   （{pos_sorted[0][1]}）")
        print(f"  负例最高分 = {neg_sorted[0][0]:.3f}   （{neg_sorted[0][1]}）")
        for score, q in neg_sorted:
            print(f"      负例 {score:>7.3f}   {q}")

    print(f"\n{LINE}")
    print("B 组（剔掉标点）之后，7 个负例里有几个直接归零？")
    print(LINE)
    zeroed = [q for s, q in neg_b if s <= 0.0]
    for s, q in sorted(neg_b, reverse=True):
        mark = "归零 → 直接返回'没找到'" if s <= 0.0 else ""
        print(f"    {s:>7.3f}   {q}  {mark}")
    print(f"\n  一共 {len(zeroed)}/{len(neg_b)} 个负例直接归零")


if __name__ == "__main__":
    main()
