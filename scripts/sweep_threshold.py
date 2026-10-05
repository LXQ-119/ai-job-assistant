"""方案 C（中文二元组 + 去标点）配上阈值，重新称一次分数线。

逻辑：
    二元组解决了"该找到的找不到"（小米那个问题从第 5 名升到第 1 名），
    但它让任何问题都能匹配上一点，所以超纲问题再也不会返回空。
    这时候必须靠**阈值**来拦。

这个脚本只做一件事：
    在二元组的分数体系下，把所有正例的最低分、所有负例的最高分排出来，
    看窗口在哪，阈值该划在哪。

跑法：
    python -m scripts.sweep_threshold
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.services.retriever as R  # noqa: E402
from app.services import rag  # noqa: E402
from scripts.compare_fix import TARGET_QUESTION, bigram_clean  # noqa: E402

LINE = "=" * 78


def main() -> None:
    cases = [
        json.loads(line)
        for line in (ROOT / "data/eval/rag_eval_set.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    R._segment = bigram_clean  # type: ignore[assignment]
    rag.index_paths([], rebuild=True)
    KB = R.KB

    pos: list[tuple[float, str]] = []
    neg: list[tuple[float, str]] = []
    for case in cases:
        raw = KB._bm25_scores(case["question"])  # noqa: SLF001
        best = max(raw) if raw else 0.0
        (neg if case["type"] == "negative" else pos).append((best, case["question"]))

    pos.sort()
    neg.sort(reverse=True)

    print(f"\n{LINE}")
    print("二元组方案下的分数分布")
    print(LINE)
    print(f"\n【正例】{len(pos)} 条，最低的 5 条：")
    for s, q in pos[:5]:
        print(f"    {s:>8.3f}   {q}")
    print(f"  ★ 正例最低分（红线）= {pos[0][0]:.3f}")

    print(f"\n【负例】{len(neg)} 条，从高到低：")
    for s, q in neg:
        print(f"    {s:>8.3f}   {q}")
    print(f"  ★ 负例最高分 = {neg[0][0]:.3f}")

    print(f"\n{LINE}")
    print("扫描阈值")
    print(LINE)
    print(f"  {'阈值':>7}{'误杀正例':>10}{'挡住负例':>10}{'拒答率':>9}")
    print("  " + "-" * 40)

    best = None
    t = 0.0
    while t <= max(neg[0][0], pos[0][0]) + 1.0:
        killed = sum(1 for s, _ in pos if s < t)
        blocked = sum(1 for s, _ in neg if s < t)
        if killed == 0:
            best = (t, blocked)
        if t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
            rate = blocked / len(neg)
            print(f"  {t:>7.1f}{killed:>10}{blocked:>10}{rate:>9.1%}")
        t = round(t + 0.5, 2)

    if best:
        print(f"\n  ★ 零误杀前提下，阈值最高可以划到 {best[0]:.1f}")
        print(f"    此时挡住 {best[1]}/{len(neg)} 个负例 "
              f"= 拒答率 {best[1] / len(neg):.1%}")

    # 小米那个问题在新体系下怎么样
    print(f"\n{LINE}")
    print("复查：小米那个问题在二元组方案下排第几")
    print(LINE)
    scores = KB._bm25_scores(TARGET_QUESTION)  # noqa: SLF001
    order = sorted(range(len(KB)), key=lambda i: scores[i], reverse=True)
    for rank, i in enumerate(order[:4], start=1):
        mark = "  ★你的答案★" if "小米" in (KB._chunks[i].heading or "") else ""
        print(f"  {rank}. {scores[i]:>8.3f}  {KB._chunks[i].heading}{mark}")

    # 不恢复：这个脚本跑完就退出，进程结束自动还原


if __name__ == "__main__":
    main()
