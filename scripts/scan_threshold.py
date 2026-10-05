"""阈值扫描：如果给检索加一条"最低分门槛"，代价和收益各是多少？

背景：
    负例（资料里根本没有的问题）现在也会返回内容。
    想加一条"分数低于 X 就不返回"的规则。

    但实测发现正例最低分 4.59、负例最高分 6.73，**区间重叠**，
    所以任何阈值都会有代价。这个脚本把代价算清楚。

    这不是失败的实验 —— 它量化了"纯关键词检索的局限"，
    而这正是后面引入向量检索的理由。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services import rag                       # noqa: E402
from app.services.retriever import KB              # noqa: E402


def main() -> int:
    KB.clear()
    rag.index_paths([], rebuild=True)

    eval_path = PROJECT_ROOT / "data" / "eval" / "rag_eval_set.jsonl"
    cases = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # 先把每道题的原始最高分算出来（只算一次）
    scored: list[tuple[str, str, float]] = []
    for case in cases:
        raw = KB._bm25_scores(case["question"])  # noqa: SLF001
        scored.append((case["type"], case["question"], max(raw) if raw else 0.0))

    positives = [r for r in scored if r[0] != "negative"]
    negatives = [r for r in scored if r[0] == "negative"]

    print("=" * 78)
    print(" 阈值扫描：给检索加一条最低分门槛，代价 vs 收益")
    print("=" * 78)
    print(f" 正例 {len(positives)} 条，负例 {len(negatives)} 条")
    print()
    print(f" {'阈值':>6} | {'正例通过':>10} | {'误杀正例':>10} | {'负例拒绝':>10} | {'漏放负例':>10} | {'综合':>8}")
    print("-" * 78)

    best = None
    thresholds = [0.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0]
    for threshold in thresholds:
        passed = [r for r in positives if r[2] >= threshold]
        killed = [r for r in positives if r[2] < threshold]
        refused = [r for r in negatives if r[2] < threshold]
        leaked = [r for r in negatives if r[2] >= threshold]

        # 综合分：正例通过率 和 负例拒绝率 的调和平均（两者都要好）
        p = len(passed) / len(positives)
        n = len(refused) / len(negatives)
        f1 = 0.0 if (p + n) == 0 else 2 * p * n / (p + n)

        print(
            f" {threshold:>6.1f} | "
            f"{len(passed):>4}/{len(positives):<5} | "
            f"{len(killed):>10} | "
            f"{len(refused):>4}/{len(negatives):<5} | "
            f"{len(leaked):>10} | "
            f"{f1:>8.3f}"
        )

        if best is None or f1 > best[1]:
            best = (threshold, f1, len(killed), len(refused))

    print("=" * 78)
    if best:
        threshold, f1, killed, refused = best
        print(f"\n 综合最优阈值：{threshold:.1f}（F1={f1:.3f}）")
        print(f"   代价：误杀 {killed} 条正例")
        print(f"   收益：正确拒绝 {refused}/{len(negatives)} 条负例")

    # 展示被误杀的正例到底是什么
    if best:
        print(f"\n 阈值 {best[0]:.1f} 会误杀这些正例：")
        for case_type, question, score in sorted(positives, key=lambda r: r[2]):
            if score < best[0]:
                print(f"   [{score:>6.3f}] ({case_type}) {question}")

        print(f"\n 阈值 {best[0]:.1f} 挡不住的负例：")
        for case_type, question, score in sorted(negatives, key=lambda r: -r[2]):
            if score >= best[0]:
                print(f"   [{score:>6.3f}] {question}")

    print()
    print("=" * 78)
    print(" 结论")
    print("=" * 78)
    print(" 单靠分数阈值做不到完美分离 —— 因为负例里的常见词（做、怎么、索引）")
    print(" 也会和文档里的词撞上，产生低分噪音。")
    print()
    print(" 这恰恰说明：**纯关键词检索有天花板**，想再往上走需要别的信号，")
    print(" 比如向量检索（语义相近但用词不同也能匹配）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
