"""诊断：正例和负例的原始 BM25 分数各是多少？

目的：为"最低分阈值"找一个有依据的数值。

为什么需要这个：
    负例（资料里根本没有的问题）现在也会返回内容，而且归一化后显示相关度 1.0。
    要修，就不能拍脑袋定阈值 —— 得先看真实分数分布，
    看正例和负例在原始分数上到底有没有可分离的界线。

这里看的是 **归一化之前的原始 BM25 分数**，因为归一化会把最高分强行拉到 1.0。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services import rag                              # noqa: E402
from app.services.retriever import KB, _segment           # noqa: E402


def main() -> int:
    # 重建索引
    KB.clear()
    rag.index_paths([], rebuild=True)

    eval_path = PROJECT_ROOT / "data" / "eval" / "rag_eval_set.jsonl"
    cases = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows: list[tuple[str, str, float, int, list[str]]] = []
    for case in cases:
        question = case["question"]
        scores = KB._bm25_scores(question)  # noqa: SLF001 - 诊断脚本，看内部分数
        top = max(scores) if scores else 0.0
        matched = [s for s in scores if s > 0]
        rows.append((case["type"], question, top, len(matched), _segment(question)))

    print("=" * 84)
    print(" 原始 BM25 最高分分布（分数越高 = 越相关）")
    print("=" * 84)
    print(f" {'类型':<12} {'最高分':>8} {'命中块数':>8}  问题")
    print("-" * 84)
    for case_type, question, top, matched, _ in sorted(rows, key=lambda r: -r[2]):
        print(f" {case_type:<12} {top:>8.3f} {matched:>8}  {question[:42]}")

    # 按类型汇总
    print()
    print("=" * 84)
    print(" 分组统计")
    print("=" * 84)
    for case_type in ("paraphrase", "multi_hop", "negative"):
        group = [r for r in rows if r[0] == case_type]
        if not group:
            continue
        scores = sorted(r[2] for r in group)
        matched_counts = [r[3] for r in group]
        print(f"\n {case_type}（{len(group)} 条）")
        print(f"   最高分: 最低 {scores[0]:.3f} ｜ 中位 {scores[len(scores)//2]:.3f} ｜ 最高 {scores[-1]:.3f}")
        print(f"   命中块数: 最少 {min(matched_counts)} ｜ 最多 {max(matched_counts)}")

    # 找分界线
    pos = sorted(r[2] for r in rows if r[0] != "negative")
    neg = sorted(r[2] for r in rows if r[0] == "negative")
    print()
    print("=" * 84)
    print(" 能不能用一条分数线分开正例和负例？")
    print("=" * 84)
    print(f"   正例最低分: {pos[0]:.3f}")
    print(f"   负例最高分: {neg[-1]:.3f}")
    if pos[0] > neg[-1]:
        mid = (pos[0] + neg[-1]) / 2
        print(f"   ✅ 可以分开！分界线取中间值 ≈ {mid:.3f}")
        print(f"      阈值设成 {mid:.2f}：正例全过、负例全拦")
    else:
        print("   ❌ 分数区间重叠，单靠分数分不开")
        print(f"      正例里有低到 {pos[0]:.3f} 的，负例里有高到 {neg[-1]:.3f} 的")
        print("      说明这些词在两边都出现了，需要换思路（比如看命中块数、或用 IDF 加权）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
