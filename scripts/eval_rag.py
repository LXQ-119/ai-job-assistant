"""检索效果评测脚本 —— 简历上那些数字就是从这里跑出来的。

用法：
    python -m scripts.eval_rag                      # 用当前 .env 配置评测
    python -m scripts.eval_rag --alpha 0 0.3 0.7 1  # 对比不同混合权重
    python -m scripts.eval_rag --with-llm           # 额外评答案质量（会花钱）

它产出三个指标：
1. Hit@k     —— 正确来源是否出现在前 k 条里（检索有没有捞到）
2. MRR       —— 正确来源排在第几位的倒数均值（排得够不够靠前）
3. 关键词覆盖率 —— 标准答案里的关键术语在检索内容里出现了多少（捞得全不全）

**为什么必须有这个脚本**：
没有它，"效果不错"就只是一句主观感受；有了它，
"命中率从 62% 提升到 89%"才是有依据的结论——而后者才是面试官想听的。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 允许 `python scripts/eval_rag.py` 和 `python -m scripts.eval_rag` 两种跑法
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_EVAL_FILE = "data/eval/rag_eval_set.jsonl"


def load_eval_set(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"评测集第 {line_number} 行不是合法 JSON：{exc}") from exc
    return cases


def rebuild_settings(alpha: float):
    """改 HYBRID_ALPHA 后清掉配置缓存，让新值立刻生效。

    因为 get_settings() 用了 lru_cache，不清理的话改环境变量不会起作用——
    这是 Python 里很典型的一个坑。
    """
    from app.config import get_settings

    os.environ["HYBRID_ALPHA"] = str(alpha)
    get_settings.cache_clear()
    return get_settings()


def evaluate(alpha: float, with_llm: bool, eval_cases: list[dict], top_k: int) -> dict:
    from app.services import rag
    from app.services.retriever import KB

    settings = rebuild_settings(alpha)

    # 每次评测都重建索引，保证不同配置之间可比
    KB.clear()
    rag.index_paths([], rebuild=True)

    hits = 0
    reciprocal_ranks: list[float] = []
    coverage: list[float] = []
    answer_coverage: list[float] = []
    latencies: list[float] = []
    total_cost = 0.0
    details: list[dict] = []

    for case in eval_cases:
        question = case["question"]
        expected_source = case.get("expected_source")
        must_contain = case.get("must_contain", [])

        started = time.perf_counter()
        retrieved = rag.retrieve(question, top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        sources = [hit.chunk.source for hit in retrieved]
        rank = None
        if expected_source and expected_source in sources:
            hits += 1
            rank = sources.index(expected_source) + 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        # 检索到的内容里覆盖了多少标准关键词
        context = "\n".join(hit.chunk.text for hit in retrieved)
        if must_contain:
            found = sum(1 for term in must_contain if term in context)
            coverage.append(found / len(must_contain))

        answer_text = ""
        if with_llm:
            result = rag.answer(question, top_k)
            answer_text = result.answer
            total_cost += result.cost_usd
            if must_contain:
                found = sum(1 for term in must_contain if term in answer_text)
                answer_coverage.append(found / len(must_contain))

        details.append(
            {
                "question": question,
                "expected_source": expected_source,
                "rank": rank,
                "top_sources": sources,
                "top_scores": [round(hit.score, 4) for hit in retrieved],
                "answer": answer_text[:400] if answer_text else "",
            }
        )

    total = len(eval_cases) or 1
    return {
        "alpha": alpha,
        "vector_enabled": KB.vector_enabled,
        "top_k": top_k,
        "cases": len(eval_cases),
        "hit_rate": round(hits / total, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
        "keyword_coverage": round(sum(coverage) / len(coverage), 4) if coverage else None,
        "answer_keyword_coverage": (
            round(sum(answer_coverage) / len(answer_coverage), 4) if answer_coverage else None
        ),
        "avg_retrieval_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "llm_cost_usd": round(total_cost, 6),
        "embedding_backend": rebuild_settings(alpha).embedding_backend,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评测检索效果")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="*",
        default=None,
        help="要对比的 HYBRID_ALPHA 取值列表，例如 --alpha 0 0.5 1",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--with-llm", action="store_true", help="额外评测生成答案（会产生 API 费用）")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = Path(__file__).resolve().parent.parent / eval_path
    if not eval_path.exists():
        print(f"找不到评测集：{eval_path}")
        return 1

    cases = load_eval_set(eval_path)
    alphas = args.alpha if args.alpha else [float(os.getenv("HYBRID_ALPHA", "0"))]

    print(f"评测集：{eval_path.name}，共 {len(cases)} 条；top_k={args.top_k}")
    print("=" * 78)
    print(f"{'alpha':>6} | {'Hit@k':>7} | {'MRR':>6} | {'关键词覆盖':>10} | {'平均检索耗时':>12} | {'向量':>4}")
    print("-" * 78)

    reports: list[dict] = []
    for alpha in alphas:
        try:
            report = evaluate(alpha, args.with_llm, cases, args.top_k)
        except Exception as exc:  # noqa: BLE001 - 评测脚本要把错误说清楚而不是抛栈
            print(f"{alpha:>6} | 评测失败：{exc}")
            continue

        reports.append(report)
        coverage = report["keyword_coverage"]
        print(
            f"{alpha:>6} | {report['hit_rate']:>7.2%} | {report['mrr']:>6.3f} | "
            f"{(f'{coverage:.2%}' if coverage is not None else '—'):>10} | "
            f"{report['avg_retrieval_ms']:>10.1f} ms | "
            f"{'是' if report['vector_enabled'] else '否':>4}"
        )

    print("=" * 78)

    if reports:
        best = max(reports, key=lambda item: (item["hit_rate"], item["mrr"]))
        print(f"最佳配置：alpha={best['alpha']}，Hit@k={best['hit_rate']:.2%}，MRR={best['mrr']:.3f}")
        if len(reports) > 1:
            worst = min(reports, key=lambda item: (item["hit_rate"], item["mrr"]))
            delta = best["hit_rate"] - worst["hit_rate"]
            if delta > 0:
                print(
                    f"对比最差配置（alpha={worst['alpha']}, Hit@k={worst['hit_rate']:.2%}）"
                    f"提升 {delta:.2%} —— 这个数字可以直接写进简历。"
                )

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(), "reports": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"详细报告已写入：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
