"""检索效果评测脚本 —— 简历上那些数字就是从这里跑出来的。

用法：
    python -m scripts.eval_rag                        # 用当前 .env 配置评测
    python -m scripts.eval_rag --alpha 0 0.3 0.5 1    # 对比不同混合权重
    python -m scripts.eval_rag --with-llm             # 额外评答案质量（会花钱）

────────────────────────────────────────────────────────────────────────
评测集长什么样（data/eval/rag_eval_set.jsonl，每行一个 JSON）
────────────────────────────────────────────────────────────────────────
    {"id": 1, "type": "paraphrase", "question": "...",
     "expected_sources": ["xxx.md"], "must_contain": ["关键词1", "关键词2"]}

**三类题目，分别考不同的能力**（这是评测集有没有区分度的关键）：

    paraphrase   改写问法 —— 不出现标题原词，考"能不能听懂人话"
    multi_hop    跨小节 —— 答案需要综合多处，考"召回够不够全"
    negative     负例   —— 资料里根本没有，考"会不会硬编"

────────────────────────────────────────────────────────────────────────
四个指标
────────────────────────────────────────────────────────────────────────
    1. Hit@k      正确来源是否出现在前 k 条（检索有没有捞到）
    2. MRR        正确来源排在第几位的倒数均值（排得够不够靠前）
    3. 关键词覆盖率 标准答案里的术语在检索内容里出现了多少（捞得全不全）
    4. 正确拒答率  负例里检索返回"空"的比例（该说没有时有没有说没有）

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

TYPE_LABELS = {
    "paraphrase": "改写问法",
    "multi_hop": "跨小节",
    "negative": "负例(答不了)",
}


def load_eval_set(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"评测集第 {line_number} 行不是合法 JSON：{exc}") from exc

        # 兼容旧格式：expected_source（单个字符串）→ expected_sources（列表）
        if "expected_sources" not in case:
            legacy = case.get("expected_source")
            case["expected_sources"] = [legacy] if legacy else []
        case.setdefault("type", "paraphrase")
        case.setdefault("must_contain", [])
        cases.append(case)

    unknown = {c["type"] for c in cases} - set(TYPE_LABELS)
    if unknown:
        raise SystemExit(f"评测集里有未知的 type：{unknown}（只支持 {list(TYPE_LABELS)}）")
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

    rebuild_settings(alpha)

    # 每次评测都重建索引，保证不同配置之间可比
    KB.clear()
    rag.index_paths([], rebuild=True)

    # 按题目类型分桶统计
    buckets: dict[str, dict] = {
        name: {"n": 0, "hits": 0, "rr": [], "coverage": [], "answer_coverage": [], "refused": 0}
        for name in TYPE_LABELS
    }
    latencies: list[float] = []
    total_cost = 0.0
    details: list[dict] = []

    for case in eval_cases:
        case_type = case["type"]
        bucket = buckets[case_type]
        bucket["n"] += 1

        question = case["question"]
        expected_sources = case["expected_sources"]
        must_contain = case["must_contain"]

        started = time.perf_counter()
        retrieved = rag.retrieve(question, top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        sources = [hit.chunk.source for hit in retrieved]
        context = "\n".join(hit.chunk.text for hit in retrieved)

        rank: int | None = None
        if case_type == "negative":
            # 负例：期望检索返回空（说明系统知道"这里没有相关内容"）
            if not retrieved:
                bucket["refused"] += 1
        else:
            # 正例：期望来源出现在前 top_k 条里
            for position, source in enumerate(sources, start=1):
                if source in expected_sources:
                    rank = position
                    bucket["hits"] += 1
                    break
            bucket["rr"].append(1.0 / rank if rank else 0.0)

            if must_contain:
                found = sum(1 for term in must_contain if term in context)
                bucket["coverage"].append(found / len(must_contain))

        answer_text = ""
        if with_llm:
            result = rag.answer(question, top_k)
            answer_text = result.answer
            total_cost += result.cost_usd
            if must_contain:
                found = sum(1 for term in must_contain if term in answer_text)
                bucket["answer_coverage"].append(found / len(must_contain))

        details.append(
            {
                "id": case.get("id"),
                "type": case_type,
                "question": question,
                "expected_sources": expected_sources,
                "rank": rank,
                "retrieved": len(retrieved),
                "top_sources": sources,
                "top_scores": [round(hit.score, 4) for hit in retrieved],
                "answer": answer_text[:400] if answer_text else "",
            }
        )

    # ---- 汇总 ----
    def summarize(bucket: dict) -> dict:
        n = bucket["n"]
        positive_n = n - (bucket["refused"] if False else 0)
        return {
            "题目数": n,
            "Hit@k": round(bucket["hits"] / n, 4) if n and bucket["rr"] else None,
            "MRR": round(sum(bucket["rr"]) / len(bucket["rr"]), 4) if bucket["rr"] else None,
            "关键词覆盖率": (
                round(sum(bucket["coverage"]) / len(bucket["coverage"]), 4)
                if bucket["coverage"]
                else None
            ),
            "正确拒答率": round(bucket["refused"] / n, 4) if n and not bucket["rr"] else None,
        }

    per_type = {name: summarize(buckets[name]) for name in TYPE_LABELS}

    positive_cases = [c for c in eval_cases if c["type"] != "negative"]
    negative_cases = [c for c in eval_cases if c["type"] == "negative"]

    all_rr = buckets["paraphrase"]["rr"] + buckets["multi_hop"]["rr"]
    all_hits = buckets["paraphrase"]["hits"] + buckets["multi_hop"]["hits"]
    all_refused = buckets["negative"]["refused"]

    return {
        "alpha": alpha,
        "vector_enabled": KB.vector_enabled,
        "top_k": top_k,
        "cases": len(eval_cases),
        "positive_cases": len(positive_cases),
        "negative_cases": len(negative_cases),
        "hit_rate": round(all_hits / len(positive_cases), 4) if positive_cases else None,
        "mrr": round(sum(all_rr) / len(all_rr), 4) if all_rr else None,
        "refusal_rate": (
            round(all_refused / len(negative_cases), 4) if negative_cases else None
        ),
        "avg_retrieval_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "llm_cost_usd": round(total_cost, 6),
        "embedding_backend": rebuild_settings(alpha).embedding_backend,
        "per_type": per_type,
        "details": details,
    }


def print_table(reports: list[dict]) -> None:
    print(f"{'alpha':>6} | {'Hit@k':>7} | {'MRR':>6} | {'拒答率':>7} | {'检索耗时':>9} | {'向量':>4}")
    print("-" * 62)
    for report in reports:
        hit = report["hit_rate"]
        mrr = report["mrr"]
        refusal = report["refusal_rate"]
        print(
            f"{report['alpha']:>6} | "
            f"{(f'{hit:.2%}' if hit is not None else '—'):>7} | "
            f"{(f'{mrr:.3f}' if mrr is not None else '—'):>6} | "
            f"{(f'{refusal:.2%}' if refusal is not None else '—'):>7} | "
            f"{report['avg_retrieval_ms']:>6.1f} ms | "
            f"{'是' if report['vector_enabled'] else '否':>4}"
        )


def print_breakdown(report: dict) -> None:
    print(f"\n按题目类型拆分（alpha={report['alpha']}）：")
    print(f"  {'类型':<16} {'题数':>4} {'Hit@k':>8} {'MRR':>7} {'关键词覆盖':>10} {'拒答率':>8}")
    print("  " + "-" * 60)
    for type_name, label in TYPE_LABELS.items():
        stats = report["per_type"][type_name]
        hit = stats["Hit@k"]
        mrr = stats["MRR"]
        cov = stats["关键词覆盖率"]
        refusal = stats["正确拒答率"]
        print(
            f"  {label:<16} {stats['题目数']:>4} "
            f"{(f'{hit:.2%}' if hit is not None else '—'):>8} "
            f"{(f'{mrr:.3f}' if mrr is not None else '—'):>7} "
            f"{(f'{cov:.2%}' if cov is not None else '—'):>10} "
            f"{(f'{refusal:.2%}' if refusal is not None else '—'):>8}"
        )


def print_failures(report: dict, limit: int = 8) -> None:
    """列出没命中的题目 —— 这才是优化的方向。"""
    failed = [
        d for d in report["details"]
        if d["type"] != "negative" and d["rank"] is None
    ]
    missed_negative = [
        d for d in report["details"] if d["type"] == "negative" and d["retrieved"] > 0
    ]
    if not failed and not missed_negative:
        print("\n✅ 所有题目都通过了。")
        return

    if failed:
        print(f"\n❌ 没命中的正例（{len(failed)} 条）—— 这是要优化的地方：")
        for item in failed[:limit]:
            print(f"   [{item['type']}] {item['question']}")
            print(f"        期望: {item['expected_sources']}")
            print(f"        实际捞到: {item['top_sources'][:3]}")
        if len(failed) > limit:
            print(f"   …… 还有 {len(failed) - limit} 条，见报告文件")

    if missed_negative:
        print(f"\n⚠️  负例误召回（{len(missed_negative)} 条）—— 资料里没有，却捞到了内容：")
        for item in missed_negative[:limit]:
            print(f"   {item['question']}")
            print(f"        捞到: {item['top_sources'][:2]}（相关度 {item['top_scores'][:2]}）")


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
    parser.add_argument("--no-details", action="store_true", help="报告里不写逐题明细（文件更小）")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = Path(__file__).resolve().parent.parent / eval_path
    if not eval_path.exists():
        print(f"找不到评测集：{eval_path}")
        return 1

    cases = load_eval_set(eval_path)
    counts = {name: sum(1 for c in cases if c["type"] == name) for name in TYPE_LABELS}

    print(f"评测集：{eval_path.name}")
    print(f"  共 {len(cases)} 条 ｜ " + " ｜ ".join(
        f"{TYPE_LABELS[k]} {v} 条" for k, v in counts.items()
    ))
    print(f"  top_k = {args.top_k}")
    print("=" * 62)

    alphas = args.alpha if args.alpha else [float(os.getenv("HYBRID_ALPHA", "0"))]

    reports: list[dict] = []
    for alpha in alphas:
        try:
            report = evaluate(alpha, args.with_llm, cases, args.top_k)
        except Exception as exc:  # noqa: BLE001 - 评测脚本要把错误说清楚而不是抛栈
            print(f"{alpha:>6} | 评测失败：{type(exc).__name__}: {exc}")
            continue
        reports.append(report)

    print_table(reports)

    if reports:
        # 展示最优配置的详细拆分
        best = max(
            reports,
            key=lambda r: ((r["hit_rate"] or 0), (r["refusal_rate"] or 0)),
        )
        print_breakdown(best)
        print_failures(best)

        if len(reports) > 1:
            worst = min(reports, key=lambda r: (r["hit_rate"] or 0))
            delta = (best["hit_rate"] or 0) - (worst["hit_rate"] or 0)
            print(f"\n最佳配置 alpha={best['alpha']}（Hit@k={best['hit_rate']:.2%}）")
            if delta > 0:
                print(
                    f"对比最差 alpha={worst['alpha']}（Hit@k={worst['hit_rate']:.2%}）"
                    f"提升 {delta:.2%} —— 这个数字可以直接写进简历。"
                )

    # ---- 写报告 ----
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "eval_file": eval_path.name,
        "top_k": args.top_k,
        "with_llm": args.with_llm,
        "reports": [
            {k: v for k, v in r.items() if k != "details" or not args.no_details}
            for r in reports
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细报告已写入：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
