"""同一个问题，两条路 —— 证明"模型能答"和"你项目让它答"是两回事。

路径 A：直接调 DeepSeek（绕过你的检索代码）
        → 只有问题，没有资料。这就是 DeepSeek API 本身的能力。

路径 B：走你项目的完整 RAG 链路（app/services/rag.py 的 answer）
        → 先检索资料柜，检索为空就直接返回，不调模型。

跑法：
    python -m scripts.two_paths
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm import chat  # noqa: E402
from app.services.rag import answer, index_paths, load_index  # noqa: E402

PLAIN_PROMPT = "你是一个乐于助人的助手。直接用中文回答问题，控制在 80 字以内。"

QUESTIONS = [
    "李白是哪个朝代的诗人？",
    "3 减 2 等于几？",
    "我给自己定的检索目标是多少？",
]

LINE = "=" * 78


def main() -> None:
    if not load_index():
        index_paths(rebuild=True)

    for question in QUESTIONS:
        print(f"\n{LINE}")
        print(f"问：{question}")
        print(LINE)

        # ---- 路径 A：绕过你的检索代码，直接问模型 ----
        result_a = chat(
            [
                {"role": "system", "content": PLAIN_PROMPT},
                {"role": "user", "content": question},
            ],
            purpose="two_paths_a",
        )
        print(f"\n【路径 A】直接调 DeepSeek（不经过你的检索代码）")
        print(f"  {result_a.text.strip()}")
        print(f"  耗时 {result_a.latency_ms:.0f} ms ｜ 花费 ${result_a.cost_usd:.6f}")

        # ---- 路径 B：走你项目的完整 RAG 链路 ----
        result_b = answer(question)
        print(f"\n【路径 B】走你项目的 RAG 链路（先检索资料柜）")
        print(f"  翻到 {result_b.retrieved} 张卡")
        print(f"  {result_b.answer.strip()[:200]}")
        print(f"  耗时 {result_b.latency_ms} ms ｜ 花费 ${result_b.cost_usd:.6f}")

    print(f"\n{LINE}")
    print("读法：")
    print("  路径 A 是 DeepSeek 本身的能力 —— 你说得对，它什么都能答。")
    print("  路径 B 是你项目的行为 —— 检索为空时它根本不叫模型。")
    print("  两条路的差别，是你写在 rag.py 里的那几行代码造成的。")


if __name__ == "__main__":
    main()
