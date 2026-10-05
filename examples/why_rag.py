"""为什么要做资料柜？—— 同一个问题问两次，对比结果。

这个脚本会问同一个问题两遍：

    第 1 遍：直接问 DeepSeek（不给任何资料）
             → 看它是不是在编，或者干脆答不上来

    第 2 遍：先去资料柜翻出相关卡片，再把卡片一起给 DeepSeek
             → 看它是不是答得准确、还带出处

用法：
    python examples/why_rag.py
    python examples/why_rag.py "你想问的问题"
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings       # noqa: E402
from app.llm import chat                  # noqa: E402
from app.services import rag              # noqa: E402
from app.services.retriever import KB     # noqa: E402

BAR = "=" * 72


def main() -> int:
    settings = get_settings()
    # 默认挑一个「只有你的笔记里才有答案」的问题
    question = sys.argv[1] if len(sys.argv) > 1 else "我实习的时候做了什么？"

    if len(KB) == 0:
        rag.index_paths([], rebuild=True)

    print(BAR)
    print(f" 问题：{question}")
    print(BAR)

    # ------------------------------------------------------------------
    # 第 1 遍：不给资料，直接问
    # ------------------------------------------------------------------
    print()
    print("【第 1 遍】直接问 DeepSeek，不给任何资料")
    print("-" * 72)
    if not settings.llm_api_key:
        print(" 没有 API Key，跳过。")
        return 0

    direct = chat(
        [
            {
                "role": "system",
                "content": "你是一个乐于助人的助手，请尽量回答用户的问题。",
            },
            {"role": "user", "content": question},
        ],
        purpose="demo_direct",
    )
    print(direct.text)
    print()
    print(f" （耗时 {direct.latency_ms:.0f} ms，成本 ${direct.cost_usd:.6f}）")

    # ------------------------------------------------------------------
    # 第 2 遍：先翻资料柜，再问
    # ------------------------------------------------------------------
    print()
    print("【第 2 遍】先翻资料柜，把翻到的卡片一起给 DeepSeek")
    print("-" * 72)
    hits = rag.retrieve(question, settings.top_k)
    print(f" 翻到 {len(hits)} 张卡片：")
    for number, hit in enumerate(hits, start=1):
        print(f"   [{number}] 相关度 {hit.score:.2f}  ← {hit.chunk.heading}")
    print()

    grounded = rag.answer(question, settings.top_k)
    print(grounded.answer)
    print()
    print(f" （耗时 {grounded.latency_ms:.0f} ms，成本 ${grounded.cost_usd:.6f}）")

    # ------------------------------------------------------------------
    print()
    print(BAR)
    print(" 对比一下两个回答的差别：")
    print(BAR)
    print("  第 1 遍：模型只能凭记忆猜 —— 它从没读过你的笔记，")
    print("          所以要么答不上来，要么编一个「听起来合理」的答案。")
    print()
    print("  第 2 遍：先把你的笔记翻出来给它看 —— 它有依据了，")
    print("          答得准，而且每句都标了 [编号] 让你能核对原文。")
    print()
    print(" 这就是「资料柜」存在的全部理由。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
