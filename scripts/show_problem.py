"""抓现场：把"模型自己编答案"这件事，用你自己的知识库现场跑出来。

这个脚本不做任何评测、不算任何指标。它只干一件事：
    拿几个**你的知识库里根本没有**的问题，去问你的系统，
    然后把「检索到了什么」和「模型回答了什么」原样打印出来。

跑法：
    python -m scripts.show_problem
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import answer, index_paths, load_index  # noqa: E402
from app.services.retriever import KB  # noqa: E402

# 这几个问题，你的知识库里一个字都没有。
# 它们对你来说是"外行话"，对你的系统来说是"超纲题"。
QUESTIONS = [
    "红烧肉怎么做才好吃？",
    "PyTorch 的自动微分是怎么实现的？",
    "我的银行卡密码是多少？",
    "Kubernetes 的 Pod 调度算法是怎么工作的？",
]

LINE = "=" * 72


def main() -> None:
    if not load_index():
        print("磁盘上没有索引，先建一次……")
        index_paths(rebuild=True)

    print(f"\n知识库里现在有 {len(KB)} 张卡片，来自 {len(KB.sources)} 个文件")
    for name in KB.sources:
        print(f"  - {name}")

    lines: list[str] = []

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n{LINE}")
        print(f"【第 {i} 个问题】{question}")
        print(LINE)

        result = answer(question)

        print(f"\n>>> 系统翻到的卡片（{result.retrieved} 张）：")
        if not result.citations:
            print("    （一张都没翻到）")
        for c in result.citations:
            print(f"    · {c.source} / {c.heading}   相关度={c.score}")

        print(f"\n>>> 系统给出的回答：")
        print(result.answer)
        print(f"\n>>> 这次回答花了：{result.latency_ms} ms")

        lines.append(f"{LINE}\n【问题】{question}\n")
        lines.append(f"翻到的卡片：{result.retrieved} 张")
        for c in result.citations:
            lines.append(f"  - {c.source} / {c.heading}  (相关度 {c.score})")
        lines.append(f"\n回答：\n{result.answer}\n")

    out = ROOT / "现场记录.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{LINE}")
    print(f"全部记录已存到：{out}")


if __name__ == "__main__":
    main()
