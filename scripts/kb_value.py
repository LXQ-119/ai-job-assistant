"""资料柜到底有没有用？—— 同一批问题，问两遍。

第一遍：不给资料，直接问模型（= 它自己脑子里的知识）
第二遍：给资料，走完整 RAG 流程

然后对比。四道题分成两类，故意这么选的：

    A 类【只有你的资料柜里才有】
        问的是"你的项目""你的笔记" —— 模型不可能知道，
        它只会瞎猜一个数字或者编一段

    B 类【模型自己就知道】
        PyTorch、红烧肉 —— 模型训练时读过无数遍

预期结果：
    A 类：不给资料 → 编；给资料 → 答对并标引用   ← 资料柜的价值在这里
    B 类：不给资料 → 答得很好；给资料 → 反而说"资料里没有"

跑法：
    python -m scripts.kb_value
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm import chat  # noqa: E402
from app.services.rag import answer, index_paths, load_index  # noqa: E402

PLAIN_PROMPT = """你是一个乐于助人的助手。
请直接、完整地回答用户的问题。用中文，控制在 250 字以内。"""

# (类别, 问题)
QUESTIONS = [
    ("只有你资料柜里有", "我给自己定的检索目标是多少？实测是多少？为什么这个数字不能用？"),
    ("只有你资料柜里有", "我的项目里，哪些设计是为了给模型留退路？"),
    ("模型自己就知道", "PyTorch 的自动微分是怎么实现的？"),
    ("模型自己就知道", "红烧肉怎么做才好吃？"),
]

LINE = "=" * 78


def without_kb(question: str) -> str:
    result = chat(
        [
            {"role": "system", "content": PLAIN_PROMPT},
            {"role": "user", "content": question},
        ],
        purpose="no_kb",
    )
    return result.text.strip()


def main() -> None:
    if not load_index():
        index_paths(rebuild=True)

    lines: list[str] = []

    for i, (kind, question) in enumerate(QUESTIONS, start=1):
        print(f"\n{LINE}")
        print(f"【第 {i} 题】{question}")
        print(f"　　类别：{kind}")
        print(LINE)

        no_kb = without_kb(question)
        print("\n───── 不给资料（纯模型自己的知识）─────")
        print(no_kb)

        with_kb = answer(question)
        print(f"\n───── 给了资料（走 RAG，翻到 {with_kb.retrieved} 张卡）─────")
        print(with_kb.answer)

        lines.append(f"{LINE}\n【{kind}】{question}\n{LINE}")
        lines.append("\n--- 不给资料 ---\n" + no_kb)
        lines.append(f"\n--- 给了资料（{with_kb.retrieved} 张卡）---\n" + with_kb.answer + "\n")

    out = ROOT / "资料柜价值对照.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{LINE}")
    print(f"全部记录已存到：{out}")


if __name__ == "__main__":
    main()
