"""对照实验：只改一句提示词，看模型会不会开始编答案。

背景：
    刚才用现在的系统问 4 个超纲问题，模型老老实实回答"资料里没有"。
    凭什么？凭 SYSTEM_PROMPT 里那一句：
        "如果参考资料不足以回答问题，直接回答'现有资料中没有提到这一点'"

    这一句就是模型"说不知道"的许可证。

这个脚本做 A/B 对照：
    A 组：现在的提示词（有那句许可证）
    B 组：只把那一句删掉，换成"尽力帮用户解决问题"
    其他一切不变：同样的知识库、同样的卡片、同样的模型、temperature=0。

跑法：
    python -m scripts.compare_prompt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm import chat  # noqa: E402
from app.services.rag import SYSTEM_PROMPT, _build_context, index_paths, load_index, retrieve  # noqa: E402
from app.services.retriever import KB  # noqa: E402

# B 组用的"坏"提示词：一个新手最可能写出来的那种。
# 注意它没有任何恶意，只是**忘了给模型留退路**。
NAIVE_PROMPT = """你是一个乐于助人的技术助手。

请尽力、完整地解答用户的问题，让用户满意。
用中文回答，条理清晰，控制在 300 字以内。"""

QUESTIONS = [
    "红烧肉怎么做才好吃？",
    "PyTorch 的自动微分是怎么实现的？",
    "Kubernetes 的 Pod 调度算法是怎么工作的？",
]

LINE = "=" * 72


def ask(system_prompt: str, question: str) -> str:
    hits = retrieve(question)
    context, _ = _build_context(hits)
    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{question}"
    result = chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        purpose="compare_prompt",
    )
    return result.text.strip()


def main() -> None:
    if not load_index():
        index_paths(rebuild=True)

    settings = get_settings()
    print(f"模型：{settings.llm_model}　知识库：{len(KB)} 张卡片")

    lines: list[str] = []

    for i, question in enumerate(QUESTIONS, start=1):
        hits = retrieve(question)
        print(f"\n{LINE}")
        print(f"【问题 {i}】{question}")
        print(f"（检索到的卡片 {len(hits)} 张，两组完全相同）")
        print(LINE)

        good = ask(SYSTEM_PROMPT, question)
        print("\n----- A 组：现在的提示词（有一句“资料里没有就说没有”）-----")
        print(good)

        bad = ask(NAIVE_PROMPT, question)
        print("\n----- B 组：只删掉那一句 -----")
        print(bad)

        lines.append(f"{LINE}\n【问题】{question}\n{LINE}")
        lines.append("\n----- A 组：现在的提示词 -----\n" + good)
        lines.append("\n----- B 组：删掉那一句 -----\n" + bad + "\n")

    out = ROOT / "对照记录.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{LINE}")
    print(f"全部记录已存到：{out}")


if __name__ == "__main__":
    main()
