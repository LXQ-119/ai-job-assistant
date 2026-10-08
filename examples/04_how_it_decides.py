"""04 - 决策到底是怎么发生的？把原始对话一字不改地摊开。

先说结论，一句话：

    **模型不能执行任何东西。它唯一的本事是「输出文字」。**

    所谓「Agent 自己做判断」，本质是：
        我们给它一份【工具说明书】（一段 JSON）
        它在需要时，输出的不是普通回答，而是一段**特殊格式的文字**
            {"name": "calculate", "arguments": "{\"expression\": \"3-2\"}"}
        我们的代码看到这段文字，才去**真的执行**工具
        执行结果塞回对话，再问它一次

    **所以「自主」的意思是：我们的代码里没有一行 if 判断"该用哪个工具"。**
    那个判断是模型自己做的 —— 它读了系统提示词和工具描述，然后自己选。

这个脚本把每一步的原始消息打印出来，包括模型返回的 tool_calls 原文。

跑法（文件名以数字开头，不能用 -m）：
    python examples/04_how_it_decides.py
    python examples/04_how_it_decides.py "你的问题"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm import chat  # noqa: E402
from app.services.agentic import (  # noqa: E402
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    execute_tool,
)
from app.services.rag import index_paths, load_index  # noqa: E402

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "3 减 2 等于几？"
LINE = "=" * 78
THIN = "-" * 78


def dump_tools() -> None:
    print(f"\n{LINE}")
    print("第 0 步：我们发给模型的【工具说明书】")
    print(LINE)
    print("\n这段 JSON 和用户问题一起发过去。模型只能靠它知道「有哪些工具、各自干什么」。\n")
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        print(f"  工具名：{fn['name']}")
        print(f"  说明：{fn['description']}")
        print(f"  参数：{json.dumps(fn['parameters'], ensure_ascii=False)}")
        print()


def main() -> None:
    settings = get_settings()
    if not load_index():
        index_paths(rebuild=True)

    dump_tools()

    print(f"{LINE}")
    print("系统提示词（决定它「什么时候该用工具」的规则）")
    print(LINE)
    for line in SYSTEM_PROMPT.splitlines():
        print(f"  {line}")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]

    print(f"\n{LINE}")
    print(f"用户问题：{QUESTION}")
    print(LINE)

    for step in range(1, 5):
        print(f"\n{THIN}")
        print(f"【第 {step} 轮】发给模型 {len(messages)} 条消息")
        print(THIN)
        for index, message in enumerate(messages, start=1):
            role = message["role"]
            preview = (message.get("content") or "")[:150].replace("\n", " ")
            print(f"  {index}. [{role}] {preview}…")

        result = chat(
            messages,
            purpose="how_it_decides",
            model=settings.llm_model,
            temperature=0,
            tools=TOOL_SCHEMAS,
        )

        print(f"\n  ← 模型返回：")
        print(f"     content = {result.text!r}")
        print(f"     tool_calls = ", end="")
        if not result.tool_calls:
            print("None")
        else:
            print()
            print(
                json.dumps(result.tool_calls, ensure_ascii=False, indent=8)[:900]
            )

        if not result.tool_calls:
            print(f"\n  ★ 模型没要求调工具 → 这就是最终回答。循环结束。")
            print(f"\n{LINE}")
            print("最终回答")
            print(LINE)
            print(result.text.strip())
            return

        print(f"\n  ★ 模型要求调工具了 —— 但注意：**它只是「说」，还没有「做」**。")
        messages.append(
            {
                "role": "assistant",
                "content": result.text or None,
                "tool_calls": result.tool_calls,
            }
        )

        for call in result.tool_calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            print(f"\n  → 现在轮到「我们的代码」真的执行：{name}({arguments})")
            payload = execute_tool(name, arguments)
            print(f"     执行结果：{json.dumps(payload, ensure_ascii=False)[:200]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )
            print(f"  → 结果作为一条 [tool] 消息塞回对话，再问模型一次")

    print("\n超过 4 轮仍未结束（这个脚本的上限）。")


if __name__ == "__main__":
    main()
