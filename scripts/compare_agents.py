"""为什么"你是谁"只有多工具 Agent 答得出来？

两个教具的核心循环代码几乎一样，差别只在**网页上的一个默认开关**：

    ui/pages/1_Agent演示.py     use_real 默认 False  →  默认用「离线假模型」
    ui/pages/2_多工具Agent.py    use_real 默认 True   →  默认用「真模型」

这个脚本把三种组合都跑一遍，看实际输出。

跑法：
    python -m scripts.compare_agents
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

import agent_core  # noqa: E402
import multi_tool_core  # noqa: E402

QUESTION = "你好，你是谁？"
LINE = "=" * 78


def run(label: str, runner, *, mock: bool) -> None:
    print(f"\n{LINE}")
    print(f"【{label}】")
    print(f"  问题：{QUESTION}")
    print(f"  假模型模式：{'是（不花钱、不联网）' if mock else '否（真模型）'}")
    print(LINE)

    final = None
    error = None
    tools_used: list[str] = []

    for event in runner(QUESTION, mock=mock, model="deepseek-flash", max_steps=5):
        kind = event["type"]
        if kind == "round":
            print(f"  ── 第 {event['index']} 轮（发给模型 {event['message_count']} 条消息）")
        elif kind == "decision":
            for call in event["calls"]:
                name = call["function"]["name"]
                tools_used.append(name)
                print(f"     模型决定调用工具：{name}  参数 {call['function']['arguments']}")
        elif kind == "tool_result":
            payload = event["result"]
            shown = payload.get("result", payload.get("error"))
            print(f"     本地执行结果：{event['name']}({event['arguments']}) → {shown}")
        elif kind == "final":
            final = event["content"]
        elif kind == "error":
            error = event["message"]

    print()
    if error:
        print(f"  ❌ 出错：{error}")
    if final is not None:
        print(f"  ✅ 最终回答：{final}")
    print(f"  （这一轮共调用工具 {len(tools_used)} 次：{tools_used or '无'}）")


def main() -> None:
    run("单工具 Agent · 假模型（这是网页的默认状态）", agent_core.run_agent_steps, mock=True)
    run("单工具 Agent · 真模型（你把侧边栏开关打开之后）", agent_core.run_agent_steps, mock=False)
    run("多工具 Agent · 真模型（这是网页的默认状态）", multi_tool_core.run_agent_steps, mock=False)

    print(f"\n{LINE}")
    print("结论")
    print(LINE)
    print("  同一个问题、几乎一样的循环代码，结果不同 —— 差别只在那个开关。")
    print("  假模型不'判断'问题，它第 1 轮无条件调用 add 工具，")
    print("  所以问它'你是谁'，它会回答一个加法算式。")


if __name__ == "__main__":
    main()
