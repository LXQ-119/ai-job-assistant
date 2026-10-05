"""02 - 多工具 Agent（命令行版）

和 01 比，多了三个能力：**选工具、处理失败、连续调用**。

用法：
    # 基本：看模型自己选对工具
    python examples/02_multi_tool_agent.py "10 减 3 等于多少"

    # 换个说法，看它能不能听懂
    python examples/02_multi_tool_agent.py "我买了 3 个苹果，吃掉了 2 个，还剩几个？"

    # 工具失败：看模型怎么处理除以零
    python examples/02_multi_tool_agent.py "100 除以 0 等于多少"

    # 连续调用：先算一步，再用结果算第二步
    python examples/02_multi_tool_agent.py "先算 3 加 5，再把结果乘以 2"

    # 不需要 API Key 的离线模式（假模型只会最简单的说法）
    python examples/02_multi_tool_agent.py "10 减 3" --mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from multi_tool_core import TOOL_SCHEMAS, run_agent_steps  # noqa: E402

BAR = "=" * 74


def main() -> int:
    parser = argparse.ArgumentParser(description="多工具 Agent（四则运算）")
    parser.add_argument("question", nargs="?", default="10 减 3 等于多少？")
    parser.add_argument("--mock", action="store_true", help="离线假模型，不需要 API Key")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    print(BAR)
    print(f" 问题：{args.question}")
    print(f" 模式：{'离线假模型（不花钱）' if args.mock else f'真模型 {args.model}'}")
    print(BAR)

    print(f"\n 这次给了模型 {len(TOOL_SCHEMAS)} 个工具，它得自己挑：")
    for schema in TOOL_SCHEMAS:
        function = schema["function"]
        print(f"   - {function['name']:9} {function['description'].split('。')[0]}")

    tool_call_count = 0
    errors: list[str] = []

    for event in run_agent_steps(
        args.question, mock=args.mock, model=args.model, max_steps=args.max_steps
    ):
        kind = event["type"]

        if kind == "round":
            print(f"\n──── 第 {event['index']} 轮：把 {event['message_count']} 条消息发给模型 ────")

        elif kind == "decision":
            tool_call_count += len(event["calls"])
            print(f" 模型决定调用 {len(event['calls'])} 个工具：")
            for call in event["calls"]:
                print(f"     选中的工具：{call['function']['name']}")
                print(f"     它填的参数：{call['function']['arguments']}")

        elif kind == "tool_result":
            payload = event["result"]
            if "error" in payload:
                errors.append(payload["error"])
                print(f"     ⚠️  工具执行失败：{payload['error']}")
                print("         → 这个错误会原样交给模型，看它怎么处理")
            else:
                print(f"     ✅ 本地执行结果：{json.dumps(payload, ensure_ascii=False)}")

        elif kind == "final":
            print(f"\n最终回答：{event['content']}")

        elif kind == "error":
            print(f"\n[失败] {event['message']}")
            return 1

    print()
    print(BAR)
    print(f" 本次统计：调用工具 {tool_call_count} 次，其中失败 {len(errors)} 次")
    print(BAR)
    print(" 回头看这三件事，就是 02 比 01 多出来的全部价值：")
    print("   1. 模型在 4 个工具里**自己选**了正确的那个")
    print("   2. 工具失败时，它**看懂错误**并用中文向你解释")
    print("   3. 复杂任务里它会**连续调用**多次（先算一步，再用结果算下一步）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
