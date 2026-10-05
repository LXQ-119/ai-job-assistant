#!/usr/bin/env python3
"""01 - 最小 Agent：让模型自己决定调用工具

这是理解 Agent 的最小完整例子。整个循环只有 5 步：

    1. 把「工具说明书」(JSON Schema) 和用户问题一起发给模型
    2. 模型**不直接回答**，而是返回一个结构化的请求：我要调用 add(a=3, b=5)
    3. 你的代码真正执行 add(3, 5)，算出 8        ← 关键：算数的不是模型，是你的代码
    4. 把 8 作为 tool 消息发回给模型
    5. 模型基于这个结果生成最终回答

这就是 function calling / tool use。所谓 Agent，本质就是把这 5 步放进一个**循环**，
并且允许模型连续调用多个工具、根据上一步结果决定下一步。

为什么这个"加法"例子值得做：
模型自己心算 3+5 也能对，但换成 1234.5678 * 9876.5432 就会开始编答案。
工具调用的意义就是**把不确定的部分交给确定性的代码**——这是所有 Agent 的立足点。

跑法：
    python examples/01_add_agent.py "3 加 5 等于多少"        # 真模型（需要 .env 里的 Key）
    python examples/01_add_agent.py "3 加 5 等于多少" --mock  # 离线假模型，不需要 Key
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

# 允许 `python examples/01_add_agent.py` 直接跑，同时能读到项目根目录的 .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


# ===========================================================================
# 第一部分：工具本身 —— 就是普通的 Python 函数
# ===========================================================================


def add(a: float, b: float) -> float:
    """两个数相加。

    注意：这个函数和"AI"没有任何关系。它就是一段确定的代码。
    Agent 的价值在于——模型知道**什么时候**该调用它。

    为什么要绕一层 Decimal，而不是直接写 a + b：

        0.1 + 0.2 == 0.30000000000000004     ← 二进制浮点的固有误差

    这不是 bug，但用户看到这串数字会以为程序坏了。
    模型发来的是**十进制**数，我们就该用十进制算。金额、数量这类场景尤其必须。
    实测：直接 a + b 时 0.1+0.2 会输出 0.30000000000000004，用 Decimal 后是 0.3。
    """
    total = _to_decimal(a) + _to_decimal(b)
    # 转回 float 是为了让 JSON 能序列化；Decimal 本身不可直接 json.dumps。
    return float(total)


def _to_decimal(value) -> Decimal:
    """把参数转成 Decimal，转换失败时抛出**人能看懂**的错误。

    错误信息的质量直接影响 Agent 能不能自救：
    如果只抛 Decimal 原生的 `[<class 'decimal.ConversionSyntax'>]`，
    模型看不懂就改不对参数；写成"参数必须是数字，收到 a='x'"它才知道该改什么。
    """
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(
            f"参数必须是数字，实际收到 {value!r}（{type(value).__name__}）"
        ) from exc


# 工具注册表：名字 -> 可执行函数。模型说"调用 add"，我们就来这里找。
TOOL_IMPLEMENTATIONS = {"add": add}


# 工具说明书：用 JSON Schema 描述工具有哪些、参数是什么、干什么用的。
# 这段文字会原样发给模型，所以 description 写得越清楚，模型选得越准。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个数字的和。凡是用户需要做加法，都必须调用这个工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个加数"},
                    "b": {"type": "number", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def execute_tool(name: str, arguments_json: str) -> dict:
    """执行模型请求的工具调用。

    关键设计：**出错时返回错误信息，而不是抛异常**。
    因为工具失败时，模型需要看到失败原因才有机会改正参数重试；
    如果直接崩掉，整个 Agent 流程就断了。
    """
    if name not in TOOL_IMPLEMENTATIONS:
        return {"error": f"未知工具：{name}。可用工具：{list(TOOL_IMPLEMENTATIONS)}"}

    try:
        kwargs = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return {"error": f"参数不是合法 JSON：{exc}"}

    if not isinstance(kwargs, dict):
        return {"error": f"参数应该是对象，实际是 {type(kwargs).__name__}"}

    try:
        # 这里是唯一的"执行"动作。不信任模型给的参数，让 Python 自己校验类型。
        return {"result": TOOL_IMPLEMENTATIONS[name](**kwargs)}
    except TypeError as exc:
        return {"error": f"参数不匹配：{exc}"}
    except Exception as exc:  # noqa: BLE001
        # 工具内部任何异常都要转成错误信息回给模型，绝不能让整个 Agent 崩掉。
        # 模型看到错误原因后，有机会改参数重试。
        return {"error": f"工具执行出错（{type(exc).__name__}）：{exc}"}


# ===========================================================================
# 第二部分：模型 —— 真模型 和 离线假模型
# ===========================================================================


def _fmt_number(value) -> str:
    """把 8.0 显示成 8、把 12345678.0 显示成 12345678。

    不用 f"{x:g}" 是因为它会在大数上退化成科学计数法（1.23457e+07），
    对给人看的答案来说反而更难读。
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class MockModel:
    """离线假模型：不需要 API Key、不花钱，用来把循环本身看清楚。

    它只模拟两轮：
      第 1 轮：假装"决定"调用 add，参数从问题里抽出来的两个数字
      第 2 轮：读出 tool 消息里的结果，拼成最终回答

    它的行为是写死的——真正的模型是**自己判断**要不要调工具。
    用假模型的唯一目的是：让你在没有 Key 的时候也能跑通整条链路。
    """

    def __init__(self, question: str) -> None:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", question)
        self.a = float(numbers[0]) if numbers else 0.0
        self.b = float(numbers[1]) if len(numbers) > 1 else 0.0
        self.turn = 0

    def respond(self, messages: list[dict]) -> dict:
        self.turn += 1

        if self.turn == 1:
            # 模型这一轮"决定"调用工具，而不是直接回答
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_mock_0001",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": json.dumps({"a": self.a, "b": self.b}),
                        },
                    }
                ],
            }

        # 第二轮：从消息历史里找出 tool 返回的结果。
        # tool 消息里放的是 JSON 字符串，必须解析出来再展示，
        # 否则会把 {"result": 8.0} 这种原始结构直接糊进答案里。
        value = None
        for message in reversed(messages):
            if message.get("role") == "tool":
                raw = message.get("content") or ""
                try:
                    parsed = json.loads(raw)
                    value = parsed.get("result", raw) if isinstance(parsed, dict) else parsed
                except json.JSONDecodeError:
                    value = raw
                break

        return {
            "content": (
                f"{_fmt_number(self.a)} + {_fmt_number(self.b)} = {_fmt_number(value)}"
            ),
            "tool_calls": None,
        }


def real_model_respond(messages: list[dict], model: str) -> dict:
    """调用真模型。

    和假模型的区别只有一点：**要不要调工具、调哪个、参数是什么，全由模型决定。**
    """
    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "没有读到 LLM_API_KEY。\n"
            "  办法一：把 .env.example 复制成 .env 并填入 Key，然后重跑\n"
            "  办法二：加 --mock 参数，用离线假模型看整个循环"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        timeout=60.0,
        max_retries=0,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_SCHEMAS,   # ← 工具说明书在这里交给模型
        temperature=0,        # 算术任务要的是稳定复现，不是创造力
    )
    message = response.choices[0].message

    tool_calls = None
    if message.tool_calls:
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]

    return {"content": message.content, "tool_calls": tool_calls}


# ===========================================================================
# 第三部分：Agent 主循环 —— 就是那 5 步，放进一个 for
# ===========================================================================


def run_agent(
    question: str,
    *,
    mock: bool = False,
    model: str = "deepseek-flash",
    max_steps: int = 5,
    verbose: bool = True,
) -> str:
    def show(text: str = "") -> None:
        if verbose:
            print(text)

    show("=" * 74)
    show(f"问题：{question}")
    show(f"模式：{'离线假模型（不花钱）' if mock else f'真模型 {model}'}")
    show("=" * 74)

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "你是一个严谨的助手。需要做算术时必须调用提供的工具，"
                "不要自己心算。拿到工具结果后再用中文回答用户。"
            ),
        },
        {"role": "user", "content": question},
    ]

    responder = MockModel(question).respond if mock else (lambda m: real_model_respond(m, model))

    for step in range(1, max_steps + 1):
        show(f"\n──── 第 {step} 轮：把 {len(messages)} 条消息发给模型 ────")
        reply = responder(messages)

        # ---- 情况 A：模型没要求调工具 → 这就是最终回答，循环结束 ----
        if not reply["tool_calls"]:
            show("模型没有请求调用工具，直接给出了回答。")
            show(f"\n最终回答：{reply['content']}")
            return reply["content"] or ""

        # ---- 情况 B：模型要求调用工具 ----
        show(f"模型决定调用 {len(reply['tool_calls'])} 个工具，而不是直接回答：")
        for call in reply["tool_calls"]:
            show(f"    工具名：{call['function']['name']}")
            show(f"    参数  ：{call['function']['arguments']}")

        # 第 3 步：把模型这一轮的"决定"记进历史（assistant 消息带 tool_calls）
        messages.append(
            {
                "role": "assistant",
                "content": reply["content"],
                "tool_calls": reply["tool_calls"],
            }
        )

        # 第 4 步：真正执行，并把结果作为 tool 消息追加进历史
        for call in reply["tool_calls"]:
            result = execute_tool(call["function"]["name"], call["function"]["arguments"])
            show(f"    本地执行结果：{json.dumps(result, ensure_ascii=False)}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # 死循环防护：模型可能一直要求调工具而永远不给最终回答
    raise RuntimeError(
        f"已经连续 {max_steps} 轮都在调用工具，仍未给出最终回答。\n"
        "真实系统里必须加这个上限，否则一次故障就能烧掉大量 token。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="最小 Agent：加法工具调用")
    parser.add_argument("question", nargs="?", default="3 加 5 等于多少？")
    parser.add_argument("--mock", action="store_true", help="用离线假模型，不需要 API Key")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-flash"))
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    try:
        run_agent(args.question, mock=args.mock, model=args.model, max_steps=args.max_steps)
    except RuntimeError as exc:
        print(f"\n[失败] {exc}")
        return 1

    print("\n" + "=" * 74)
    print("回看这 5 步：模型负责「决定调什么」，代码负责「真的算」，")
    print("两者靠一份 JSON Schema 契约连接——这就是 function calling 的全部。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
