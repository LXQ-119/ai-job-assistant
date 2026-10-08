"""Agent 核心：工具定义 + 决策循环。

供两处共用：
    examples/01_add_agent.py  —— 命令行版，把步骤打印到终端
    ui/pages/1_Agent演示.py   —— 网页版，把步骤渲染成界面

设计要点：核心逻辑产出**结构化的步骤事件**（dict），而不是直接 print。
这样同一份逻辑就能被终端和网页用完全不同的方式展示，而不用写两遍。
这就是"业务逻辑"和"展示层"的分离。
"""

from __future__ import annotations

import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

# 教具脚本设计成能独立运行（不 import app/ 里的东西），所以自己加载 .env。
#
# 这是真实踩到的坑：只有 app/config.py 里调用了 load_dotenv，
# 而这里的 real_model_respond 是直接 os.getenv("LLM_API_KEY") 读环境变量。
# 结果网页上默认开着"真模型"，却一直报"没有读到 LLM_API_KEY" ——
# 明明 .env 里填了 Key。**配置必须只有一个加载入口，否则一定会漏。**
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ===========================================================================
# 工具
# ===========================================================================


def _to_decimal(value) -> Decimal:
    """转 Decimal，失败时给出**人能看懂**的错误。

    错误信息的质量决定模型能不能自救：抛 decimal.ConversionSyntax 它看不懂，
    告诉它"参数必须是数字，实际收到 'x'"它才知道该改什么。
    """
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(
            f"参数必须是数字，实际收到 {value!r}（{type(value).__name__}）"
        ) from exc


def add(a, b) -> float:
    """两个数相加（十进制精确计算，避免 0.1+0.2=0.30000000000000004）。"""
    return float(_to_decimal(a) + _to_decimal(b))


TOOL_IMPLEMENTATIONS = {"add": add}

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
    """执行模型请求的工具调用。出错时返回错误信息，而不是抛异常。"""
    if name not in TOOL_IMPLEMENTATIONS:
        return {"error": f"未知工具：{name}。可用工具：{list(TOOL_IMPLEMENTATIONS)}"}
    try:
        kwargs = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return {"error": f"参数不是合法 JSON：{exc}"}
    if not isinstance(kwargs, dict):
        return {"error": f"参数应该是对象，实际是 {type(kwargs).__name__}"}
    try:
        return {"result": TOOL_IMPLEMENTATIONS[name](**kwargs)}
    except TypeError as exc:
        return {"error": f"参数不匹配：{exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"工具执行出错（{type(exc).__name__}）：{exc}"}


# ===========================================================================
# 模型
# ===========================================================================


def _fmt_number(value) -> str:
    """8.0 显示成 8，12345678.0 显示成 12345678（不要科学计数法）。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class MockModel:
    """离线假模型：不需要 API Key，用来把循环本身看清楚。

    它的"决定"是写死的——真正的模型是**自己判断**要不要调工具。
    """

    def __init__(self, question: str) -> None:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", question)
        self.a = float(numbers[0]) if numbers else 0.0
        self.b = float(numbers[1]) if len(numbers) > 1 else 0.0
        self.turn = 0

    def respond(self, messages: list[dict]) -> dict:
        self.turn += 1
        if self.turn == 1:
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
            "content": f"{_fmt_number(self.a)} + {_fmt_number(self.b)} = {_fmt_number(value)}",
            "tool_calls": None,
        }


def real_model_respond(messages: list[dict], model: str) -> dict:
    """调用真模型。**要不要调工具、调哪个、参数是什么，全由模型决定。**"""
    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "没有读到 LLM_API_KEY。请把 .env.example 复制成 .env 并填入 Key，"
            "或改用离线模式（不需要 Key）。"
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
        tools=TOOL_SCHEMAS,
        temperature=0,
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
# 主循环：产出结构化步骤事件
# ===========================================================================

SYSTEM_PROMPT = (
    "你是一个严谨的助手。需要做算术时必须调用提供的工具，不要自己心算。"
    "拿到工具结果后再用中文回答用户。"
)


def run_agent_steps(
    question: str,
    *,
    mock: bool = False,
    model: str = "deepseek-flash",
    max_steps: int = 5,
) -> Iterator[dict]:
    """跑一次 Agent，逐步 yield 事件。

    事件类型：
        start        开始，带问题与模式
        round        进入第 N 轮
        decision     模型决定调用工具（而不是直接回答）
        tool_result  本地执行工具的结果
        final        最终回答
        error        失败（含超过 max_steps 的死循环防护）
    """
    yield {
        "type": "start",
        "question": question,
        "mode": "mock" if mock else "real",
        "model": model,
    }

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    if mock:
        responder = MockModel(question).respond
    else:
        responder = lambda msgs: real_model_respond(msgs, model)  # noqa: E731

    for step in range(1, max_steps + 1):
        yield {"type": "round", "index": step, "message_count": len(messages)}

        try:
            reply = responder(messages)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": str(exc)}
            return

        # 模型没要求调工具 → 这就是最终回答
        if not reply["tool_calls"]:
            yield {"type": "final", "content": reply["content"] or ""}
            return

        yield {"type": "decision", "calls": reply["tool_calls"]}
        messages.append(
            {
                "role": "assistant",
                "content": reply["content"],
                "tool_calls": reply["tool_calls"],
            }
        )

        for call in reply["tool_calls"]:
            result = execute_tool(call["function"]["name"], call["function"]["arguments"])
            yield {
                "type": "tool_result",
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "result": result,
            }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    yield {
        "type": "error",
        "message": (
            f"已经连续 {max_steps} 轮都在调用工具，仍未给出最终回答。"
            "真实系统里必须加这个上限，否则一次故障就能烧掉大量 token。"
        ),
    }
