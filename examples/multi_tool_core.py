"""02 - 多工具 Agent：让模型在四个工具里自己选，并且能处理失败。

和 01 的区别 —— 这就是这一课的全部价值：

    01：只有 1 个工具 add
        → 模型没有"选择"这个动作，只证明了"会调用工具"

    02：有 4 个工具（加减乘除）
        → 模型必须**判断该用哪个** —— 这才是真正的决策
        → 工具**会失败**（除以零）—— 模型必须看懂错误并处理
        → 能**连续调用**（先算 A，再用 A 的结果算 B）

这三个能力，是所有真实 Agent 的地基：

    选工具  →  处理失败  →  多步编排
    （本文）   （本文）     （本文已支持，多轮循环天然支持）
"""

from __future__ import annotations

import json
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Iterator

# ===========================================================================
# 四个工具
# ===========================================================================


def _to_decimal(value, name: str) -> Decimal:
    """把参数转成 Decimal，失败时给出**人能看懂**的错误。

    错误信息的质量直接决定模型能不能自救。
    抛 decimal 原生的 `[<class 'decimal.ConversionSyntax'>]`，模型看不懂；
    写成"参数 a 必须是数字，实际收到 '十'" 它才知道该改什么。
    """
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(
            f"参数 {name} 必须是数字，实际收到 {value!r}（{type(value).__name__}）"
        ) from exc


def add(a, b) -> float:
    """加法。用 Decimal 保证十进制精确（0.1 + 0.2 = 0.3，不是 0.30000000000000004）。"""
    return float(_to_decimal(a, "a") + _to_decimal(b, "b"))


def subtract(a, b) -> float:
    """减法。"""
    return float(_to_decimal(a, "a") - _to_decimal(b, "b"))


def multiply(a, b) -> float:
    """乘法。"""
    return float(_to_decimal(a, "a") * _to_decimal(b, "b"))


def divide(a, b) -> float:
    """除法。

    注意这里的**错误处理**：除数为零时抛出一个含义清楚的错误，
    而不是让 Python 抛 ZeroDivisionError。

    为什么重要：execute_tool() 会把错误信息原样交给模型，
    模型看到"除数不能为零"就知道该怎么向用户解释；
    看到 "division by zero" 也能懂，但不如中文直观。
    """
    divisor = _to_decimal(b, "b")
    if divisor == 0:
        raise ValueError("除数不能为零 —— 数学上除以零没有定义，请换一个非零的除数")
    return float(_to_decimal(a, "a") / divisor)


TOOL_IMPLEMENTATIONS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


# ===========================================================================
# 工具说明书
# ===========================================================================
# 这 4 段 JSON 会原样发给模型。模型**只能靠这些描述**来决定该用哪个工具，
# 所以 description 写得越清楚，它选得越准。
#
# 特别注意每条都写了"什么时候用"，而不只是"这是什么" ——
# 因为模型需要的是**决策依据**，不是名词解释。

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个数字的和。当用户需要求和、累加、总共多少时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个加数"},
                    "b": {"type": "number", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "计算两个数字的差（a 减 b）。当用户需要求差、剩余多少、少了多少时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "被减数"},
                    "b": {"type": "number", "description": "减数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个数字的积。当用户需要求积、几倍、每份多少乘以份数时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个乘数"},
                    "b": {"type": "number", "description": "第二个乘数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "计算 a 除以 b 的商。当用户需要求商、平均分、每份多少时使用。除数为零时工具会返回错误。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "被除数"},
                    "b": {"type": "number", "description": "除数，不能为零"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def execute_tool(name: str, arguments_json: str) -> dict:
    """执行模型请求的工具调用。

    **核心设计：出错时返回错误信息，而不是抛异常。**

    因为模型看到失败原因才有机会改正重试；如果直接崩掉，整个 Agent 就断了。
    这就是 02 要教的第一件事：工具失败是常态，不是意外。
    """
    if name not in TOOL_IMPLEMENTATIONS:
        return {
            "error": f"未知工具：{name}",
            "可用工具": list(TOOL_IMPLEMENTATIONS),
        }

    try:
        kwargs = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return {"error": f"参数不是合法 JSON：{exc}。请重新给出参数。"}

    if not isinstance(kwargs, dict):
        return {"error": f"参数应该是对象，实际是 {type(kwargs).__name__}"}

    try:
        result = TOOL_IMPLEMENTATIONS[name](**kwargs)
        # 整数结果就别显示成 8.0 —— 给模型看的数字越干净，它转述得越准
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return {"result": result}
    except TypeError as exc:
        return {"error": f"参数不匹配：{exc}。请检查参数名和个数。"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}：{exc}"}


# ===========================================================================
# 离线假模型（不需要 API Key，用来把循环看清楚）
# ===========================================================================


class MockModel:
    """离线假模型：靠关键词猜该用哪个工具。

    它的"选择"是写死的规则，**真模型是自己判断**。
    存在的意义：没有 Key 的时候也能跑通整条链路、看清循环。

    它也是 02 最好的反面教材 —— 你会发现它只能处理最简单的说法，
    稍微换个问法（"我买了 3 个，吃掉 2 个"）它就懵了，而真模型不会。
    """

    OPERATIONS = [
        (("加", "和", "总共", "一共", "+", "求和"), "add"),
        (("减", "差", "少了", "剩", "-", "去掉"), "subtract"),
        (("乘", "积", "倍", "*", "×"), "multiply"),
        (("除", "商", "平均", "每份", "/", "÷"), "divide"),
    ]

    def __init__(self, question: str) -> None:
        self.numbers = re.findall(r"-?\d+(?:\.\d+)?", question)
        self.a = float(self.numbers[0]) if self.numbers else 0.0
        self.b = float(self.numbers[1]) if len(self.numbers) > 1 else 0.0
        self.tool = self._guess_tool(question)
        self.turn = 0

    @classmethod
    def _guess_tool(cls, question: str) -> str:
        for keywords, tool in cls.OPERATIONS:
            if any(keyword in question for keyword in keywords):
                return tool
        return "add"

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
                            "name": self.tool,
                            "arguments": json.dumps({"a": self.a, "b": self.b}),
                        },
                    }
                ],
            }

        # 第二轮：从历史里找出 tool 返回的内容
        payload: object = None
        for message in reversed(messages):
            if message.get("role") == "tool":
                raw = message.get("content") or "{}"
                try:
                    parsed = json.loads(raw)
                    payload = parsed.get("result", parsed) if isinstance(parsed, dict) else parsed
                except json.JSONDecodeError:
                    payload = raw
                break

        # 工具报错时，假模型只会照念；真模型会解释原因并给你建议
        if isinstance(payload, dict) and "error" in payload:
            return {
                "content": f"工具报错了：{payload['error']}",
                "tool_calls": None,
            }
        return {
            "content": f"结果是 {payload}",
            "tool_calls": None,
        }


def real_model_respond(messages: list[dict], model: str) -> dict:
    """调用真模型。**选哪个工具、参数是什么，全由模型决定。**"""
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
# 主循环（产出结构化事件，供命令行和网页共用）
# ===========================================================================

SYSTEM_PROMPT = (
    "你是一个严谨的助手。凡是需要做四则运算，都必须调用提供的工具，不要自己心算。"
    "根据用户的意思选择正确的工具（求和用 add、求差用 subtract、求积用 multiply、求商用 divide）。"
    "如果工具返回错误，请看懂错误原因，用中文向用户解释清楚，并给出建议。"
    "拿到工具结果后再回答用户。"
)


def run_agent_steps(
    question: str,
    *,
    mock: bool = False,
    model: str = "deepseek-flash",
    max_steps: int = 6,
) -> Iterator[dict]:
    """跑一次 Agent，逐步 yield 事件。

    事件类型：
        start        开始
        round        进入第 N 轮
        decision     模型决定调用工具（含它选了哪个）
        tool_result  本地执行结果（可能是成功，也可能是错误）
        final        最终回答
        error        失败（含超过 max_steps 的防护）
    """
    yield {
        "type": "start",
        "question": question,
        "mode": "mock" if mock else "real",
        "model": model,
        "tools": [schema["function"]["name"] for schema in TOOL_SCHEMAS],
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
