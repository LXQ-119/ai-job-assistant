"""Agentic RAG：让模型自己决定走哪条路。

对比三种做法，一眼看清区别：

┌─────────────────────────────────────────────────────────────────────┐
│ 纯大模型                                                            │
│   问题 → 模型直接答                                                 │
│   问"我给自己定的检索目标是多少" → 它不知道，只能编                  │
├─────────────────────────────────────────────────────────────────────┤
│ 写死的 RAG（本项目原来的做法）                                       │
│   问题 →【强制查资料柜】→ 查到就答 / 查不到说"没有"                  │
│   问"3 减 2 等于几" → "资料里没有提到"  ← 明明会却答不了             │
├─────────────────────────────────────────────────────────────────────┤
│ Agentic RAG（本文件）                                               │
│   问题 → 模型先判断 → ┌─ 该查我的笔记 → 调 search_notes（带引用）    │
│                       ├─ 该算数       → 调 calculate（精确计算）     │
│                       └─ 我自己就会   → 直接答，但必须标明          │
│                                          "不来自你的笔记"            │
└─────────────────────────────────────────────────────────────────────┘

**最关键的设计不是"多两个工具"，而是"来源必须可分辨"。**
两条路都能走通，但用户必须能一眼看出哪句话有出处、哪句话没有 ——
否则 Agent 的灵活性会把 RAG 的可追溯性吃干净。
"""

from __future__ import annotations

import ast
import json
import operator
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Iterator, Sequence

from ..llm import LLMError, chat
from .rag import retrieve
from .retriever import KB

# ===========================================================================
# 工具 1：查用户的笔记
# ===========================================================================

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_notes",
        "description": (
            "在用户自己的面经笔记里检索。"
            "凡是问到他本人的东西（他的面经、他的项目、他的笔记、他的实习经历），"
            "都必须调用这个工具 —— 这些信息你不可能知道，只能去他的笔记里查。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索用的关键词或问题，用中文",
                }
            },
            "required": ["query"],
        },
    },
}


def search_notes(query: str) -> dict:
    """查资料柜。注意：这里复用写死 RAG 的那套检索，没有另起一套。"""
    hits = retrieve(query)

    if not hits:
        return {
            "found": False,
            "message": (
                f"在用户笔记（共 {len(KB)} 张卡片）里没有找到与「{query}」相关的内容。"
                "请如实告诉用户他的笔记里没有，然后可以补充你自己的知识，"
                "但必须标明那部分不来自笔记。绝对不要编造笔记里没有的内容。"
            ),
        }

    return {
        "found": True,
        "count": len(hits),
        "results": [
            {
                "编号": index,
                "来源": f"{hit.chunk.source} · {hit.chunk.heading}",
                "内容": hit.chunk.text,
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }


# ===========================================================================
# 工具 2：计算器
# ===========================================================================

CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "精确计算四则运算表达式。需要算数时必须调用这个工具，不要自己心算 —— "
            "模型心算大数会出错，而且错得和答对一样自信。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": '要计算的算式，例如 "3+5"、"(12-4)*3"、"100/8"',
                }
            },
            "required": ["expression"],
        },
    },
}

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> Decimal:
    """只允许数字和四则运算。

    为什么不用 eval()？因为 eval 会执行任意代码 ——
    模型给的参数是不可信输入，`__import__('os').system(...)` 也是一段合法表达式。
    用 ast 白名单遍历，能执行的东西就只有加减乘除。
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字")
        # 走 str 再转 Decimal：0.1 在二进制浮点里存的是 0.1000000000000000055…，
        # 直接 Decimal(0.1) 会把这个误差原样带进来。
        return Decimal(str(node.value))

    if isinstance(node, ast.BinOp):
        func = _BINARY_OPS.get(type(node.op))
        if func is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}（只支持 + - * / %）")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise ZeroDivisionError("除数不能为 0")
        return func(left, right)

    if isinstance(node, ast.UnaryOp):
        func = _UNARY_OPS.get(type(node.op))
        if func is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return func(_eval_node(node.operand))

    raise ValueError(f"不支持的写法：{type(node).__name__}（只支持数字和四则运算）")


def calculate(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return {"error": f"表达式语法错误：{exc.msg}"}
    try:
        value = _eval_node(tree)
    except ZeroDivisionError as exc:
        return {"error": str(exc)}
    except (ValueError, InvalidOperation, DivisionByZero) as exc:
        return {"error": str(exc)}

    # 整数就不显示小数点：8 而不是 8.0
    if value == value.to_integral_value():
        return {"result": str(int(value))}
    return {"result": str(value.normalize())}


# ===========================================================================
# 系统提示词 —— 这一课的重点其实在这里
# ===========================================================================

SYSTEM_PROMPT = """你是「AI 求职助手」。你手上有两个工具，但并不是每次都要用。

【必须调用 search_notes 的情况】
问题涉及用户本人的东西：他的面经、他的项目、他的笔记、他的实习经历、他的简历。
这些信息你不可能知道，只能去他的笔记里查。
查到之后，回答里必须用 [编号] 标出每条结论的出处。

【必须调用 calculate 的情况】
需要做四则运算时一律调工具，不要自己心算。

【两个都不需要时】
用你自己的知识直接回答。但开头必须先写一句：
⚠️ 以下内容不来自你的笔记，是我自己的知识：
绝对不要把"你自己的知识"和"用户笔记里的内容"混在一起说 ——
用户必须能一眼分清哪句话有出处、哪句话没有。

【查了但没查到】
如实说"你的笔记里没有这方面的记录"，然后可以补充你自己的知识，
同样要加上面那句标注。绝对不要编造用户笔记里没有的内容。"""


# ===========================================================================
# 工具分发
# ===========================================================================

IMPLEMENTATIONS = {
    "search_notes": search_notes,
    "calculate": calculate,
}


def execute_tool(name: str, arguments_json: str) -> dict:
    """执行模型请求的工具调用。出错时返回错误信息，而不是抛异常。

    为什么返回错误而不是抛？因为**错误也是给模型的输入**：
    它看到"除数不能为 0"，就有机会换个数重试或者向用户解释。
    直接抛异常则整个流程崩掉，模型连补救的机会都没有。
    """
    func = IMPLEMENTATIONS.get(name)
    if func is None:
        return {"error": f"未知工具：{name}。可用工具：{list(IMPLEMENTATIONS)}"}

    try:
        kwargs = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return {"error": f"参数不是合法 JSON：{exc}"}

    if not isinstance(kwargs, dict):
        return {"error": f"参数应该是对象，实际是 {type(kwargs).__name__}"}

    try:
        return func(**kwargs)
    except TypeError as exc:
        return {"error": f"参数不匹配：{exc}"}
    except Exception as exc:  # noqa: BLE001 - 工具出错不能拖垮整个流程
        return {"error": f"工具执行出错（{type(exc).__name__}）：{exc}"}


# ===========================================================================
# 主循环
# ===========================================================================

TOOL_SCHEMAS: Sequence[dict] = [SEARCH_TOOL, CALC_TOOL]


def run(
    question: str,
    *,
    model: str | None = None,
    max_steps: int = 5,
) -> Iterator[dict]:
    """跑一次 Agentic RAG，逐步 yield 事件。

    事件类型：
        start        开始，带可用工具清单
        round        进入第 N 轮
        decision     模型决定调用工具（含它选了哪个、参数是什么）
        tool_result  本地真正执行的结果
        final        最终回答（附这次走了哪条路、总成本、总轮数）
        error        失败（含超过 max_steps 的防死循环）
    """
    yield {
        "type": "start",
        "question": question,
        "model": model,
        "tools": [schema["function"]["name"] for schema in TOOL_SCHEMAS],
        "kb_chunks": len(KB),
    }

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    total_cost = 0.0
    total_tokens = 0
    citations: list[dict] = []
    used: list[str] = []

    for step in range(1, max_steps + 1):
        yield {"type": "round", "index": step}

        try:
            # temperature=0：要的是可复现的判断，不是创造力
            result = chat(
                messages,
                purpose="agentic_rag",
                model=model,
                temperature=0,
                tools=TOOL_SCHEMAS,
            )
        except LLMError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        total_cost += result.cost_usd
        total_tokens += result.prompt_tokens + result.completion_tokens

        # 模型没要求调工具 → 这就是最终回答
        if not result.tool_calls:
            yield {
                "type": "final",
                "content": result.text.strip(),
                "steps": step,
                "cost_usd": round(total_cost, 6),
                "tokens": total_tokens,
                "citations": citations,
                "route": _describe_route(used),
                "tools_used": used,
            }
            return

        yield {"type": "decision", "calls": result.tool_calls}
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
            payload = execute_tool(name, arguments)
            used.append(name)

            # 把检索到的卡片留给界面渲染引用，让"可追溯"在 Agent 模式下也成立
            if name == "search_notes" and payload.get("found"):
                for item in payload["results"]:
                    citations.append(
                        {
                            "source": item["来源"].split(" · ")[0],
                            "heading": item["来源"].split(" · ")[-1],
                            "text": item["内容"],
                        }
                    )

            yield {
                "type": "tool_result",
                "name": name,
                "arguments": arguments,
                "result": payload,
            }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )

    yield {
        "type": "error",
        "message": (
            f"已经连续 {max_steps} 轮都在调用工具，仍未给出最终回答，已强制停止。"
            "真实系统必须加这个上限，否则一次故障就能烧掉大量 token。"
        ),
    }


def _describe_route(used: list[str]) -> str:
    """把"调了哪些工具"翻译成一句人话，界面上直接显示。"""
    if not used:
        return "模型自己答（未查笔记、未计算）"
    labels = []
    if "search_notes" in used:
        labels.append("查了你的笔记")
    if "calculate" in used:
        labels.append("调了计算器")
    return " → ".join(labels)
