"""🧮 教具 02 · 多工具 Agent —— 让模型在四个工具里自己挑，并且能处理失败。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ui"))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import style  # noqa: E402
from multi_tool_core import TOOL_SCHEMAS, run_agent_steps  # noqa: E402

style.hero(
    "🧮 教具 02 · 多工具 Agent",
    "从 1 个工具到 4 个工具 —— 模型第一次有了「选择」这个动作",
    ["4 个工具里自己挑", "工具会失败", "能连续调用两步"],
)

style.hint(
    "和 01 的本质区别：<strong>01 只有一个工具，模型没得选</strong>；"
    "这里它必须在加减乘除里判断该用哪个。<br>"
    "再加两件事：工具<strong>会失败</strong>（除以零），"
    "以及能<strong>连续调用</strong>（先算 A，再用 A 的结果算 B）。"
)

with st.sidebar:
    st.divider()
    st.markdown("##### ⚙️ Agent 设置")
    use_real = st.toggle(
        "使用真模型",
        value=True,
        help="打开 = 真模型，由模型自己判断该用哪个工具；"
        "关闭 = 离线假模型（靠关键词猜，不需要 Key）",
    )
    if use_real:
        st.success("真模型：判断由模型做出")
    else:
        st.warning("假模型：只会处理最简单的说法（加/减/乘/除 关键词）")
    model = st.text_input("模型名", value="deepseek-flash", disabled=not use_real)
    max_steps = st.slider("最大轮数（防死循环）", 1, 10, 6)

st.markdown("### 这次给模型 4 个工具，它得自己挑")
columns = st.columns(4)
for column, schema in zip(columns, TOOL_SCHEMAS):
    function = schema["function"]
    with column:
        st.markdown(f"**`{function['name']}`**")
        st.caption(function["description"])

with st.expander("📋 完整的工具说明书（JSON Schema）"):
    st.caption("这 4 段描述会原样发给模型。它只能靠这些文字决定该用哪个工具。")
    st.json(TOOL_SCHEMAS)

st.markdown("### 试试这四种情况")

CASES = [
    ("🎯 选对工具", "10 减 3 等于多少？"),
    ("🗣️ 换个说法", "我买了 3 个苹果，吃掉了 2 个，还剩几个？"),
    ("💥 工具失败", "100 除以 0 等于多少？"),
    ("🔗 连续两步", "先算 3 加 5，再把结果乘以 2，最后是多少？"),
]

if "q_multi" not in st.session_state:
    st.session_state.q_multi = CASES[0][1]

columns = st.columns(4)
for column, (label, text) in zip(columns, CASES):
    with column:
        if st.button(label, width="stretch", key=f"multi_{label}"):
            st.session_state.q_multi = text
            st.rerun()
        st.caption(f"`{text}`")

question = st.text_input("问题", key="q_multi")
run = st.button("▶ 运行 Agent", type="primary", width="stretch")

if run:
    st.divider()
    st.markdown("### 执行过程")

    final_answer = None
    error_message = None
    tool_calls: list[str] = []
    tool_errors: list[str] = []

    for event in run_agent_steps(
        question, mock=not use_real, model=model, max_steps=max_steps
    ):
        kind = event["type"]

        if kind == "start":
            mode = "离线假模型" if event["mode"] == "mock" else f"真模型 {event['model']}"
            st.info(f"**问题**：{event['question']}　｜　**模式**：{mode}")

        elif kind == "round":
            st.markdown(
                f"**第 {event['index']} 轮**　"
                f"<span style='color:#94a3b8;font-size:.82em'>"
                f"（发送 {event['message_count']} 条消息）</span>",
                unsafe_allow_html=True,
            )

        elif kind == "decision":
            for call in event["calls"]:
                tool_calls.append(call["function"]["name"])
                style.step(
                    f"🧠 模型决定调用 {call['function']['name']}，而不是直接回答",
                    call["function"]["arguments"],
                    "decide",
                )

        elif kind == "tool_result":
            payload = event["result"]
            if "error" in payload:
                tool_errors.append(payload["error"])
                style.step("💥 工具失败", payload["error"], "decide")
                st.caption(
                    "这个错误是**返回给模型**的，不是抛异常。"
                    "模型看到原因后，才有机会向用户解释清楚。"
                )
            else:
                style.step(
                    "✅ 本地执行成功",
                    f"{event['name']}({event['arguments']}) → {payload['result']}",
                    "run",
                )

        elif kind == "final":
            final_answer = event["content"]

        elif kind == "error":
            error_message = event["message"]

    st.divider()
    if error_message:
        st.error(f"❌ {error_message}")

    if final_answer is not None:
        st.markdown("### ✅ 最终回答")
        st.success(final_answer)

        col1, col2, col3 = st.columns(3)
        col1.metric("调用工具次数", len(tool_calls))
        col2.metric("工具失败次数", len(tool_errors))
        col3.metric("用到的工具", "、".join(tool_calls) if tool_calls else "无")

        if tool_errors:
            st.info(
                "**这一轮发生了什么**：工具失败了，但 Agent 没有崩 —— "
                "它读懂了错误信息，然后用自己的话向你解释了原因。\n\n"
                "如果把 `execute_tool()` 里的错误处理去掉（改成直接抛异常），"
                "这个对话就会直接中断。**这就是 02 要教的核心。**"
            )
        elif len(tool_calls) > 1:
            st.info(
                "**这一轮发生了什么**：模型**连续调用了两次工具** —— "
                "先用第一步的结果，再算第二步。\n\n"
                "这不是代码写死的顺序，是模型自己根据上一步结果决定的。"
            )
