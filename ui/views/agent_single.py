"""🤖 教具 01 · 单工具 Agent —— 看清"决策循环"本身。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ui"))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import style  # noqa: E402
from agent_core import TOOL_SCHEMAS, run_agent_steps  # noqa: E402

style.hero(
    "🤖 教具 01 · 单工具 Agent",
    "模型负责「决定调什么」，代码负责「真的算」—— 两者靠一份 JSON Schema 契约连接",
    ["function calling", "只有 1 个工具 add", "决策循环看得见"],
)

style.hint(
    "这一页的重点<strong>不是「能算加法」</strong>，而是让你看清 Agent 的循环：<br>"
    "发消息 → 模型说要调工具 → 本地真的执行 → 结果回传 → 模型给出最终回答。"
)

with st.sidebar:
    st.divider()
    st.markdown("##### ⚙️ Agent 设置")
    use_real = st.toggle(
        "使用真模型",
        value=True,
        help="打开 = 真模型，由模型自己决定要不要调工具；"
        "关闭 = 离线假模型（不需要 Key、不花钱，但它的「决定」是写死的）",
    )
    if use_real:
        st.success("真模型：决策由模型做出")
    else:
        st.warning(
            "假模型**不判断问题**，第 1 轮无条件调用 add。"
            "问它「你好，你是谁」会得到「0 + 0 = 0」。"
        )
    model = st.text_input("模型名", value="deepseek-flash", disabled=not use_real)
    max_steps = st.slider("最大轮数（防死循环）", 1, 8, 5)

with st.expander("📋 发给模型的「工具说明书」（JSON Schema）"):
    st.caption("这段 JSON 会和用户问题一起发给模型。description 写得越清楚，模型选得越准。")
    st.json(TOOL_SCHEMAS)

st.markdown("### 给 Agent 发一个问题")

SUGGESTIONS = [
    "3 加 5 等于多少？",
    "我买了 3 个苹果，又买了 5 个，一共几个？",
    "12345678 加 87654322 等于多少？",
    "0.1 加 0.2 等于多少？",
    "你好，你是谁？",
]

if "q_single" not in st.session_state:
    st.session_state.q_single = SUGGESTIONS[0]

cols = st.columns(len(SUGGESTIONS))
for column, suggestion in zip(cols, SUGGESTIONS):
    with column:
        if st.button(suggestion, width="stretch", key=f"single_{suggestion}"):
            st.session_state.q_single = suggestion
            st.rerun()

question = st.text_input("问题", key="q_single")
run = st.button("▶ 运行 Agent", type="primary", width="stretch")

if run:
    st.divider()
    st.markdown("### 执行过程")

    final_answer = None
    error_message = None
    tool_call_count = 0

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
                f"（把 {event['message_count']} 条消息发给模型）</span>",
                unsafe_allow_html=True,
            )

        elif kind == "decision":
            tool_call_count += len(event["calls"])
            for call in event["calls"]:
                style.step(
                    f"🧠 模型没有直接回答，而是决定调用：{call['function']['name']}",
                    call["function"]["arguments"],
                    "decide",
                )

        elif kind == "tool_result":
            payload = event["result"]
            if "error" in payload:
                style.step("⚠️ 工具返回错误", payload["error"], "decide")
                st.caption(
                    "错误是以结构化信息返回给模型的，模型有机会改参数重试，"
                    "而不是让程序崩掉。"
                )
            else:
                style.step(
                    "✅ 本地代码真正执行了工具",
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
        st.caption(
            f"这一轮共调用工具 {tool_call_count} 次。"
            "打开「使用真模型」时，上面的工具名和参数就都是模型自己决定的。"
        )
