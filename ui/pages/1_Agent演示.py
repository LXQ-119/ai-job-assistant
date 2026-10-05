"""🤖 Agent 演示 —— 在网页里给加法 Agent 发问题，并看清它每一步在干什么。

这是 Streamlit 的**多页应用**：只要在入口脚本旁边建一个 pages/ 目录，
里面的每个 .py 就会自动出现在左侧边栏。不需要改主入口。

页面里你能看到：
    1. 发给模型的「工具说明书」（JSON Schema）—— 模型靠它才知道有 add 这个工具
    2. 模型这一轮**决定调用工具**（而不是直接回答），以及它自己解析出的参数
    3. 本地代码真正执行的结果
    4. 模型基于工具结果给出的最终回答
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from agent_core import TOOL_SCHEMAS, run_agent_steps  # noqa: E402

st.set_page_config(page_title="Agent 演示", page_icon="🤖", layout="wide")

st.title("🤖 加法 Agent 演示")
st.caption(
    "模型负责「决定调什么」，代码负责「真的算」——两者靠一份 JSON Schema 契约连接。"
)

# ---------------------------------------------------------------------------
# 侧边栏设置
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("⚙️ Agent 设置")

    use_real = st.toggle(
        "使用真模型",
        value=False,
        help="关闭 = 离线假模型（不需要 API Key、不花钱）；"
        "打开 = 真模型，由模型自己决定要不要调工具（需要 .env 里的 LLM_API_KEY）",
    )
    if use_real:
        st.success("真模型模式：决策由模型做出")
    else:
        st.info("离线假模型：决策是写死的，用于看清循环")

    model = st.text_input("模型名", value="deepseek-flash", disabled=not use_real)
    max_steps = st.slider("最大轮数（防死循环）", 1, 8, 5)

# ---------------------------------------------------------------------------
# 工具说明书
# ---------------------------------------------------------------------------

with st.expander("📋 发给模型的「工具说明书」（JSON Schema）", expanded=False):
    st.caption(
        "这段 JSON 会和用户问题一起发给模型。description 写得越清楚，模型选得越准。"
    )
    st.json(TOOL_SCHEMAS)

# ---------------------------------------------------------------------------
# 提问
# ---------------------------------------------------------------------------

st.markdown("### 给 Agent 发一个问题")

SUGGESTIONS = [
    "3 加 5 等于多少？",
    "我买了 3 个苹果，又买了 5 个，一共几个？",
    "12345678 加 87654322 等于多少？",
    "0.1 加 0.2 等于多少？",
    "你好，你是谁？",
]

question = st.text_input("问题", value="3 加 5 等于多少？", label_visibility="collapsed")

st.caption("点一下就能填入：")
cols = st.columns(len(SUGGESTIONS))
for column, suggestion in zip(cols, SUGGESTIONS):
    if column.button(suggestion, width="stretch"):
        question = suggestion
        st.session_state["pending_question"] = suggestion
        st.rerun()

run = st.button("▶ 运行 Agent", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# 渲染每一步
# ---------------------------------------------------------------------------

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
            mode_label = "离线假模型" if event["mode"] == "mock" else f"真模型 {event['model']}"
            st.info(f"**问题**：{event['question']}　｜　**模式**：{mode_label}")

        elif kind == "round":
            st.markdown(
                f"#### 第 {event['index']} 轮　"
                f"<span style='color:gray;font-size:0.8em'>"
                f"（把 {event['message_count']} 条消息发给模型）</span>",
                unsafe_allow_html=True,
            )

        elif kind == "decision":
            tool_call_count += len(event["calls"])
            st.warning("模型**没有直接回答**，而是决定调用工具：")
            for call in event["calls"]:
                col1, col2 = st.columns([1, 2])
                col1.metric("工具名", call["function"]["name"])
                col2.code(call["function"]["arguments"], language="json")

        elif kind == "tool_result":
            st.success("本地代码真正执行了工具：")
            payload = event["result"]
            if "error" in payload:
                st.error(f"工具返回错误：{payload['error']}")
                st.caption("注意：错误是以结构化信息返回给模型的，模型有机会改参数重试，而不是让程序崩掉。")
            else:
                st.markdown(f"`{event['name']}({event['arguments']})` → **{payload['result']}**")

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
            "如果打开「使用真模型」，上面的工具名和参数就都是模型自己决定的。"
        )
