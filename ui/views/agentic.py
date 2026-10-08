"""🧭 智能助手（03 · Agentic RAG）—— 一个框，三条路，模型自己选。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import style  # noqa: E402

style.hero(
    "🧭 智能助手",
    "一个框，三条路 —— 模型自己决定：查你的笔记、调计算器，还是直接回答",
    ["Agentic RAG", "function calling", "来源可分辨", "03"],
)

# ---------------------------------------------------------------------------
# 三条路说明
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)
with col1:
    style.card(
        "📚 查你的笔记",
        "问到你自己的事（面经、项目、实习经历）时，它去 <code>data/knowledge</code> "
        "里检索，答案带 <code>[编号]</code> 出处。",
    )
with col2:
    style.card(
        "🧮 调计算器",
        "需要算数时交给工具精确计算。模型心算大数会出错，"
        "而且错得和答对一样自信。",
    )
with col3:
    style.card(
        "💬 直接回答",
        "通用常识它自己就会。但这种回答<strong>必须标注</strong>"
        "「不来自你的笔记」，你才分得清哪句有出处。",
    )

common.sidebar()
st.divider()

# ---------------------------------------------------------------------------
# 提问
# ---------------------------------------------------------------------------

st.markdown("### 试试这四种情况")

CASES = [
    ("📚 查笔记", "我从小米那次面试里学到了什么？"),
    ("🧮 算数", "1234 乘以 5678 等于多少？"),
    ("💬 直接答", "李白是哪个朝代的诗人？"),
    ("🔍 查了没有", "我简历上写的期望薪资是多少？"),
]

if "agent_question" not in st.session_state:
    st.session_state.agent_question = CASES[0][1]

cols = st.columns(4)
for column, (label, text) in zip(cols, CASES):
    with column:
        if st.button(label, width="stretch"):
            st.session_state.agent_question = text
            st.session_state.pop("agent_result", None)
            st.rerun()
        st.caption(f"`{text}`")

question = st.text_input("问题", key="agent_question")
run = st.button("▶ 让模型自己判断", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# 渲染执行过程
# ---------------------------------------------------------------------------


def render_trace(events: list[dict]) -> None:
    """把模型的决策过程一步步摊开。

    这是整个页面最有价值的地方：**用户能看见模型想了什么**，
    而不是只看到一个黑盒吐出来的答案。
    """
    for event in events:
        kind = event["type"]

        if kind == "round":
            st.markdown(f"**第 {event['index']} 轮**")

        elif kind == "decision":
            for call in event["calls"]:
                name = call["function"]["name"]
                if name == "search_notes":
                    style.step(
                        "📚 模型决定：去查你的笔记", call["function"]["arguments"], "decide"
                    )
                elif name == "calculate":
                    style.step(
                        "🧮 模型决定：调计算器", call["function"]["arguments"], "decide"
                    )
                else:
                    style.step(
                        f"🔧 模型决定调用：{name}", call["function"]["arguments"], "decide"
                    )

        elif kind == "tool_result":
            payload = event["result"]
            if "error" in payload:
                style.step("⚠️ 工具返回错误", payload["error"], "decide")
            elif event["name"] == "search_notes":
                if payload.get("found"):
                    names = "　".join(
                        f"`[{item['编号']}]` {item['来源']}" for item in payload["results"]
                    )
                    style.step(f"📚 翻到 {payload['count']} 张卡片", names, "run")
                else:
                    style.step("📚 你的笔记里没有相关内容", "模型会如实告知，不会编", "decide")
            elif event["name"] == "calculate":
                style.step("🧮 计算结果", f"{payload.get('result')}", "run")

        elif kind == "error":
            st.error(event["message"])


if run and question.strip():
    with st.spinner("模型正在判断该走哪条路……（可能要十几秒）"):
        st.session_state.agent_result = common.api_post(
            "/agent/ask", {"question": question}, timeout=300
        )

result = st.session_state.get("agent_result")

if result:
    st.divider()
    st.markdown("### 执行过程")
    render_trace(result["events"])

    final = result.get("final")
    if final:
        st.divider()
        st.markdown("### 最终回答")

        kind = "plain"
        if final["tools_used"]:
            kind = "calc" if final["tools_used"] == ["calculate"] else "default"
        st.markdown(style.route_badge(f"🧭 {final['route']}", kind), unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("模型调用轮数", final["steps"])
        m2.metric("消耗 token", f"{final['tokens']:,}")
        m3.metric("本次成本", f"${final['cost_usd']:.6f}")

        st.markdown(final["content"])

        if final["citations"]:
            st.markdown("#### 引用的笔记")
            common.show_citations(final["citations"])

    if result.get("error"):
        st.error(result["error"])

elif not run:
    st.divider()
    style.hint(
        "点上面的按钮选一个问题，或自己输入。<br>"
        "执行过程会一步步摊开：模型调了什么工具、参数是什么、工具返回了什么。"
    )
