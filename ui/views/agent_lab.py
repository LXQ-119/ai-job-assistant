"""🎓 Agent 原理演示 —— 从 1 个工具到 4 个工具。

为什么要合成一页：两个演示是**递进关系**，放在一起才看得出这条路径。

    01 只有一个工具 add
        → 模型没有"选择"这个动作，只证明了"会调用工具"
        → 好处是干扰少，能专注看清循环本身

    02 有四个工具
        → 模型必须**判断该用哪个**，这才是真正的决策
        → 工具**会失败**（除以零），模型得看懂错误
        → 能**连续调用**（先算 A，再用 A 的结果算 B）

这一页讲的东西，正是 🧭 智能助手（03）的底层机制 ——
03 只是把工具换成了「查笔记」和「计算器」，循环一模一样。
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ui"))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import style  # noqa: E402
from agent_core import TOOL_SCHEMAS as SINGLE_TOOLS  # noqa: E402
from agent_core import run_agent_steps as run_single  # noqa: E402
from multi_tool_core import TOOL_SCHEMAS as MULTI_TOOLS  # noqa: E402
from multi_tool_core import run_agent_steps as run_multi  # noqa: E402

style.hero(
    "🎓 Agent 原理演示",
    "从 1 个工具到 4 个工具 —— 看清模型是怎么「做决定」的",
    ["function calling", "决策循环", "工具选择", "容错与编排"],
)

style.hint(
    "这一页讲的是 <strong>🧭 智能助手（03）的底层机制</strong>："
    "03 只是把工具换成了「查笔记」和「计算器」，循环一模一样。<br>"
    "两个演示是递进关系，建议按顺序看。"
)

# ---------------------------------------------------------------------------
# 共同的渲染逻辑
# ---------------------------------------------------------------------------


def render_events(events, *, on_decision, on_tool_result) -> tuple[str | None, str | None, int]:
    """把事件流渲染出来。两个演示共用，只是措辞和统计口径不同。"""
    final = None
    error = None
    calls = 0

    for event in events:
        kind = event["type"]

        if kind == "start":
            mode = "离线假模型" if event["mode"] == "mock" else f"真模型 {event['model']}"
            st.info(f"**问题**：{event['question']}　｜　**模式**：{mode}")

        elif kind == "round":
            extra = (
                f"（把 {event['message_count']} 条消息发给模型）"
                if "message_count" in event
                else ""
            )
            st.markdown(
                f"**第 {event['index']} 轮**　"
                f"<span style='color:#94a3b8;font-size:.82em'>{extra}</span>",
                unsafe_allow_html=True,
            )

        elif kind == "decision":
            calls += len(event["calls"])
            on_decision(event["calls"])

        elif kind == "tool_result":
            on_tool_result(event)

        elif kind == "final":
            final = event["content"]

        elif kind == "error":
            error = event["message"]

    return final, error, calls


# ---------------------------------------------------------------------------
# 01 单工具
# ---------------------------------------------------------------------------


def render_single() -> None:
    style.hint(
        "<strong>01 的重点不是「能算加法」</strong>，而是让你看清 Agent 的循环：<br>"
        "发消息 → 模型说要调工具 → 本地真的执行 → 结果回传 → 模型给出最终回答。"
    )

    with st.expander("⚙️ 设置", expanded=False):
        c1, c2, c3 = st.columns([2, 2, 3])
        use_real = c1.toggle(
            "使用真模型",
            value=True,
            key="s_real",
            help="关闭 = 离线假模型（不需要 Key、不花钱，但「决定」是写死的）",
        )
        model = c2.text_input("模型名", value="deepseek-flash", disabled=not use_real, key="s_model")
        max_steps = c3.slider("最大轮数（防死循环）", 1, 8, 5, key="s_steps")

    if not use_real:
        st.warning(
            "假模型**不判断问题**，第 1 轮无条件调用 add。"
            "问它「你好，你是谁」会得到「0 + 0 = 0」—— 那不是在回答你，是在做加法。"
        )

    with st.expander("📋 发给模型的「工具说明书」（JSON Schema）"):
        st.caption("这段 JSON 会和用户问题一起发给模型。description 写得越清楚，模型选得越准。")
        st.json(SINGLE_TOOLS)

    st.markdown("#### 给 Agent 发一个问题")

    suggestions = [
        "3 加 5 等于多少？",
        "我买了 3 个苹果，又买了 5 个，一共几个？",
        "12345678 加 87654322 等于多少？",
        "0.1 加 0.2 等于多少？",
        "你好，你是谁？",
    ]

    if "q_single" not in st.session_state:
        st.session_state.q_single = suggestions[0]

    cols = st.columns(len(suggestions))
    for column, suggestion in zip(cols, suggestions):
        with column:
            if st.button(suggestion, width="stretch", key=f"s_btn_{suggestion}"):
                st.session_state.q_single = suggestion
                st.rerun()

    question = st.text_input("问题", key="q_single")
    run = st.button("▶ 运行单工具 Agent", type="primary", width="stretch", key="s_run")

    if not run:
        return

    st.divider()
    st.markdown("### 执行过程")

    def on_decision(calls: list[dict]) -> None:
        for call in calls:
            style.step(
                f"🧠 模型没有直接回答，而是决定调用：{call['function']['name']}",
                call["function"]["arguments"],
                "decide",
            )

    def on_tool_result(event: dict) -> None:
        payload = event["result"]
        if "error" in payload:
            style.step("⚠️ 工具返回错误", payload["error"], "decide")
        else:
            style.step(
                "✅ 本地代码真正执行了工具",
                f"{event['name']}({event['arguments']}) → {payload['result']}",
                "run",
            )

    final, error, calls = render_events(
        run_single(question, mock=not use_real, model=model, max_steps=max_steps),
        on_decision=on_decision,
        on_tool_result=on_tool_result,
    )

    st.divider()
    if error:
        st.error(f"❌ {error}")
    if final is not None:
        st.markdown("### ✅ 最终回答")
        st.success(final)
        st.caption(
            f"这一轮共调用工具 {calls} 次。"
            "打开「使用真模型」时，工具名和参数就都是模型自己决定的。"
        )


# ---------------------------------------------------------------------------
# 02 多工具
# ---------------------------------------------------------------------------


def render_multi() -> None:
    style.hint(
        "和 01 的本质区别：<strong>01 只有一个工具，模型没得选</strong>；"
        "这里它必须在加减乘除里判断该用哪个。<br>"
        "再加两件事：工具<strong>会失败</strong>（除以零），"
        "以及能<strong>连续调用</strong>（先算 A，再用 A 的结果算 B）。"
    )

    with st.expander("⚙️ 设置", expanded=False):
        c1, c2, c3 = st.columns([2, 2, 3])
        use_real = c1.toggle(
            "使用真模型",
            value=True,
            key="m_real",
            help="关闭 = 离线假模型（靠关键词猜，不需要 Key）",
        )
        model = c2.text_input("模型名", value="deepseek-flash", disabled=not use_real, key="m_model")
        max_steps = c3.slider("最大轮数（防死循环）", 1, 10, 6, key="m_steps")

    if not use_real:
        st.warning("假模型：只会处理最简单的说法（加/减/乘/除 关键词）")

    st.markdown("#### 这次给模型 4 个工具，它得自己挑")
    columns = st.columns(4)
    for column, schema in zip(columns, MULTI_TOOLS):
        function = schema["function"]
        with column:
            st.markdown(f"**`{function['name']}`**")
            st.caption(function["description"])

    with st.expander("📋 完整的工具说明书（JSON Schema）"):
        st.caption("这 4 段描述会原样发给模型。它只能靠这些文字决定该用哪个工具。")
        st.json(MULTI_TOOLS)

    st.markdown("#### 试试这四种情况")

    cases = [
        ("🎯 选对工具", "10 减 3 等于多少？"),
        ("🗣️ 换个说法", "我买了 3 个苹果，吃掉了 2 个，还剩几个？"),
        ("💥 工具失败", "100 除以 0 等于多少？"),
        ("🔗 连续两步", "先算 3 加 5，再把结果乘以 2，最后是多少？"),
    ]

    if "q_multi" not in st.session_state:
        st.session_state.q_multi = cases[0][1]

    columns = st.columns(4)
    for column, (label, text) in zip(columns, cases):
        with column:
            if st.button(label, width="stretch", key=f"m_btn_{label}"):
                st.session_state.q_multi = text
                st.rerun()
            st.caption(f"`{text}`")

    question = st.text_input("问题", key="q_multi")
    run = st.button("▶ 运行多工具 Agent", type="primary", width="stretch", key="m_run")

    if not run:
        return

    st.divider()
    st.markdown("### 执行过程")

    used_tools: list[str] = []
    tool_errors: list[str] = []

    def on_decision(calls: list[dict]) -> None:
        for call in calls:
            used_tools.append(call["function"]["name"])
            style.step(
                f"🧠 模型决定调用 {call['function']['name']}，而不是直接回答",
                call["function"]["arguments"],
                "decide",
            )

    def on_tool_result(event: dict) -> None:
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

    final, error, _ = render_events(
        run_multi(question, mock=not use_real, model=model, max_steps=max_steps),
        on_decision=on_decision,
        on_tool_result=on_tool_result,
    )

    st.divider()
    if error:
        st.error(f"❌ {error}")

    if final is not None:
        st.markdown("### ✅ 最终回答")
        st.success(final)

        col1, col2, col3 = st.columns(3)
        col1.metric("调用工具次数", len(used_tools))
        col2.metric("工具失败次数", len(tool_errors))
        col3.metric("用到的工具", "、".join(used_tools) if used_tools else "无")

        if tool_errors:
            st.info(
                "**这一轮发生了什么**：工具失败了，但 Agent 没有崩 —— "
                "它读懂了错误信息，然后用自己的话向你解释了原因。\n\n"
                "如果把 `execute_tool()` 里的错误处理去掉（改成直接抛异常），"
                "这个对话就会直接中断。**这就是 02 要教的核心。**"
            )
        elif len(used_tools) > 1:
            st.info(
                "**这一轮发生了什么**：模型**连续调用了两次工具** —— "
                "先用第一步的结果，再算第二步。\n\n"
                "这不是代码写死的顺序，是模型自己根据上一步结果决定的。"
            )


# ---------------------------------------------------------------------------
# 两个标签页
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["🤖 01 · 单工具（先看清循环）", "🧮 02 · 多工具（模型开始做选择）"])

with tab1:
    render_single()

with tab2:
    render_multi()
