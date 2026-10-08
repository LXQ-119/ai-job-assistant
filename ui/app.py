"""Streamlit 入口 + 导航。

为什么手动定义页面，而不是靠 `pages/` 目录自动发现？

    自动发现会把目录里每个 .py **平等地**塞进侧边栏。结果是侧边栏长这样：

        app              ← 主页面，名字来自文件名，很丑
        Agent演示         ← 这是什么？
        多工具Agent       ← 又是什么？
        智能助手

    面试官点进来会想"这几个是什么关系"。**说不清楚就是减分。**

    手动分组之后，侧边栏能表达出真实的两层结构：

        🎯 产品       面经问答 / 智能助手 / 简历解析 / 评测与成本
        🎓 教学演示    01 单工具 Agent / 02 多工具 Agent

    这样"教学演示"就从一个说不清的窗口，变成了一个**有意为之的分层**。

启动：
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ui"))

import style  # noqa: E402

st.set_page_config(page_title="AI 求职助手", page_icon="🎯", layout="wide")
style.apply()

nav = st.navigation(
    {
        "🎯 产品": [
            st.Page("views/chat.py", title="面经问答", icon="💬", default=True),
            st.Page("views/agentic.py", title="智能助手（03）", icon="🧭"),
            st.Page("views/resume.py", title="简历解析", icon="📄"),
            st.Page("views/metrics.py", title="评测与成本", icon="📊"),
        ],
        "🎓 教学演示": [
            st.Page("views/agent_lab.py", title="Agent 原理演示", icon="🎓"),
        ],
    }
)

nav.run()
