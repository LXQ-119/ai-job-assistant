"""💬 面经问答 —— 写死的 RAG（先检索，再生成，检索为空就拒答）。"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import style  # noqa: E402

style.hero(
    "💬 面经问答",
    "把你的笔记变成随时能查的知识库 —— 每个答案都带出处，查不到就老实说没有",
    ["写死的 RAG", "BM25 检索", "SSE 流式", "引用可核对"],
)

style.hint(
    "这条路是<strong>固定</strong>的：任何问题都先检索资料柜，翻到就照资料答，"
    "翻不到就直接说「没找到」—— <strong>不会调用模型去自由发挥</strong>。<br>"
    "想要「模型自己判断该不该查」的那种，去左边的 <strong>🧭 智能助手</strong>。"
)

top_k = common.sidebar(top_k_slider=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 空状态：给几个能立刻点的例子。一片空白对第一次来的人太不友好。
if not st.session_state.messages:
    st.markdown("")
    samples = [
        "我从小米那次面试里学到了什么？",
        "我怎么防止模型编答案？",
        "检索命中率是怎么评测的？",
    ]
    cols = st.columns(3)
    for column, sample in zip(cols, samples):
        with column:
            if st.button(sample, width="stretch", key=f"sample_{sample}"):
                st.session_state.pending = sample
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        common.show_citations(message.get("citations", []))

question = st.chat_input("例如：我给自己定的检索目标是多少？") or st.session_state.pop(
    "pending", None
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        buffer = ""
        citations: list[dict] = []
        try:
            for event in common.stream_answer(question, top_k):
                if event["type"] == "sources":
                    citations = event.get("citations", [])
                elif event["type"] == "delta":
                    buffer += event["text"]
                    placeholder.markdown(buffer + "▌")
                elif event["type"] == "error":
                    st.error(event["message"])
            placeholder.markdown(buffer or "_（没有返回内容）_")
            common.show_citations(citations)
        except httpx.HTTPStatusError as exc:
            st.error(f"请求失败：{exc.response.status_code} {exc.response.text[:200]}")
        except httpx.RequestError:
            st.error("连不上后端，请确认 `python -m app.main` 正在运行。")

    st.session_state.messages.append(
        {"role": "assistant", "content": buffer, "citations": citations}
    )
