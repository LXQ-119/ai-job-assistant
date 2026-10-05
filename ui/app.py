"""Streamlit 前端。

它通过 HTTP 调用 FastAPI，**不直接 import app 里的 service**。
这个边界很重要：前后端分离后，同一套 API 可以同时给网页、命令行、
甚至别人的程序用；将来换前端（比如换成 React）也不需要动后端一行代码。

启动：
    streamlit run ui/app.py
"""

from __future__ import annotations

import json

import httpx
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = httpx.Timeout(120.0, connect=5.0)

st.set_page_config(page_title="AI 求职助手", page_icon="🎯", layout="wide")


# ---------------------------------------------------------------------------
# 与后端通信
# ---------------------------------------------------------------------------


def api_get(path: str) -> dict | None:
    try:
        response = httpx.get(f"{API_BASE}{path}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"接口返回 {exc.response.status_code}：{exc.response.text[:300]}")
    except httpx.RequestError:
        st.error(
            f"连不上后端 {API_BASE}。请先在另一个终端启动：`python -m app.main`"
        )
    return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        response = httpx.post(f"{API_BASE}{path}", json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        st.error(f"接口返回 {exc.response.status_code}：{detail}")
    except httpx.RequestError:
        st.error(
            f"连不上后端 {API_BASE}。请先在另一个终端启动：`python -m app.main`"
        )
    return None


def stream_answer(question: str, top_k: int):
    """消费 SSE 流式接口，逐段产出事件。"""
    with httpx.stream(
        "POST",
        f"{API_BASE}/kb/stream",
        json={"question": question, "top_k": top_k},
        timeout=TIMEOUT,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[len("data: ") :]
            if raw.strip() == "[DONE]":
                break
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

st.sidebar.title("🎯 AI 求职助手")
st.sidebar.caption("简历解析 + 面经知识库问答")

health = api_get("/health")
if health:
    if health["api_key_configured"]:
        st.sidebar.success(f"模型：{health['model']}")
    else:
        st.sidebar.error("未配置 LLM_API_KEY —— 请先填 .env")
    st.sidebar.metric("知识库块数", health["knowledge_chunks"])
    st.sidebar.caption(f"向量检索：{health['embedding_backend']}")

st.sidebar.divider()
top_k = st.sidebar.slider("检索条数 top_k", 1, 10, 4)

if st.sidebar.button("重建知识库索引", width="stretch"):
    with st.spinner("正在读取 data/knowledge 并切块……"):
        result = api_post("/kb/index", {"paths": [], "rebuild": True})
    if result:
        st.sidebar.success(
            f"索引完成：{result['indexed_files']} 个文件 / {result['total_chunks']} 个块"
        )

status = api_get("/kb/status")
if status:
    if status["warnings"]:
        for warning in status["warnings"]:
            st.sidebar.warning(warning)
    if status["sources"]:
        with st.sidebar.expander(f"已索引文件（{len(status['sources'])}）"):
            for name in status["sources"]:
                st.write(f"- {name}")


# ---------------------------------------------------------------------------
# 主区域
# ---------------------------------------------------------------------------

tab_chat, tab_resume, tab_metrics = st.tabs(["💬 面经问答", "📄 简历解析", "📊 评测与成本"])

# --- 问答 -------------------------------------------------------------------

with tab_chat:
    st.subheader("问面经知识库")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                label = citation["source"]
                if citation.get("heading"):
                    label = f"{label} · {citation['heading']}"
                with st.expander(f"来源：{label}（相关度 {citation['score']}）"):
                    st.text(citation["text"])

    question = st.chat_input("例如：RAG 的命中率怎么评测？")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            buffer = ""
            citations: list[dict] = []
            try:
                for event in stream_answer(question, top_k):
                    if event["type"] == "sources":
                        citations = event.get("citations", [])
                    elif event["type"] == "delta":
                        buffer += event["text"]
                        placeholder.markdown(buffer + "▌")
                    elif event["type"] == "error":
                        st.error(event["message"])
                placeholder.markdown(buffer or "_（没有返回内容）_")

                for citation in citations:
                    label = citation["source"]
                    if citation.get("heading"):
                        label = f"{label} · {citation['heading']}"
                    with st.expander(f"来源：{label}（相关度 {citation['score']}）"):
                        st.text(citation["text"])
            except httpx.HTTPStatusError as exc:
                st.error(f"请求失败：{exc.response.status_code} {exc.response.text[:200]}")
            except httpx.RequestError:
                st.error("连不上后端，请确认 `python -m app.main` 正在运行。")

        st.session_state.messages.append(
            {"role": "assistant", "content": buffer, "citations": citations}
        )

# --- 简历 -------------------------------------------------------------------

with tab_resume:
    st.subheader("把简历变成结构化数据")

    uploaded = st.file_uploader("上传简历（.md / .txt / .pdf / .docx）", type=["md", "txt", "pdf", "docx"])
    pasted = st.text_area("或者直接粘贴简历正文", height=200)

    if st.button("开始解析", type="primary"):
        if uploaded is not None:
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                response = httpx.post(f"{API_BASE}/resume/upload", files=files, timeout=TIMEOUT)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                st.error(f"解析失败：{exc.response.text[:400]}")
                data = None
            except httpx.RequestError:
                st.error("连不上后端。")
                data = None
        elif pasted.strip():
            data = api_post("/resume/parse", {"text": pasted})
        else:
            st.warning("请先上传文件或粘贴内容。")
            data = None

        if data:
            profile = data["profile"]
            col1, col2, col3 = st.columns(3)
            col1.metric("耗时", f"{data['latency_ms']:.0f} ms")
            col2.metric("成本", f"${data['cost_usd']:.6f}")
            col3.metric("尝试次数", data["attempts"])

            if data["attempts"] > 1:
                st.info("第一次输出没通过校验，触发了修复重试——这在录屏里是很好的素材。")

            st.markdown("### 结构化结果")
            st.json(profile)

            if profile.get("skills"):
                st.markdown("### 技能标签")
                st.write("　".join(f"`{s}`" for s in profile["skills"]))

# --- 指标 -------------------------------------------------------------------

with tab_metrics:
    st.subheader("调用埋点与成本")

    data = api_get("/metrics")
    if data:
        summary = data["summary"]
        rate = data["usd_to_cny"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("调用次数", summary["calls"])
        col2.metric("总成本", f"${summary['cost_usd']:.4f}", f"≈ ¥{summary['cost_usd'] * rate:.3f}")
        latency = summary["latency_ms"]
        col3.metric("P50 延迟", f"{latency['p50']} ms" if latency["p50"] else "—")
        col4.metric("P95 延迟", f"{latency['p95']} ms" if latency["p95"] else "—")

        col5, col6, col7 = st.columns(3)
        col5.metric("总 token", summary["total_tokens"])
        cache_rate = summary["cache_hit_rate"]
        col6.metric("缓存命中率", f"{cache_rate:.1%}" if cache_rate is not None else "—")
        success = summary["success_rate"]
        col7.metric("成功率", f"{success:.1%}" if success is not None else "—")

        if summary["by_purpose"]:
            st.markdown("### 按场景拆分")
            st.dataframe(
                [
                    {"场景": purpose, **values}
                    for purpose, values in summary["by_purpose"].items()
                ],
                width="stretch",
            )

        if data["recent"]:
            with st.expander("最近调用明细"):
                st.dataframe(data["recent"], width="stretch")

        if st.button("清空埋点"):
            api_post("/metrics/reset", {})
            st.rerun()
