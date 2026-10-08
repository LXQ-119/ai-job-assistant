"""页面之间共用的东西：后端连接、侧边栏、引用渲染。

抽出来的理由很实在：6 个页面里有 5 个都要调后端、都有侧边栏、都要渲染引用。
复制 5 遍的话，改一次后端地址得改 5 个文件 —— 总有一个会漏。
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

import style

# 后端地址可配置：默认本机 8000，也可以用环境变量指到别的机器。
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(240.0, connect=5.0)


def api_get(path: str, *, quiet: bool = False) -> dict | None:
    try:
        response = httpx.get(f"{API_BASE}{path}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        if not quiet:
            st.error(f"接口返回 {exc.response.status_code}：{exc.response.text[:300]}")
    except httpx.RequestError:
        if not quiet:
            st.error(f"连不上后端 {API_BASE}。请先启动：`python -m app.main`")
    return None


def api_post(path: str, payload: dict, *, timeout: float | None = None) -> dict | None:
    try:
        response = httpx.post(
            f"{API_BASE}{path}",
            json=payload,
            timeout=timeout or TIMEOUT,
        )
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
        st.error(f"连不上后端 {API_BASE}。请先启动：`python -m app.main`")
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


def show_citations(citations: list[dict]) -> None:
    """渲染引用来源。按 (文件, 标题) 去重，避免同一个块重复显示。"""
    if not citations:
        return
    seen: set[tuple] = set()
    for citation in citations:
        key = (citation.get("source"), citation.get("heading"))
        if key in seen:
            continue
        seen.add(key)
        style.source(
            citation.get("source", "?"),
            citation.get("heading") or "（无标题）",
            citation.get("text", ""),
        )


def sidebar(*, top_k_slider: bool = False) -> int:
    """统一的侧边栏。返回 top_k（不需要时返回默认 4）。"""
    top_k = 4
    with st.sidebar:
        st.markdown("### 🎯 AI 求职助手")
        st.caption("简历解析 · 面经知识库 · 成本监控")

        health = api_get("/health", quiet=True)
        if health:
            if health["api_key_configured"]:
                st.success(f"模型在线 ｜ `{health['model']}`", icon="✅")
                if health.get("provider"):
                    st.caption(
                        f"服务商：{health['provider']}　·　"
                        "换模型去左侧「⚙️ 模型设置」"
                    )
            else:
                st.error("当前服务商还没配 API Key —— 去「⚙️ 模型设置」填")

            c1, c2 = st.columns(2)
            c1.metric("知识库块数", health["knowledge_chunks"])
            c2.metric(
                "向量检索", "开" if health["embedding_backend"] != "none" else "关"
            )
        else:
            st.error("后端未响应")

        st.divider()

        if top_k_slider:
            st.markdown("##### ⚙️ 检索设置")
            top_k = st.slider("检索条数 top_k", 1, 10, 4, help="每次送几张卡片给模型")

        if st.button("🔄 重建知识库索引", width="stretch"):
            with st.spinner("正在读取 data/knowledge 并切块……"):
                result = api_post("/kb/index", {"paths": [], "rebuild": True})
            if result:
                st.success(
                    f"完成：{result['indexed_files']} 个文件 / {result['total_chunks']} 个块"
                )

        status = api_get("/kb/status", quiet=True)
        if status:
            for warning in status["warnings"]:
                st.warning(warning)
            if status["sources"]:
                with st.expander(f"📁 已索引文件（{len(status['sources'])}）"):
                    for name in status["sources"]:
                        st.markdown(f"- `{name}`")

    return top_k
