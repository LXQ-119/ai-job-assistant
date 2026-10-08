"""📄 简历解析 —— 大模型抽取 + Pydantic 契约校验 + 失败自动修复重试。"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import style  # noqa: E402

style.hero(
    "📄 简历解析",
    "把一份简历变成确定的字段 —— 靠的不是提示词，是 Pydantic 输出契约",
    ["Pydantic 校验", "失败自动修复重试", "结构化抽取"],
)

style.hint(
    "模型返回的 JSON 只保证「是合法 JSON」，<strong>不保证字段符合预期</strong>"
    "（比如工作年限会返回 '3年' 这种字符串）。<br>"
    "所以这里用 Pydantic 定义输出契约，校验失败时把<strong>具体报错 + 模型的原始输出</strong>"
    "一起回传让它修正重试。"
)

common.sidebar()

uploaded = st.file_uploader(
    "上传简历（.md / .txt / .pdf / .docx）", type=["md", "txt", "pdf", "docx"]
)
pasted = st.text_area("或者直接粘贴简历正文", height=180)

if st.button("▶ 开始解析", type="primary"):
    data = None

    if uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue())}
        try:
            response = httpx.post(
                f"{common.API_BASE}/resume/upload", files=files, timeout=common.TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            st.error(f"解析失败：{exc.response.text[:400]}")
        except httpx.RequestError:
            st.error("连不上后端。")
    elif pasted.strip():
        data = common.api_post("/resume/parse", {"text": pasted})
    else:
        st.warning("请先上传文件或粘贴内容。")

    if data:
        profile = data["profile"]

        m1, m2, m3 = st.columns(3)
        m1.metric("耗时", f"{data['latency_ms']:.0f} ms")
        m2.metric("成本", f"${data['cost_usd']:.6f}")
        m3.metric("尝试次数", data["attempts"])

        if data["attempts"] > 1:
            st.info(
                "第一次输出没通过校验，触发了**修复重试** —— 这正是 Pydantic 契约的价值。"
            )

        st.markdown("#### 结构化结果")
        st.json(profile)

        if profile.get("skills"):
            st.markdown("#### 技能标签")
            style.pills(profile["skills"])
