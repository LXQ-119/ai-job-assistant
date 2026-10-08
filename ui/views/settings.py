"""⚙️ 模型设置 —— 把模型接入做成可插拔的。

为什么这件事值得单独做一页：

    **几乎所有主流服务商都兼容 OpenAI 协议。**
    所以"支持多家模型"不需要为每家写适配器 ——
    只要把 base_url / api_key / model 三样做成**运行时配置**，
    同一套代码就能接：DeepSeek、OpenAI、通义、智谱、Kimi、硅基流动、
    本地 Ollama / vLLM / LM Studio……

配置存两层：

    .env                      默认值（首次启动用）
    data/runtime/model.json   网页上切的那份，**不用重启就生效**

第二层在 .gitignore 里 —— 因为它含 API Key。
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import style  # noqa: E402

style.hero(
    "⚙️ 模型设置",
    "换一家模型不用改代码、不用重启 —— 因为大家都兼容同一套协议",
    ["OpenAI 兼容协议", "运行时切换", "先测再切"],
)

data = common.api_get("/providers")
if not data:
    st.stop()

active = data["active"]
presets = data["presets"]

# ---------------------------------------------------------------------------
# 当前生效的配置
# ---------------------------------------------------------------------------

st.markdown("### 当前正在用")

c1, c2, c3, c4 = st.columns(4)
c1.metric("服务商", active["provider_name"].split("（")[0])
c2.metric("模型", active["model"])
c3.metric("密钥", active["api_key_masked"] if active["api_key_configured"] else "未配置")
c4.metric("配置来源", active["source_label"])

style.hint(
    f"接口地址：<code>{active['base_url']}</code><br>"
    f"运行时配置文件：<code>{data['runtime_file']}</code><br>"
    "这个文件<strong>含 API Key</strong>，已经在 .gitignore 里 —— "
    "把密钥写进会被提交的文件，是这类项目最常见的事故。"
)

st.divider()

# ---------------------------------------------------------------------------
# 切换
# ---------------------------------------------------------------------------

st.markdown("### 换一个")

labels = [p["name"] for p in presets]
default_index = next(
    (i for i, p in enumerate(presets) if p["key"] == active["provider"]), 0
)
choice = st.selectbox("选择服务商", labels, index=default_index)
preset = presets[labels.index(choice)]

if preset["note"]:
    style.hint(preset["note"])

# 每个预设用独立的 key，这样切换服务商时输入框会跟着换成对应的默认值。
# 如果共用一个 key，Streamlit 会保留上一次输入的内容 —— 这是很常见的坑。
c1, c2 = st.columns([3, 2])
base_url = c1.text_input(
    "接口地址 base_url",
    value=preset["base_url"],
    key=f"url_{preset['key']}",
    placeholder="https://api.example.com/v1",
)
default_model = preset["models"][0] if preset["models"] else ""
if preset["models"]:
    model = c2.selectbox("模型名", preset["models"], key=f"model_{preset['key']}")
else:
    model = c2.text_input("模型名", key=f"model_{preset['key']}", placeholder="模型名")

if preset["needs_key"]:
    api_key = st.text_input(
        "API Key",
        type="password",
        key=f"key_{preset['key']}",
        placeholder="sk-...（本地模型这一栏可以留空）",
        help="Key 只会写进 data/runtime/model.json，不会进 git，也不会出现在日志里",
    )
else:
    api_key = ""
    st.info("这个服务商不需要 API Key。", icon="🔓")

b1, b2, b3 = st.columns([2, 2, 2])

payload = {
    "provider": preset["key"],
    "base_url": base_url,
    "api_key": api_key,
    "model": model,
}

if b1.button("🔌 测试连接", width="stretch"):
    with st.spinner("正在发一个最小请求……"):
        result = common.api_post("/providers/test", payload, timeout=60)
    if result:
        st.session_state.provider_test = result

if b2.button("✅ 切换到这个模型", type="primary", width="stretch"):
    if not base_url.strip():
        st.error("接口地址不能为空。")
    elif not model.strip():
        st.error("模型名不能为空。")
    else:
        result = common.api_post("/providers/switch", payload)
        if result:
            st.success(
                f"已切换到 **{result['active']['provider_name']}** ／ "
                f"`{result['active']['model']}` —— **立即生效，不用重启**"
            )
            st.session_state.pop("provider_test", None)
            st.rerun()

if b3.button("↩️ 回到 .env 默认", width="stretch"):
    result = common.api_post("/providers/reset", {})
    if result:
        st.info(f"已回到 .env 默认：`{result['active']['model']}`")
        st.rerun()

# 测试结果
test = st.session_state.get("provider_test")
if test:
    if test["ok"]:
        st.success(
            f"✅ 连接成功 ｜ {test['latency_ms']} ms ｜ 模型回：{test['reply']}"
        )
    else:
        st.error(f"❌ 连接失败：{test['message']}")
        st.caption(
            "常见原因：base_url 写错（有的服务商必须带 `/v1`，有的必须不带）、"
            "Key 无效或没额度、模型名不存在。"
        )

st.divider()

# ---------------------------------------------------------------------------
# 预设清单
# ---------------------------------------------------------------------------

st.markdown("### 内置的服务商预设")

rows = []
for item in presets:
    rows.append(
        {
            "服务商": item["name"],
            "base_url": item["base_url"] or "（自己填）",
            "常见模型": "、".join(item["models"]) or "（自己填）",
            "要 Key": "需要" if item["needs_key"] else "不需要",
        }
    )
st.dataframe(rows, width="stretch")

style.hint(
    "<strong>为什么加一家服务商不用写新代码？</strong><br>"
    "因为这个项目调模型走的是 OpenAI SDK，而主流服务商都实现同一套 HTTP 协议。"
    "所谓「接入新模型」，本质只是换三个字符串："
    "<code>base_url</code>、<code>api_key</code>、<code>model</code>。<br>"
    "代码层面唯一要做的事，是把这三样从「启动时读死的配置」"
    "变成「调用时现读的配置」—— 也就是这页在做的事。"
)
