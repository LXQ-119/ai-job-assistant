"""📊 评测与成本 —— 每次调用都埋点，量化到能写进简历。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import style  # noqa: E402

style.hero(
    "📊 评测与成本",
    "不测量就无法优化 —— 每次调用的 token、成本、延迟都记在这里",
    ["P50 / P95", "缓存命中率", "峰谷计价", "按场景拆分"],
)

style.hint(
    "延迟要盯 <strong>P95 而不是平均值</strong>：平均值会把少数很慢的请求藏起来，"
    "而用户只记得那几次卡顿。<br>"
    "成本要区分<strong>缓存命中/未命中</strong>，还要区分 <strong>DeepSeek 的高峰/非高峰</strong>"
    "—— 高峰单价是非高峰的两倍。"
)

common.sidebar()

data = common.api_get("/metrics")
if data:
    summary = data["summary"]
    rate = data["usd_to_cny"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("调用次数", summary["calls"])
    col2.metric(
        "总成本", f"${summary['cost_usd']:.4f}", f"≈ ¥{summary['cost_usd'] * rate:.3f}"
    )
    latency = summary["latency_ms"]
    col3.metric("P50 延迟", f"{latency['p50']} ms" if latency["p50"] else "—")
    col4.metric("P95 延迟", f"{latency['p95']} ms" if latency["p95"] else "—")

    col5, col6, col7 = st.columns(3)
    col5.metric("总 token", f"{summary['total_tokens']:,}")
    cache_rate = summary["cache_hit_rate"]
    col6.metric("缓存命中率", f"{cache_rate:.1%}" if cache_rate is not None else "—")
    success = summary["success_rate"]
    col7.metric("成功率", f"{success:.1%}" if success is not None else "—")

    if summary["by_purpose"]:
        st.markdown("#### 按场景拆分")
        st.dataframe(
            [
                {"场景": purpose, **values}
                for purpose, values in summary["by_purpose"].items()
            ],
            width="stretch",
        )

    if data["recent"]:
        with st.expander("🕒 最近调用明细"):
            st.dataframe(data["recent"], width="stretch")

    if st.button("清空埋点"):
        common.api_post("/metrics/reset", {})
        st.rerun()

st.divider()
st.markdown("#### 🔬 检索效果评测")
style.hint(
    "跑 <code>python -m scripts.eval_rag</code> 得到 30 条评测集的完整报告。"
    "当前配置（中文二元组分词 + 最低分阈值 11.5）："
)

e1, e2, e3 = st.columns(3)
e1.metric("负例拒答率", "100%", "0% → 100%", delta_color="normal")
e2.metric("关键词覆盖率", "93.75%", "84.38% → 93.75%")
e3.metric("MRR", "0.859", "0.935 → 0.859")

style.hint(
    "Hit@k 目前是 100%，但<strong>这个数字没有区分度</strong> —— "
    "知识库里只有 3 个文件，翻出任何一张卡都来自这 3 个文件。<br>"
    "三次踩坑的完整记录见 <code>INTERVIEW.md</code> 第七节。"
)
