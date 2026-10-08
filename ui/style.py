"""统一的视觉层 —— 所有页面共用一套样式和组件。

Streamlit 的默认样式"能用但不好看"，主要是三个问题：

    1. 没有视觉层次 —— 标题、正文、指标长得都差不多，眼睛不知道该看哪
    2. 没有留白和分组 —— 信息糊成一片
    3. 没有品牌感 —— 看起来像内部工具，不像一个产品

这个文件用一段 CSS 把上面三点补掉。**改一处，所有页面生效** ——
这就是"样式要集中管理"的意义（分散写 CSS 的项目，改个颜色要翻十个文件）。
"""

from __future__ import annotations

from typing import Iterable

import streamlit as st

# ---------------------------------------------------------------------------
# 设计变量：颜色只在这里定义一次
# ---------------------------------------------------------------------------

PRIMARY = "#4f46e5"
PRIMARY_LIGHT = "#7c3aed"
ACCENT = "#a855f7"
INK = "#0f172a"
INK_SOFT = "#475569"
MUTED = "#94a3b8"
LINE = "#e2e8f0"
OK = "#10b981"
WARN = "#f59e0b"
BAD = "#ef4444"

_CSS = f"""
<style>
/* ===================== 全局 ===================== */
.stApp {{
  background:
    radial-gradient(1100px 520px at 8% -12%, #eef2ff 0%, rgba(238,242,255,0) 62%),
    radial-gradient(880px 460px at 96% -4%, #f5f3ff 0%, rgba(245,243,255,0) 58%),
    #f8fafc;
}}
.block-container {{ padding-top: 2rem; padding-bottom: 3.5rem; max-width: 1150px; }}

html, body, [class*="css"] {{
  font-family: "Inter", "PingFang SC", "Microsoft YaHei", -apple-system,
               BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

h1 {{ font-size: 1.9rem !important; font-weight: 760 !important;
      letter-spacing: -0.022em; color: {INK}; }}
h2 {{ font-size: 1.24rem !important; font-weight: 700 !important;
      letter-spacing: -0.014em; color: {INK}; margin-top: .3rem !important; }}
h3 {{ font-size: 1.04rem !important; font-weight: 650 !important; color: #1e293b; }}
p, li {{ color: {INK_SOFT}; line-height: 1.75; }}

hr {{ border-color: {LINE}; margin: 1.4rem 0; }}

/* ===================== 英雄区 ===================== */
.hero {{
  position: relative; overflow: hidden;
  padding: 1.55rem 1.75rem 1.5rem;
  border-radius: 20px;
  background: linear-gradient(118deg, {PRIMARY} 0%, {PRIMARY_LIGHT} 52%, {ACCENT} 100%);
  box-shadow: 0 18px 44px -20px rgba(79,70,229,.78);
  margin-bottom: 1.5rem;
}}
.hero::after {{
  content: ""; position: absolute; right: -70px; top: -90px;
  width: 260px; height: 260px; border-radius: 50%;
  background: rgba(255,255,255,.13);
}}
.hero h1 {{
  color: #fff !important; margin: 0 0 .3rem 0 !important;
  font-size: 1.78rem !important; position: relative; z-index: 1;
}}
.hero p {{ color: rgba(255,255,255,.9); margin: 0; font-size: .93rem;
           position: relative; z-index: 1; }}
.hero .tags {{ margin-top: .85rem; display: flex; gap: .4rem;
               flex-wrap: wrap; position: relative; z-index: 1; }}
.hero .tag {{
  font-size: .74rem; padding: .2rem .62rem; border-radius: 999px;
  background: rgba(255,255,255,.17);
  border: 1px solid rgba(255,255,255,.3);
  color: #fff; white-space: nowrap;
}}

/* ===================== 指标卡 ===================== */
div[data-testid="stMetric"] {{
  background: #fff;
  border: 1px solid {LINE};
  border-radius: 14px;
  padding: .85rem 1rem .8rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.04);
  transition: box-shadow .16s ease, transform .16s ease;
}}
div[data-testid="stMetric"]:hover {{
  box-shadow: 0 10px 26px -14px rgba(15,23,42,.28);
  transform: translateY(-1px);
}}
div[data-testid="stMetricLabel"] p {{
  font-size: .76rem !important; color: {MUTED} !important;
  font-weight: 600 !important; letter-spacing: .02em;
}}
div[data-testid="stMetricValue"] {{
  font-size: 1.42rem !important; font-weight: 720 !important; color: {INK} !important;
}}

/* ===================== 展开框 ===================== */
div[data-testid="stExpander"] {{
  border: 1px solid {LINE} !important;
  border-radius: 12px !important;
  background: #fff !important;
  overflow: hidden;
}}
div[data-testid="stExpander"] summary {{
  font-size: .86rem !important; font-weight: 600 !important; color: {INK_SOFT} !important;
}}
div[data-testid="stExpander"] summary:hover {{ color: {PRIMARY} !important; }}

/* ===================== 按钮 ===================== */
.stButton > button, .stDownloadButton > button {{
  border-radius: 11px !important;
  border: 1px solid {LINE} !important;
  font-weight: 600 !important;
  font-size: .87rem !important;
  transition: all .15s ease;
  background: #fff;
}}
.stButton > button:hover {{
  border-color: {PRIMARY} !important;
  color: {PRIMARY} !important;
  background: #f5f3ff !important;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(115deg, {PRIMARY}, {PRIMARY_LIGHT}) !important;
  border: none !important; color: #fff !important;
  box-shadow: 0 8px 20px -10px rgba(79,70,229,.85);
}}
.stButton > button[kind="primary"]:hover {{
  filter: brightness(1.07); color: #fff !important;
}}

/* ===================== 标签页 ===================== */
.stTabs [data-baseweb="tab-list"] {{
  gap: .35rem; border-bottom: 1px solid {LINE};
}}
.stTabs [data-baseweb="tab"] {{
  height: 42px; border-radius: 10px 10px 0 0;
  padding: 0 1.05rem; font-weight: 620; font-size: .92rem; color: {MUTED};
}}
.stTabs [aria-selected="true"] {{ color: {PRIMARY} !important; background: #eef2ff; }}
/* 默认那条高亮横杠是 Streamlit 自带的红色，和主题冲突，必须覆盖 */
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {PRIMARY} !important; }}
.stTabs [data-baseweb="tab-border"] {{ background-color: {LINE} !important; }}

/* ===================== 侧边栏导航 ===================== */
section[data-testid="stSidebar"] nav a {{
  border-radius: 10px !important;
  padding: .34rem .6rem !important;
  transition: background .14s ease;
}}
section[data-testid="stSidebar"] nav a:hover {{
  background: #eef2ff !important;
}}
section[data-testid="stSidebar"] nav a[aria-current="page"] {{
  background: linear-gradient(115deg, #eef2ff, #f5f3ff) !important;
  box-shadow: inset 3px 0 0 {PRIMARY};
}}
section[data-testid="stSidebar"] nav span {{
  font-size: .885rem !important; font-weight: 560 !important;
}}
section[data-testid="stSidebar"] h3 {{ font-size: 1rem !important; }}

/* ===================== 聊天气泡 ===================== */
div[data-testid="stChatMessage"] {{
  background: #fff;
  border: 1px solid {LINE};
  border-radius: 15px;
  padding: .95rem 1.1rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.035);
}}

/* ===================== 自定义组件 ===================== */
.card {{
  background: #fff; border: 1px solid {LINE}; border-radius: 15px;
  padding: 1.05rem 1.2rem; margin-bottom: .85rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.04);
}}
.card.flat {{ box-shadow: none; background: #fcfdff; }}
.card .t {{ font-weight: 680; color: {INK}; font-size: .97rem; margin-bottom: .3rem; }}
.card .d {{ color: {INK_SOFT}; font-size: .87rem; line-height: 1.72; }}

.route {{
  display: inline-flex; align-items: center; gap: .4rem;
  font-size: .82rem; font-weight: 640;
  padding: .28rem .7rem; border-radius: 999px;
  background: #eef2ff; color: {PRIMARY}; border: 1px solid #ddd6fe;
}}
.route.plain {{ background: #f1f5f9; color: {INK_SOFT}; border-color: {LINE}; }}
.route.calc  {{ background: #ecfdf5; color: #047857; border-color: #a7f3d0; }}
.route.miss  {{ background: #fffbeb; color: #b45309; border-color: #fde68a; }}

.step {{
  border-left: 3px solid {LINE}; padding: .1rem 0 .1rem .95rem;
  margin: .55rem 0;
}}
.step.decide {{ border-left-color: {PRIMARY_LIGHT}; }}
.step.run    {{ border-left-color: {OK}; }}
.step .h {{
  font-size: .86rem; font-weight: 660; color: {INK}; margin-bottom: .2rem;
}}
.step .b {{
  font-size: .82rem; color: {INK_SOFT};
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  word-break: break-all;
}}

.src {{
  border: 1px solid {LINE}; border-left: 3px solid #c7d2fe;
  border-radius: 10px; padding: .6rem .8rem; margin-bottom: .45rem;
  background: #fcfdff;
}}
.src .h {{ font-size: .8rem; font-weight: 660; color: {PRIMARY}; margin-bottom: .22rem; }}
.src .b {{ font-size: .81rem; color: {INK_SOFT}; line-height: 1.68;
           max-height: 190px; overflow: auto; white-space: pre-wrap; }}

.hint {{
  font-size: .84rem; color: {MUTED}; line-height: 1.7;
  padding: .1rem 0 .1rem .9rem; border-left: 2px solid {LINE};
}}
.pill {{
  display:inline-block; font-size:.74rem; font-weight:600;
  padding:.15rem .55rem; border-radius:999px; margin:.1rem .22rem .1rem 0;
  background:#f1f5f9; color:{INK_SOFT}; border:1px solid {LINE};
}}
</style>
"""


def apply() -> None:
    """每个页面开头调一次。"""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 组件
# ---------------------------------------------------------------------------


def hero(title: str, subtitle: str, tags: Iterable[str] = ()) -> None:
    """顶部渐变横幅。比 st.title 有辨识度得多。"""
    chips = "".join(f'<span class="tag">{tag}</span>' for tag in tags)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>'
        f'<div class="tags">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str, *, flat: bool = False) -> None:
    cls = "card flat" if flat else "card"
    st.markdown(
        f'<div class="{cls}"><div class="t">{title}</div>'
        f'<div class="d">{body}</div></div>',
        unsafe_allow_html=True,
    )


def hint(text: str) -> None:
    st.markdown(f'<div class="hint">{text}</div>', unsafe_allow_html=True)


def route_badge(text: str, kind: str = "default") -> str:
    """返回 HTML 字符串，方便调用方自己拼进别的块里。"""
    cls = "route" if kind == "default" else f"route {kind}"
    return f'<span class="{cls}">{text}</span>'


def step(title: str, body: str, kind: str = "") -> None:
    cls = f"step {kind}".strip()
    st.markdown(
        f'<div class="{cls}"><div class="h">{title}</div>'
        f'<div class="b">{body}</div></div>',
        unsafe_allow_html=True,
    )


def source(source: str, heading: str, text: str) -> None:
    st.markdown(
        f'<div class="src"><div class="h">{source} · {heading}</div>'
        f'<div class="b">{text}</div></div>',
        unsafe_allow_html=True,
    )


def pills(items: Iterable[str]) -> None:
    html = "".join(f'<span class="pill">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)
