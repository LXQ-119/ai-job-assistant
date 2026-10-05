"""RAG 的最小骨架 —— 去掉所有工程细节，只剩核心。

看完这个文件你就会发现：**RAG 的代码结构是固定的两段。**

    ┌─ 离线（建索引，只做一次）────────────────────┐
    │   读文档  →  切块  →  存起来                 │
    └──────────────────────────────────────────────┘

    ┌─ 在线（每次提问都跑一遍）────────────────────┐
    │   问题  →  检索出最相关的几块  →             │
    │   拼进提示词  →  让模型回答                  │
    └──────────────────────────────────────────────┘

**所有 RAG 系统都是这个骨架。** 区别只在于每一步做得多细致：

    这个文件（教学版）          项目里的工程版
    ─────────────────────      ─────────────────────────────
    按空行粗略切块             三层策略：标题面包屑 + 段落累积 + 硬切重叠
    数重叠的词（很粗糙）        BM25 算法（IDF + 词频饱和 + 长度归一化）
    没考虑中文分词              用 jieba 分词 + 停用词过滤
    没有引用                    每个结论标 [1][2]，可点开核对原文
    没处理"找不到"              检索为空时直接返回固定话术，不浪费调用
    没有重试                    指数退避重试 + 成本埋点 + 降级
    同步调用                    SSE 流式输出（首字延迟 3s → 0.6s）

用法：
    python examples/rag_minimal.py
    python examples/rag_minimal.py "你的问题"
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


# ===========================================================================
# 离线：建索引（读文档 → 切块 → 存起来）
# ===========================================================================


def load_documents(folder: Path) -> str:
    """把所有 .md 文件读成一大段文本。"""
    parts = []
    for path in sorted(folder.glob("*.md")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def split_into_chunks(text: str, size: int = 500) -> list[str]:
    """切块：按空行分段，累积到大约 size 字就切一块。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) < size:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


# ===========================================================================
# 在线：检索（问题 → 找出最相关的几块）
# ===========================================================================


def keywords(text: str) -> set[str]:
    """把文本变成一组"特征词"。

    中文没有空格，所以这里用**二字组合**当特征（"检索算法" → 检索/索算/算法）。
    项目里用的是 jieba 分词，比这个准得多。
    """
    lower = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", lower))          # 英文单词
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", lower)          # 只留汉字
    tokens |= {cjk[i : i + 2] for i in range(len(cjk) - 1)}  # 二字组合
    return tokens


def search(question: str, chunks: list[str], k: int = 3) -> list[tuple[float, str]]:
    """检索：算问题里的词在每块里出现了几个，取前 k 名。"""
    question_keys = keywords(question)
    scored: list[tuple[float, str]] = []
    for chunk in chunks:
        overlap = len(question_keys & keywords(chunk))
        scored.append((overlap, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


# ===========================================================================
# 在线：生成（拼提示词 → 让模型回答）
# ===========================================================================

SYSTEM_PROMPT = """你是一个严谨的助手。

规则：
1. 只能依据【参考资料】回答，不要使用资料之外的知识。
2. 资料不足以回答时，直接说"现有资料中没有提到这一点"，绝对不要编造。
3. 每个结论后面用 [编号] 标注来源。
4. 用中文回答，控制在 200 字以内。"""


def answer(question: str, chunks: list[str], k: int = 3) -> str:
    """完整的在线流程：检索 → 拼提示词 → 调模型。"""
    hits = search(question, chunks, k)

    # 把检索到的内容编号，拼成"参考资料"
    context = "\n\n".join(
        f"[{number}] {text}" for number, (_, text) in enumerate(hits, start=1)
    )

    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        timeout=60.0,
    )
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-flash"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】\n{question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


# ===========================================================================
# 串起来
# ===========================================================================


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else "切块的时候 overlap 有什么用？"

    print("=" * 70)
    print(" RAG 最小骨架演示")
    print("=" * 70)

    # ---- 离线部分 ----
    print("\n【离线】读文档 → 切块")
    text = load_documents(PROJECT_ROOT / "data" / "knowledge")
    chunks = split_into_chunks(text)
    print(f"  读了 {len(text)} 个字，切成 {len(chunks)} 块")
    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"   第 {i} 块（{len(chunk)} 字）：{chunk[:45].replace(chr(10), ' ')}…")

    # ---- 在线部分 ----
    print(f"\n【在线】问题：{question}")

    print("\n  第 1 步：检索（算每块有几个词和问题重合）")
    hits = search(question, chunks, 3)
    for rank, (score, chunk) in enumerate(hits, start=1):
        print(f"   第 {rank} 名  重合 {score:.0f} 个词  {chunk[:45].replace(chr(10), ' ')}…")

    print("\n  第 2 步：拼提示词（把前 3 块编号后塞进去）")
    print("   ┌─────────────────────────────────────────────┐")
    print("   │ system: 只能依据参考资料，不足就说不知道        │")
    print("   │ user:   【参考资料】[1]… [2]… [3]…          │")
    print(f"   │         【问题】{question[:16]}…    │")
    print("   └─────────────────────────────────────────────┘")

    print("\n  第 3 步：让模型回答")
    if not os.getenv("LLM_API_KEY"):
        print("   没有 API Key，跳过。")
        return 0
    print()
    print(answer(question, chunks, 3))

    print()
    print("=" * 70)
    print(" 这 60 行就是 RAG 的全部逻辑。")
    print(" 项目里那几百行，都是在把这 3 步做得更细致、更稳、更省。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
