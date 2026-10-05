"""看看「资料柜」是怎么翻的 —— 把检索过程一步步打印出来。

这是理解 RAG 最直观的一个例子。运行它，你会看到：

    1. 资料柜里到底有什么（文档被切成了多少张"卡片"）
    2. 你的问题被切成了哪些词
    3. 每张卡片各自得了多少分
    4. 最后选了哪几张交给大模型

用法：
    python examples/retrieval_demo.py
    python examples/retrieval_demo.py "切块的时候 overlap 有什么用"
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings                    # noqa: E402
from app.services import rag                           # noqa: E402
from app.services.retriever import KB, _segment        # noqa: E402

BAR = "=" * 70


def main() -> int:
    settings = get_settings()
    question = sys.argv[1] if len(sys.argv) > 1 else "BM25 的 IDF 是干什么用的？"

    # --- 0. 确保资料柜里有东西 ---
    if len(KB) == 0:
        print("资料柜是空的，先建索引……")
        rag.index_paths([], rebuild=True)

    # --- 1. 资料柜里有什么 ---
    print(BAR)
    print(" 第 1 步：资料柜里有什么？")
    print(BAR)
    print(f" 你的知识库文件被切成了 {len(KB)} 张「卡片」（chunk）。每张卡片是这样的：\n")
    for i, chunk in enumerate(KB._chunks, start=1):  # noqa: SLF001 - 演示脚本，直接看内部
        title = chunk.heading or "(文档开头，没有小标题)"
        preview = chunk.text[:38].replace("\n", " ")
        print(f"   {i:2d}. [{title}]")
        print(f"       {preview}…")
    print()
    print(" 为什么要切成小卡片？")
    print("   一份笔记几万字，全塞给模型又贵又慢还容易看花眼。")
    print("   切成小卡片，就能只挑最相关的几张给它。")

    # --- 2. 问题被切成词 ---
    print()
    print(BAR)
    print(" 第 2 步：你的问题被切成了哪些词？")
    print(BAR)
    tokens = _segment(question)
    print(f" 原问题：{question}")
    print(f" 切成词：{tokens}")
    print()
    print(" 这里用 jieba 做中文分词。像「的」「是」「用」这种没信息量的词，")
    print(" 会被停用词表过滤掉 —— 因为它们每张卡片里都有，区分不出哪张重要。")

    # --- 3. 每张卡片得分 ---
    print()
    print(BAR)
    print(" 第 3 步：拿这些词去和每张卡片比对，算分")
    print(BAR)
    scores = KB._bm25_scores(question)  # noqa: SLF001
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    top = scores[0] if scores else 1.0

    print(f" {'卡片':<4} {'得分':>7}   {'条形图':<24} 标题")
    print(" " + "-" * 66)
    for index, score in ranked:
        chunk = KB._chunks[index]  # noqa: SLF001
        title = chunk.heading.split(">")[-1].strip() if chunk.heading else "(开头)"
        bar = "█" * int(round(score / top * 20)) if top > 0 else ""
        print(f" {index + 1:<4} {score:>7.3f}   {bar:<24} {title}")

    print()
    print(" 这就是 BM25 算法在做的事：数关键词。")
    print("  - 你问的词，在某张卡片里出现越多，得分越高")
    print("  - 但这个词在越多卡片里都出现，说明它越没用，权重就越低（这就是 IDF）")

    # --- 4. 选出前几名 ---
    top_k = settings.top_k
    print()
    print(BAR)
    print(f" 第 4 步：取分数最高的 {top_k} 张，交给大模型")
    print(BAR)
    hits = rag.retrieve(question, top_k)
    for number, hit in enumerate(hits, start=1):
        print(f" [卡片 {number}] 相关度 {hit.score:.2f}   来自：{hit.chunk.heading}")
        print(f"    {hit.chunk.text[:90].replace(chr(10), ' ')}…")
        print()

    print(BAR)
    print(" 接下来会发生什么？")
    print(BAR)
    print(f" 这 {len(hits)} 张卡片会被拼进提示词，和你的问题一起发给大模型：")
    print()
    print("     【参考资料】")
    print("     [1] （来源：xxx.md）BM25 是经典的关键词检索算法……")
    print("     [2] （来源：xxx.md）……")
    print()
    print(f"     【问题】{question}")
    print()
    print(" 模型只能依据这些参考资料回答，答不了就说「资料里没有」。")

    # --- 5. 真的问一次 ---
    print()
    print(BAR)
    print(" 第 5 步：真的问一次试试（需要 API Key）")
    print(BAR)
    if not settings.llm_api_key:
        print(" 没有配置 LLM_API_KEY，跳过。配好后重跑本脚本就能看到回答。")
        return 0

    result = rag.answer(question, top_k)
    print(f"\n 模型回答：\n {result.answer}\n")
    print(f" （耗时 {result.latency_ms:.0f} ms，成本 ${result.cost_usd:.6f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
