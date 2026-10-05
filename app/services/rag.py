"""RAG 问答：检索 → 拼提示词 → 生成 → 带引用返回。

提示词设计（这一段的每一句都能在面试里解释"为什么"）：
- 明确"只依据参考资料" → 划定知识边界
- 资料不足时必须说"没有提到" → 给模型一条诚实的退路，
  否则它一定会编（幻觉往往不是能力问题，而是没给它"说不知道"的选项）
- 强制标注 [1][2] → 让引用可核对，也让用户能自己验证
- 要求简洁 → 控制输出 token，直接等于控制成本
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..config import get_settings
from ..llm import LLMError, chat, chat_stream
from ..schemas import Citation, RagAnswer
from .documents import DocumentError, load_and_chunk
from .retriever import KB, ScoredChunk

SYSTEM_PROMPT = """你是一个严谨的技术知识库助手。

必须遵守的规则：
1. 只能依据【参考资料】回答，不要使用资料之外的知识。
2. 如果参考资料不足以回答问题，直接回答"现有资料中没有提到这一点"，
   并说明资料里最接近的内容是什么。绝对不要编造。
3. 每个结论后面用 [编号] 标注来源，例如 [1] 或 [1][3]。
4. 用中文回答，条理清晰，控制在 300 字以内。
5. 不要复述参考资料原文，要用自己的话总结。"""


def index_paths(paths: list[str] | None = None, *, rebuild: bool = False) -> dict:
    """为知识库建索引。paths 为空表示用 data/knowledge 下的全部文件。"""
    settings = get_settings()
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild:
        KB.clear()

    targets: list[Path] = []
    if paths:
        targets = [Path(p) for p in paths]
    else:
        targets = [settings.knowledge_dir]

    if not targets:
        return {"indexed_files": 0, "total_chunks": len(KB), "files": KB.sources}

    chunks, processed = load_and_chunk(
        targets, size=settings.chunk_size, overlap=settings.chunk_overlap
    )

    if rebuild:
        KB.clear()
    KB.add_chunks(chunks)
    KB.save(settings.index_path)

    return {
        "indexed_files": len(processed),
        "total_chunks": len(KB),
        "files": KB.sources,
        "warnings": KB.warnings,
    }


def load_index() -> bool:
    """启动时从磁盘恢复索引，避免每次重启都要重新读文件。"""
    settings = get_settings()
    return KB.load(settings.index_path)


def _build_context(hits: list[ScoredChunk]) -> tuple[str, list[Citation]]:
    """把检索结果拼成带编号的参考资料，同时产出结构化引用。"""
    blocks: list[str] = []
    citations: list[Citation] = []

    for number, hit in enumerate(hits, start=1):
        label = f"{hit.chunk.source}"
        if hit.chunk.heading:
            label = f"{label} · {hit.chunk.heading}"
        blocks.append(f"[{number}] （来源：{label}）\n{hit.chunk.text}")
        citations.append(
            Citation(
                source=hit.chunk.source,
                heading=hit.chunk.heading,
                chunk_index=hit.chunk.index,
                score=round(hit.score, 4),
                text=hit.chunk.text,
            )
        )

    return "\n\n".join(blocks), citations


def retrieve(question: str, top_k: int | None = None) -> list[ScoredChunk]:
    settings = get_settings()
    return KB.search(question, top_k=top_k or settings.top_k)


def _no_hits_message() -> str:
    """检索为空时该对用户说什么。

    这里**必须区分两种情况**，否则会误导用户：

        ① 知识库真的是空的（还没建索引）
           → 提示去建索引，这是对的

        ② 知识库有内容，但这个问题一张卡片都没匹配到
           → 必须说"没找到相关内容"，而不是"知识库是空的"

    之前两处都写死了 ① 的说法。结果用户问一个超出知识库范围的问题
    （比如知识库全是 AI 面试笔记，却问了一句算术题），
    会看到"知识库还是空的，请先建索引" —— 于是跑去反复重建索引，
    而真正的原因是"知识库里没有相关内容"。这是个真实的误导。
    """
    total = len(KB)
    if total == 0:
        return (
            "知识库还是空的。请先把资料放进 data/knowledge 目录，"
            "然后调用 /kb/index 建索引（或在前端点「重建知识库索引」）。"
        )
    return (
        f"我在知识库（共 {total} 张卡片，来自 {len(KB.sources)} 个文件）里"
        f"没有找到与这个问题相关的内容。\n\n"
        f"可能的原因：\n"
        f"  1. 这个问题超出了知识库覆盖的范围 —— 知识库里装了什么，就只能答什么；\n"
        f"  2. 你的问法和资料里的用词差别太大 —— 可以换一种说法再试。\n\n"
        f"如果确实需要回答这类问题，请把相关资料补充进 data/knowledge 目录，"
        f"然后重建索引。"
    )


def answer(question: str, top_k: int | None = None) -> RagAnswer:
    """非流式问答。适合被程序调用（比如评测脚本）。"""
    hits = retrieve(question, top_k)

    if not hits:
        # 检索为空时不要浪费一次模型调用——这也是一种成本控制
        return RagAnswer(
            answer=_no_hits_message(),
            citations=[],
            retrieved=0,
            model=get_settings().llm_model,
            latency_ms=0.0,
            cost_usd=0.0,
        )

    context, citations = _build_context(hits)
    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{question}"

    result = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        purpose="rag_answer",
    )

    return RagAnswer(
        answer=result.text.strip(),
        citations=citations,
        retrieved=len(hits),
        model=result.model,
        latency_ms=round(result.latency_ms, 1),
        cost_usd=round(result.cost_usd, 6),
    )


def answer_stream(question: str, top_k: int | None = None) -> Iterator[dict]:
    """流式问答，产出结构化事件，供 SSE 直接转发给前端。

    事件顺序：
      1. {"type": "sources", ...}  —— 先把引用推给前端，用户能立刻看到"在查什么"
      2. {"type": "delta", "text": "..."}  —— 正文增量
      3. {"type": "done", ...}  —— 收尾，带耗时统计
      4. 出错时 {"type": "error", "message": "..."}
    """
    settings = get_settings()
    hits = retrieve(question, top_k)

    if not hits:
        yield {
            "type": "sources",
            "citations": [],
            "retrieved": 0,
            "warnings": KB.warnings,
        }
        yield {
            "type": "delta",
            "text": _no_hits_message(),
        }
        yield {"type": "done", "latency_ms": 0.0}
        return

    context, citations = _build_context(hits)
    yield {
        "type": "sources",
        "citations": [c.model_dump() for c in citations],
        "retrieved": len(hits),
        "warnings": KB.warnings,
    }

    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{question}"

    try:
        for piece in chat_stream(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            purpose="rag_stream",
        ):
            yield {"type": "delta", "text": piece}
    except LLMError as exc:
        yield {"type": "error", "message": str(exc)}
        return

    yield {"type": "done", "latency_ms": None, "model": settings.llm_model}
