"""简历结构化：把自由格式的简历文本变成确定的字段。

这里演示一个**真实工程问题**：
大模型返回的 JSON 只保证"是合法 JSON"，**不保证字段符合你的预期**——
可能少字段、类型错（年限返回 "3年" 而不是 3）、或者整段包在 ```json 里。

所以流程必须是：调用 → 解析 → Pydantic 校验 → **失败则带着错误信息重试**。
这个"修复重试"模式是新人项目和工程项目的典型差别。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ..config import get_settings
from ..llm import LLMError, LLMResult, chat
from ..schemas import ResumeProfile
from .documents import read_document

EXTRACTION_PROMPT = """你是一个简历信息抽取引擎。

任务：从用户提供的简历文本中抽取结构化信息。

输出要求：
- 只输出一个 JSON 对象，不要输出任何解释文字，不要用 Markdown 代码块包裹。
- 严格使用以下字段：
  name (字符串，姓名；找不到就空字符串)
  years_of_experience (数字，工作或实习年限；应届生写 0)
  education (字符串，最高学历 + 学校 + 专业)
  skills (字符串数组，技术栈关键词，每个元素一个词)
  projects (字符串数组，每段项目经历压缩成一句话)
  highlights (字符串数组，只放**可量化**的亮点，例如"命中率从 62% 提升到 89%")
- 文本里没有的信息，用空字符串或空数组，不要猜测、不要编造。"""

# 模型有时会把 JSON 包在 Markdown 代码块里，这里剥掉。
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

_EXPECTED_KEYS = {"name", "years_of_experience", "education", "skills", "projects", "highlights"}


@dataclass
class ResumeOutcome:
    profile: ResumeProfile
    attempts: int
    model: str
    latency_ms: float
    cost_usd: float
    source: str
    text_chars: int


def _load_profile(raw_text: str) -> ResumeProfile:
    """把模型输出变成 ResumeProfile，任何一步不符合预期都抛异常。"""
    cleaned = _FENCE_RE.sub("", raw_text).strip()
    data = json.loads(cleaned)  # 可能抛 json.JSONDecodeError

    if not isinstance(data, dict):
        raise ValueError(f"模型返回的不是 JSON 对象，而是 {type(data).__name__}")

    # 容错：模型有时会套一层外壳，比如 {"resume": {...}}
    if not (_EXPECTED_KEYS & set(data.keys())):
        nested = [v for v in data.values() if isinstance(v, dict)]
        if len(nested) == 1:
            data = nested[0]

    return ResumeProfile.model_validate(data)  # 可能抛 ValidationError


def parse_resume_text(text: str, *, max_attempts: int = 2) -> ResumeOutcome:
    """解析简历文本。max_attempts>1 时启用修复重试。"""
    trimmed = text.strip()
    if len(trimmed) < 10:
        raise LLMError("简历内容太短（少于 10 个字符），无法解析。")

    settings = get_settings()
    messages: list[dict] = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": trimmed[:20000]},  # 截断，防止超出上下文并控制成本
    ]

    last_error: Exception | None = None
    total_latency = 0.0
    total_cost = 0.0
    result: LLMResult | None = None

    for attempt in range(1, max_attempts + 1):
        # temperature=0：抽取任务要的是稳定复现，不是创造力
        result = chat(messages, purpose="resume_parse", json_mode=True, temperature=0.0)
        total_latency += result.latency_ms
        total_cost += result.cost_usd

        try:
            profile = _load_profile(result.text)
            return ResumeOutcome(
                profile=profile,
                attempts=attempt,
                model=result.model,
                latency_ms=round(total_latency, 1),
                cost_usd=round(total_cost, 6),
                source="text",
                text_chars=len(trimmed),
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            # 关键：把模型的错误输出 + 具体校验错误一起回传，
            # 并明确要求"只输出 JSON"。这比单纯重试一次有效得多。
            messages = messages + [
                {"role": "assistant", "content": result.text[:2000]},
                {
                    "role": "user",
                    "content": (
                        f"上面的输出不符合要求，校验报错：{exc}\n\n"
                        "请重新输出**且只输出**一个合法的 JSON 对象，"
                        "字段名必须完全一致，不要加解释、不要加代码块标记。"
                    ),
                },
            ]

    raise LLMError(
        f"简历解析失败：连续 {max_attempts} 次输出都未通过校验。最后一次错误：{last_error}"
    )


def parse_resume_file(path: Path) -> ResumeOutcome:
    """从文件解析简历，支持 .md/.txt/.pdf/.docx。"""
    text = read_document(path)
    if not text.strip():
        raise LLMError(f"文件 {path.name} 解析后没有文字内容。扫描件 PDF 需要先做 OCR。")
    outcome = parse_resume_text(text)
    outcome.source = path.name
    return outcome
