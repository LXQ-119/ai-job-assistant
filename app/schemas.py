"""跨层共用的数据结构。

用 Pydantic 定义一次，FastAPI 会自动拿它做三件事：
请求校验、响应序列化、生成 Swagger 文档。
换句话说，接口文档不用手写，也不会和实现脱节。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class HealthResponse(BaseModel):
    status: str
    model: str
    api_key_configured: bool
    embedding_backend: str
    knowledge_chunks: int


# ---------------------------------------------------------------------------
# 简历
# ---------------------------------------------------------------------------


class ResumeProfile(BaseModel):
    """简历结构化结果。

    字段是**写死**的，这是刻意的：给模型一个明确的输出契约，
    比让它自由发挥更容易得到稳定结果，也让下游代码有确定的字段可读。
    """

    name: str = Field(default="", description="姓名，没有就留空字符串")
    years_of_experience: float = Field(default=0.0, description="工作或实习年限，应届生填 0")
    education: str = Field(default="", description="最高学历 + 学校 + 专业，一句话")
    skills: list[str] = Field(default_factory=list, description="技术栈关键词，逐个列出")
    projects: list[str] = Field(default_factory=list, description="项目经历，每条一句话")
    highlights: list[str] = Field(
        default_factory=list, description="可量化的亮点，例如'把命中率从 62% 提到 89%'"
    )


class ResumeTextRequest(BaseModel):
    """直接提交简历正文的请求体。"""

    text: str = Field(min_length=10, max_length=50000)


class ParseResumeResponse(BaseModel):
    profile: ResumeProfile
    model: str
    attempts: int = Field(description="实际尝试次数，>1 说明第一次输出没通过校验并触发了修复重试")
    latency_ms: float
    cost_usd: float


# ---------------------------------------------------------------------------
# 知识库检索
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """引用来源。

    带引用是抑制幻觉最实用的手段：用户能自己核对，模型也更难瞎编。
    """

    source: str
    heading: str = ""
    chunk_index: int
    score: float
    text: str


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RagAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved: int
    model: str
    latency_ms: float
    cost_usd: float


class IndexRequest(BaseModel):
    """索引请求。路径为空表示重建 data/knowledge 下的全部文件。"""

    paths: list[str] = Field(default_factory=list)
    rebuild: bool = False


class IndexResponse(BaseModel):
    indexed_files: int
    total_chunks: int
    files: list[str]
    latency_ms: float


class MetricsResponse(BaseModel):
    summary: dict
    recent: list[dict]
    usd_to_cny: float
