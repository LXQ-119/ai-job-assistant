"""HTTP 接口层。

这一层只做三件事：校验入参、调用 service、把结果或异常转成 HTTP 响应。
**业务逻辑一律不写在这里**——否则接口和逻辑会缠在一起，
将来想做 CLI 或定时任务就得复制一遍代码。

另外注意 index 接口做了路径限制（只允许 data 目录内），
这是最基本的防护：否则用户能通过 API 读取服务器上任意文件。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from ..config import get_settings, mask_secret
from ..llm import LLMError, describe_connection
from ..metrics import METRICS
from ..schemas import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    MetricsResponse,
    ParseResumeResponse,
    RagAnswer,
    RagQueryRequest,
    ResumeTextRequest,
)
from ..services import rag, resume as resume_service
from ..services.documents import DocumentError, SUPPORTED_SUFFIXES
from ..services.retriever import KB

router = APIRouter()


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        api_key_configured=bool(settings.llm_api_key),
        embedding_backend=settings.embedding_backend,
        knowledge_chunks=len(KB),
    )


@router.get("/debug/connection", summary="查看当前的模型连接信息（密钥已打码）")
def debug_connection() -> dict:
    settings = get_settings()
    return {
        "connection": describe_connection(),
        "api_key": mask_secret(settings.llm_api_key),
        "embedding_backend": settings.embedding_backend,
        "hybrid_alpha": settings.hybrid_alpha,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "warnings": KB.warnings,
    }


@router.get("/metrics", response_model=MetricsResponse, summary="调用埋点与成本统计")
def metrics() -> MetricsResponse:
    settings = get_settings()
    return MetricsResponse(
        summary=METRICS.summary(),
        recent=METRICS.recent(limit=20),
        usd_to_cny=settings.usd_to_cny,
    )


@router.post("/metrics/reset", summary="清空埋点数据")
def metrics_reset() -> dict:
    METRICS.reset()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------


def _safe_paths(raw_paths: list[str]) -> list[Path]:
    """只允许操作 data 目录内的路径，挡住 ../ 之类的路径穿越。"""
    settings = get_settings()
    allowed_root = settings.data_dir.resolve()
    resolved: list[Path] = []

    for raw in raw_paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (settings.data_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if allowed_root not in candidate.parents and candidate != allowed_root:
            raise HTTPException(
                status_code=400,
                detail=f"路径 {raw} 不在允许范围内（只允许 {allowed_root} 目录内的文件）。",
            )
        resolved.append(candidate)
    return resolved


@router.post("/kb/index", response_model=IndexResponse, summary="为知识库建索引")
def kb_index(request: IndexRequest) -> IndexResponse:
    import time

    started = time.perf_counter()
    try:
        paths = _safe_paths(request.paths) if request.paths else []
        result = rag.index_paths([str(p) for p in paths], rebuild=request.rebuild)
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return IndexResponse(
        indexed_files=result["indexed_files"],
        total_chunks=result["total_chunks"],
        files=result["files"],
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@router.get("/kb/status", summary="知识库状态")
def kb_status() -> dict:
    settings = get_settings()
    return {
        "chunks": len(KB),
        "sources": KB.sources,
        "vector_enabled": KB.vector_enabled,
        "hybrid_alpha": settings.hybrid_alpha,
        "warnings": KB.warnings,
        "index_path": str(settings.index_path),
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
    }


@router.post("/kb/query", response_model=RagAnswer, summary="知识库问答（非流式）")
def kb_query(request: RagQueryRequest) -> RagAnswer:
    try:
        return rag.answer(request.question, request.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/kb/stream", summary="知识库问答（SSE 流式）")
def kb_stream(request: RagQueryRequest) -> StreamingResponse:
    def event_stream():
        try:
            for event in rag.answer_stream(request.question, request.top_k):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except LLMError as exc:
            payload = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 让 Nginx 不要缓冲，否则流式会变成一次性返回
        },
    )


# ---------------------------------------------------------------------------
# 简历
# ---------------------------------------------------------------------------


def _to_response(outcome: resume_service.ResumeOutcome) -> ParseResumeResponse:
    return ParseResumeResponse(
        profile=outcome.profile,
        model=outcome.model,
        attempts=outcome.attempts,
        latency_ms=outcome.latency_ms,
        cost_usd=outcome.cost_usd,
    )


@router.post("/resume/parse", response_model=ParseResumeResponse, summary="解析简历文本")
def resume_parse(request: ResumeTextRequest) -> ParseResumeResponse:
    try:
        return _to_response(resume_service.parse_resume_text(request.text))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/resume/upload", response_model=ParseResumeResponse, summary="上传简历文件并解析")
async def resume_upload(file: UploadFile = File(...)) -> ParseResumeResponse:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix!r}，请上传 {', '.join(sorted(SUPPORTED_SUFFIXES))}。",
        )

    payload = await file.read()
    max_bytes = 10 * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(status_code=400, detail="文件超过 10MB 上限。")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # 存成临时文件，只为让解析库能按后缀识别格式；解析完立即删除。
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, dir=settings.data_dir
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)

    try:
        outcome = await run_in_threadpool(resume_service.parse_resume_file, temp_path)
        outcome.source = file.filename or temp_path.name
        return _to_response(outcome)
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
