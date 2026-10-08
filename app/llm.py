"""LLM 网关：全项目**只有这一个文件**直接调用模型。

为什么值得单独抽一层：
1. 换供应商只改配置 —— openai SDK 兼容任何 OpenAI 协议的端点
   （DeepSeek / 通义 / 智谱 / 硅基流动 / 本地 vLLM / Ollama 都能接）。
   而且现在支持**运行时切换**：网页上选好服务商，下一次调用立刻生效，不用重启。
   见 app/providers.py
2. 重试、超时、埋点、错误归一化都集中在这里，业务代码完全不用关心
3. 单元测试时可以只 mock 这一层

注意：这里刻意把 SDK 自带的自动重试关掉（max_retries=0），
改用我们自己的重试循环，这样每一次尝试都能被记录、被观察到。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Sequence

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from . import providers
from .config import estimate_cost_usd, get_settings, is_peak_now
from .metrics import METRICS
from .providers import get_active
from .schemas import ChatMessage

# 这些异常值得重试；其余的（比如 401 鉴权失败、400 参数错误）重试只是浪费时间。
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)


class LLMError(RuntimeError):
    """对外统一的模型调用异常。业务层只需要捕获这一种。"""


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_ms: float
    cost_usd: float
    # 模型要求调用的工具（function calling）。None 表示它没要求调工具。
    # 注意：这里只是「模型的请求」，工具**还没有被执行** ——
    # 执行是业务层的事，模型碰不到你的代码。
    tool_calls: list[dict] | None = None
    # 结束原因：stop=正常说完，length=被 max_tokens 截断，tool_calls=要调工具。
    # length 值得警惕 —— 答案可能是半句话。
    finish_reason: str = ""


_client_cache: dict[tuple[str, str, float], OpenAI] = {}


def _is_local(url: str) -> bool:
    """本地服务（Ollama / vLLM / LM Studio）不需要 Key。"""
    return "127.0.0.1" in url or "localhost" in url or "0.0.0.0" in url


def _client(*, override_base_url: str | None = None, override_api_key: str | None = None) -> OpenAI:
    """按 (base_url, api_key, timeout) 复用客户端，避免每次调用都新建连接池。

    缓存键里含 base_url 和 api_key —— 所以在网页上切了服务商之后，
    下一次调用会自动新建客户端，**不需要重启服务**。

    override_* 只在"测试连接"时用：验一下某个还没保存的配置通不通，
    而**不覆盖当前正在用的配置**（否则填错一个字段就把能用的配置毁了）。
    """
    settings = get_settings()
    active = get_active()

    base_url = (override_base_url or active.base_url).strip().rstrip("/")
    if override_api_key is None:
        api_key = active.effective_key
    else:
        # 显式传了 key（哪怕是空串），就按传进来的算
        key = override_api_key.strip()
        api_key = key or ("not-needed" if _is_local(base_url) else "")

    if not api_key:
        raise LLMError(
            f"「{base_url}」还没有配 API Key。"
            "可以打开网页上的「⚙️ 模型设置」填，或者改 .env 里的 LLM_API_KEY。"
        )

    cache_key = (base_url, api_key, settings.llm_timeout_s)
    client = _client_cache.get(cache_key)
    if client is None:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # 重试由我们自己控制，见 chat()
        )
        _client_cache[cache_key] = client
    return client


def describe_connection() -> str:
    """给日志/健康检查用的连接信息，密钥永远打码。"""
    return providers.describe()


def _extract_usage(usage) -> tuple[int, int, int]:
    """从 SDK 的 usage 对象里取出 (输入, 输出, 缓存命中输入)。

    缓存命中数藏在嵌套的 prompt_tokens_details 里，
    而它和未命中部分的**单价不同**，所以必须单独取出来。
    """
    if usage is None:
        return 0, 0, 0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0

    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return prompt, completion, cached


def chat(
    messages: Sequence[ChatMessage] | Sequence[dict],
    *,
    purpose: str = "chat",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
    tools: Sequence[dict] | None = None,
    override_base_url: str | None = None,
    override_api_key: str | None = None,
) -> LLMResult:
    """同步调用模型，带重试与埋点。

    json_mode=True 时要求模型输出 JSON 对象。注意：这只保证"是合法 JSON"，
    **不保证字段符合你的 schema**，所以调用方仍必须用 Pydantic 校验。

    tools 传入工具说明书（JSON Schema 列表）时，模型可能不直接回答，
    而是返回 tool_calls 要求调用工具。**它只能"要求"，真正执行的是调用方。**

    override_* 只在"测试连接"时用，不影响当前生效的配置。
    """
    settings = get_settings()
    used_model = model or get_active().model
    payload = [m.model_dump() if isinstance(m, ChatMessage) else m for m in messages]

    started = time.perf_counter()
    peak = is_peak_now()
    last_error: Exception | None = None

    for attempt in range(1, settings.llm_max_retries + 2):
        try:
            kwargs: dict = {
                "model": used_model,
                "messages": payload,
                "temperature": settings.llm_temperature if temperature is None else temperature,
                "max_tokens": max_tokens or settings.llm_max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if tools:
                kwargs["tools"] = list(tools)

            response = _client(
                override_base_url=override_base_url,
                override_api_key=override_api_key,
            ).chat.completions.create(**kwargs)

            if not response.choices:
                raise LLMError("模型返回了空 choices，通常是上游异常，请重试。")

            message = response.choices[0].message
            text = message.content or ""
            finish_reason = getattr(response.choices[0], "finish_reason", "") or ""

            # max_tokens 太小的话，模型会在「还没说出话」的时候就被截断。
            # 真实踩到的：max_tokens=16 时问它「回复两个字：可用」，
            # content 是空字符串，completion_tokens 正好等于 16 ——
            # **没有任何报错**，调用方拿到一个空答案，还以为是模型的问题。
            # 这种"静默失败"必须变成显式异常，否则排查起来毫无线索。
            if finish_reason == "length" and not text:
                raise LLMError(
                    f"模型输出被 max_tokens={kwargs['max_tokens']} 截断，返回内容为空"
                    "（finish_reason=length）。请把这个值调大。"
                )

            raw_calls = getattr(message, "tool_calls", None)
            tool_calls = (
                [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in raw_calls
                ]
                if raw_calls
                else None
            )
            prompt_tokens, completion_tokens, cached_tokens = _extract_usage(response.usage)
            latency_ms = (time.perf_counter() - started) * 1000
            cost = estimate_cost_usd(
                used_model, prompt_tokens, completion_tokens, cached_tokens, peak=peak
            )

            METRICS.record(
                purpose=purpose,
                model=used_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                peak=peak,
                ok=True,
            )
            return LLMResult(
                text=text,
                model=used_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

        except RETRYABLE_ERRORS as exc:
            last_error = exc
            if attempt <= settings.llm_max_retries:
                # 指数退避：0.5s、1s、2s…… 上限 4s。
                # 不加退避会在限流时把配额打得更狠。
                time.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))
                continue
        except APIError as exc:
            # 非重试类错误（鉴权、参数、额度）立刻失败，并保留原始信息便于定位。
            last_error = exc
            break
        except Exception as exc:  # noqa: BLE001 - 兜底，避免未捕获异常直接 500
            last_error = exc
            break

    latency_ms = (time.perf_counter() - started) * 1000
    METRICS.record(
        purpose=purpose,
        model=used_model,
        latency_ms=latency_ms,
        peak=peak,
        ok=False,
        error=str(last_error)[:300],
    )
    raise LLMError(f"模型调用失败（{used_model}，用时 {latency_ms:.0f}ms）：{last_error}") from last_error


def chat_stream(
    messages: Sequence[ChatMessage] | Sequence[dict],
    *,
    purpose: str = "stream",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """流式调用，逐段 yield 文本增量。

    流式是体验上的硬需求：首字延迟从"等 3 秒"变成"0.6 秒就开始出字"。
    代价是 usage 统计更麻烦——这里用 stream_options 让服务端在最后一段带上用量。
    """
    settings = get_settings()
    used_model = model or get_active().model
    payload = [m.model_dump() if isinstance(m, ChatMessage) else m for m in messages]

    started = time.perf_counter()
    peak = is_peak_now()
    emitted_any = False
    usage_holder: list = []

    def _run(include_usage: bool) -> Iterator[str]:
        nonlocal emitted_any
        kwargs: dict = {
            "model": used_model,
            "messages": payload,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": True,
        }
        if include_usage:
            kwargs["stream_options"] = {"include_usage": True}

        stream = _client().chat.completions.create(**kwargs)
        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage_holder.append(chunk.usage)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                # 必须在这里标记：一旦已经吐过内容，就不能再重试，
                # 否则用户会看到重复输出。
                emitted_any = True
                yield piece

    try:
        try:
            yield from _run(include_usage=True)
        except APIError as exc:
            # 少数兼容端点不认 stream_options。
            # 只有在还没吐出任何内容时才能安全重试，否则会重复输出。
            if emitted_any or "stream_options" not in str(exc).lower():
                raise
            yield from _run(include_usage=False)

        # 记录埋点：如果服务端没返回 usage，token 记 0，但延迟仍然有记录。
        usage = usage_holder[0] if usage_holder else None
        prompt_tokens, completion_tokens, cached_tokens = _extract_usage(usage)
        latency_ms = (time.perf_counter() - started) * 1000
        METRICS.record(
            purpose=purpose,
            model=used_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            latency_ms=latency_ms,
            cost_usd=estimate_cost_usd(
                used_model, prompt_tokens, completion_tokens, cached_tokens, peak=peak
            ),
            peak=peak,
            ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        METRICS.record(
            purpose=purpose,
            model=used_model,
            latency_ms=(time.perf_counter() - started) * 1000,
            peak=peak,
            ok=False,
            error=str(exc)[:300],
        )
        raise LLMError(f"流式调用失败（{used_model}）：{exc}") from exc


# ---------------------------------------------------------------------------
# 向量化
# ---------------------------------------------------------------------------
# 关键事实：DeepSeek 官方 API **不提供 embeddings 接口**。
# 所以向量检索要么用别的服务，要么用本地模型。默认关闭（EMBEDDING_BACKEND=none），
# 此时系统纯用 BM25 关键词检索，零额外依赖即可跑通。


def embed_texts(texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
    """把文本转成向量，供向量检索使用。"""
    settings = get_settings()
    backend = settings.embedding_backend

    if backend == "none":
        raise LLMError(
            "EMBEDDING_BACKEND=none，向量检索未启用。\n"
            "可选方案：\n"
            "  1) 用别的服务商：EMBEDDING_BACKEND=api，并配好 EMBEDDING_BASE_URL / "
            "EMBEDDING_MODEL / EMBEDDING_API_KEY\n"
            "  2) 用本地模型：pip install sentence-transformers，然后 "
            "EMBEDDING_BACKEND=local、EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5"
        )

    if backend == "local":
        return _embed_local(texts, settings.embedding_model, batch_size)

    if backend == "api":
        if not settings.embedding_base_url or not settings.embedding_model:
            raise LLMError(
                "EMBEDDING_BACKEND=api 时，必须同时配置 EMBEDDING_BASE_URL 和 EMBEDDING_MODEL。"
            )
        client = OpenAI(
            api_key=settings.embedding_api_key or settings.llm_api_key,
            base_url=settings.embedding_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = client.embeddings.create(model=settings.embedding_model, input=batch)
            # 按 index 排序，避免服务端乱序返回导致向量和文本错位。
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        return vectors

    raise LLMError(f"未知的 EMBEDDING_BACKEND：{backend!r}（可选 none / api / local）")


def _embed_local(texts: Sequence[str], model_name: str, batch_size: int) -> list[list[float]]:
    """本地向量模型。首次运行会下载权重（几百 MB），之后离线可用。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise LLMError(
            "EMBEDDING_BACKEND=local 需要先安装：pip install sentence-transformers"
        ) from exc

    if not model_name:
        raise LLMError("EMBEDDING_BACKEND=local 时必须配置 EMBEDDING_MODEL。")
    model = SentenceTransformer(model_name)
    vectors = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True)
    return [list(map(float, vec)) for vec in vectors]
