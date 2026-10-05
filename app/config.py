"""集中配置。

设计原则：代码里不出现任何密钥，也不硬编码任何价格——全部来自 .env。
这样密钥只留在本机（.env 被 .gitignore 排除），而本地 / Docker / 服务器
三种环境只需要换 .env，代码一行都不用改。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# load_dotenv 默认不覆盖已存在的环境变量，
# 所以命令行里临时设置的变量（比如 CI 里注入的）优先级高于 .env。
load_dotenv(PROJECT_ROOT / ".env")


def _str(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _int(key: str, default: int) -> int:
    raw = _str(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 必须是整数，当前是 {raw!r}") from exc


def _float(key: str, default: float) -> float:
    raw = _str(key, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {key} 必须是数字，当前是 {raw!r}") from exc


# ---------------------------------------------------------------------------
# 价格表
# ---------------------------------------------------------------------------
# 来源：https://api-docs.deepseek.com/quick_start/pricing/
# 官方规则：非高峰价 = 高峰价的一半。所以这里只存非高峰价，高峰时把 base 乘 2。
# 单位：美元 / 100 万 token。


@dataclass(frozen=True)
class ModelPrice:
    cache_hit: float   # 命中前缀缓存的输入 token
    input_miss: float  # 未命中缓存的输入 token
    output: float


MODEL_PRICES: dict[str, ModelPrice] = {
    "deepseek-flash": ModelPrice(cache_hit=0.003, input_miss=0.15, output=0.60),
    "deepseek-v4-pro": ModelPrice(cache_hit=0.022, input_miss=0.66, output=1.98),
    # 旧名字仍被接受：服务端由 V4.1-Flash 提供，按 Flash 价计费。
    "deepseek-v4-flash": ModelPrice(cache_hit=0.003, input_miss=0.15, output=0.60),
    "deepseek-v4-flash-vision-exp": ModelPrice(cache_hit=0.003, input_miss=0.15, output=0.60),
}

# 未收录的模型 id 用这个兜底，避免成本统计直接崩掉。
_FALLBACK_PRICE = ModelPrice(cache_hit=0.003, input_miss=0.15, output=0.60)


def is_peak_now(now: datetime | None = None) -> bool:
    """判断当前是否处于 DeepSeek 的高峰计价时段。

    高峰时段：UTC 周一至周五 01:00-04:00 与 06:00-10:00。
    中国法定节假日全天算非高峰——这里没有节假日日历，简化为只按星期判断，
    因此节假日会略微高估成本（宁可高估，不要低估）。
    """
    moment = now or datetime.now(timezone.utc)
    if moment.weekday() >= 5:  # 周六周日全天非高峰
        return False
    hour = moment.hour
    return 1 <= hour < 4 or 6 <= hour < 10


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    *,
    peak: bool | None = None,
) -> float:
    """按真实计费口径估算一次调用的成本。

    注意缓存命中的输入 token 和未命中的是**两个不同单价**，
    普通 Demo 常常忽略这一点，导致成本算高一截。
    """
    price = MODEL_PRICES.get(model, _FALLBACK_PRICE)
    multiplier = 2.0 if (is_peak_now() if peak is None else peak) else 1.0

    cached = max(0, min(cached_tokens, prompt_tokens))
    miss = max(0, prompt_tokens - cached)

    cost = (
        cached * price.cache_hit
        + miss * price.input_miss
        + max(0, completion_tokens) * price.output
    ) / 1_000_000
    return cost * multiplier


# ---------------------------------------------------------------------------
# 运行配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    # --- 对话模型 ---
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_s: float
    llm_max_retries: int

    # --- 向量化（DeepSeek 不提供 embeddings，所以默认关闭，只跑 BM25）---
    embedding_backend: str      # none | api | local
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str

    # --- 检索参数（这几个值就是面试里"你怎么调的"的答案）---
    chunk_size: int
    chunk_overlap: int
    top_k: int
    candidate_k: int
    hybrid_alpha: float         # 1.0 = 纯向量，0.0 = 纯 BM25，中间值 = 混合

    # --- 展示 ---
    usd_to_cny: float

    # --- 路径 ---
    data_dir: Path
    knowledge_dir: Path
    index_path: Path
    reports_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取配置。用 lru_cache 保证整个进程只解析一次 .env。"""
    data_dir = Path(_str("DATA_DIR", str(PROJECT_ROOT / "data")))
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    return Settings(
        llm_base_url=_str("LLM_BASE_URL", "https://api.deepseek.com"),
        llm_api_key=_str("LLM_API_KEY", ""),
        llm_model=_str("LLM_MODEL", "deepseek-flash"),
        llm_temperature=_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_int("LLM_MAX_TOKENS", 2048),
        llm_timeout_s=_float("LLM_TIMEOUT_S", 60.0),
        llm_max_retries=_int("LLM_MAX_RETRIES", 2),
        embedding_backend=_str("EMBEDDING_BACKEND", "none").lower(),
        embedding_base_url=_str("EMBEDDING_BASE_URL", ""),
        embedding_api_key=_str("EMBEDDING_API_KEY", ""),
        embedding_model=_str("EMBEDDING_MODEL", ""),
        chunk_size=_int("CHUNK_SIZE", 600),
        chunk_overlap=_int("CHUNK_OVERLAP", 100),
        top_k=_int("TOP_K", 4),
        candidate_k=_int("CANDIDATE_K", 20),
        hybrid_alpha=_float("HYBRID_ALPHA", 0.0),
        usd_to_cny=_float("USD_TO_CNY", 7.2),
        data_dir=data_dir,
        knowledge_dir=data_dir / "knowledge",
        index_path=data_dir / "index" / "kb_index.json",
        reports_dir=PROJECT_ROOT / "reports",
    )


def mask_secret(secret: str) -> str:
    """日志里永远只出现打码后的密钥。

    这是必须养成的习惯：一旦密钥进了日志文件，就等于泄露。
    """
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * 6}{secret[-4:]}"
