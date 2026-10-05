"""调用埋点与成本统计。

这是本项目跟"教程级 Demo"的分水岭：每次模型调用都留一条记录，
于是所有优化都有数字支撑，简历上才写得出
"P95 延迟从 4.2s 降到 1.8s""单次问答成本 ¥0.02" 这种句子。

实现刻意保持极简：进程内环形缓冲 + 一把锁。
生产环境该换成 Prometheus / Langfuse，但那属于第二阶段的事。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field

# 只保留最近 N 条明细，防止长时间运行把内存吃满。
_MAX_RECORDS = 1000


@dataclass
class CallRecord:
    ts: float
    purpose: str            # resume_parse / rag_answer / chat ... 用来分场景统计成本
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_ms: float
    cost_usd: float
    peak: bool
    ok: bool
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class Metrics:
    """线程安全的调用记录器。

    FastAPI 默认用线程池执行同步端点，所以这里必须加锁。
    """

    def __init__(self, max_records: int = _MAX_RECORDS) -> None:
        self._lock = threading.Lock()
        self._records: deque[CallRecord] = deque(maxlen=max_records)

    # -- 写入 ---------------------------------------------------------------

    def record(
        self,
        *,
        purpose: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        peak: bool = False,
        ok: bool = True,
        error: str = "",
    ) -> CallRecord:
        rec = CallRecord(
            ts=time.time(),
            purpose=purpose,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            peak=peak,
            ok=ok,
            error=error,
        )
        with self._lock:
            self._records.append(rec)
        return rec

    # -- 读取 ---------------------------------------------------------------

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            items = list(self._records)[-limit:]
        return [r.as_dict() for r in reversed(items)]

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self) -> dict:
        """聚合指标：调用数、成功率、token、成本、延迟分位数。"""
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "calls": 0,
                "success_rate": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_hit_rate": None,
                "cost_usd": 0.0,
                "latency_ms": {"p50": None, "p95": None, "max": None},
                "by_purpose": {},
            }

        ok_records = [r for r in records if r.ok]
        latencies = sorted(r.latency_ms for r in ok_records)
        prompt_tokens = sum(r.prompt_tokens for r in records)
        cached_tokens = sum(r.cached_tokens for r in records)

        by_purpose: dict[str, dict] = {}
        for r in records:
            bucket = by_purpose.setdefault(
                r.purpose,
                {"calls": 0, "cost_usd": 0.0, "tokens": 0, "errors": 0, "latency_ms_sum": 0.0},
            )
            bucket["calls"] += 1
            bucket["cost_usd"] += r.cost_usd
            bucket["tokens"] += r.prompt_tokens + r.completion_tokens
            bucket["latency_ms_sum"] += r.latency_ms
            if not r.ok:
                bucket["errors"] += 1

        for bucket in by_purpose.values():
            bucket["avg_latency_ms"] = round(bucket.pop("latency_ms_sum") / bucket["calls"], 1)
            bucket["cost_usd"] = round(bucket["cost_usd"], 6)

        return {
            "calls": len(records),
            "success_rate": round(len(ok_records) / len(records), 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": sum(r.completion_tokens for r in records),
            "total_tokens": prompt_tokens + sum(r.completion_tokens for r in records),
            "cache_hit_rate": round(cached_tokens / prompt_tokens, 4) if prompt_tokens else None,
            "cost_usd": round(sum(r.cost_usd for r in records), 6),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": round(latencies[-1], 1) if latencies else None,
            },
            "by_purpose": by_purpose,
        }


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """最近秩法算分位数，不引入 numpy 依赖。"""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 1)
    index = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return round(sorted_values[index], 1)


# 全局单例：整个进程共用一份埋点数据。
METRICS = Metrics()
