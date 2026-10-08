"""同一句话，两个模型 —— 量一量「首字延迟」到底差多少。

为什么要测这个：

    教科书说"SSE 让首字延迟从 3 秒降到 0.6 秒"。
    但我实测下来只有 1.3 倍提升 —— 因为**首字延迟的大头不是"生成"，
    而是"模型在想"**。

    deepseek-v4-pro 在正式回答前会先产生一段**推理 token**。
    那段推理用户**看不到**，但它在实实在在地占时间。
    所以模型选得越"重"，SSE 能救回来的就越少。

    这直接影响一个产品决策：**面向交互场景，模型不是越大越好。**

跑法：
    python -m scripts.model_latency
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm import chat_stream  # noqa: E402

QUESTION = "用三句话说说什么是检索增强生成（RAG）。"
MODELS = ["deepseek-flash", "deepseek-v4-pro"]
LINE = "=" * 74

# 官方单价（美元 / 100 万 token，非高峰）
PRICES = {
    "deepseek-flash": {"in": 0.15, "out": 0.60},
    "deepseek-v4-pro": {"in": 0.66, "out": 1.98},
}


def measure(model: str) -> dict:
    started = time.perf_counter()
    first_at: float | None = None
    pieces = 0

    for piece in chat_stream(
        [{"role": "user", "content": QUESTION}],
        purpose="model_latency",
        model=model,
        temperature=0,
    ):
        if first_at is None:
            first_at = (time.perf_counter() - started) * 1000
        pieces += 1

    total = (time.perf_counter() - started) * 1000
    return {
        "model": model,
        "first_ms": first_at or 0.0,
        "total_ms": total,
        "pieces": pieces,
    }


def main() -> None:
    print(f"\n{LINE}")
    print(f"问题：{QUESTION}")
    print("（同一句话，两个模型，都走流式）")
    print(LINE)

    rows = []
    for model in MODELS:
        print(f"\n正在跑 {model} ……")
        row = measure(model)
        rows.append(row)
        print(
            f"  首字 {row['first_ms']:>7.0f} ms ｜ "
            f"总耗时 {row['total_ms']:>7.0f} ms ｜ 推了 {row['pieces']} 个片段"
        )

    print(f"\n{LINE}")
    print("对比")
    print(LINE)
    print(f"  {'模型':<20}{'首字延迟':>12}{'总耗时':>12}{'单价(入/出)':>18}")
    print("  " + "-" * 62)
    for row in rows:
        price = PRICES.get(row["model"], {})
        price_text = (
            f"${price['in']:.2f}/${price['out']:.2f}" if price else "—"
        )
        print(
            f"  {row['model']:<20}{row['first_ms']:>10.0f} ms"
            f"{row['total_ms']:>10.0f} ms{price_text:>18}"
        )

    if len(rows) == 2:
        fast, slow = sorted(rows, key=lambda r: r["first_ms"])
        ratio = slow["first_ms"] / max(fast["first_ms"], 1)
        print(
            f"\n  → 「首字」这一项，{fast['model']} 比 {slow['model']} "
            f"快 {ratio:.1f} 倍"
        )
        print(f"  → 但「总耗时」只差 {abs(slow['total_ms'] - fast['total_ms']):.0f} ms")

    print(f"""
{LINE}
结论
{LINE}

  1. **SSE 只能救「生成」那一段，救不了「模型在想」那一段。**
     模型越重（推理 token 越多），首字延迟越大，SSE 的相对收益越小。

  2. **所以模型选择不只是成本问题，也是体验问题。**
     面向交互场景（用户盯着屏幕等），flash 往往比 pro 更合适 ——
     哪怕 pro 答得更准。

  3. **这也是为什么专业产品会把「首字延迟」当成一个独立指标盯着，
     而不是只看总耗时。** 用户感知的是前者。
""")


if __name__ == "__main__":
    main()
