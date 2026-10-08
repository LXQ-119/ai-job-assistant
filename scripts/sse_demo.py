"""SSE 到底改变了什么？用同一个问题跑两次，量给你看。

先纠正一个常见误解：

    ❌ "SSE 是大模型的一种输出形式"
    ✅ "SSE 是 HTTP 的一种传输方式，用在「后端 → 浏览器」这一段"

    大模型**无论如何**都是一小段一小段生成的 —— 这是它的工作原理
    （自回归：一次预测下一个 token，再拿这个 token 去预测下一个）。
    模型根本不知道 SSE 是什么东西。

    区别在中间那一段：

        没有 SSE：后端把模型吐的所有片段**攒完**，再一次性返回
                  → 你盯着空白等 2 秒，然后全文突然出现

        有 SSE  ：后端收到一段就**立刻转发**
                  → 你 0.6 秒后就开始看到字

    ★ 关键：**总耗时几乎一样，变的是「你什么时候开始看到东西」。**
      这就是「感知延迟」和「真实延迟」的区别。

跑法：
    python -m scripts.sse_demo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm import chat, chat_stream  # noqa: E402

# 支持从命令行传问题和模型：
#   python -m scripts.sse_demo "写一篇 800 字的文章" deepseek-flash
QUESTION = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "用三句话说说什么是检索增强生成（RAG）。"
)
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
MESSAGES = [{"role": "user", "content": QUESTION}]
LINE = "=" * 76


def main() -> None:
    print(f"\n{LINE}")
    print(f"问题：{QUESTION}")
    print(f"模型：{MODEL or '（当前生效的）'}")
    print(LINE)

    # ---------------- 非流式 ----------------
    print("\n【跑法 A】非流式（后端攒完再一起给）")
    started = time.perf_counter()
    result = chat(
        MESSAGES,
        purpose="sse_demo_block",
        model=MODEL,
        temperature=0,
        max_tokens=3000,
    )
    total_block = (time.perf_counter() - started) * 1000

    print(f"  用户看到的：先空白 {total_block:.0f} ms，然后全文一次性出现")
    print(f"  首字延迟 = 总耗时 = {total_block:.0f} ms")
    print(f"  正文长度：{len(result.text)} 字")

    # ---------------- 流式 ----------------
    print(f"\n{LINE}")
    print("【跑法 B】流式（后端收到一段就转发一段）")
    print(LINE)
    print("\n  用户看到的（边收边打，只打前 120 字，免得刷屏）：")

    started = time.perf_counter()
    first_at: float | None = None
    pieces = 0
    shown = ""

    for piece in chat_stream(
        MESSAGES, purpose="sse_demo_stream", model=MODEL, temperature=0
    ):
        if first_at is None:
            first_at = (time.perf_counter() - started) * 1000
        pieces += 1
        if len(shown) < 120:
            shown += piece
            print(f"\r  {shown}", end="", flush=True)
    total_stream = (time.perf_counter() - started) * 1000
    print()
    print(f"  首字延迟 = {first_at:.0f} ms　（用户这么早就开始看到东西了）")
    print(f"  总耗时   = {total_stream:.0f} ms")
    print(f"  一共推了 {pieces} 个片段")

    # ---------------- 对比 ----------------
    print(f"\n{LINE}")
    print("对比")
    print(LINE)
    print(f"  {'':<16}{'首字延迟':>12}{'总耗时':>12}")
    print("  " + "-" * 40)
    print(f"  {'非流式':<14}{total_block:>10.0f} ms{total_block:>10.0f} ms")
    print(f"  {'流式 (SSE)':<13}{first_at:>10.0f} ms{total_stream:>10.0f} ms")
    print(
        f"\n  → 首字延迟快了约 {total_block / max(first_at, 1):.1f} 倍，"
        f"而总耗时只差 {abs(total_stream - total_block):.0f} ms"
    )
    print("\n  ★ 所以 SSE 并没有让模型变快 —— 它只是让「等待」从『盯着空白』")
    print("    变成了『看着字往外冒』。人脑对这两件事的感受完全不同。")

    # ---------------- SSE 事件长什么样 ----------------
    print(f"\n{LINE}")
    print("SSE 在网络上是长这样的（后端发给浏览器的原始格式）")
    print(LINE)
    print("""
  data: {"type":"sources","citations":[...],"retrieved":4}

  data: {"type":"delta","text":"检索"}

  data: {"type":"delta","text":"增强"}

  data: {"type":"delta","text":"生成"}

  data: {"type":"done","latency_ms":1873}

  data: [DONE]

  规则很简单：每条消息以 "data: " 开头，以**两个换行**结束。
  浏览器（EventSource）收到一条就触发一次回调 —— 前端就是这么"一个字一个字打出来"的。
""")


if __name__ == "__main__":
    main()
