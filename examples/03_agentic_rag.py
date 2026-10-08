"""03 - Agentic RAG：让模型自己决定走哪条路。

命令行版。跑三类问题，看它分别走哪条：

    ① 问用户自己的东西  →  应该调 search_notes（查笔记，带引用）
    ② 纯算术            →  应该调 calculate（不心算）
    ③ 通用常识          →  应该直接答，并标明"不来自你的笔记"

跑法（文件名以数字开头，不能用 -m，直接跑文件）：
    python examples/03_agentic_rag.py
    python examples/03_agentic_rag.py "你的问题"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.agentic import run  # noqa: E402
from app.services.rag import index_paths, load_index  # noqa: E402

DEFAULT_QUESTIONS = [
    ("问用户自己的东西", "我从小米那次面试里学到了什么？"),
    ("纯算术", "1234 乘以 5678 等于多少？"),
    ("通用常识", "李白是哪个朝代的诗人？"),
    ("查了但没查到", "我简历上写的期望薪资是多少？"),
]

LINE = "=" * 78


def show(label: str, question: str) -> None:
    print(f"\n{LINE}")
    print(f"【{label}】{question}")
    print(LINE)

    final = None
    for event in run(question):
        kind = event["type"]

        if kind == "round":
            print(f"  ── 第 {event['index']} 轮")

        elif kind == "decision":
            for call in event["calls"]:
                print(f"   🧠 模型决定调用：{call['function']['name']}")
                print(f"      参数：{call['function']['arguments']}")

        elif kind == "tool_result":
            payload = event["result"]
            if "error" in payload:
                print(f"   ⚠️  工具返回错误：{payload['error']}")
            elif event["name"] == "search_notes":
                if payload.get("found"):
                    print(f"   📚 翻到 {payload['count']} 张卡片：")
                    for item in payload["results"]:
                        preview = item["内容"].replace("\n", " ")[:52]
                        print(f"        [{item['编号']}] {item['来源']}")
                        print(f"            {preview}…")
                else:
                    print("   📚 笔记里没有相关内容")
            else:
                print(f"   🧮 计算结果：{payload.get('result')}")

        elif kind == "final":
            final = event

        elif kind == "error":
            print(f"   ❌ {event['message']}")

    if final:
        print(f"\n   ✅ 走了哪条路：{final['route']}")
        print(f"   💰 {final['steps']} 轮 ｜ {final['tokens']} tokens "
              f"｜ ${final['cost_usd']}")
        print(f"\n   最终回答：\n{final['content']}")


def main() -> None:
    if not load_index():
        print("磁盘上没有索引，先建一次……")
        index_paths(rebuild=True)

    if len(sys.argv) > 1:
        show("你问的", sys.argv[1])
        return

    for label, question in DEFAULT_QUESTIONS:
        show(label, question)


if __name__ == "__main__":
    main()
