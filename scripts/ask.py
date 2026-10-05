"""命令行提问工具 —— 不重建索引，直接问。

用法：
    python -m scripts.ask "我从小米那次面试里学到了什么？"
    python -m scripts.ask "红烧肉怎么做？"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import answer, index_paths, load_index  # noqa: E402

LINE = "=" * 78


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    if not load_index():
        print("磁盘上没有索引，先建一次……")
        index_paths(rebuild=True)

    question = sys.argv[1]
    result = answer(question)

    print(f"\n{LINE}")
    print(f"问：{question}")
    print(LINE)
    print(f"\n翻到 {result.retrieved} 张卡：")
    for c in result.citations:
        print(f"  · {c.source} / {c.heading}  (相关度 {c.score})")
    if not result.citations:
        print("  （一张都没翻到 → 触发最低分阈值）")
    print(f"\n答：\n{result.answer}")
    print(f"\n耗时 {result.latency_ms} ms ｜ 花费 ${result.cost_usd}")


if __name__ == "__main__":
    main()
