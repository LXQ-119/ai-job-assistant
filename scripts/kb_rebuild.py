"""重建知识库索引，并可选地问一个问题验证。

用法：
    python -m scripts.kb_rebuild
    python -m scripts.kb_rebuild "我在小米的面试怎么样？"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.rag import answer, index_paths  # noqa: E402
from app.services.retriever import KB  # noqa: E402

LINE = "=" * 78


def main() -> None:
    info = index_paths(rebuild=True)

    print(f"\n{LINE}")
    print("重建完成")
    print(LINE)
    print(f"  文件数：{info['indexed_files']}")
    print(f"  卡片数：{info['total_chunks']}")
    if info.get("warnings"):
        print(f"  警告：{info['warnings']}")

    current = None
    for chunk in KB._chunks:  # noqa: SLF001
        if chunk.source != current:
            current = chunk.source
            count = sum(1 for c in KB._chunks if c.source == current)  # noqa: SLF001
            print(f"\n  【{current}】 → {count} 张卡")
        print(f"     #{chunk.index:<3} {chunk.heading or '(无标题)'}   （{len(chunk.text)} 字）")

    if len(sys.argv) > 1:
        question = sys.argv[1]
        print(f"\n{LINE}")
        print(f"验证提问：{question}")
        print(LINE)
        result = answer(question)
        print(f"\n翻到 {result.retrieved} 张卡：")
        for c in result.citations:
            print(f"  · {c.source} / {c.heading}  (相关度 {c.score})")
        print(f"\n回答：\n{result.answer}")


if __name__ == "__main__":
    main()
