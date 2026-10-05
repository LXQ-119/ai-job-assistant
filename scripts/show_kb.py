"""把"资料柜"完全拆开：从文件夹到卡片，一层一层看。

资料柜的真身只有三样东西：
    1. 一个文件夹        data/knowledge/
    2. 里面几个普通文本文件  .md / .txt / .pdf / .docx
    3. 一个索引文件      data/index/kb_index.json（切好的卡片存这里）

跑法：
    python -m scripts.show_kb
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.services.rag import index_paths, load_index  # noqa: E402
from app.services.retriever import KB  # noqa: E402

LINE = "=" * 78


def main() -> None:
    settings = get_settings()

    print(f"\n{LINE}")
    print("第 1 层：资料柜在哪")
    print(LINE)
    print(f"  文件夹：{settings.knowledge_dir}")
    print(f"  索引文件：{settings.index_path}")
    print(f"  切块大小：{settings.chunk_size} 字（重叠 {settings.chunk_overlap} 字）")

    print(f"\n  文件夹里有什么：")
    for f in sorted(settings.knowledge_dir.iterdir()):
        if f.is_file():
            print(f"    · {f.name}   （{f.stat().st_size / 1024:.1f} KB）")

    print(f"\n  支持丢进去的格式：.md / .txt / .pdf / .docx")
    print(f"  ★ 也就是说：你往这个文件夹里扔一个文件，它就进资料柜了。")

    if not load_index():
        index_paths(rebuild=True)

    print(f"\n{LINE}")
    print("第 2 层：索引文件里存了什么")
    print(LINE)
    print(f"  索引文件大小：{settings.index_path.stat().st_size / 1024:.1f} KB")
    print(f"  它不是数据库，就是一个 JSON 文本文件。")
    print(f"  打开它你会看到：切好的每一张卡片 + 每张卡片对应的向量（现在没用向量，所以是 null）")

    print(f"\n{LINE}")
    print(f"第 3 层：切成了 {len(KB)} 张卡片")
    print(LINE)

    current_source = None
    for chunk in KB._chunks:  # noqa: SLF001
        if chunk.source != current_source:
            current_source = chunk.source
            same = [c for c in KB._chunks if c.source == current_source]  # noqa: SLF001
            print(f"\n┌── 文件：{current_source}   → 切成了 {len(same)} 张卡片")
            print("│")
        heading = chunk.heading or "（无标题）"
        preview = chunk.text.replace("\n", " ")[:58]
        print(f"│  卡片 #{chunk.index:<3} 「{heading}」")
        print(f"│            {len(chunk.text)} 字 ｜ {preview}…")
    print("└──")

    print(f"\n{LINE}")
    print("第 4 层：一张卡片的完整长相")
    print(LINE)
    sample = KB._chunks[1]  # noqa: SLF001
    print(f"  来源文件：{sample.source}")
    print(f"  所属标题：{sample.heading}")
    print(f"  卡片编号：#{sample.index}")
    print(f"\n  ── 实际拿去送给模型的内容 ──")
    for line in sample.text.splitlines():
        print(f"  {line}")

    print(f"\n{LINE}")
    print("第 5 层：送给模型之前，真正拼出来的提示词长什么样")
    print(LINE)
    from app.services.rag import _build_context, retrieve  # noqa: E402

    hits = retrieve("切块的时候为什么要保留重复内容？")
    context, _ = _build_context(hits)
    print(f"\n  你问：切块的时候为什么要保留重复内容？")
    print(f"  系统翻出 {len(hits)} 张卡，拼成下面这段塞给模型：\n")
    for line in context.splitlines()[:22]:
        print(f"  │ {line}")
    print("  │ ……（后面还有）")


if __name__ == "__main__":
    main()
