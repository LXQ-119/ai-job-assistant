"""把简历文件读成纯文本 —— 方便在终端里看内容、或者喂给别的脚本。

用法：
    python -m scripts.resume_to_text "C:\\path\\to\\简历.pdf"
    python -m scripts.resume_to_text "简历.pdf" --save
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.documents import DocumentError, read_document  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = (ROOT / path).resolve()

    if not path.exists():
        print(f"找不到文件：{path}")
        return

    try:
        text = read_document(path)
    except DocumentError as exc:
        print(f"读取失败：{exc}")
        return

    print(f"\n{'=' * 74}")
    print(f"文件：{path.name}　（{len(text)} 字）")
    print("=" * 74)
    print(text)

    if "--save" in sys.argv:
        out = ROOT / "data" / "resumes" / f"{path.stem}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\n已存到：{out}")


if __name__ == "__main__":
    main()
