"""文档加载与切块。

切块（chunking）是 RAG 里最容易被忽略、但对效果影响最大的一步。
策略分三层：
1. 先按 Markdown 标题切 —— 尊重语义边界，并把"父级标题 > 子标题"作为面包屑
   拼进每个块，这样单独看一个块也知道它属于哪一节
2. 再按段落累积到接近 chunk_size —— 尽量不把一段话从中间截断
3. 超长段落硬切并保留 overlap —— 防止答案正好落在切口上

overlap 的作用常被误解：它不是"多存点字"，而是保证跨块的语义在两侧都完整出现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class DocumentError(RuntimeError):
    """读文档失败，信息面向使用者（会说清楚装哪个包）。"""


@dataclass
class Chunk:
    source: str        # 来源文件名，回答时用来做引用
    heading: str       # 标题面包屑，例如 "RAG > 检索策略"
    text: str
    index: int         # 在整篇里的序号

    @property
    def retrieval_text(self) -> str:
        """真正参与检索/向量化的文本：把标题拼在前面。

        这一步很小但收益明显——标题里往往就是关键词，
        不拼上去的话，正文里只提了一次的术语很容易被漏召回。
        """
        return f"{self.heading}\n{self.text}" if self.heading else self.text

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "heading": self.heading,
            "text": self.text,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            source=data["source"],
            heading=data.get("heading", ""),
            text=data["text"],
            index=int(data.get("index", 0)),
        )


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def read_document(path: Path) -> str:
    """把各种格式的文档读成纯文本。"""
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentError("解析 PDF 需要先安装：pip install pypdf") from exc
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages)

    if suffix == ".docx":
        try:
            import docx
        except ImportError as exc:
            raise DocumentError("解析 Word 需要先安装：pip install python-docx") from exc
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)

    raise DocumentError(
        f"不支持的文件类型 {suffix!r}。当前支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def collect_files(paths: list[Path]) -> list[Path]:
    """展开目录，过滤掉不支持的扩展名，并跳过隐藏文件。"""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    if not candidate.name.startswith("."):
                        files.append(candidate)
        elif path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(path)
    # 去重并保持稳定顺序，避免每次索引顺序不同导致结果不可复现
    return sorted(set(files))


# ---------------------------------------------------------------------------
# 切块
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    *,
    source: str,
    size: int = 600,
    overlap: int = 100,
) -> list[Chunk]:
    """把一篇文本切成若干块。"""
    size = max(100, size)
    # overlap 不能超过块大小的一半，否则块会长到停不下来
    overlap = max(0, min(overlap, size // 2))

    chunks: list[Chunk] = []
    for heading, body in _split_sections(text):
        for piece in _pack(body, size, overlap):
            if piece.strip():
                chunks.append(
                    Chunk(source=source, heading=heading, text=piece.strip(), index=len(chunks))
                )
    return chunks


def load_and_chunk(
    paths: list[Path],
    *,
    size: int = 600,
    overlap: int = 100,
) -> tuple[list[Chunk], list[str]]:
    """读取并切块，返回 (所有块, 成功处理的文件名)。

    单个文件失败不会中断整体索引——只跳过它。
    这一点很重要：真实数据里总有几个坏文件，不能让一个坏文件毁掉整次索引。
    """
    chunks: list[Chunk] = []
    processed: list[str] = []

    for path in collect_files(paths):
        try:
            text = read_document(path)
        except DocumentError:
            raise
        except Exception:  # noqa: BLE001 - 单个文件解析失败只跳过
            continue
        if not text.strip():
            continue
        chunks.extend(chunk_text(text, source=path.name, size=size, overlap=overlap))
        processed.append(path.name)

    return chunks, processed


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切分，返回 (标题面包屑, 正文) 列表。"""
    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_heading, body))

    in_code_block = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        # 代码块里的 "# 注释" 不是标题，必须排除
        match = None if in_code_block else _HEADING_RE.match(line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, match.group(2)))
            current_heading = " > ".join(title for _, title in heading_stack)
            continue

        buffer.append(line)

    flush()

    if not sections:
        stripped = text.strip()
        return [("", stripped)] if stripped else []
    return sections


def _pack(text: str, size: int, overlap: int) -> list[str]:
    """把段落累积成不超过 size 的块。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(paragraph, size, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= size:
            current = candidate
        else:
            pieces.append(current)
            # 用上一块的尾部作为衔接上下文
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{paragraph}" if tail else paragraph

    if current:
        pieces.append(current)
    return pieces


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """超长段落（比如没有空行的长文）按固定窗口硬切。"""
    step = max(1, size - overlap)
    windows = [text[i : i + size] for i in range(0, len(text), step)]
    return [w for w in windows if w.strip()]
