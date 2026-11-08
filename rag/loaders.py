"""Read company documents (PDF, Word, Markdown, text) into pages of plain text."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


@dataclass(frozen=True)
class Page:
    source: str   # file name
    page: int     # 1-based page (or section) number
    text: str


def load_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [Page(path.name, i + 1, (p.extract_text() or "").strip())
                for i, p in enumerate(reader.pages)]
    if suffix == ".docx":
        import docx
        doc = docx.Document(str(path))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Page(path.name, 1, text)]
    if suffix in {".md", ".txt"}:
        return [Page(path.name, 1, path.read_text(encoding="utf-8"))]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_folder(folder: Path) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            pages.extend(p for p in load_file(path) if p.text)
    return pages
