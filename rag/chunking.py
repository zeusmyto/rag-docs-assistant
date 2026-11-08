"""Split pages into overlapping chunks that keep paragraphs together."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Page


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source} p.{self.page}"


def _paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def chunk_pages(pages: list[Page], max_chars: int = 500, overlap: int = 1) -> list[Chunk]:
    """Group paragraphs up to max_chars; repeat the last `overlap` paragraph(s)
    at the start of the next chunk so an answer spanning two chunks is not lost."""
    if max_chars < 100:
        raise ValueError("max_chars too small")
    chunks: list[Chunk] = []
    for page in pages:
        paras = _paragraphs(page.text)
        current: list[str] = []
        n = 0

        def flush():
            nonlocal n
            if current:
                chunks.append(Chunk(f"{page.source}#{page.page}-{n}", page.source, page.page, "\n".join(current)))
                n += 1

        for para in paras:
            # very long paragraph: hard-split it
            while len(para) > max_chars:
                flush()
                current = []
                chunks.append(Chunk(f"{page.source}#{page.page}-{n}", page.source, page.page, para[:max_chars]))
                n += 1
                para = para[max_chars:]
            if current and len("\n".join(current + [para])) > max_chars:
                flush()
                current = current[-overlap:] if overlap else []
                if len("\n".join(current + [para])) > max_chars:
                    current = []
            current.append(para)
        flush()
    return chunks
