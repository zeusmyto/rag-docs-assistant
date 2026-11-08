"""Ingest documents and answer questions with citations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_pages
from .llm import NO_ANSWER
from .loaders import load_folder
from .store import VectorStore


@dataclass
class Answer:
    question: str
    text: str
    sources: list[tuple[Chunk, float]] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return NO_ANSWER.lower() in self.text.lower()

    def pretty(self) -> str:
        lines = [self.text, ""]
        for i, (chunk, score) in enumerate(self.sources, start=1):
            lines.append(f"  [{i}] {chunk.citation}  (score {score:.2f})")
        return "\n".join(lines)


def ingest(folder: Path, embedder, index_dir: Path, max_chars: int = 500) -> VectorStore:
    pages = load_folder(folder)
    if not pages:
        raise SystemExit(f"No supported documents in {folder}")
    chunks = chunk_pages(pages, max_chars=max_chars)
    store = VectorStore(embedder.name)
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    store.save(index_dir)
    return store


class Assistant:
    """Retrieve the k best chunks, then let the LLM answer from them only.

    If the best match is below `min_score`, we refuse without calling the LLM:
    cheaper, and it avoids confident answers to questions the documents
    do not cover."""

    def __init__(self, store: VectorStore, embedder, llm, k: int = 4, min_score: float = 0.25):
        if store.embedder_name != embedder.name:
            raise ValueError(f"Index built with {store.embedder_name}, not {embedder.name}. Re-run ingest.")
        self.store, self.embedder, self.llm = store, embedder, llm
        self.k, self.min_score = k, min_score

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        return self.store.search(self.embedder.embed([question])[0], self.k)

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits or hits[0][1] < self.min_score:
            return Answer(question, NO_ANSWER, hits)
        text = self.llm.answer(question, [c for c, _ in hits])
        return Answer(question, text, hits)
