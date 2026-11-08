"""A tiny vector store: one numpy matrix + JSON metadata on disk.

Enough for thousands of chunks on a laptop. For millions, swap in
pgvector or Qdrant behind the same two methods (add / search).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, embedder_name: str):
        self.embedder_name = embedder_name
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        self.chunks.extend(chunks)
        self.matrix = vectors if self.matrix.size == 0 else np.vstack([self.matrix, vectors])

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = self.matrix @ query_vector.reshape(-1)  # vectors are normalized -> cosine
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.matrix)
        meta = {"embedder": self.embedder_name, "chunks": [asdict(c) for c in self.chunks]}
        (folder / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        meta = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        store = cls(meta["embedder"])
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store.matrix = np.load(folder / "vectors.npy")
        return store
