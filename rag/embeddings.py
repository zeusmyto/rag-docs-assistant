"""Embedding back-ends.

- OllamaEmbedder: local model (default nomic-embed-text), data never leaves the server.
- HashingEmbedder: no model at all, deterministic. Used for tests and offline demos.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import requests


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434", batch: int = 32):
        self.model, self.url, self.batch = model, url.rstrip("/"), batch
        self.name = f"ollama:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = requests.post(f"{self.url}/api/embed",
                              json={"model": self.model, "input": texts[i:i + self.batch]}, timeout=120)
            r.raise_for_status()
            vectors.extend(r.json()["embeddings"])
        return _normalize(np.asarray(vectors, dtype=np.float32))


STOPWORDS = set("""
les des une est que qui quel quelle quels quelles pour par dans sur avec sont aux ces ses son
combien comment jour jours peut on faire etre avoir cette elle ils leur plus moins tout tous
the and for are what which how many does with from this that can
""".split())


def tokens(text: str) -> list[str]:
    """Lower-case, accent-free word tokens without stop words."""
    import unicodedata
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9@.]+", folded)
            if (len(t) > 2 or t.isdigit()) and t not in STOPWORDS]


class HashingEmbedder:
    """Bag of words hashed into a fixed-size vector (with accent folding).
    Not semantic, but good enough to test the pipeline end to end."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing:{dim}"

    @staticmethod
    def tokens(text: str) -> list[str]:
        return tokens(text)

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self.tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                m[row, h % self.dim] += 1.0
        return _normalize(m)


def make_embedder(kind: str, **kw):
    if kind == "ollama":
        return OllamaEmbedder(**kw)
    if kind == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedder: {kind}")
