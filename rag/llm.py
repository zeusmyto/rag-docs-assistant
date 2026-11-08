"""Answer generation: local model through Ollama, or Claude through the Anthropic API."""
from __future__ import annotations

import os
import re

import requests

from .chunking import Chunk

NO_ANSWER = "Je ne trouve pas cette information dans les documents."

SYSTEM_PROMPT = f"""You answer questions about a company's internal documents.
Rules:
- Use ONLY the numbered sources below. Do not use outside knowledge.
- Cite every fact with its source number in brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"
- Answer in the language of the question, briefly."""


def build_prompt(question: str, sources: list[Chunk]) -> str:
    blocks = [f"[{i}] ({c.citation})\n{c.text}" for i, c in enumerate(sources, start=1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


class OllamaLLM:
    def __init__(self, model: str = "llama3.1:8b", url: str = "http://localhost:11434"):
        self.model, self.url = model, url.rstrip("/")

    def answer(self, question: str, sources: list[Chunk]) -> str:
        r = requests.post(f"{self.url}/api/chat", timeout=300, json={
            "model": self.model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": build_prompt(question, sources)}],
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class ClaudeLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def answer(self, question: str, sources: list[Chunk]) -> str:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json",
        }, json={
            "model": self.model, "max_tokens": 800, "temperature": 0, "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_prompt(question, sources)}],
        })
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()


class ExtractiveLLM:
    """No model: returns the source sentence that shares the most words with
    the question. For tests and offline demos, not for production."""

    def answer(self, question: str, sources: list[Chunk]) -> str:
        from .embeddings import tokens
        q = set(tokens(question))
        best, best_score, best_idx = None, 0, 0
        for idx, chunk in enumerate(sources, start=1):
            for sentence in re.split(r"(?<=[.!?;])\s+|\n", chunk.text):
                if sentence.lstrip().startswith("#"):
                    continue
                score = len(q & set(tokens(sentence)))
                if score > best_score:
                    best, best_score, best_idx = sentence.strip(), score, idx
        if not best:
            return NO_ANSWER
        return f"{best} [{best_idx}]"


def make_llm(kind: str, **kw):
    return {"ollama": OllamaLLM, "claude": ClaudeLLM, "extractive": ExtractiveLLM}[kind](**kw)
