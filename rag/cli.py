"""Command line.

  python -m rag ingest sample_docs
  python -m rag ask "Combien de jours de vacances ?"
  python -m rag eval evals/handbook.jsonl --min-hit 0.9

Back-ends are chosen with --embedder (ollama|hashing) and --llm (ollama|claude|extractive).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evaluation
from .embeddings import make_embedder
from .llm import make_llm
from .pipeline import Assistant, ingest
from .store import VectorStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag", description="Answer questions from your own documents, with sources.")
    p.add_argument("--index", default=".rag_index", help="index folder (default .rag_index)")
    p.add_argument("--embedder", default="ollama", choices=["ollama", "hashing"])
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--llm", default="ollama", choices=["ollama", "claude", "extractive"])
    p.add_argument("--model", default=None, help="chat model (default llama3.1:8b for Ollama)")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("-k", type=int, default=4, help="chunks retrieved per question")
    p.add_argument("--min-score", type=float, default=0.25, help="refuse below this similarity")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest", help="index a folder of documents")
    s.add_argument("folder")
    s = sub.add_parser("ask", help="ask one question")
    s.add_argument("question")
    s = sub.add_parser("eval", help="run an evaluation file")
    s.add_argument("cases")
    s.add_argument("--min-hit", type=float, default=0.0)
    s.add_argument("--min-facts", type=float, default=0.0)
    s.add_argument("--min-refusal", type=float, default=0.0)
    s.add_argument("--json", action="store_true", help="print the summary as JSON")
    return p


def _embedder(args):
    if args.embedder == "ollama":
        return make_embedder("ollama", model=args.embed_model, url=args.ollama_url)
    return make_embedder("hashing")


def _llm(args):
    if args.llm == "ollama":
        return make_llm("ollama", model=args.model or "llama3.1:8b", url=args.ollama_url)
    if args.llm == "claude":
        return make_llm("claude", model=args.model)
    return make_llm("extractive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = Path(args.index)
    embedder = _embedder(args)

    if args.cmd == "ingest":
        store = ingest(Path(args.folder), embedder, index)
        print(f"Indexed {len(store.chunks)} chunks from {args.folder} -> {index}")
        return 0

    assistant = Assistant(VectorStore.load(index), embedder, _llm(args), k=args.k, min_score=args.min_score)
    if args.cmd == "ask":
        print(assistant.ask(args.question).pretty())
        return 0

    thresholds = {"hit@k": args.min_hit, "facts": args.min_facts, "refusal_ok": args.min_refusal}
    summary, passed, text = evaluation.run(assistant, Path(args.cases), thresholds)
    print(json.dumps(summary) if args.json else text)
    if not passed:
        print("\nQuality gate FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
