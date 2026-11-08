"""Measure the assistant instead of guessing.

Each line of the eval file is a JSON object:
  {"question": "...", "expected_source": "handbook.md", "must_include": ["25 jours"], "answerable": true}
  {"question": "...", "answerable": false}      # out-of-scope: the assistant must refuse

Metrics
- hit@k        : the expected document is among the k retrieved chunks
- mrr          : 1 / rank of the first chunk from the expected document
- facts        : the answer contains every `must_include` string (accent/case-insensitive)
- refusal_ok   : refuses out-of-scope questions, and does NOT refuse answerable ones
- cited        : answerable answers contain at least one [n] citation
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .pipeline import Assistant


def fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


@dataclass
class CaseResult:
    question: str
    answerable: bool
    hit: bool | None
    rr: float | None
    facts: bool | None
    refusal_ok: bool
    cited: bool | None
    answer: str


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_case(assistant: Assistant, case: dict) -> CaseResult:
    answerable = case.get("answerable", True)
    ans = assistant.ask(case["question"])
    hit = rr = facts = cited = None
    if answerable:
        sources = [c.source for c, _ in ans.sources]
        expected = case.get("expected_source")
        hit = expected in sources if expected else None
        rr = (1.0 / (sources.index(expected) + 1)) if expected in sources else 0.0
        facts = all(fold(s) in fold(ans.text) for s in case.get("must_include", []))
        cited = bool(re.search(r"\[\d+\]", ans.text))
        refusal_ok = not ans.refused
    else:
        refusal_ok = ans.refused
    return CaseResult(case["question"], answerable, hit, rr, facts, refusal_ok, cited, ans.text)


def summarize(results: list[CaseResult]) -> dict[str, float]:
    def rate(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else 1.0
    answerable = [r for r in results if r.answerable]
    return {
        "cases": len(results),
        "hit@k": rate([r.hit for r in answerable]),
        "mrr": rate([r.rr for r in answerable]),
        "facts": rate([r.facts for r in answerable]),
        "cited": rate([r.cited for r in answerable]),
        "refusal_ok": rate([r.refusal_ok for r in results]),
    }


def report(results: list[CaseResult], summary: dict[str, float]) -> str:
    mark = {True: "ok ", False: "FAIL", None: " - "}
    lines = ["hit  facts cite refuse  question", "---- ----- ---- ------  --------"]
    for r in results:
        lines.append(f"{mark[r.hit]} {mark[r.facts]}  {mark[r.cited]} {mark[r.refusal_ok]}    {r.question[:70]}")
    lines.append("")
    lines.append("  ".join(f"{k}={v}" for k, v in summary.items()))
    return "\n".join(lines)


def run(assistant: Assistant, cases_path: Path, thresholds: dict[str, float] | None = None) -> tuple[dict, bool, str]:
    results = [evaluate_case(assistant, c) for c in load_cases(cases_path)]
    summary = summarize(results)
    passed = all(summary.get(k, 0) >= v for k, v in (thresholds or {}).items())
    return summary, passed, report(results, summary)
