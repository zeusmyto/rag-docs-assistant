import json
from pathlib import Path

import numpy as np
import pytest

from rag import llm as llm_mod
from rag.chunking import Chunk, chunk_pages
from rag.cli import main
from rag.embeddings import HashingEmbedder, OllamaEmbedder, tokens
from rag.evaluation import evaluate_case, summarize
from rag.llm import NO_ANSWER, ClaudeLLM, ExtractiveLLM, build_prompt
from rag.loaders import Page, load_folder
from rag.pipeline import Assistant, ingest
from rag.store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "sample_docs"
EVALS = ROOT / "evals" / "handbook.jsonl"


# --- loading and chunking ---------------------------------------------------

def test_loads_markdown_word_and_pdf():
    sources = {p.source for p in load_folder(DOCS)}
    assert sources == {"reglement_interne.md", "securite_informatique.docx", "guide_accueil.pdf"}


def test_chunks_respect_max_size_and_keep_source():
    text = "\n\n".join(f"Paragraphe {i} " + "mot " * 30 for i in range(10))
    chunks = chunk_pages([Page("doc.md", 3, text)], max_chars=400)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.source == "doc.md" and c.page == 3 for c in chunks)
    assert chunks[0].citation == "doc.md p.3"


def test_overlap_repeats_last_paragraph():
    text = "\n\n".join(["A" * 150, "B" * 150, "C" * 150])
    chunks = chunk_pages([Page("d", 1, text)], max_chars=320, overlap=1)
    assert chunks[1].text.startswith("B")


def test_long_paragraph_is_split():
    chunks = chunk_pages([Page("d", 1, "x" * 1200)], max_chars=500)
    assert len(chunks) == 3


# --- embeddings and store ---------------------------------------------------

def test_tokens_fold_accents_and_drop_stopwords():
    assert tokens("Quel est le télétravail ?") == ["teletravail"]


def test_hashing_embedder_is_normalized_and_deterministic():
    e = HashingEmbedder()
    a, b = e.embed(["vacances annuelles", "vacances annuelles"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5


def test_store_search_and_roundtrip(tmp_path):
    e = HashingEmbedder()
    chunks = [Chunk("1", "a.md", 1, "vacances 25 jours"), Chunk("2", "b.md", 1, "mot de passe 14 caracteres")]
    store = VectorStore(e.name)
    store.add(chunks, e.embed([c.text for c in chunks]))
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    best, score = loaded.search(e.embed(["mot de passe"])[0], k=1)[0]
    assert best.source == "b.md" and score > 0


def test_ollama_embedder_calls_embed_endpoint(monkeypatch):
    calls = []

    class R:
        def raise_for_status(self): pass
        def json(self): return {"embeddings": [[3.0, 4.0]] * len(calls[-1]["input"])}

    def fake_post(url, json, timeout):
        calls.append(json)
        assert url.endswith("/api/embed")
        return R()

    monkeypatch.setattr("rag.embeddings.requests.post", fake_post)
    m = OllamaEmbedder(batch=2).embed(["a", "b", "c"])
    assert m.shape == (3, 2) and np.allclose(m[0], [0.6, 0.8]) and len(calls) == 2


# --- generation -------------------------------------------------------------

def test_prompt_numbers_sources():
    p = build_prompt("Q ?", [Chunk("1", "a.md", 2, "texte A"), Chunk("2", "b.pdf", 1, "texte B")])
    assert "[1] (a.md p.2)" in p and "[2] (b.pdf p.1)" in p and p.endswith("Question: Q ?")


def test_extractive_picks_best_sentence_and_cites():
    src = [Chunk("1", "a.md", 1, "Le badge ouvre de 6 h a 22 h. Le salaire est verse le 25.")]
    assert ExtractiveLLM().answer("Quand le salaire est-il verse ?", src) == "Le salaire est verse le 25. [1]"


def test_claude_request_shape(monkeypatch):
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"content": [{"type": "text", "text": "25 jours [1]"}]}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return R()

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    out = ClaudeLLM(model="claude-test", api_key="k").answer("Q", [Chunk("1", "a.md", 1, "t")])
    assert out == "25 jours [1]"
    assert seen["headers"]["x-api-key"] == "k" and seen["body"]["model"] == "claude-test"
    assert "ONLY the numbered sources" in seen["body"]["system"]


# --- assistant and evaluation -----------------------------------------------

@pytest.fixture()
def assistant(tmp_path):
    e = HashingEmbedder()
    store = ingest(DOCS, e, tmp_path / "idx")
    return Assistant(store, e, ExtractiveLLM(), k=4, min_score=0.1)


def test_answers_with_source(assistant):
    ans = assistant.ask("Combien de jours de vacances pour un collaborateur à 100 % ?")
    assert "25" in ans.text and ans.sources[0][0].source == "reglement_interne.md"


def test_refuses_out_of_scope_without_calling_llm(tmp_path):
    class Boom:
        def answer(self, *a):
            raise AssertionError("LLM must not be called")
    e = HashingEmbedder()
    store = ingest(DOCS, e, tmp_path / "idx")
    ans = Assistant(store, e, Boom(), min_score=0.1).ask("Qui a gagné la Coupe du monde 2018 ?")
    assert ans.text == NO_ANSWER and ans.refused


def test_index_and_embedder_must_match(tmp_path):
    store = VectorStore("ollama:nomic-embed-text")
    with pytest.raises(ValueError):
        Assistant(store, HashingEmbedder(), ExtractiveLLM())


def test_evaluation_metrics(assistant):
    cases = [json.loads(line) for line in EVALS.read_text(encoding="utf-8").splitlines()]
    summary = summarize([evaluate_case(assistant, c) for c in cases])
    assert summary["hit@k"] == 1.0
    assert summary["facts"] >= 0.8
    assert summary["refusal_ok"] >= 0.8


def test_cli_quality_gate(tmp_path, capsys):
    idx = str(tmp_path / "idx")
    assert main(["--embedder", "hashing", "--index", idx, "ingest", str(DOCS)]) == 0
    code = main(["--embedder", "hashing", "--llm", "extractive", "--index", idx, "--min-score", "0.1",
                 "eval", str(EVALS), "--min-hit", "0.9", "--min-facts", "0.8", "--min-refusal", "0.8"])
    assert code == 0
    assert "hit@k=1.0" in capsys.readouterr().out
