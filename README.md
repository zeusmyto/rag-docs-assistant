# rag-docs-assistant

![tests](https://github.com/zeusmyto/rag-docs-assistant/actions/workflows/ci.yml/badge.svg)

**An assistant that answers questions from a company's own documents (PDF, Word, Markdown), cites its sources, refuses when the answer is not in the documents, and comes with an evaluation suite that measures all of this.**

Runs fully local with Ollama (no data leaves the server) or with Claude through the Anthropic API.

```
$ python -m rag --embedder hashing --llm extractive --min-score 0.1 ask "Peut-on copier des données clients dans ChatGPT ?"
Données clients : il est interdit de copier des données clients dans des outils d'IA publics. [1]

  [1] securite_informatique.docx p.1  (score 0.42)
```

(Offline demo mode shown above. With Ollama or Claude the answer is written by the model, still with `[n]` citations.)

## How it works

```
documents ─► load (PDF / DOCX / MD) ─► chunk (paragraphs, overlap) ─► embed ─► vector index
                                                                                   │
question ─► embed ─► top-k chunks ─► best score < threshold ? ── yes ─► "I can't find it"
                                          │ no
                                          ▼
                              LLM answers ONLY from the numbered sources, with [n] citations
```

Design choices, and why:

- **Refuse before calling the model.** If no chunk is similar enough, the assistant answers "not in the documents" without calling the LLM. Cheaper, and no confident invented answers.
- **Numbered sources + strict prompt.** The model must cite `[n]` for every fact, so a user can check the answer in the original document.
- **Swappable back-ends.** Embeddings: Ollama (`nomic-embed-text`) or a model-free hashing embedder for tests. LLM: Ollama, Claude, or an extractive fallback.
- **Simple index.** A numpy matrix + JSON on disk: enough for thousands of chunks. For more, replace `VectorStore` with pgvector or Qdrant (same `add` / `search` methods).

## Evaluation: measure, don't guess

`evals/handbook.jsonl` lists questions with the expected document, facts the answer must contain, and out-of-scope questions the assistant must refuse.

```
$ python -m rag --embedder hashing --llm extractive --min-score 0.1 eval evals/handbook.jsonl
hit  facts cite refuse  question
---- ----- ---- ------  --------
ok  ok   ok  ok     Combien de jours de vacances pour un collaborateur à 100 % ?
ok  ok   ok  ok     Quelle longueur minimale pour un mot de passe ?
...
 -   -    -  ok     Qui a gagné la Coupe du monde de football 2018 ?

cases=11  hit@k=1.0  mrr=1.0  facts=1.0  cited=1.0  refusal_ok=1.0
```

| Metric | Meaning |
|---|---|
| `hit@k` | the right document is among the retrieved chunks |
| `mrr` | how high it ranks (1 = first) |
| `facts` | the answer contains the expected facts |
| `cited` | the answer cites its sources |
| `refusal_ok` | refuses out-of-scope questions, answers in-scope ones |

The same command with `--min-hit 0.9 --min-facts 0.8 --min-refusal 0.8` returns a non-zero exit code below those thresholds, so **CI blocks a change that makes answers worse** (see `.github/workflows/ci.yml`). The refusal threshold (`--min-score`) depends on the embedding model: tune it with the eval set, not by feel.

## Quick start (local, private)

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text && ollama pull llama3.1:8b

python -m rag ingest path/to/your/documents
python -m rag ask "Combien de jours de télétravail ?"
python -m rag eval evals/handbook.jsonl
```

With Claude instead of a local model:

```bash
export ANTHROPIC_API_KEY=...        # ANTHROPIC_MODEL to choose the model
python -m rag --llm claude ask "..."
```

Offline demo (no model at all): add `--embedder hashing --llm extractive --min-score 0.1`.

The sample documents in `sample_docs/` describe a **fictional** company.

## Tests

```bash
pip install pytest && python -m pytest -q
```

---

Built by [Adel Touhami](https://www.linkedin.com/in/adel-touhami-b35a352a) with Claude Code. MIT license.
