# Healthcare RAG Assistant

A local educational document assistant using Python, Sentence Transformers, FAISS, and Llama 3.1 through Ollama. It retrieves evidence from three CDC diabetes articles, generates answers, and displays source references.

This is an ongoing learning and portfolio project, not a clinically validated system or a tool for personal diagnosis or treatment.

## Current features

- Keyword baseline over six fictional clinic passages.
- MiniLM embeddings: 384-dimensional normalized vectors.
- FAISS `IndexFlatIP` for exact cosine-similarity search.
- CDC article extraction with source URLs, review dates, and fingerprints.
- Paragraph/sentence chunking: 180 content-token budget, zero overlap, validated against the 256-token model limit.
- Local `llama3.1:8b` generation with requested citations and missing-evidence handling.
- Structured JSON answers with source-ID and evidence-quote validation.
- Persistent FAISS index with corpus/model consistency checks.
- Interactive terminal sessions with optional verbose diagnostics.
- Retrieval evaluation and saved answers for manual review.

The included CDC snapshot contains **3 documents and 20 chunks**. LangChain is not used; this project connects the underlying components directly.

## Architecture

```mermaid
flowchart LR
    A[Saved CDC HTML] --> B[Extract sections]
    B --> C[Chunks and source metadata]
    C --> D[MiniLM embeddings]
    D --> E[FAISS index]
    Q[Question] --> F[MiniLM question embedding]
    F --> E
    E --> G[Top 3 passages]
    G --> H[Local Llama via Ollama]
    Q --> H
    H --> I[Answer with citation labels]
    G --> J[Source URL reference key]
```

## Setup

Developed on Apple Silicon with 24 GB RAM and Python 3.14; other platforms have not been verified. Install [Ollama](https://ollama.com/download) separately and keep its service running.

```bash
git clone https://github.com/santhosh-madha/healthcare-rag-assistant.git
cd healthcare-rag-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
python persistent_search.py --build
python structured_healthcare.py "What is insulin resistance?"
```

Run the CDC development evaluation (requires Ollama for generation):

```bash
python evaluate_healthcare.py --retrieval-only
python evaluate_healthcare.py
python evaluate_healthcare.py --structured
python evaluate_healthcare.py --structured --questions cdc_holdout_questions.json
```

The eight-question CDC set covers six answerable and two missing-information cases. Strict annotated retrieval hits were 5/6 at rank one and 6/6 within three. The age question's first result is also relevant but is not included in its evidence annotations, so the strict rank-one miss is not a proven retrieval failure. This is a development set, not a held-out benchmark.

Review of the earlier free-form baseline found contradictory refusals on the typo and age cases, incomplete citation coverage, and an unsupported extrapolation in the testing answer. Both missing-information cases declined appropriately. Reports are saved locally; citation-label checks validate IDs only, not factual support. No aggregate answer-accuracy claim is made.

The first embedding run downloads MiniLM into `.cache/models`. Ollama stores its model separately. Inference is local; initial downloads require internet access. `requirements-lock.txt` records the development environment; `requirements.txt` lists direct dependencies.

The processed CDC snapshot is included, so article downloading and extraction are optional for trying the assistant.

## Learning stages

```bash
python retrieve.py "How to call off my booking?"
python semantic_search.py "How to call off my booking?"
python evaluate_retrieval.py
python rag_assistant.py "How can I request a copy of my records?"
python evaluate_answers.py
python search_healthcare.py "Can type 2 diabetes have no noticeable symptoms?"
python healthcare_assistant.py "What is insulin resistance?"
```

Fictional-clinic tests and CDC searches use separate collections. Generation reports go into `evaluation_runs/`, excluded from Git by default. Debug output exposes vectors, retrieval results, and evidence for learning.

## Evaluation so far

On **five answerable fictional-clinic development questions**:

| Method | Hit@1 | Hit@3 |
|---|---:|---:|
| Keyword | 3/5 (60%) | 3/5 (60%) |
| Semantic | 4/5 (80%) | 5/5 (100%) |

Hit@1 checks whether expected evidence ranks first; Hit@3 checks whether it appears in the first three. An additional missing-information question is inspected separately. These are not held-out benchmark results or medical accuracy measurements.

An eight-question generation run produced four supported answers to five answerable questions, appropriate refusals for two missing-information questions, and an overly terse refusal for a partial-evidence question. The typo case failed despite relevant evidence being retrieved. Review was AI-assisted, not independently human-verified.

Three CDC generation smoke checks completed: two produced supported main answers, and one declined an unavailable clinic-phone-number question. One answer added an uncited statement; the refusal did not match the requested wording exactly. Subsequent structured runs are summarized below.

## Sources and rebuilding

- [CDC: Diabetes Basics](https://www.cdc.gov/diabetes/about/index.html)
- [CDC: Type 2 Diabetes](https://www.cdc.gov/diabetes/about/about-type-2-diabetes.html)
- [CDC: Symptoms of Diabetes](https://www.cdc.gov/diabetes/signs-symptoms/index.html)

Source text is attributed to its original publisher. This project is not affiliated with or endorsed by CDC. The included snapshot may differ from current pages; source review dates are not download dates.

The automated downloader encountered HTTP 403 responses, so browser-saved HTML was used. To reproduce extraction, save the articles under `data/raw/manual_cdc/` with these filenames:

```text
Diabetes Basics _ Diabetes _ CDC.html
Type 2 Diabetes _ Diabetes _ CDC.html
Symptoms of Diabetes _ Diabetes _ CDC.html
```

Then run:

```bash
python extract_documents.py
python chunk_documents.py
```

Rebuilding replaces processed outputs and can change chunk IDs and search results. Raw browser downloads, environments, model caches, and local evaluation reports are excluded from Git.

## Limitations and next stages

- Citation labels are generated by the model; support and coverage are not automatically verified.
- Similarity is not confidence; nearest-neighbor search can return irrelevant evidence.
- Sentence splitting is heuristic; headings may become separated from their paragraphs.
- Earlier assistants and evaluation rebuild the index each run; the structured assistant now loads a saved index (see below).
- Prompt changes have caused regressions, including incorrect refusals.
- Next: an interface, broader evaluation, and stronger assessment of whether evidence supports each claim.

Earlier exercises are preserved in [learning notes](docs/learning-notes.md); these describe the initial stages, not current feature completeness.

## Saved-index structured assistant

Build document embeddings once, then query with checked evidence quotes:

```bash
python persistent_search.py --build
python structured_healthcare.py "What is insulin resistance?"
python -m unittest discover -v
```

The structured assistant loads the saved local index and embeds only the question. Rebuild after changing chunks or embedding models. Generated index files are ignored by Git. Model startup still occurs for every command. The evaluation runner and earlier assistants retain their in-memory indexing paths. Validation checks JSON structure, source IDs, and quoted text after whitespace normalization (including spaces before commas and periods). It does not verify semantic entailment or whether a refusal is justified. Model-weight revisions are not pinned.

## Interactive terminal session

```bash
python structured_healthcare.py
# Optional debug details:
python structured_healthcare.py --verbose
```

The model and saved index load once per session. Ask independent questions and type `/exit` to quit. There is no conversation memory. The session uses its initial document snapshot until restarted. Default output shows answers, evidence quotes, and source URLs; verbose mode additionally prints retrieval internals and raw JSON. Library startup messages may still appear. A quoted question still runs once and exits.

## Structured evaluation checkpoint

The eight-question development run passed structure/source/quote validation on all eight responses; annotated retrieval Hit@1 was 5/6 and Hit@3 was 6/6 for answerable questions. A first six-question held-out run had four answerable questions with Hit@1 and Hit@3 of 4/4, plus two missing-information cases; all six responses passed validation. These small runs preceded the punctuation-spacing fix and do not establish general answer accuracy.

AI-assisted review found the main answers supported and missing-information questions appropriately declined, with a minor precision issue in an additional claim about insulin release. Review was not independently human-verified. These sets are now inspected regression sets, not fresh holdouts for future tuning.

All 28 local unit/integration tests pass, covering quote validation, persistent-index consistency, and interactive session behavior. A user-run regression confirmed that quotes differing only in spaces before punctuation now pass. Matching a quote still does not prove it supports the generated claim.
