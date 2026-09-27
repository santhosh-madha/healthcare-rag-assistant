# Commands and folder guide

Run these from the repository root with the virtual environment activated. The data snapshots are unchanged. The web and structured terminal apps now use hybrid retrieval and JSON schema; evaluation defaults remain available for baseline comparisons.

## Main application (unchanged commands)

```bash
python persistent_search.py --build
python persistent_search.py "What is insulin resistance?"
python structured_healthcare.py
python web_assistant.py
python -m unittest discover -v
```

For one test module: `python -m unittest tests.test_structured_validation -v`.

## Ingestion

```bash
python -m healthcare_rag.ingestion.download_documents
python -m healthcare_rag.ingestion.extract_documents
python -m healthcare_rag.ingestion.chunk_documents
```

The automated CDC downloader previously received HTTP 403 responses. The extractor accepts browser-saved files in `data/raw/manual_cdc/`:

- `Diabetes Basics _ Diabetes _ CDC.html`
- `Type 2 Diabetes _ Diabetes _ CDC.html`
- `Symptoms of Diabetes _ Diabetes _ CDC.html`

Extraction and chunking replace processed files. Run `python persistent_search.py --build` afterward. Keep source attribution and review the extracted text.

## Evaluation and local reviews

```bash
python -m healthcare_rag.evaluation.evaluate_healthcare --retrieval-only
python -m healthcare_rag.evaluation.evaluate_healthcare --structured --questions data/evaluation/cdc_challenge_questions.json
python -m healthcare_rag.evaluation.review_answers evaluation_runs/YOUR_REPORT.json
python -m healthcare_rag.evaluation.review_answers evaluation_runs/reviews/YOUR_REPORT_review.json --summary
```

Question files moved from the root to `data/evaluation/`. Existing generated reports and private review files remain in `evaluation_runs/`; they are not moved or modified by this cleanup.

## Optional experiments

```bash
python -m healthcare_rag.experiments.compare_retrieval data/evaluation/cdc_challenge_questions.json data/evaluation/cdc_evaluation_questions.json data/evaluation/cdc_holdout_questions.json
python -m healthcare_rag.evaluation.evaluate_healthcare --structured --retriever hybrid --schema --questions data/evaluation/cdc_challenge_questions.json
python -m healthcare_rag.evaluation.evaluate_healthcare --structured --retriever hybrid --prompt-variant atomic --questions data/evaluation/cdc_challenge_questions.json
python -m healthcare_rag.experiments.inspect_claim_support --id birth_partial
python -m healthcare_rag.experiments.evaluate_claim_support --dry-run
```

The atomic prompt and support judge did not justify promotion to app defaults. Historical findings are preserved in [development history](development-history.md); claim-support examples have their own [guide](claim-support-guide.md).

## Earlier learning stages

```bash
python -m healthcare_rag.retrieve "How to call off my booking?"
python -m healthcare_rag.semantic_search "How to call off my booking?"
python -m healthcare_rag.rag_assistant "How can I request a copy of my records?"
python -m healthcare_rag.examples.evaluate_retrieval
python -m healthcare_rag.examples.evaluate_answers
python -m healthcare_rag.examples.search_healthcare "What is insulin resistance?"
python -m healthcare_rag.healthcare_assistant "What is insulin resistance?"
```

Shared early components remain in the main package because current code imports their search, tokenization, context-building, or Ollama functions. They are not duplicate copies of the root launchers.

## MedQuAD data expansion

See [MedQuAD setup](medquad.md) to prepare the separate dataset and index. After preparation:

```bash
python -m healthcare_rag.medquad_search "What are the symptoms of Adult Acute Myeloid Leukemia?"
python -m healthcare_rag.medquad_search --benchmark
```

### Select a collection

Start `python web_assistant.py` and choose CDC or MedQuAD in the document collection menu. Restart an already-running server to load this update. MedQuAD appears only when its prepared corpus and matching saved index load successfully; CDC remains available if MedQuAD is missing.

```bash
python structured_healthcare.py --collection medquad "What are the symptoms of Adult Acute Myeloid Leukemia?"
python structured_healthcare.py --collection cdc "What is insulin resistance?"
```

Both collections use hybrid retrieval, schema-constrained generation, and quote matching. These mechanical checks do not establish factual support.
