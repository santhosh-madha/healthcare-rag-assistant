# MedQuAD data expansion

The separate learning corpus contains **500 answer records from 481 source documents**, split into **1,020 chunks**. The web assistant now offers a collection selector. The terminal accepts `--collection medquad`; CDC remains the default. Both use the existing generation and quote-validation pipeline with separate saved indexes.

## Reproduce

Run from the project root with the virtual environment active. Clone only if the raw folder does not exist:

```bash
git clone https://github.com/abachaa/MedQuAD.git data/raw/medquad
git -C data/raw/medquad checkout 577bd37b96c02d1833b2c9eed2de9f96964e96cb
python -m healthcare_rag.ingestion.import_medquad --limit 500 --seed 42
python -m healthcare_rag.ingestion.chunk_medquad
python -m healthcare_rag.medquad_search --build
python -m healthcare_rag.medquad_search "What are the symptoms of Adult Acute Myeloid Leukemia?"
python -m healthcare_rag.medquad_search --benchmark
```

Importing and chunking refuse to overwrite their existing outputs. Once prepared, use the search and benchmark commands directly. Raw data, processed MedQuAD snapshots, indexes, and evaluation reports are Git-ignored; the code and this reproduction guide are tracked.

## Data flow and citations

XML answers → normalized answer records → token-budgeted chunks → normalized 384-dimensional embeddings → saved FAISS index → dense/BM25 reciprocal-rank fusion.

The importer excludes subsets 10–12, whose answers were removed upstream for reuse restrictions. It rejects missing content or citation URLs and deduplicates normalized exact answers before seeded sampling. The scan found 15,803 usable unique records; it selected 500. This is a random sample, not a balanced medical collection; genetics-related collections dominate it.

Each record retains its original source URL, title, XML filename, collection, record ID, and SHA-256 fingerprint. A manifest records the dataset revision, selection seed, transformations, and import time. Import time is not a clinical review date: unavailable review dates remain null. Historical source URLs are preserved but have not been checked for current availability or updated content.

Dataset questions live in a separate questions.json file. They are not concatenated into embedded answer text. Chunks prefer paragraph/sentence boundaries; 28 answers needed a word-boundary fallback for long passages. There is no overlap. The maximum observed chunk length is 182 tokens including special tokens, below the model's 256-token limit. Exact character offsets preserve traceability to each normalized answer.

The separate saved index is in data/index/medquad. Its model settings, corpus fingerprint, and index fingerprint are checked using the existing persistence layer. The CDC index is preserved.

## Initial local retrieval check

- Embedding and saving 1,020 vectors: approximately 6.73 seconds, excluding model loading.
- Seed-42 sample of 30 paired dataset questions: source-record Hit@1 = 25/30; Hit@3 = 28/30.
- Mean retrieval time: approximately 16.9 ms; p95: 17.8 ms on the development Mac.

Timing includes question embedding, BM25 computation, and fusion, but excludes model/index loading and answer generation. Hits mean the original paired answer record was retrieved. Other records may also answer the question. These are small paired-question smoke-test results, not independent answer accuracy or clinical validation. Reports are saved locally under evaluation_runs.

## Attribution

[MedQuAD](https://github.com/abachaa/MedQuAD) by Asma Ben Abacha and Dina Demner-Fushman is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its license is copied into each imported snapshot. Cite: *A Question-Entailment Approach to Question Answering*, BMC Bioinformatics (2019), [DOI](https://doi.org/10.1186/s12859-019-3119-4).

This project changes the data through XML extraction, whitespace normalization, deduplication, sampling, and chunking. Neither the dataset authors nor the original institutions endorse this project. The historical dataset is used for an educational retrieval demonstration.

## End-to-end check

See [the ten-question smoke check](medquad-smoke-check.md) for generation, quote validation, timing, and evidence-inspection results.
