# Architecture

Two stages connect source documents to cited answers. Each collection keeps its own corpus and FAISS index; the embedding model is shared.

```mermaid
flowchart TB
  subgraph preparation[Prepare once per snapshot]
    CDC[Saved CDC HTML] --> Extract[Extract sections]
    XML[MedQuAD XML] --> Clean[Normalize, filter and deduplicate]
    Clean --> Sample[Seeded sample: 500 answer records]
    Sample --> Questions[Separate paired questions for evaluation]
    Sample --> Chunk[Token-budgeted chunks with source metadata]
    Extract --> Chunk
    Chunk --> Embed[MiniLM: normalized 384-dimensional vectors]
    Embed --> Save[Separate saved FAISS indexes]
    Chunk --> Metadata[Text, IDs, URLs and fingerprints]
  end
  subgraph answering[For each question]
    UI[Web or terminal: choose collection] --> Query[Question]
    Query --> QE[MiniLM query embedding]
    QE --> Dense[FAISS semantic top 10]
    Save --> Dense
    Query --> BM25[BM25 keyword top 10]
    Metadata --> BM25
    Dense --> RRF[Reciprocal rank fusion: final top 3]
    BM25 --> RRF
    RRF --> Context[Evidence labeled S1, S2, S3]
    Metadata --> Context
    Context --> LLM[Ollama / Llama 3.1 8B + JSON schema]
    Query --> LLM
    LLM --> Check[Validate structure, source labels and quote matches]
    Check --> Answer[Answer and quotes with original source links]
    Check --> Withhold[Withhold invalid output]
    LLM --> Refusal[Insufficient-evidence response]
  end
```

## Code map

| Responsibility | Implementation |
|---|---|
| Import and chunk MedQuAD | `healthcare_rag/ingestion/import_medquad.py`, `chunk_medquad.py` |
| Encode and persist vectors | `semantic_search.py`, `persistent_search.py` |
| Choose corpus/index | `collections.py` |
| Combine semantic and keyword retrieval | `hybrid_search.py` |
| Generate and validate answers | `structured_healthcare.py`, `structured_schema.py`, `rag_assistant.py` |
| Serve the interface | `web_assistant.py`, `web/index.html` |
| End-to-end smoke checks | `evaluation/medquad_smoke.py` |

Paths in the table are relative to `healthcare_rag/` unless shown otherwise. The interface file is at repository-root `web/index.html`.

FAISS stores vectors; accompanying metadata maps vector positions back to passages. Source labels are assigned per question. The system does not fine-tune Llama, retain conversation memory, or verify clinical correctness. A real quote can still be insufficient to support a claim; validation failures are withheld rather than displayed as answers.
