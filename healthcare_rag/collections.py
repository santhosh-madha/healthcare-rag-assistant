"""Explicit collection paths keep independent corpora and indexes together."""
from healthcare_rag.paths import PROJECT
from healthcare_rag.persistent_search import read_corpus, load_index

COLLECTIONS = {
    "cdc": ("CDC diabetes", PROJECT / "data/processed/cdc_chunks.json", PROJECT / "data/index"),
    "medquad": ("MedQuAD medical topics", PROJECT / "data/processed/medquad/chunks.json", PROJECT / "data/index/medquad"),
}


def load_collection(model, name):
    if name not in COLLECTIONS:
        raise ValueError("Unknown collection.")
    _, path, folder = COLLECTIONS[name]
    passages, corpus_hash = read_corpus(path)
    return load_index(model, passages, corpus_hash, folder=folder), passages
