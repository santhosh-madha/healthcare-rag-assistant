"""Save CDC document vectors once and reuse them for question retrieval."""

import argparse
import hashlib
import json
from pathlib import Path

# Apply thread settings before importing FAISS.
from semantic_search import CACHE, MODEL_NAME, SentenceTransformer, build_index, search
import faiss

PROJECT = Path(__file__).parent
CHUNKS_FILE = PROJECT / "data" / "processed" / "cdc_chunks.json"
INDEX_FOLDER = PROJECT / "data" / "index"


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


def model_settings(model):
    return {
        "model_name": MODEL_NAME,
        "dimension": model.get_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
        "normalize_embeddings": True,
        "index_type": "IndexFlatIP",
    }


def read_corpus(path=CHUNKS_FILE):
    raw = path.read_bytes()
    dataset = json.loads(raw)
    if dataset["embedding_model"] != MODEL_NAME:
        raise ValueError("Chunking model differs from the search model.")
    if not dataset["passages"]:
        raise ValueError("The corpus contains no passages.")
    return dataset["passages"], fingerprint(raw)


def save_index(model, passages, corpus_hash, folder=INDEX_FOLDER):
    """Write a local vector index and its ordered passage metadata."""
    index = build_index(model, passages)
    folder.mkdir(parents=True, exist_ok=True)
    temporary_index = folder / "cdc.faiss.tmp"
    temporary_metadata = folder / "metadata.json.tmp"
    faiss.write_index(index, str(temporary_index))
    metadata = {
        "schema_version": 1,
        "corpus_sha256": corpus_hash,
        "index_sha256": fingerprint(temporary_index.read_bytes()),
        "model_settings": model_settings(model),
        "passages": passages,
    }
    temporary_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    # Each replacement is atomic, but the pair is not. If interrupted between
    # replacements, checksum/metadata checks refuse a mismatched pair.
    temporary_index.replace(folder / "cdc.faiss")
    temporary_metadata.replace(folder / "metadata.json")
    return index


def load_index(model, passages, corpus_hash, folder=INDEX_FOLDER):
    """Validate metadata and checksum before reading our locally built index."""
    rebuild = "Run: python3 persistent_search.py --build"
    index_file = folder / "cdc.faiss"
    metadata_file = folder / "metadata.json"
    if not index_file.exists() or not metadata_file.exists():
        raise ValueError(f"No complete saved index found. {rebuild}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported metadata version. {rebuild}")
    if metadata["corpus_sha256"] != corpus_hash:
        raise ValueError(f"The chunk file changed. {rebuild}")
    if metadata["model_settings"] != model_settings(model):
        raise ValueError(f"Embedding settings changed. {rebuild}")
    if metadata["passages"] != passages:
        raise ValueError(f"Passage metadata or ordering changed. {rebuild}")
    if fingerprint(index_file.read_bytes()) != metadata["index_sha256"]:
        raise ValueError(f"Index checksum mismatch. {rebuild}")
    index = faiss.read_index(str(index_file))
    if (index.ntotal != len(passages)
            or index.d != model.get_embedding_dimension()
            or not isinstance(index, faiss.IndexFlatIP)
            or index.metric_type != faiss.METRIC_INNER_PRODUCT):
        raise ValueError(f"Index properties do not match metadata. {rebuild}")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question in quotes")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build and args.question is not None:
        parser.error("Use --build or a question, not both.")
    if not args.build and not (args.question or "").strip():
        parser.error("Provide a question or use --build.")
    try:
        passages, corpus_hash = read_corpus()
        print("Loading the embedding model...", flush=True)
        model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE), device="cpu")
        if args.build:
            index = save_index(model, passages, corpus_hash)
            print(f"Saved {index.ntotal} vectors of dimension {index.d} to {INDEX_FOLDER}")
            return
        index = load_index(model, passages, corpus_hash)
        print(f"Loaded {index.ntotal} saved vectors. Document embeddings were not rebuilt.")
        results = search(model, index, passages, args.question, limit=3)
        for rank, result in enumerate(results, 1):
            print(f"\n{rank}. {result['title']} — {result['section_heading']}")
            print(f"Chunk: {result['chunk_id']}")
            print(f"Similarity: {result['score']:.3f}")
            print(f"Source: {result['source_url']}")
            print(result["text"])
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
