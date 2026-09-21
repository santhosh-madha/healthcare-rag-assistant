"""Lesson 2: inspect embeddings and search them with FAISS."""

import argparse
import os
from pathlib import Path

# Keep processing simple for our small dataset.
# These settings also avoid a crash observed on this Mac.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
from sentence_transformers import SentenceTransformer

from retrieve import load_passages


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = Path(__file__).parent / ".cache" / "models"


def build_index(model, passages):
    """Convert passage text into vectors and store them in FAISS."""

    texts = [passage["text"] for passage in passages]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    # print("\n--- Passage embeddings ---")
    # print("Array shape:", vectors.shape)
    # print("First passage:", passages[0]["text"])
    # print("First 10 numbers:", vectors[0][:10])
    # print("All 384 numbers for the first passage:")
    # print(vectors[0])

    # Normalized vectors make inner product equal cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # print("\n--- FAISS index ---")
    # print("Index type:", type(index).__name__)
    # print("Number of stored vectors:", index.ntotal)
    # print("Numbers per vector:", index.d)
    # print("First 10 numbers stored at position 0:")
    # print(index.reconstruct(0)[:10])

    return index


def search(model, index, passages, question, limit=3):
    """Find the passage vectors most similar to the question vector."""

    if not question.strip() or not passages or limit <= 0:
        return []

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, positions = index.search(
        query_vector,
        min(limit, len(passages)),
    )

    print("\n--- Question embedding ---")
    print("Question:", question)
    print("Array shape:", query_vector.shape)
    print("First 10 numbers:", query_vector[0][:10])

    print("\n--- Raw FAISS results ---")
    print("Scores:", scores)
    print("Positions:", positions)

    print("\n--- Positions mapped to passages ---")
    results = []

    for score, position in zip(scores[0], positions[0]):
        passage = passages[int(position)]

        print(
            f"Position {position} -> "
            f"{passage['source']}, passage {passage['passage']} "
            f"| Similarity: {score:.3f}"
        )

        results.append({
            **passage,
            "score": float(score),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        help="Question to search for, enclosed in quotes",
    )
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("Please enter a non-empty question.")

    passages = load_passages()

    if not passages:
        parser.error("No sample passages found in data/sample.")

    print(
        "Loading embedding model "
        "(first run downloads model files)...",
        flush=True,
    )

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(CACHE),
        device="cpu",
    )

    index = build_index(model, passages)
    results = search(model, index, passages, args.question)

    print("\n--- Retrieved passages ---")
    print(
        "Scores measure similarity, not confidence. "
        "Results may not answer the question.\n"
    )

    for rank, result in enumerate(results, 1):
        print(
            f"{rank}. "
            f"[{result['source']}, passage {result['passage']}]"
        )
        print(f"   Cosine similarity: {result['score']:.3f}")
        print(f"   {result['text']}\n")


if __name__ == "__main__":
    main()