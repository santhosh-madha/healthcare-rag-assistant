"""Search the CDC chunks and display their original sources."""

import argparse
import json
from pathlib import Path

from semantic_search import (
    CACHE,
    MODEL_NAME,
    SentenceTransformer,
    build_index,
    search,
)


CHUNKS_FILE = (
    Path(__file__).parent
    / "data"
    / "processed"
    / "cdc_chunks.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Your question in quotes")
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("Please enter a non-empty question.")

    dataset = json.loads(
        CHUNKS_FILE.read_text(encoding="utf-8")
    )
    passages = dataset["passages"]

    if not passages:
        parser.error("The dataset contains no chunks.")

    if dataset["embedding_model"] != MODEL_NAME:
        parser.error(
            "The chunking model differs from the search model. "
            "Recreate the chunks with the current model."
        )

    print("Loading the embedding model...", flush=True)

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(CACHE),
        device="cpu",
    )

    index = build_index(model, passages)

    results = search(
        model,
        index,
        passages,
        args.question,
        limit=3,
    )

    print("\n--- Healthcare search results ---")

    for rank, result in enumerate(results, 1):
        print(f"\n{rank}. {result['title']}")
        print(f"Section: {result['section_heading']}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Similarity: {result['score']:.3f}")
        print(f"Source: {result['source_url']}")
        print(f"Source reviewed: {result['source_reviewed_date']}")
        print(f"\n{result['text']}")


if __name__ == "__main__":
    main()