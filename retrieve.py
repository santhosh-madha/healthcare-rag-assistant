"""Lesson 1: retrieve passages using shared words. No model or API needed."""

import argparse
import re
from pathlib import Path


DOCUMENTS = Path(__file__).parent / "data" / "sample"
STOP_WORDS = {"a", "an", "the", "is", "are", "to", "of", "and", "in", "for", "i", "my", "how", "do", "can", "what"}


def tokenize(text):
    """Normalize text so 'Appointment?' and 'appointment' match."""
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


def load_passages():
    """Treat each paragraph as a chunk, preserving its source and position."""
    passages = []
    for path in sorted(DOCUMENTS.glob("*.txt")):
        for number, paragraph in enumerate(path.read_text().strip().split("\n\n"), 1):
            if paragraph.strip():
                passages.append({"source": path.name, "passage": number, "text": paragraph.strip()})
    return passages


def retrieve(question, passages, limit=3):
    """Rank by unique shared words; this score is not a confidence percentage."""
    query_words = tokenize(question)
    results = []
    for passage in passages:
        shared_words = query_words & tokenize(passage["text"])
        if shared_words:
            results.append({**passage, "score": len(shared_words), "matches": sorted(shared_words)})
    return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to search for, enclosed in quotes")
    args = parser.parse_args()
    passages = load_passages()
    print(f"Searching {len(passages)} passages from fictional learning documents.\n")
    results = retrieve(args.question, passages)
    if not results:
        print("No matching passages found. Try words used in the documents.")
    for rank, result in enumerate(results, 1):
        print(f"{rank}. [{result['source']}, passage {result['passage']}]")
        print(f"   Shared words: {', '.join(result['matches'])} | Score: {result['score']}")
        print(f"   {result['text']}\n")


if __name__ == "__main__":
    main()
