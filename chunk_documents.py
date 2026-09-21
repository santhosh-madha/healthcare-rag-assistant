"""Split CDC sections at paragraph or sentence boundaries within a token budget."""

import json
import re
from pathlib import Path

from semantic_search import CACHE, MODEL_NAME, SentenceTransformer


PROJECT = Path(__file__).parent
INPUT_FILE = PROJECT / "data" / "processed" / "cdc_documents.json"
OUTPUT_FILE = PROJECT / "data" / "processed" / "cdc_chunks.json"

CHUNK_TOKENS = 180
OVERLAP_TOKENS = 0


def count_tokens(text, tokenizer):
    """Count full text for splitting; do not encode an oversized section."""
    return len(tokenizer(
        text, add_special_tokens=False, truncation=False, verbose=False
    )["input_ids"])


def split_section(text, tokenizer):
    """Pack complete paragraphs, splitting long ones at sentence boundaries."""
    units = []
    for paragraph in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.DOTALL):
        if count_tokens(paragraph.group(), tokenizer) <= CHUNK_TOKENS:
            units.append((paragraph.start(), paragraph.end()))
            continue

        # Simple sentence heuristic; abbreviations can require manual review.
        for sentence in re.finditer(
            r"\S.*?(?:[.!?](?=\s|$)|$)", paragraph.group(), re.DOTALL
        ):
            start = paragraph.start() + sentence.start()
            end = paragraph.start() + sentence.end()
            if count_tokens(text[start:end], tokenizer) > CHUNK_TOKENS:
                raise ValueError(
                    "A sentence exceeds the chunk budget. Inspect: "
                    + text[start:end][:120]
                )
            units.append((start, end))

    chunks = []
    current_start = current_end = None
    for start, end in units:
        if current_start is None:
            current_start, current_end = start, end
        elif count_tokens(text[current_start:end], tokenizer) <= CHUNK_TOKENS:
            current_end = end
        else:
            chunks.append({
                "text": text[current_start:current_end],
                "character_start": current_start,
                "character_end": current_end,
            })
            current_start, current_end = start, end

    if current_start is not None:
        chunks.append({
            "text": text[current_start:current_end],
            "character_start": current_start,
            "character_end": current_end,
        })
    return chunks


def main():
    if CHUNK_TOKENS <= 0 or OVERLAP_TOKENS != 0:
        raise SystemExit("This splitter requires a positive budget and zero overlap.")

    documents = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    print("Loading the embedding model's tokenizer...", flush=True)

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(CACHE),
        device="cpu",
    )
    tokenizer = model.tokenizer

    print("Model input limit:", model.max_seq_length, "tokens")

    passages = []

    for document in documents:
        passage_number = 0

        for section_number, section in enumerate(
            document["sections"], 1
        ):
            chunks = split_section(section["text"], tokenizer)

            for chunk in chunks:
                passage_number += 1

                # Include special tokens in the actual length check.
                token_count = len(
                    tokenizer(
                        chunk["text"],
                        add_special_tokens=True,
                        truncation=False,
                    )["input_ids"]
                )

                if token_count > model.max_seq_length:
                    raise ValueError(
                        "Chunk exceeds model input limit. "
                        "Reduce CHUNK_TOKENS."
                    )

                passages.append({
                    "chunk_id": (
                        f"{document['id']}:"
                        f"chunk-{passage_number:03d}"
                    ),
                    "source": document["id"],
                    "passage": passage_number,
                    "title": document["title"],
                    "source_url": document["source_url"],
                    "source_reviewed_date": document[
                        "source_reviewed_date"
                    ],
                    "source_sha256": document["source_sha256"],
                    "section_number": section_number,
                    "section_heading": section["heading"],
                    "token_count": token_count,
                    **chunk,
                })

        print(
            f"{document['title']}: "
            f"{passage_number} chunks"
        )

    if not passages:
        raise SystemExit("No chunks were created.")

    report = {
        "chunking_strategy": "paragraph_sentence_no_overlap_v1",
        "embedding_model": MODEL_NAME,
        "model_input_limit": model.max_seq_length,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "passages": passages,
    }

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved {len(passages)} chunks to {OUTPUT_FILE}")
    print("\nFirst chunk:")
    print(json.dumps(passages[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
