"""Run our question set and save generated answers for human review."""

import json
from datetime import datetime, timezone
from pathlib import Path

from retrieve import load_passages
from semantic_search import (
    CACHE,
    MODEL_NAME,
    SentenceTransformer,
    build_index,
    search,
)
from rag_assistant import (
    GENERATION_MODEL,
    SYSTEM_PROMPT,
    build_context,
    generate_answer,
)


PROJECT = Path(__file__).parent


def main():
    questions = json.loads(
        (PROJECT / "evaluation_questions.json").read_text(
            encoding="utf-8"
        )
    )

    # Add two generation checks without changing our retrieval test file.
    extra_questions = [
        {
            "question": "Can I cancel my appointment by calling reception?",
            "category": "partial evidence",
        },
        {
            "question": "What time does the clinic close?",
            "category": "missing information",
        },
    ]

    existing = {item["question"] for item in questions}

    for item in extra_questions:
        if item["question"] not in existing:
            questions.append(item)
            existing.add(item["question"])

    passages = load_passages()

    if not passages:
        raise SystemExit("No passages found in data/sample.")

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(CACHE),
        device="cpu",
    )
    index = build_index(model, passages)

    timestamp = datetime.now(timezone.utc)
    output_folder = PROJECT / "evaluation_runs"
    output_folder.mkdir(exist_ok=True)

    output_file = output_folder / (
        timestamp.strftime("%Y%m%dT%H%M%S_%fZ") + ".json"
    )

    report = {
        "created_at": timestamp.isoformat(),
        "embedding_model": MODEL_NAME,
        "generation_model": GENERATION_MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "passages": passages,
        "results": [],
    }

    for number, test in enumerate(questions, 1):
        question = test["question"]
        print(f"\nQuestion {number}/{len(questions)}: {question}")

        results = search(model, index, passages, question, limit=3)
        context = build_context(results)

        answer = None
        error = None

        try:
            answer = generate_answer(question, context)
            print("Answer:", answer)
        except RuntimeError as exception:
            error = str(exception)
            print("Error:", error)

        report["results"].append({
            "test": test,
            "retrieved_passages": results,
            "context_sent": context,
            "answer": answer,
            "error": error,
            "human_review": {
                "answers_question": None,
                "claims_supported": None,
                "citations_support_claims": None,
                "missing_information_handled": None,
                "notes": "",
            },
        })

        # Save after every question so completed results are preserved.
        output_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nSaved report: {output_file}")


if __name__ == "__main__":
    main()