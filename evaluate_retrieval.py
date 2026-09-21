"""Compare keyword and semantic retrieval on our learning questions."""

import json
from pathlib import Path

from retrieve import load_passages, retrieve
from semantic_search import (
    CACHE,
    MODEL_NAME,
    SentenceTransformer,
    build_index,
    search,
)


QUESTION_FILE = Path(__file__).parent / "evaluation_questions.json"


def matches_expected(result, test):
    """A match must have both the correct filename and passage number."""
    return (
        result["source"] == test["expected_source"]
        and result["passage"] == test["expected_passage"]
    )


def evaluate_results(results, test):
    """Check whether the expected passage appears first or in the top three."""
    hit_at_1 = bool(results) and matches_expected(results[0], test)

    hit_at_3 = any(
        matches_expected(result, test)
        for result in results[:3]
    )

    return hit_at_1, hit_at_3


def show_results(method, results):
    """Display source locations in ranked order."""
    print(f"\n{method} results:")

    if not results:
        print("  No results.")
        return

    for rank, result in enumerate(results, 1):
        print(
            f"  {rank}. {result['source']}, "
            f"passage {result['passage']}"
        )


def main():
    tests = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    passages = load_passages()

    if not tests:
        raise SystemExit("The evaluation file contains no questions.")

    if not passages:
        raise SystemExit("No passages found in data/sample.")

    print("Loading the embedding model once for all questions...")

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(CACHE),
        device="cpu",
    )

    index = build_index(model, passages)

    totals = {
        "Keyword": {"hit_at_1": 0, "hit_at_3": 0},
        "Semantic": {"hit_at_1": 0, "hit_at_3": 0},
    }

    answerable_count = 0
    missing_information_count = 0

    for number, test in enumerate(tests, 1):
        question = test["question"]

        print("\n" + "=" * 60)
        print(f"Question {number}: {question}")
        print(f"Category: {test['category']}")

        results_by_method = {
            "Keyword": retrieve(question, passages, limit=3),
            "Semantic": search(
                model,
                index,
                passages,
                question,
                limit=3,
            ),
        }

        for method, results in results_by_method.items():
            show_results(method, results)

        if test["expected_source"] is None:
            missing_information_count += 1
            print("\nExpected: no supporting passage.")
            print(
                "Inspect these results manually. "
                "Neither method checks whether the question is answerable."
            )
            continue

        answerable_count += 1

        print(
            f"\nExpected: {test['expected_source']}, "
            f"passage {test['expected_passage']}"
        )

        for method, results in results_by_method.items():
            hit_at_1, hit_at_3 = evaluate_results(results, test)

            totals[method]["hit_at_1"] += int(hit_at_1)
            totals[method]["hit_at_3"] += int(hit_at_3)

            print(
                f"{method}: "
                f"Hit@1 = {'YES' if hit_at_1 else 'NO'}, "
                f"Hit@3 = {'YES' if hit_at_3 else 'NO'}"
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"Answerable questions scored: {answerable_count}")
    print(
        "Missing-information questions inspected separately: "
        f"{missing_information_count}"
    )

    if answerable_count:
        for method, counts in totals.items():
            hit_1 = counts["hit_at_1"]
            hit_3 = counts["hit_at_3"]

            print(f"\n{method}:")
            print(
                f"  Hit@1: {hit_1}/{answerable_count} "
                f"({hit_1 / answerable_count:.0%})"
            )
            print(
                f"  Hit@3: {hit_3}/{answerable_count} "
                f"({hit_3 / answerable_count:.0%})"
            )

    print(
        "\nThis small learning set measures retrieval hits, "
        "not answer accuracy or medical reliability."
    )


if __name__ == "__main__":
    main()