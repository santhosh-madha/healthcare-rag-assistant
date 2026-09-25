"""Evaluate baseline or structured CDC answers on the same questions."""

import argparse
import hashlib
import io
import json
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from healthcare_assistant import (
    CHUNKS_FILE,
    HEALTHCARE_PROMPT,
    build_healthcare_context,
)
from rag_assistant import GENERATION_MODEL, generate_answer
from semantic_search import (
    CACHE,
    MODEL_NAME,
    SentenceTransformer,
    build_index,
    search,
)
from structured_healthcare import STRUCTURED_PROMPT, validate_response


PROJECT = Path(__file__).parent


def normalize(text):
    return " ".join(text.lower().split())


def evidence_hit(passage, evidence):
    """Check annotated source text, rather than sequential chunk IDs."""
    return any(
        passage["source"] == item["source"]
        and normalize(item["quote"]) in normalize(passage["text"])
        for item in evidence
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--structured",
        action="store_true",
        help="Generate JSON answers and validate their evidence quotes",
    )
    modes.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation",
    )

    parser.add_argument(
    "--questions",
    type=Path,
    default=PROJECT / "cdc_evaluation_questions.json",
    help="Path to the evaluation question JSON file",
)
    parser.add_argument("--retriever", choices=("dense", "hybrid"), default="dense")
    parser.add_argument("--prompt-variant", choices=("original", "atomic"), default="original")
    parser.add_argument("--schema", action="store_true", help="Constrain structured generation with JSON Schema")
    args = parser.parse_args()
    if args.schema and not args.structured:
        parser.error("--schema requires --structured")
    if args.prompt_variant != "original" and not args.structured:
        parser.error("--prompt-variant atomic requires --structured")
    retrieve_fn = search
    retrieval_settings = {"method": "dense", "limit": 3}
    if args.retriever == "hybrid":
        from hybrid_search import search as retrieve_fn, K1, B, RRF_K, CANDIDATES
        retrieval_settings = {"method": "hybrid", "limit": 3, "bm25_k1": K1,
                              "bm25_b": B, "rrf_k": RRF_K, "candidates": CANDIDATES}

    mode = (
        "structured" if args.structured
        else "retrieval" if args.retrieval_only
        else "baseline"
    )
    prompt = STRUCTURED_PROMPT if args.structured else HEALTHCARE_PROMPT
    if args.prompt_variant == "atomic":
        from experimental_prompt import ATOMIC_PROMPT
        prompt = ATOMIC_PROMPT

    raw_corpus = CHUNKS_FILE.read_bytes()
    corpus = json.loads(raw_corpus)
    passages = corpus["passages"]

    if not args.questions.is_file():
        parser.error(f"Question file not found: {args.questions}")

    test_set = json.loads(
        args.questions.read_text(encoding="utf-8")
    )
    tests = test_set["questions"]

    if not passages or not tests:
        parser.error("The corpus or question set is empty.")

    if corpus["embedding_model"] != MODEL_NAME:
        parser.error("Rebuild chunks using the current embedding model.")

    if len({test["id"] for test in tests}) != len(tests):
        parser.error("Duplicate question IDs.")

    for test in tests:
        if test["answerable"] and not any(
            evidence_hit(passage, test["evidence"])
            for passage in passages
        ):
            parser.error(
                f"Evidence annotation no longer matches: {test['id']}"
            )

    print(f"Mode: {mode}")
    print("Loading model and building one index...", flush=True)

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(CACHE),
        device="cpu",
    )

    with redirect_stdout(io.StringIO()):
        index = build_index(model, passages)

    now = datetime.now(timezone.utc)
    folder = PROJECT / "evaluation_runs"
    folder.mkdir(exist_ok=True)

    filename = (
        f"cdc_{mode}_"
        + now.strftime("%Y%m%dT%H%M%S_%fZ")
        + ".json"
    )
    output_file = folder / filename

    report = {
        "created_at": now.isoformat(),
        "mode": mode,
        "retriever": args.retriever,
        "retrieval_settings": retrieval_settings,
        "embedding_model": MODEL_NAME,
        "generation_model": GENERATION_MODEL,
        "system_prompt": prompt,
        "prompt_variant": args.prompt_variant,
        "response_format": "schema" if args.schema else ("json" if args.structured else None),
        "corpus_sha256": hashlib.sha256(raw_corpus).hexdigest(),
        "corpus": corpus,
        "test_set": test_set,
        "results": [],
    }

    for test in tests:
        print(f"\nRunning {test['id']}: {test['question']}", flush=True)

        with redirect_stdout(io.StringIO()):
            results = retrieve_fn(
                model, index, passages, test["question"], limit=3
            )

        context = build_healthcare_context(results)

        raw_response = None
        structured_response = None
        generation_error = None
        validation_error = None
        validation_passed = None

        response_format = "json" if args.structured else None
        if args.schema:
            from structured_schema import response_schema
            response_format = response_schema(len(results))

        if not args.retrieval_only:
            try:
                raw_response = generate_answer(
                    test["question"],
                    context,
                    system_prompt=prompt,
                    response_format=response_format,
                )
            except RuntimeError as error:
                generation_error = str(error)

            if args.structured and generation_error is None:
                try:
                    structured_response = validate_response(
                        raw_response, results
                    )
                    validation_passed = True
                except ValueError as error:
                    validation_error = str(error)
                    validation_passed = False

        entry = {
            "test": test,
            "retrieved_passages": results,
            "context_sent": context,
            "hit_at_1": (
                bool(results) and evidence_hit(results[0], test["evidence"])
                if test["answerable"] else None
            ),
            "hit_at_3": (
                any(evidence_hit(p, test["evidence"]) for p in results)
                if test["answerable"] else None
            ),
            "response_format_sent": response_format,
            "raw_response": raw_response,
            "structured_response": structured_response,
            "generation_error": generation_error,
            "validation_error": validation_error,
            "validation_passed": validation_passed,
            "review": {
                "reviewer": None,
                "behavior_pass": None,
                "claims_supported": None,
                "notes": "",
            },
        }

        report["results"].append(entry)

        # Preserve each completed result, even if a later request fails.
        output_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if generation_error:
            print("Generation error:", generation_error)
        elif validation_error:
            print("Validation error:", validation_error)
        elif structured_response is not None:
            print("Status:", structured_response["status"])
            for claim in structured_response["claims"]:
                print(f"  {claim['text']} [{claim['source']}]")
        elif raw_response is not None:
            print(raw_response)

    rows = report["results"]
    answerable = [row for row in rows if row["test"]["answerable"]]

    summary = {
        "mode": mode,
        "questions": len(rows),
        "answerable_questions": len(answerable),
        "hit_at_1": sum(row["hit_at_1"] for row in answerable),
        "hit_at_3": sum(row["hit_at_3"] for row in answerable),
        "generation_errors": sum(
            row["generation_error"] is not None for row in rows
        ),
        "validation_errors": (
            sum(row["validation_passed"] is False for row in rows)
            if args.structured else None
        ),
        "validation_passes": (
            sum(row["validation_passed"] is True for row in rows)
            if args.structured else None
        ),
    }

    report["summary"] = summary
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {output_file}")
    print("Validation passes do not establish answer correctness.")

    if summary["generation_errors"] or summary["validation_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()