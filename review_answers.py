"""Create and summarize local human-review sheets for saved generation runs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

DIMENSIONS = ("correctness", "evidence_support", "completeness", "refusal_behavior")
FAILURES = {"retrieval_failure", "unsupported_claim", "incorrect_answer", "incomplete_answer",
            "unnecessary_refusal", "missing_refusal", "formatting_error", "generation_error"}
RUBRIC = """# Answer review guide

Review the answer against the retrieved passages AND the question's annotated evidence.
Use pass, fail, or na for each dimension. Leave null until reviewed.

- correctness: Does the answer accurately address the question according to this document snapshot?
- evidence_support: Does the cited evidence support EVERY factual claim, including qualifications?
- completeness: Does the answer cover the requested information without a material omission?
- refusal_behavior: Was answering or declining appropriate for the evidence actually retrieved?

For a justified refusal, use na for the first three dimensions and pass for refusal_behavior.
For an unnecessary refusal with an answer in the retrieved passages, use fail for completeness
and refusal_behavior; use na for correctness/evidence_support if there are no factual claims.
For a generation/validation error, use na where no answer can be assessed and record the error category.
A refusal can be appropriate given poor retrieval even when the full corpus contains an answer:
record retrieval_failure separately. Review hit metrics as hints, not ground truth.

Failure categories (multiple allowed): retrieval_failure, unsupported_claim, incorrect_answer,
incomplete_answer, unnecessary_refusal, missing_refusal, formatting_error, generation_error.
Record evidence and rationale in notes. Set reviewer to your name and completed to true only
when all four dimensions are filled. Mechanical quote matches do not establish support.
These are human judgments on this snapshot, not medical validation or a general accuracy benchmark.
"""


def create_review(report_path):
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if report.get("mode") == "retrieval" or not any(
        row.get("raw_response") or row.get("generation_error") for row in report["results"]
    ):
        raise ValueError("Choose a generation run, not a retrieval-only report.")
    sheet = {"schema_version": 1, "report": report_path.name,
             "report_sha256": hashlib.sha256(raw).hexdigest(), "reviewer": "",
             "review_type": "human", "items": []}
    sections = [RUBRIC]
    for row in report["results"]:
        test = row["test"]
        sheet["items"].append({"id": test["id"], "completed": False,
            **{key: None for key in DIMENSIONS}, "failure_categories": [], "notes": ""})
        sections.append(f"\n## {test['id']}: {test['question']}\n")
        sections.append("### Expected behavior and annotated evidence\n\n" + json.dumps(test, indent=2, ensure_ascii=False))
        sections.append("\n### Model response (may be invalid or unsupported)\n\n" + (row.get("raw_response") or "No response."))
        sections.append("\n### Mechanical checks\n\n" + json.dumps({key: row.get(key) for key in
            ("hit_at_1", "hit_at_3", "generation_error", "validation_error", "validation_passed")}, indent=2))
        for number, passage in enumerate(row["retrieved_passages"], 1):
            sections.append(f"\n### S{number}: {passage.get('title', passage.get('source', 'Source'))}\n\n"
                            + passage["text"] + "\n\n" + passage.get("source_url", ""))
    return sheet, "\n".join(sections)


def summarize(sheet):
    rows = sheet["items"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate review IDs.")
    done = []
    for row in rows:
        if not isinstance(row.get("completed"), bool):
            raise ValueError("completed must be true or false.")
        if not row["completed"]:
            continue
        if not sheet.get("reviewer", "").strip():
            raise ValueError("Set reviewer before marking reviews complete.")
        if any(row.get(key) not in ("pass", "fail", "na") for key in DIMENSIONS):
            raise ValueError(f"{row['id']}: fill all dimensions with pass, fail, or na.")
        tags = row.get("failure_categories")
        if not isinstance(tags, list) or any(tag not in FAILURES for tag in tags):
            raise ValueError(f"{row['id']}: invalid failure category.")
        if ("fail" in [row[key] for key in DIMENSIONS] or tags) and not row.get("notes", "").strip():
            raise ValueError(f"{row['id']}: explain failures in notes.")
        done.append(row)
    return {"total": len(rows), "reviewed": len(done), "unreviewed": len(rows)-len(done),
            "dimensions": {key: {rating: sum(row[key] == rating for row in done)
                                  for rating in ("pass", "fail", "na")} for key in DIMENSIONS},
            "failure_categories": dict(Counter(tag for row in done for tag in set(row["failure_categories"])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Saved evaluation JSON, or review JSON with --summary")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    try:
        if args.summary:
            print(json.dumps(summarize(json.loads(args.report.read_text())), indent=2))
            print("Only completed reviews counted; na and unreviewed entries are not passes.")
            return
        sheet, evidence = create_review(args.report)
        folder = Path(__file__).parent / "evaluation_runs" / "reviews"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / (args.report.stem + "_review.json")
        guide = target.with_suffix(".md")
        if target.exists() or guide.exists():
            raise ValueError(f"Review already exists; edit it to resume: {target}")
        with guide.open("x", encoding="utf-8") as output:
            output.write(evidence)
        with target.open("x", encoding="utf-8") as output:
            json.dump(sheet, output, indent=2, ensure_ascii=False)
            output.write("\n")
        print(f"Read evidence: {guide}\nFill scores: {target}")
        print("Original report unchanged. No scores have been assigned automatically.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
