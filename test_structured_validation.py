"""Test evidence validation without running an embedding model or Llama."""

import json
import unittest

from structured_healthcare import validate_response


class TestStructuredValidation(unittest.TestCase):
    def setUp(self):
        # Synthetic evidence isolates validation from healthcare knowledge.
        self.passages = [
            {
                "text": "Patients can book appointments through the portal."
            }
        ]

    def make_response(
        self,
        text="Patients can book appointments through the portal.",
        source="S1",
        quote="Patients can book appointments through the portal.",
    ):
        return {
            "status": "answered",
            "claims": [
                {
                    "text": text,
                    "source": source,
                    "quote": quote,
                }
            ],
        }

    def validate(self, response):
        return validate_response(
            json.dumps(response),
            self.passages,
        )

    def test_valid_answer_passes(self):
        result = self.validate(self.make_response())
        self.assertEqual(result["status"], "answered")

    def test_malformed_json_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_response("This is not JSON.", self.passages)

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(self.make_response(source="S9"))

    def test_fabricated_quote_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(
                self.make_response(
                    quote="Patients can cancel appointments by email."
                )
            )

    def test_empty_quote_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate(self.make_response(quote="   "))

    def test_missing_field_is_rejected(self):
        response = self.make_response()
        del response["claims"][0]["source"]

        with self.assertRaises(ValueError):
            self.validate(response)

    def test_answered_without_claims_is_rejected(self):
        with self.assertRaises(ValueError):
            self.validate({
                "status": "answered",
                "claims": [],
            })

    def test_refusal_with_claims_is_rejected(self):
        response = self.make_response()
        response["status"] = "insufficient_evidence"

        with self.assertRaises(ValueError):
            self.validate(response)

    def test_empty_refusal_passes_structure_check(self):
        result = self.validate({
            "status": "insufficient_evidence",
            "claims": [],
        })

        self.assertEqual(result["status"], "insufficient_evidence")

    def test_whitespace_differences_are_allowed(self):
        result = self.validate(
            self.make_response(
                quote=(
                    "Patients can book appointments\n"
                    "through   the portal."
                )
            )
        )

        self.assertEqual(result["status"], "answered")

    def test_real_quote_does_not_prove_claim_support(self):
        # The quote is real, but it says nothing about parking.
        # Our mechanical validator intentionally cannot detect this.
        response = self.make_response(
            text="Parking is free at the clinic."
        )

        result = self.validate(response)

        self.assertEqual(
            result["claims"][0]["text"],
            "Parking is free at the clinic.",
        )

    def test_spacing_before_commas_and_periods_is_allowed_both_ways(self):
        compact = "Appointments, records, and directions are available."
        spaced = "Appointments , records\t, and directions are available ."
        for passage, quote in ((spaced, compact), (compact, spaced)):
            with self.subTest(passage=passage):
                self.passages = [{"text": passage}]
                response = self.make_response(quote=quote)
                result = self.validate(response)
                self.assertEqual(result, response)
                self.assertEqual(self.passages[0]["text"], passage)

    def test_reported_insulin_quote_spacing_regression(self):
        self.passages = [{"text": (
            "When there isn't enough insulin or cells stop responding to insulin , "
            "too much blood sugar stays in your bloodstream."
        )}]
        quote = (
            "When there isn't enough insulin or cells stop responding to insulin, "
            "too much blood sugar stays in your bloodstream."
        )
        self.assertEqual(self.validate(self.make_response(quote=quote))["status"], "answered")

    def test_changed_words_or_punctuation_still_fail(self):
        self.passages = [{"text": "Patients can book, change, or cancel appointments ."}]
        for quote in (
            "Patients cannot book, change, or cancel appointments.",
            "Patients can book; change, or cancel appointments.",
            "Patients can book change or cancel appointments.",
            "Patients can book, change, or cancel appointments!",
        ):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.validate(self.make_response(quote=quote))

    def test_quote_from_wrong_existing_source_is_rejected(self):
        self.passages.append({"text": "The information desk provides directions ."})
        with self.assertRaises(ValueError):
            self.validate(self.make_response(source="S2"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
