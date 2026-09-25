import unittest
from review_answers import DIMENSIONS, summarize


class TestReviewSummary(unittest.TestCase):
    def sheet(self):
        return {"reviewer": "Tester", "items": [
            {"id": "a", "completed": True, **{key: "pass" for key in DIMENSIONS},
             "failure_categories": [], "notes": ""},
            {"id": "b", "completed": False, **{key: None for key in DIMENSIONS}}]}

    def test_unreviewed_not_counted_as_pass(self):
        result = summarize(self.sheet())
        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["unreviewed"], 1)
        self.assertEqual(result["dimensions"]["correctness"]["pass"], 1)

    def test_na_not_counted_as_pass(self):
        sheet = self.sheet()
        sheet["items"][0]["correctness"] = "na"
        self.assertEqual(summarize(sheet)["dimensions"]["correctness"], {"pass": 0, "fail": 0, "na": 1})

    def test_incomplete_review_rejected(self):
        sheet = self.sheet()
        sheet["items"][0]["completeness"] = None
        with self.assertRaises(ValueError):
            summarize(sheet)

    def test_failures_need_explanation(self):
        sheet = self.sheet()
        sheet["items"][0]["evidence_support"] = "fail"
        with self.assertRaises(ValueError):
            summarize(sheet)

    def test_requires_reviewer(self):
        sheet = self.sheet()
        sheet["reviewer"] = ""
        with self.assertRaises(ValueError):
            summarize(sheet)
