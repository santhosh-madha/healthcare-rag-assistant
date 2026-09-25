"""Check that the browser service uses validated structured responses."""
import unittest
from unittest.mock import patch
import web_assistant as app


class TestWebAssistant(unittest.TestCase):
    def setUp(self):
        self.sources = [{"title": "Example", "section_heading": "Overview",
                         "source_reviewed_date": None, "text": "The portal provides records ."}]

    def test_valid_response_returns_original_evidence(self):
        raw = '{"status":"answered","claims":[{"text":"Records are available.","source":"S1","quote":"The portal provides records."}]}'
        with patch.object(app, "search", return_value=self.sources), patch.object(app, "generate_answer", return_value=raw) as generate:
            result = app.ask(None, None, self.sources, "Where are records?")
        self.assertEqual(result["sources"], self.sources)
        self.assertEqual(result["response"]["status"], "answered")
        self.assertEqual(generate.call_args.kwargs["system_prompt"], app.STRUCTURED_PROMPT)
        self.assertEqual(generate.call_args.kwargs["response_format"], "json")

    def test_invalid_quote_is_withheld(self):
        raw = '{"status":"answered","claims":[{"text":"Made up.","source":"S1","quote":"Invented evidence."}]}'
        with patch.object(app, "search", return_value=self.sources), patch.object(app, "generate_answer", return_value=raw):
            with self.assertRaises(ValueError):
                app.ask(None, None, self.sources, "Question?")

    def test_refusal_preserves_evidence_for_review(self):
        with patch.object(app, "search", return_value=self.sources), patch.object(app, "generate_answer", return_value='{"status":"insufficient_evidence","claims":[]}'):
            result = app.ask(None, None, self.sources, "Unknown?")
        self.assertEqual(result["response"]["claims"], [])
        self.assertEqual(result["sources"], self.sources)


if __name__ == "__main__":
    unittest.main()
