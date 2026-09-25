"""Integration checks for saved-index use without invoking real models."""

import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import MagicMock, patch

import structured_healthcare as app


class TestStructuredPersistence(unittest.TestCase):
    def test_loads_index_without_rebuilding_documents(self):
        passage = {"title": "Example", "section_heading": "Booking",
                   "source_reviewed_date": None, "text": "Use the portal.",
                   "source_url": "https://example.org"}
        model = MagicMock()
        index = MagicMock(ntotal=1)
        response = json.dumps({"status": "answered", "claims": [{
            "text": "Use the portal.", "source": "S1", "quote": "Use the portal."}]})
        with patch("sys.argv", ["structured_healthcare.py", "How to book?"]), \
                patch.object(app, "read_corpus", return_value=([passage], "hash")), \
                patch.object(app, "SentenceTransformer", return_value=model), \
                patch.object(app, "load_index", return_value=index) as load, \
                patch.object(app, "search", return_value=[passage]), \
                patch.object(app, "generate_answer", return_value=response) as generate, \
                patch("semantic_search.build_index", side_effect=AssertionError("Must not rebuild")), \
                redirect_stdout(io.StringIO()) as output:
            app.main()
        load.assert_called_once_with(model, [passage], "hash")
        generate.assert_called_once()
        model.encode.assert_not_called()
        self.assertIn("Document embeddings were not rebuilt", output.getvalue())
        self.assertIn("Source IDs and quote matches passed", output.getvalue())

    def test_stale_index_stops_before_search_and_generation(self):
        with patch("sys.argv", ["structured_healthcare.py", "Question"]), \
                patch.object(app, "read_corpus", return_value=([{}], "new-hash")), \
                patch.object(app, "SentenceTransformer"), \
                patch.object(app, "load_index", side_effect=ValueError("The chunk file changed. Rebuild.")), \
                patch.object(app, "search") as search, \
                patch.object(app, "generate_answer") as generate, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as caught:
            app.main()
        self.assertEqual(caught.exception.code, 2)
        search.assert_not_called()
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
