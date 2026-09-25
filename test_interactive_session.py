"""Verify resource reuse, independent requests, and session error recovery."""

import io
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import MagicMock, patch

import structured_healthcare as app


class TestInteractiveSession(unittest.TestCase):
    def run_session(self, inputs, effects=None):
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", ["structured_healthcare.py"]))
            stack.enter_context(patch("builtins.input", side_effect=inputs))
            stack.enter_context(patch.object(app, "read_corpus", return_value=([{}], "hash")))
            model = stack.enter_context(patch.object(app, "SentenceTransformer"))
            load = stack.enter_context(patch.object(app, "load_index", return_value=MagicMock(ntotal=1)))
            answer = stack.enter_context(patch.object(app, "answer_question", side_effect=effects))
            output = stack.enter_context(redirect_stdout(io.StringIO()))
            app.main()
            return model, load, answer, output.getvalue()

    def test_load_once_skip_blank_and_handle_two_questions(self):
        model, load, answer, _ = self.run_session(["  ", "First?", "Second?", "/exit"])
        model.assert_called_once()
        load.assert_called_once()
        self.assertEqual([c.args[3] for c in answer.call_args_list], ["First?", "Second?"])

    def test_generation_error_does_not_end_session(self):
        _, _, answer, output = self.run_session(
            ["First?", "Second?", "/exit"], [RuntimeError("Ollama unavailable"), None])
        self.assertEqual(answer.call_count, 2)
        self.assertIn("Ollama unavailable", output)

    def test_eof_and_interrupt_exit_cleanly(self):
        for event in [EOFError(), KeyboardInterrupt()]:
            with self.subTest(event=type(event).__name__):
                _, _, answer, output = self.run_session([event])
                answer.assert_not_called()
                self.assertIn("Goodbye", output)

    def test_verbose_controls_debug_without_changing_request(self):
        passage = {"title": "Example", "section_heading": "Example",
                   "source_reviewed_date": None, "text": "Example."}
        def noisy_search(*args, **kwargs):
            print("DEBUG VECTORS")
            return [passage]
        calls = []
        for verbose in [False, True]:
            with patch.object(app, "search", side_effect=noisy_search), \
                    patch.object(app, "generate_answer", return_value='{"status":"insufficient_evidence","claims":[]}') as generate, \
                    redirect_stdout(io.StringIO()) as output:
                app.answer_question(None, None, [passage], "Question?", verbose)
            self.assertEqual("DEBUG VECTORS" in output.getvalue(), verbose)
            self.assertEqual("--- Raw model response ---" in output.getvalue(), verbose)
            calls.append(generate.call_args)
        self.assertEqual(calls[0], calls[1])


if __name__ == "__main__":
    unittest.main()
