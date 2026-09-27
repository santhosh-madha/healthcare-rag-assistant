import unittest
from healthcare_rag.experiments.evaluate_claim_support import judge_input, parse_judgment, summarize


class TestSupportJudge(unittest.TestCase):
    def test_labels_and_rationales_not_sent(self):
        text = judge_input({'claim':'A','quote':'B','expected_label':'SECRET_LABEL','rationale':'SECRET_REASON'}, {'text':'C'})
        self.assertNotIn('SECRET', text)
        self.assertNotIn('expected_label', text)

    def test_invalid_output_is_not_a_judgment(self):
        for raw in ('not JSON', '{"label":"maybe","reason":"x"}', '{"label":"supported","reason":""}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_judgment(raw)

    def test_errors_and_false_acceptances_are_separate(self):
        summary=summarize([
            {'expected_label':'unsupported','judgment':{'label':'supported'}},
            {'expected_label':'supported','judgment':None},
            {'expected_label':'partially_supported','judgment':{'label':'partially_supported'}}])
        self.assertEqual(summary['errors'],1)
        self.assertEqual(summary['agreements_with_draft'],1)
        self.assertEqual(summary['unsafe_acceptances_vs_draft'],1)
