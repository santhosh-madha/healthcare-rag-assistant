"""Ensure web and terminal share retrieval, prompt, schema and validation."""
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from healthcare_rag import structured_healthcare as cli, web_assistant as web, hybrid_search
from healthcare_rag.structured_schema import response_schema


class TestAppPipeline(unittest.TestCase):
    def test_both_apps_use_hybrid_and_identical_generation_request(self):
        self.assertIs(cli.search, hybrid_search.search)
        self.assertIs(web.search, hybrid_search.search)
        sources=[{'title':'Example','section_heading':'Overview','text':'Example evidence.',
                  'source_reviewed_date':None,'source_url':'https://example.org'}]
        raw=json.dumps({'status':'answered','claims':[{'text':'Example evidence.',
                        'source':'S1','quote':'Example evidence.'}]})
        calls=[]
        for module in (cli, web):
            with patch.object(module,'search',return_value=sources), \
                 patch.object(module,'generate_answer',return_value=raw) as generate, \
                 redirect_stdout(io.StringIO()):
                if module is cli:
                    cli.answer_question(None,None,sources,'Question?')
                else:
                    web.ask(None,None,sources,'Question?')
                calls.append(generate.call_args)
        self.assertEqual(calls[0],calls[1])
        self.assertEqual(calls[0].kwargs['response_format'],response_schema(1))

    def test_cli_still_rejects_fabricated_quotes(self):
        sources=[{'title':'Example','section_heading':'Overview','text':'Real evidence.',
                  'source_reviewed_date':None}]
        raw=json.dumps({'status':'answered','claims':[{'text':'Wrong.', 'source':'S1','quote':'Invented.'}]})
        with patch.object(cli,'search',return_value=sources), patch.object(cli,'generate_answer',return_value=raw), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'Validation failed'):
                cli.answer_question(None,None,sources,'Question?')
