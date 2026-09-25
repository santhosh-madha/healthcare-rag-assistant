import json
import unittest
from unittest.mock import MagicMock, patch
from rag_assistant import generate_answer
from structured_schema import response_schema
from structured_healthcare import validate_response


class TestSchemaIntegration(unittest.TestCase):
    def test_schema_reaches_ollama_without_changing_prompt(self):
        schema = response_schema(2)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            'done': True, 'message': {'content': '{"status":"insufficient_evidence","claims":[]}'}}).encode()
        with patch('rag_assistant.urlopen', return_value=response) as request:
            generate_answer('Question?', 'Evidence', system_prompt='Original prompt', response_format=schema)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload['format'], schema)
        self.assertEqual(payload['messages'][0]['content'], 'Original prompt')
        self.assertEqual(payload['options']['num_predict'], 300)

    def test_labels_and_status_claim_limits(self):
        answered, refusal = response_schema(2)['anyOf']
        self.assertEqual(answered['properties']['claims']['items']['properties']['source']['enum'], ['S1', 'S2'])
        self.assertEqual(answered['properties']['claims']['minItems'], 1)
        self.assertEqual(answered['properties']['claims']['maxItems'], 2)
        self.assertEqual(refusal['properties']['claims']['maxItems'], 0)
        with self.assertRaises(ValueError):
            response_schema(0)

    def test_shape_conformant_fabricated_quote_still_rejected(self):
        raw = json.dumps({'status': 'answered', 'claims': [
            {'text': 'Example claim.', 'source': 'S1', 'quote': 'Invented evidence.'}]})
        with self.assertRaises(ValueError):
            validate_response(raw, [{'text': 'Actual evidence.'}])
