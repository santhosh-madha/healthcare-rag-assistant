"""Collection routing must not mix evidence between independent corpora."""
import io
import json
import unittest
from email.message import Message
from types import SimpleNamespace
from unittest.mock import patch
from healthcare_rag import web_assistant as web
from healthcare_rag import collections as catalog


class TestCollections(unittest.TestCase):
    def request(self, data):
        cdc = (object(), [{'text': 'CDC'}])
        medquad = (object(), [{'text': 'MedQuAD'}])
        handler = object.__new__(web.make_handler(None, *cdc, {'cdc': cdc, 'medquad': medquad}))
        body = json.dumps(data).encode()
        handler.headers = Message()
        handler.headers['Origin'] = 'http://127.0.0.1:8000'
        handler.headers['Content-Type'] = 'application/json'
        handler.headers['Content-Length'] = str(len(body))
        handler.server = SimpleNamespace(server_port=8000)
        handler.path = '/ask'
        handler.rfile = io.BytesIO(body)
        responses = []
        handler.send = lambda status, body: responses.append((status, json.loads(body)))
        with patch.object(web, 'ask', return_value={'response': {}, 'sources': []}) as ask:
            handler.do_POST()
        return cdc, medquad, ask, responses

    def test_medquad_routes_to_its_own_index_and_passages(self):
        _, medquad, ask, responses = self.request({'question': 'Example?', 'collection': 'medquad'})
        ask.assert_called_once_with(None, *medquad, 'Example?')
        self.assertEqual(responses[0][1]['collection'], 'medquad')

    def test_default_remains_cdc(self):
        cdc, _, ask, responses = self.request({'question': 'Example?'})
        ask.assert_called_once_with(None, *cdc, 'Example?')
        self.assertEqual(responses[0][0], 200)

    def test_unknown_or_invalid_collection_never_generates(self):
        for name in ['missing', '../medquad', [], None]:
            with self.subTest(name=name):
                _, _, ask, responses = self.request({'question': 'Example?', 'collection': name})
                ask.assert_not_called()
                self.assertEqual(responses[0][0], 400)

    def test_loader_uses_matching_corpus_and_index_paths(self):
        with patch.object(catalog, 'read_corpus', return_value=(['passage'], 'hash')) as read, patch.object(catalog, 'load_index', return_value='index') as load:
            self.assertEqual(catalog.load_collection('model', 'medquad'), ('index', ['passage']))
            read.assert_called_once_with(catalog.COLLECTIONS['medquad'][1])
            load.assert_called_once_with('model', ['passage'], 'hash', folder=catalog.COLLECTIONS['medquad'][2])
