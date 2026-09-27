"""Protect path resolution and fixture locations after moving modules."""
import os
import tempfile
import unittest
from pathlib import Path
from healthcare_rag.paths import PROJECT
from healthcare_rag.persistent_search import CHUNKS_FILE, INDEX_FOLDER, read_corpus
from healthcare_rag.semantic_search import CACHE
from healthcare_rag.web_assistant import PAGE
from healthcare_rag.examples.evaluate_retrieval import QUESTION_FILE


class TestRepositoryLayout(unittest.TestCase):
    def test_assets_resolve_outside_repository_cwd(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                passages, digest = read_corpus()
                self.assertEqual(len(passages), 20)
                self.assertEqual(len(digest), 64)
                self.assertTrue(PAGE.is_file())
                self.assertTrue(QUESTION_FILE.is_file())
                self.assertEqual(CHUNKS_FILE, PROJECT/'data/processed/cdc_chunks.json')
                self.assertEqual(INDEX_FOLDER, PROJECT/'data/index')
                self.assertEqual(CACHE, PROJECT/'.cache/models')
            finally:
                os.chdir(original)
