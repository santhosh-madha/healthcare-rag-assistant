"""Check stale/corrupt index handling without model downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from persistent_search import load_index, save_index
import faiss
import numpy as np


class FakeModel:
    max_seq_length = 256

    def get_embedding_dimension(self):
        return 2


class TestPersistentSearch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.model = FakeModel()
        self.passages = [{"text": "first"}, {"text": "second"}]
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0], [0, 1]], dtype="float32"))
        with patch("persistent_search.build_index", return_value=index):
            save_index(self.model, self.passages, "corpus-v1", self.folder)

    def load(self, passages=None, corpus_hash="corpus-v1"):
        return load_index(self.model, self.passages if passages is None else passages,
                          corpus_hash, self.folder)

    def test_round_trip_without_document_encoding(self):
        with patch("persistent_search.build_index", side_effect=AssertionError("Must not rebuild")):
            index = self.load()
        scores, positions = index.search(np.array([[0, 1]], dtype="float32"), 1)
        self.assertEqual(int(positions[0][0]), 1)
        self.assertEqual(float(scores[0][0]), 1)

    def test_changed_corpus_rejected(self):
        with self.assertRaisesRegex(ValueError, "chunk file changed"):
            self.load(corpus_hash="corpus-v2")

    def test_reordered_passages_rejected(self):
        with self.assertRaisesRegex(ValueError, "ordering changed"):
            self.load(passages=list(reversed(self.passages)))

    def test_changed_model_settings_rejected(self):
        self.model.max_seq_length = 128
        with self.assertRaisesRegex(ValueError, "Embedding settings changed"):
            self.load()

    def test_corrupted_file_rejected_before_faiss_reads_it(self):
        with (self.folder / "cdc.faiss").open("ab") as f:
            f.write(b"corruption")
        with patch("persistent_search.faiss.read_index", side_effect=AssertionError("Must not read")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.load()

    def test_incomplete_pair_rejected(self):
        (self.folder / "metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "No complete saved index"):
            self.load()

    def test_unknown_schema_rejected(self):
        path = self.folder / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["schema_version"] = 99
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Unsupported metadata"):
            self.load()


if __name__ == "__main__":
    unittest.main()
