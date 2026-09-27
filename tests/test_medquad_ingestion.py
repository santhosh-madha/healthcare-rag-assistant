import tempfile
import unittest
from pathlib import Path
from healthcare_rag.ingestion.import_medquad import read_records, select_records
from healthcare_rag.ingestion.chunk_medquad import split_answer


class WordTokenizer:
    def __call__(self,text,**kwargs):
        return {'input_ids':list(range(len(text.split())))}


class TestMedquadIngestion(unittest.TestCase):
    def test_skips_missing_duplicates_and_excluded_subsets(self):
        xml='''<Document url="https://example.org/topic"><Focus>Topic</Focus><QAPairs>
        <QAPair><Question qid="1" qtype="information">Secret test question?</Question><Answer>Answer text.</Answer></QAPair>
        <QAPair><Question qid="2">Duplicate?</Question><Answer>Answer text.</Answer></QAPair>
        <QAPair><Question qid="3">Missing?</Question><Answer/></QAPair>
        </QAPairs></Document>'''
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for collection in ('1_CancerGov_QA','10_MPlus_ADAM_QA'):
                (root/collection).mkdir()
                (root/collection/'one.xml').write_text(xml)
            records,questions,counts,_=read_records(root)
        self.assertEqual(len(records),1)
        self.assertNotIn('Secret',records[0]['text'])
        self.assertEqual(questions[records[0]['record_id']],'Secret test question?')
        self.assertEqual(counts['duplicate_answer'],1)
        self.assertEqual(counts['missing_answer'],1)
        self.assertEqual(counts['excluded_subset_files'],1)
        self.assertEqual(records[0]['source_url'],'https://example.org/topic')

    def test_selection_is_reproducible_independent_of_input_order(self):
        records=[{'record_id':str(n)} for n in range(20)]
        self.assertEqual(select_records(records,5,42),select_records(list(reversed(records)),5,42))
        with self.assertRaises(ValueError):select_records(records,21,42)

    def test_fallback_preserves_words_offsets_and_budget(self):
        text=' '.join('word'+str(i) for i in range(401))
        chunks,strategy=split_answer(text,WordTokenizer())
        self.assertEqual(strategy,'word_boundary_fallback')
        self.assertEqual(' '.join(c['text'] for c in chunks),text)
        for c in chunks:
            self.assertEqual(c['text'],text[c['character_start']:c['character_end']])
            self.assertLessEqual(len(c['text'].split()),180)
