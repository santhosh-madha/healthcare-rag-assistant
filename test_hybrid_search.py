import unittest
from hybrid_search import bm25_scores, fuse
from compare_retrieval import metrics


class TestHybridSearch(unittest.TestCase):
    def test_rare_exact_term_outweighs_common_term(self):
        passages = [{'text': 'diabetes prediabetes'}, {'text': 'diabetes diabetes'}, {'text': 'diabetes'}]
        scores = bm25_scores(passages, 'prediabetes')
        self.assertGreater(scores[0], 0)
        self.assertEqual(scores[1:], [0, 0])

    def test_lexical_can_recover_dense_miss_without_duplicate(self):
        passages = [{'chunk_id': str(i), 'text': 'example'} for i in range(3)]
        dense = [{**passages[0], 'score': .9}, {**passages[1], 'score': .8}]
        result = fuse(dense, passages, [0, 2, 3])
        self.assertEqual(result[0]['chunk_id'], '1')
        self.assertEqual(len({p['chunk_id'] for p in result}), 3)
        self.assertIsNone(next(p for p in result if p['chunk_id']=='2')['score'])
        self.assertEqual(result[0]['score'], .8)  # Never replace cosine with RRF.

    def test_zero_lexical_matches_preserve_dense_order(self):
        passages = [{'chunk_id': 'a'}, {'chunk_id': 'b'}]
        dense = [{**passages[1], 'score': .9}, {**passages[0], 'score': .5}]
        self.assertEqual([p['chunk_id'] for p in fuse(dense, passages, [0, 0])], ['b', 'a'])

    def test_complete_annotation_coverage_requires_each_excerpt(self):
        test = {'evidence': [{'source': 'a', 'quote': 'first'}, {'source': 'b', 'quote': 'second'}]}
        score = metrics([{'source': 'a', 'text': 'first'}], test)
        self.assertTrue(score['hit_at_3'])
        self.assertFalse(score['all_annotations_at_3'])
