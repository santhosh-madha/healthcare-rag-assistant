"""Compare frozen dense and hybrid retrieval; no answer generation."""
import argparse
import hashlib
import io
import json
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from evaluate_healthcare import evidence_hit
from hybrid_search import bm25_scores, fuse, CANDIDATES, K1, B, RRF_K
from persistent_search import read_corpus, load_index
from semantic_search import CACHE, MODEL_NAME, SentenceTransformer, search


def metrics(results, test):
    evidence = test['evidence']
    return {
        'hit_at_1': bool(results) and evidence_hit(results[0], evidence),
        'hit_at_3': any(evidence_hit(p, evidence) for p in results[:3]),
        # Each listed annotation must match. This is NOT semantic coverage:
        # alternative annotations on older sets can describe the same fact.
        'all_annotations_at_3': all(any(evidence_hit(p, [e]) for p in results[:3]) for e in evidence),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('questions', type=Path, nargs='+')
    args = parser.parse_args()
    passages, corpus_hash = read_corpus()
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE), device='cpu')
    index = load_index(model, passages, corpus_hash)
    report = {'corpus_sha256': corpus_hash, 'embedding_model': MODEL_NAME,
              'settings': {'bm25_k1': K1, 'bm25_b': B, 'rrf_k': RRF_K, 'candidates': CANDIDATES}, 'sets': []}
    for path in args.questions:
        raw = path.read_bytes()
        dataset = json.loads(raw)
        rows = []
        for test in dataset['questions']:
            with redirect_stdout(io.StringIO()):
                dense = search(model, index, passages, test['question'], limit=CANDIDATES)
            hybrid = fuse(dense, passages, bm25_scores(passages, test['question']))
            rows.append({'test': test, 'dense_results': dense[:3], 'hybrid_results': hybrid,
                         'dense': metrics(dense[:3], test) if test['answerable'] else None,
                         'hybrid': metrics(hybrid, test) if test['answerable'] else None})
        answerable = [row for row in rows if row['test']['answerable']]
        summary = {'answerable': len(answerable)}
        for mode in ('dense', 'hybrid'):
            summary[mode] = {key: sum(row[mode][key] for row in answerable)
                             for key in ('hit_at_1', 'hit_at_3', 'all_annotations_at_3')}
        report['sets'].append({'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                               'summary': summary, 'results': rows})
        print(path.name, json.dumps(summary))
        for row in answerable:
            if row['dense'] != row['hybrid']:
                print(' ', row['test']['id'], row['dense'], '->', row['hybrid'])
    folder = Path(__file__).parent / 'evaluation_runs'
    folder.mkdir(exist_ok=True)
    target = folder / ('retrieval_comparison_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.json')
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print('Saved:', target)
    print('Inspected regression sets; no general accuracy claim. All-annotation matching is stricter than any-evidence hits.')


if __name__ == '__main__':
    main()
