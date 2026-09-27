"""Inspect labeled claim/quote examples without calling a model."""
import argparse
import hashlib
import json
from collections import Counter

from healthcare_rag.structured_healthcare import normalize_whitespace

from healthcare_rag.paths import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', help='Show one example with its draft label and rationale')
    args = parser.parse_args()
    raw = (PROJECT / 'data/processed/cdc_chunks.json').read_bytes()
    dataset = json.loads((PROJECT / 'data/evaluation/claim_support_examples.json').read_text())
    if dataset['corpus_sha256'] != hashlib.sha256(raw).hexdigest():
        parser.error('Corpus changed; recheck examples before using these labels.')
    passages = {p['chunk_id']: p for p in json.loads(raw)['passages']}
    examples = dataset['examples']
    if len({e['id'] for e in examples}) != len(examples):
        parser.error('Duplicate example IDs.')
    for e in examples:
        p = passages.get(e['chunk_id'])
        if (not p or p['source'] != e['source'] or not e['quote'].strip()
                or normalize_whitespace(e['quote']) not in normalize_whitespace(p['text'])):
            parser.error(f"Quote/source mismatch: {e['id']}")
        if e['expected_label'] not in dataset['rubric']:
            parser.error(f"Unknown label: {e['id']}")
    if args.id:
        selected = next((e for e in examples if e['id'] == args.id), None)
        if selected is None:
            parser.error('Unknown example ID.')
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        print('\nFull source passage for context:\n' + passages[selected['chunk_id']]['text'])
    else:
        print(f'{len(examples)} quotes match their cited stored passages.')
        print('Draft support labels:', dict(Counter(e['expected_label'] for e in examples)))
        for e in examples:
            print(f"  {e['id']}: {e['expected_label']}")
    print('\nAI-assisted labels, pending human review; quote matches do not verify claim support.')


if __name__ == '__main__':
    main()
