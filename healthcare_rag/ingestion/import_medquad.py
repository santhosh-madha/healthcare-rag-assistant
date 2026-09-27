"""Import a reproducible MedQuAD answer subset; keep questions separate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

from healthcare_rag.paths import PROJECT

EXCLUDED = {'10_MPlus_ADAM_QA', '11_MPlusDrugs_QA', '12_MPlusHerbsSupplements_QA'}
REPOSITORY = 'https://github.com/abachaa/MedQuAD'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def element_text(element):
    if element is None:
        return ''
    # XML entities are decoded by ElementTree; normalize layout only.
    text = ''.join(element.itertext()).strip()
    paragraphs = re.split(r'\n\s*\n', text)
    return '\n\n'.join(' '.join(p.split()) for p in paragraphs if p.strip())


def read_records(folder):
    records, questions, counts, skipped = [], {}, Counter(), []
    seen_answers, seen_ids = set(), set()
    files = sorted(folder.glob('*/*.xml'))
    for path in files:
        relative = path.relative_to(folder).as_posix()
        counts['xml_files_seen'] += 1
        if path.parent.name in EXCLUDED:
            counts['excluded_subset_files'] += 1
            continue
        raw = path.read_bytes()
        if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
            raise ValueError(f'Unsupported XML declaration: {relative}')
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            counts['malformed_xml_files'] += 1
            skipped.append({'file': relative, 'reason': 'malformed_xml'})
            continue
        url = root.get('url', '').strip()
        parsed = urlsplit(url)
        title = element_text(root.find('Focus'))
        for pair in root.findall('./QAPairs/QAPair'):
            counts['qa_pairs_seen'] += 1
            qnode = pair.find('Question')
            question = element_text(qnode)
            answer = element_text(pair.find('Answer'))
            qid = qnode.get('qid', '') if qnode is not None else ''
            reason = None
            if not answer: reason = 'missing_answer'
            elif not question or not qid: reason = 'missing_question_or_id'
            elif not title: reason = 'missing_title'
            elif parsed.scheme not in ('http','https') or not parsed.netloc: reason = 'missing_or_invalid_source_url'
            answer_hash = digest(' '.join(answer.lower().split()).encode())
            record_id = 'medquad:' + path.parent.name + ':' + path.stem + ':' + qid
            if not reason and record_id in seen_ids: reason = 'duplicate_record_id'
            if not reason and answer_hash in seen_answers: reason = 'duplicate_answer'
            if reason:
                counts[reason] += 1
                skipped.append({'file': relative, 'question_id': qid, 'reason': reason})
                continue
            seen_ids.add(record_id)
            seen_answers.add(answer_hash)
            records.append({'record_id': record_id, 'document_id': path.parent.name+':'+path.stem,
                'title': title, 'source_url': url, 'source_reviewed_date': None,
                'source_collection': path.parent.name, 'source_file': relative,
                'source_sha256': digest(raw), 'answer_sha256': digest(answer.encode()),
                'question_id': qid, 'question_type': qnode.get('qtype', 'unknown'), 'text': answer})
            questions[record_id] = question
    counts['usable_unique_records'] = len(records)
    return records, questions, dict(counts), skipped


def select_records(records, limit, seed):
    if limit <= 0 or limit > len(records):
        raise ValueError(f'Requested {limit} records; {len(records)} usable records available.')
    return sorted(random.Random(seed).sample(sorted(records,key=lambda r:r['record_id']),limit),key=lambda r:r['record_id'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PROJECT/'data/raw/medquad')
    parser.add_argument('--output', type=Path, default=PROJECT/'data/processed/medquad')
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args=parser.parse_args()
    try:
        if not (args.input/'LICENSE.txt').is_file():
            raise ValueError('Expected the MedQuAD repository with LICENSE.txt; see docs/medquad.md.')
        if args.output.exists():
            raise ValueError('Output already exists. Use a new --output folder to preserve the snapshot.')
        records, questions, counts, skipped=read_records(args.input)
        selected=select_records(records,args.limit,args.seed)
        revision=subprocess.run(['git','-C',str(args.input),'rev-parse','HEAD'],capture_output=True,text=True)
        imported=datetime.now(timezone.utc).isoformat()
        for r in selected: r['imported_at']=imported
        metadata={'dataset':'MedQuAD','repository':REPOSITORY,
            'revision':revision.stdout.strip() if revision.returncode==0 else None,
            'license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
            'attribution':'Asma Ben Abacha and Dina Demner-Fushman. A Question-Entailment Approach to Question Answering. BMC Bioinformatics (2019).',
            'paper_url':'https://doi.org/10.1186/s12859-019-3119-4',
            'imported_at':imported,'selection_seed':args.seed,'selection':'Random sample from sorted, exact-answer-deduplicated usable records',
            'changes':'XML text extraction, whitespace normalization, answer deduplication and sampling; questions stored separately.',
            'excluded_subsets':sorted(EXCLUDED),'scan_counts':counts,'selected_records':len(selected),
            'selected_unique_documents':len({r['document_id'] for r in selected}),
            'selected_unique_source_urls':len({r['source_url'] for r in selected}),
            'selected_by_collection':dict(Counter(r['source_collection'] for r in selected)),
            'selected_by_question_type':dict(Counter(r['question_type'] for r in selected))}
        args.output.mkdir(parents=True)
        outputs={'records.json':{'metadata':metadata,'records':selected},
                 'questions.json':{'purpose':'Paired dataset questions for source-record retrieval checks; not independent clinical QA labels.',
                    'questions':[{'id':r['record_id'],'question':questions[r['record_id']], 'expected_record_id':r['record_id']} for r in selected]},
                 'manifest.json':{**metadata,'skipped':skipped}}
        for name,data in outputs.items():
            (args.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
        (args.output/'LICENSE.txt').write_bytes((args.input/'LICENSE.txt').read_bytes())
        print(json.dumps({k:v for k,v in metadata.items() if k.startswith('selected') or k=='scan_counts'},indent=2))
        print('Saved:',args.output)
    except (OSError,ValueError) as error:
        parser.error(str(error))


if __name__=='__main__': main()
