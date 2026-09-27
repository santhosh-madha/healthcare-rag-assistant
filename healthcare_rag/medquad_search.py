"""Build/search the separate MedQuAD index and run a paired-question smoke test."""
import argparse
import io
import json
import math
import random
from contextlib import redirect_stdout
from datetime import datetime, timezone
import time

from healthcare_rag.paths import PROJECT
from healthcare_rag.persistent_search import read_corpus, save_index, load_index
from healthcare_rag.semantic_search import CACHE, MODEL_NAME, SentenceTransformer
from healthcare_rag.hybrid_search import search

FOLDER=PROJECT/'data/processed/medquad'
INDEX=PROJECT/'data/index/medquad'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question',nargs='?')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--build',action='store_true')
    modes.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if args.question and (args.build or args.benchmark):parser.error('Use a question, --build, or --benchmark separately.')
    if not args.build and not args.benchmark and not (args.question or '').strip():parser.error('Provide a question, --build, or --benchmark.')
    try:
        passages,corpus_hash=read_corpus(FOLDER/'chunks.json')
        model=SentenceTransformer(MODEL_NAME,cache_folder=str(CACHE),device='cpu')
        if args.build:
            start=time.perf_counter()
            index=save_index(model,passages,corpus_hash,folder=INDEX)
            result={'vectors':index.ntotal,'dimensions':index.d,'embedding_and_save_seconds':time.perf_counter()-start,
                    'corpus_sha256':corpus_hash,'embedding_model':MODEL_NAME}
            (INDEX/'build_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2));return
        index=load_index(model,passages,corpus_hash,folder=INDEX)
        if not args.benchmark:
            results=search(model,index,passages,args.question)
            for i,r in enumerate(results,1):
                print(f"\n[S{i}] {r['title']} — {r['section_heading']}\nRecord: {r['record_id']}\nSource: {r['source_url']}\n{r['text']}")
            return
        questions=json.loads((FOLDER/'questions.json').read_text())['questions']
        selected=random.Random(42).sample(questions,min(30,len(questions)))
        rows=[]
        for q in selected:
            start=time.perf_counter()
            with redirect_stdout(io.StringIO()):results=search(model,index,passages,q['question'])
            elapsed=time.perf_counter()-start
            rows.append({'question':q,'retrieval_seconds':elapsed,
                'hit_at_1':results[0]['record_id']==q['expected_record_id'],
                'hit_at_3':any(r['record_id']==q['expected_record_id'] for r in results),
                'retrieved_chunk_ids':[r['chunk_id'] for r in results]})
        times=sorted(r['retrieval_seconds'] for r in rows)
        summary={'questions':len(rows),'hit_at_1':sum(r['hit_at_1'] for r in rows),'hit_at_3':sum(r['hit_at_3'] for r in rows),
            'mean_retrieval_seconds':sum(times)/len(times),'p95_retrieval_seconds':times[math.ceil(.95*len(times))-1]}
        report={'summary':summary,'corpus_sha256':corpus_hash,'model':MODEL_NAME,'sample_seed':42,
            'scope':'Paired dataset questions, exact source-record hits. Includes query encoding and BM25 computation, excludes model/index loading. No generation. Not independent QA accuracy; alternatives may also answer a question.',
            'results':rows}
        output=PROJECT/'evaluation_runs'/('medquad_retrieval_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(summary,indent=2));print('Saved:',output);print(report['scope'])
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        parser.error(str(error)+ '\nPrepare: python -m healthcare_rag.ingestion.import_medquad; python -m healthcare_rag.ingestion.chunk_medquad; python -m healthcare_rag.medquad_search --build')


if __name__=='__main__':main()
