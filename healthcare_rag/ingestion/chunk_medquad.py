"""Chunk imported MedQuAD answers without indexing dataset questions."""
import argparse
import json
from pathlib import Path
import re

from healthcare_rag.ingestion.chunk_documents import split_section, count_tokens, CHUNK_TOKENS
from healthcare_rag.paths import PROJECT
from healthcare_rag.semantic_search import CACHE, MODEL_NAME, SentenceTransformer


def split_answer(text, tokenizer):
    try:
        return split_section(text,tokenizer), 'paragraph_sentence'
    except ValueError:
        # Long lists/sentences: exact contiguous word-boundary slices, zero overlap.
        words=list(re.finditer(r'\S+',text))
        chunks=[]
        first=0
        while first < len(words):
            start=words[first].start()
            low, high=first+1,len(words)
            if count_tokens(text[start:words[first].end()],tokenizer)>CHUNK_TOKENS:
                raise ValueError('Single word exceeds token budget.')
            best=low
            while low<=high:
                middle=(low+high)//2
                if count_tokens(text[start:words[middle-1].end()],tokenizer)<=CHUNK_TOKENS:
                    best=middle;low=middle+1
                else: high=middle-1
            end=words[best-1].end()
            chunks.append({'text':text[start:end],'character_start':start,'character_end':end})
            first=best
        return chunks,'word_boundary_fallback'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=PROJECT/'data/processed/medquad')
    args=parser.parse_args()
    target=args.folder/'chunks.json'
    if target.exists(): parser.error('chunks.json exists; preserve it or choose a new folder.')
    data=json.loads((args.folder/'records.json').read_text())
    model=SentenceTransformer(MODEL_NAME,cache_folder=str(CACHE),device='cpu')
    passages=[]; fallback=0
    for record in data['records']:
        chunks,strategy=split_answer(record['text'],model.tokenizer)
        fallback+=strategy=='word_boundary_fallback'
        for number,chunk in enumerate(chunks,1):
            tokens=len(model.tokenizer(chunk['text'],add_special_tokens=True,truncation=False,verbose=False)['input_ids'])
            if tokens>model.max_seq_length: raise ValueError('Chunk exceeds embedding model limit.')
            passages.append({'chunk_id':record['record_id']+f':chunk-{number:04d}',
                'source':record['record_id'],'record_id':record['record_id'],'document_id':record['document_id'],
                'passage':number,'title':record['title'],'source_url':record['source_url'],
                'source_reviewed_date':None,'source_sha256':record['source_sha256'],
                'source_file':record['source_file'],'source_collection':record['source_collection'],
                'section_heading':record['question_type'],'token_count':tokens,'chunking_strategy':strategy,**chunk})
    target.write_text(json.dumps({'embedding_model':MODEL_NAME,'metadata':data['metadata'],
        'model_input_limit':model.max_seq_length,'chunk_tokens':CHUNK_TOKENS,'overlap_tokens':0,
        'fallback_records':fallback,'passages':passages},indent=2,ensure_ascii=False)+'\n')
    print(f"Saved {len(passages)} chunks from {len(data['records'])} records; fallback splitting used for {fallback} records.")
    print('Maximum tokens including special tokens:',max(p['token_count'] for p in passages))
    print('Saved:',target)


if __name__=='__main__':main()
