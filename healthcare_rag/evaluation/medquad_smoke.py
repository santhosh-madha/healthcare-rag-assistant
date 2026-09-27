"""Run ten fixed development questions through the MedQuAD answer pipeline."""
import io
import hashlib
import json
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from healthcare_rag import web_assistant as app
from healthcare_rag.collections import load_collection
from healthcare_rag.paths import PROJECT

CASES = [
    ('aml', 'What symptoms of adult acute myeloid leukemia are listed in the documents?', 'answered'),
    ('hairy', 'Is there a standard staging system for hairy cell leukemia?', 'answered'),
    ('thyroid', 'Why might someone not notice the symptoms of hypothyroidism?', 'answered'),
    ('thyroid_typo', 'What are common symptoms of hypothyrodism?', 'answered'),
    ('alpers', "What are the goals of the research on Alpers' disease described in the documents?", 'answered'),
    ('alcohol', 'Can alcohol use disorder range from mild to severe?', 'answered'),
    ('parasite', 'What is neurocysticercosis?', 'answered'),
    ('parasite_cause', 'Which parasite causes cysticercosis?', 'answered'),
    ('price', 'What is the exact dollar price of a medical appointment at Cedar Learning Clinic?', 'insufficient_evidence'),
    ('phone', 'What is the phone number of Cedar Learning Clinic?', 'insufficient_evidence'),
]


def main():
    model = app.SentenceTransformer(app.MODEL_NAME, cache_folder=str(app.CACHE), device='cpu')
    index, passages = load_collection(model, 'medquad')
    rows = []
    for case_id, question, expected in CASES:
        print('Running:', case_id, flush=True)
        row = {'id': case_id, 'question': question, 'expected_status': expected}
        started = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            sources = app.search(model, index, passages, question, limit=3)
        row['retrieval_seconds'] = time.perf_counter() - started
        row['sources'] = sources
        started = time.perf_counter()
        try:
            raw = app.generate_answer(question, app.build_healthcare_context(sources),
                                      system_prompt=app.STRUCTURED_PROMPT,
                                      response_format=app.response_schema(len(sources)))
            row['raw_response'] = raw
            row['response'] = app.validate_response(raw, sources)
            row['validation_pass'] = True
        except (ValueError, RuntimeError, OSError) as error:
            row['validation_pass'] = False
            row['error'] = str(error)
        row['generation_and_validation_seconds'] = time.perf_counter() - started
        row['total_seconds'] = row['retrieval_seconds'] + row['generation_and_validation_seconds']
        rows.append(row)
    report = {'embedding_model': app.MODEL_NAME, 'generation_model': 'llama3.1:8b',
              'corpus_sha256': hashlib.sha256((PROJECT / 'data/processed/medquad/chunks.json').read_bytes()).hexdigest(),
              'scope': 'Ten fixed development smoke cases; historical document support, not clinical accuracy. Timing excludes model/index startup. Semantic review is separate from mechanical validation.',
              'cases': rows}
    folder = PROJECT / 'evaluation_runs'
    folder.mkdir(exist_ok=True)
    path = folder / ('medquad_e2e_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Saved:', path)
    for row in rows:
        print(row['id'], row.get('response', {}).get('status', 'error'), round(row['total_seconds'], 2), row.get('error', ''))


if __name__ == '__main__':
    main()
