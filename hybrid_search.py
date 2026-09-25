"""Experimental BM25 + dense retrieval using reciprocal rank fusion."""
import math
import re
from collections import Counter

from retrieve import STOP_WORDS
from semantic_search import search as dense_search

# Fixed before the first comparison; no per-question routing or tuning.
K1 = 1.5
B = 0.75
RRF_K = 60
CANDIDATES = 10


def terms(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]


def bm25_scores(passages, question):
    """Score exact terms, with term-frequency saturation and length normalization."""
    documents = [Counter(terms(p['text'])) for p in passages]
    if not documents:
        return []
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / len(lengths) or 1
    frequency = Counter(term for doc in documents for term in doc)
    query = set(terms(question))
    scores = []
    for doc, length in zip(documents, lengths):
        total = 0.0
        for term in query:
            tf = doc[term]
            if not tf:
                continue
            idf = math.log(1 + (len(documents) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            total += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / average))
        scores.append(total)
    return scores


def fuse(dense, passages, lexical_scores, limit=3):
    """Fuse ranks, keeping cosine similarity separate from the fusion score."""
    by_id = {p['chunk_id']: p for p in passages}
    if len(by_id) != len(passages):
        raise ValueError('Hybrid search requires unique chunk IDs.')
    lexical = sorted(range(len(passages)), key=lambda i: (-lexical_scores[i], i))
    lexical = [i for i in lexical if lexical_scores[i] > 0][:CANDIDATES]
    combined = {}
    for rank, passage in enumerate(dense[:CANDIDATES], 1):
        combined[passage['chunk_id']] = {**passage, 'dense_rank': rank, 'bm25_rank': None,
                                        'fusion_score': 1 / (RRF_K + rank)}
    for rank, position in enumerate(lexical, 1):
        passage = passages[position]
        item = combined.setdefault(passage['chunk_id'], {**by_id[passage['chunk_id']], 'score': None,
                                    'dense_rank': None, 'fusion_score': 0.0})
        item['bm25_rank'] = rank
        item['fusion_score'] += 1 / (RRF_K + rank)
    return sorted(combined.values(), key=lambda p: (-p['fusion_score'], p['chunk_id']))[:max(0, limit)]


def search(model, index, passages, question, limit=3):
    if not question.strip() or not passages or limit <= 0:
        return []
    dense = dense_search(model, index, passages, question, limit=CANDIDATES)
    return fuse(dense, passages, bm25_scores(passages, question), limit)
