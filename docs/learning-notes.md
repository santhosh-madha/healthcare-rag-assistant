# Healthcare Knowledge Assistant

A learning project that will grow into a document research assistant with retrieval, generated answers, verifiable citations, and measured evaluation.

## Lesson 1: keyword retrieval

This version searches six paragraphs from three **fictional, synthetic clinic administration documents**. They are teaching fixtures, not medical sources or real clinic policies. No patient data, model, API key, package installation, or network connection is required.

It returns original passages and their source locations. It does not generate answers yet. Shared-word scores are rankings, not confidence or correctness estimates. Keyword overlap cannot determine whether a passage actually answers a question.

## Run

From this project folder, using Python 3.10 or newer:

```sh
python3 retrieve.py "How do I cancel an appointment?"
```

## How it works

1. `load_passages()` reads the text files and splits them at blank lines. Each paragraph becomes a **chunk** with its filename and paragraph number.
2. `tokenize()` lowercases text, extracts words, and removes some common words.
3. `retrieve()` counts the unique words shared by the question and each chunk, then returns up to three matching chunks.
4. `main()` prints those chunks with source locations and matching words.

The pipeline today is: **question → keyword search → source passages**.

The future RAG pipeline is: **question → retrieval → passages plus question sent to a language model → answer with supporting citations**.

## First experiment

Run these questions and inspect both the ranking and the source text:

```sh
python3 retrieve.py "How do I cancel an appointment?"
python3 retrieve.py "How do I call off a booking?"
python3 retrieve.py "Where is the parking garage?"
```

The first two questions have similar meanings, but our search cannot recognize synonyms. The parking question asks about something absent from the collection. Notice that missing evidence and different wording can both produce no results.

Then add one fictional paragraph about parking to `data/sample/visiting.txt`, separated by a blank line, and rerun the parking question. No other code changes should be necessary.

Before moving on, explain in your own words: What is a chunk? Why keep the source? Why did the second question fail? Why isn't a shared-word score proof of a correct answer?

## Planned stages

- [x] Transparent keyword retrieval baseline
- [x] Embeddings and FAISS: search by meaning and compare with this baseline
- [ ] Curated public healthcare sources with source URLs and dates
- [ ] Generate answers using retrieved context
- [ ] Validate citations and handle insufficient evidence
- [ ] Build a held-out evaluation set and report results and failures
- [ ] Add an interface, deployment, architecture diagram, and demo

We will choose the model and dependencies at the relevant stage. Dataset sizes and quality metrics will be reported only after measurement.

## Lesson 2: semantic retrieval

Activate your environment and install the dependencies if needed:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python semantic_search.py "How to call off my booking?"
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` into `.cache/models`. Model inference then runs on your CPU; it does not require an API key. The model cache is excluded from Git. `requirements-lock.txt` records the full package versions installed during development on this Mac with Python 3.14; other platforms may need different versions.

### Follow the data

1. We reuse `load_passages()` from lesson 1, including source information.
2. Sentence Transformers converts each complete passage into an **embedding**, a list of numbers. It encodes the question using the same model. We do not remove stop words for this model.
3. We normalize each vector to length one. This makes the inner product used by FAISS equivalent to **cosine similarity**, which compares vector direction.
4. FAISS stores the passage vectors in an `IndexFlatIP` index and finds the three most similar to the question vector. This simple index searches every vector exactly.
5. The returned positions identify the original passages, so we can display their text and source.

These numbers are learned representations, not human-written categories. Higher similarity means closer vectors; a score of 0.6 does **not** mean 60% correct. The general-purpose model is a teaching baseline, not a medically validated model.

### Compare the approaches

```sh
python retrieve.py "How to call off my booking?"
python semantic_search.py "How to call off my booking?"
python semantic_search.py "How do I cancel an appointment?"
python semantic_search.py "Where is the parking garage?"
```

Inspect whether the cancellation passage ranks first for both appointment questions. Then inspect the parking results: none of our documents describes parking, but nearest-neighbor search still returns passages. Retrieving the closest passage does not establish that an answer exists. We have not added an evidence check or a calibrated rejection threshold yet.

The FAISS index is rebuilt in memory each run so every step remains visible. Persistent indexing, larger documents, evaluation, and generated answers will come later. This is still retrieval, not the full RAG pipeline.

The script uses one CPU thread and disables tokenizer parallelism after a native-library crash occurred with default settings on the development Mac. This small collection does not need parallel processing.

Official reference: [Sentence Transformers quickstart](https://www.sbert.net/docs/quickstart.html).
