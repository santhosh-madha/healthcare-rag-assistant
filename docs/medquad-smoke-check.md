# MedQuAD end-to-end smoke check

Run: 2026-09-27. Ten fixed development cases against the 500-record snapshot. Hybrid retrieval, Llama 3.1 8B, structured schema, and existing quote validator—the same pipeline used by the web service. Model/index startup is excluded from timings. This run exercises the pipeline directly, not browser interaction.

Results: eight answers, two appropriate missing-information refusals, ten mechanical validation passes, and no execution errors. These selected cases are not a held-out benchmark or a clinical accuracy measure.

Latency: mean 4.92s, median 4.20s, range 1.73–9.50s. Single local run; no concurrency/load test.

| Case | Status | Seconds | Evidence inspection |
|---|---|---:|---|
| aml | answered | 9.50 | Listed symptoms are supported by the quote; relevant source selected. |
| hairy | answered | 6.60 | Both staging statements are directly supported by their quotes. |
| thyroid | answered | 3.84 | Slow development explanation is directly supported. |
| thyroid_typo | answered | 7.12 | Recognized the misspelling; listed symptoms are supported. Answer omits two items but does not claim an exhaustive list. |
| alpers | answered | 4.57 | Supported with passage context: preceding sentence resolves these disorders to a group including Alpers disease. Quote alone leaves the referent implicit. |
| alcohol | answered | 3.60 | Supported; preceding sentence defines AUD as alcohol use disorder. |
| parasite | answered | 6.81 | Overall answer supported by the full passage. First quote names cysticercosis, not neurocysticercosis; neighboring sentences establish the relationship. First claim needs broader quoted evidence to stand alone. |
| parasite_cause | answered | 3.68 | Parasite attribution directly supported by quote. |
| price | insufficient_evidence | 1.75 | Retrieved passages provide no exact Cedar Learning Clinic price; refusal appropriate for evidence. |
| phone | insufficient_evidence | 1.73 | Retrieved passages provide no Cedar Learning Clinic phone number; refusal appropriate for evidence. |

The qualitative notes are an assistant inspection of the saved passages, not an independently human-verified assessment. All eight answers were supported at the full-passage level in this inspection. The neurocysticercosis case illustrates why an exact quote match is not sufficient: a claim may need context beyond its individual quote. Alpers and AUD quotes also use references resolved by surrounding text. No prompt or validator was changed to optimize these ten cases.

Reproduce:

```bash
python -m healthcare_rag.evaluation.medquad_smoke
```

Requires the prepared MedQuAD index and Ollama running with llama3.1:8b. Each run saves raw responses, retrieved passages, mechanical outcomes, and timings under the ignored evaluation_runs directory. Qualitative inspection is separate and is not automatically reproduced by the script. Outputs can vary between runs.
