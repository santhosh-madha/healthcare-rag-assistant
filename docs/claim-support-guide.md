# Learning to check claim support

A real quote can be attached to an unsupported answer. This exercise separates **quote matching** (does the text occur in the source?) from **support** (does that text establish the claim?).

The twelve examples in `data/evaluation/claim_support_examples.json` use genuine excerpts from the saved CDC snapshot. Four draft labels are supported, four partially supported, and four unsupported. Labels and rationales were authored with AI assistance; they are not human-verified ground truth. Some claims are deliberately flawed constructed examples, not healthcare advice. Others adapt failures observed in earlier runs. These are learning/development examples, not an independent test set.

## Label rules

- **Supported:** every factual proposition is established by the quote, with qualifications preserved.
- **Partially supported:** a multi-part claim has a supported proposition plus an unsupported proposition.
- **Unsupported:** the claim as written is not established. This includes a single statement changing “usually” to “always” or “thought to be” to “definitely.” Topic overlap alone is insufficient.

Read the quoted text first. Use the full source passage only to resolve a pronoun or heading, not to fill in missing substantive evidence. Other documents and personal medical knowledge are outside this exercise. An unsupported label means this quote is inadequate, not necessarily that the claim is false.

## Try it

```bash
python -m healthcare_rag.experiments.inspect_claim_support
python -m healthcare_rag.experiments.inspect_claim_support --id birth_partial
python -m healthcare_rag.experiments.inspect_claim_support --id always_unsupported
```

For each item, identify the propositions in the claim, locate support in the quote, then check scope, negation, and uncertainty. Read the draft label and rationale and decide whether you agree. In `birth_partial`, a quote about later risk does not also establish what happens after birth. In `always_unsupported`, “usually” cannot establish “always.”

The inspector checks quote membership and dataset/source consistency only. It does not predict support labels. No judge model has been added to the assistant. A future judge should receive the claim, quote, and permitted context without the expected label or rationale; compare its outputs with reviewed labels, report errors by class, and reserve fresh examples for testing after development.
