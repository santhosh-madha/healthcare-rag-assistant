"""Opt-in generation experiment; does not change the interactive assistants."""
from structured_healthcare import STRUCTURED_PROMPT

ATOMIC_PROMPT = STRUCTURED_PROMPT + """

Evidence and answer selection:
- First check the question's premise. If a supplied passage explicitly contradicts
  it, answer with the supported correction. A false premise alone is not a reason
  to return insufficient_evidence. If neither an answer nor a correction is
  supported, return insufficient_evidence.
- Make each claim atomic: one fact supported in full by one continuous quote
  from one source. For comparisons or two-part questions, use two separate
  claims, each with its own source and quote. Do not attach a type 1-only quote
  to a sentence that also describes type 2.
- The source field must contain exactly one label, such as S1. Never combine labels.
- Return at most two claims. Choose only the facts requested; omit extra facts.
  Do not repeat the same claim with a different source.
- A list that omits something does not prove that thing is absent or impossible.
  Do not add negative claims based only on an omission from a list.
- Preserve the source's uncertainty and scope. Do not turn "thought to be" into
  a certain cause, "usually" into "always", or "may" into a definite outcome.
- Before returning JSON, check each entire claim against its own quote. If any
  part is unsupported, narrow the claim to what that quote actually supports.
  Keep exact source wording in quotes, including punctuation.
"""
