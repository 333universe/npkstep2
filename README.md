# NPKmath

## Step 1: a verifier you can trust
- npkmath/engine.py    exact symbolic check + 50-digit numeric cross-check. REFUTED always carries a
                       counterexample. matrix_verify reads its inputs.
- npkmath/signals.py   fixed: stale freshness, summary side effects, proofs decaying, REFUTED shown as
                       "verified". Adds claim_holds + timeless fields.
- npkmath/pipeline.py  emits signals with MEASURED time/precision and the claim text.
- npkmath/benchmark.py 53 labelled claims. Goal: 0 critical errors, 0 false assurance.

## Step 2: does verified context help an AI?
- npkmath/dataset.py   60-question set: benchmark claims + corrupted copies (usually FALSE) + meaning-
                       preserving rewrites (TRUE). Engine-labelled, balanced so guessing one answer ~48%.
- npkmath/llm_eval.py  prompts, copy-paste batches (any chat AI) or API function, scoring, paired
                       comparison, always-guess floor, REAL feedback logging, verifier-overrides-model loop.
- Conditions: baseline (claim alone) vs grounded (claim + verified facts about OTHER related claims;
  the target and its own family are never included, so no answer leakage).

## Run (Colab)
Upload this zip, then open run_step1.ipynb or run_step2.ipynb and run top to bottom.

## Honest limits
- Labels come from the engine (validated on the 53-claim benchmark, no failures found). A bug in the
  engine would be copied into the labels.
- One model, 60 questions, one run: a rough guide, not proof. Repeat with other models and seeds.
- MockModel is a plumbing test only. Its numbers say nothing about any real AI.
- Some "related" context from different families can be topically close to the target.

## Tests
python -m pytest -q    (your original 27 tests are unchanged in tests/test_signals.py)
