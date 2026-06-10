"""Evaluation harness: golden set, metrics, local judge, scorecard.

Measurement comes before optimization. This package defines:

- ``golden``  — the hand-authored golden question set (schema + loader)
- ``metrics`` — retrieval metrics (hit@k, MRR, context precision/recall) as
  pure functions
- ``judge``   — an Ollama-backed LLM judge for answer faithfulness and
  correctness; the judge model is deliberately different from the generation
  model to avoid self-evaluation bias
- ``run``     — drive the golden set through the pipeline and emit a scorecard

The retrieval metrics and the golden schema carry no model dependencies, so
they run in CI; the judge and the full run require a local Ollama.
"""
