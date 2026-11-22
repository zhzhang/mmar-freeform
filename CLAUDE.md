# Project context

Research code for a NeurIPS 2026 paper on whether audio-language models can judge the
correctness of free-form answers to audio-reasoning questions **without a reference
answer**. Deadline is imminent. Prefer small, verifiable changes over refactors.

## Layout

- `run_experiment.py` and `mmar_models.py` generate freeform answers.
- `run_judges.py`, `grader.py`, and `mmar_api.py` judge them.
- `alt_test.py` is the Calderon et al. alt-test. `krippendorff.py` is Krippendorff's
  alpha for the human pass/fail labels.
- `analysis.ipynb` produces the paper figures.

## Conventions

- Package manager is `uv`, not pip. Use `uv add`, `uv run`.
- Never commit `.env`, API keys, `mmar/`, `outputs/`, or any `predictions.jsonl`.
- Never change `primary_judge` in `manifest.json`; it drives the difficulty ranking.
