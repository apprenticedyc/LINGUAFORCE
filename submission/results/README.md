# Benchmark Results

`human_llm/` contains formal zero-shot runs and scoring summaries.
`human_supervised/` contains word, character, and constant predictions on
the same 634 dialogues. These are the results used by the current paper.

`human_llm/metrics.json` combines all methods. Common-subset metrics,
validity counts, per-label scores, and paired pressure bootstrap intervals
are supplied. See the subdirectory READMEs for settings and interpretation.

`human_llm/robustness.json` adds the post-hoc negative-control analysis and
paired uncertainty estimates for intensity, strategies, and dimensions.
Its generated tables are in `human_llm/robustness_tables.tex`.

`qwen3_4b/` contains the separate matched original/LoRA Qwen3-4B diagnostic.
Both runs received the same 634 test requests with `/no_think`. The directory
includes the exact answer text, prompt, available settings, output coverage,
and paired common-subset scores and intervals. Run `code/score_qwen4b.py`
from the supplement to regenerate its metrics.
