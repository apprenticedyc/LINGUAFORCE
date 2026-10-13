# Human-label benchmark predictions

Test outputs contain only the formal 634-dialogue runs. Development smoke
tests and aborted runs are excluded from this supplement.
A run is complete only when all 634
unique test IDs are present; the scorer refuses incomplete files.

All three formal runs now contain 634 unique test dialogues. `metrics.json`
reports all task-valid outputs; `common_subset_metrics.json` aligns the
sample across systems (634 for pressure/intensity/strategies, 295 for
complete dimension profiles). `pressure_bootstrap.json` contains 1,000-draw
paired intervals. `result_tables.tex` is generated from those metrics.
The exact prompt and public settings are saved as `prompt.txt` and
`protocol.json`.

Formal-run token usage: GPT 666,581; Qwen 620,612; DeepSeek 608,684.
These totals exclude smoke tests, the abandoned GPT default-reasoning run,
and any unpersisted requests. They are token counts, not billing statements.

Models tested: GPT-6 Astra through the configured company gateway,
Qwen3.8-Max through DashScope, and DeepSeek-Flash through DeepSeek's API.
Both the requested and returned model names are saved. Alias names are not
immutable model versions.

`code/run_human_llm.py` sends the same English task definitions and dialogue
text to all three models. Human labels are excluded from requests. GPT uses
the Responses interface with a short JSON-format instruction. Qwen and
DeepSeek use chat completions at temperature 0 with thinking disabled.
The final GPT test run uses provider-default sampling and fixed low reasoning; these settings
are not claimed to be identical across providers. The output budget is
1,600 tokens (including GPT reasoning tokens). The GPT resumed run retries
transport failures up to three times with 5/10/15-second delays. Schema-invalid responses are
retained and flagged rather than silently excluded or replaced.

Scoring accepts the unambiguous key alias `D3_Logical_False_Dilemma` as
`D3_Logical_Trap_False_Dilemma` across all providers. This changes only a
field name, not its prediction; raw responses and strict-schema validity
remain unchanged. Out-of-range dimension values are not clipped. Task-level
metrics include coverage, and separate common-subset metrics align the
evaluated dialogues across systems.

Scoring also closes exactly one missing outer JSON brace when the last field
is complete and the JSON error points to the end. Nine GPT outputs needed
this syntax-only normalization; no missing label values are supplied. Both
normalizations were introduced after output-format inspection and apply to
every provider. They do not modify prompts, rerun generations, or use gold
label values. Raw strict-validation flags are retained.

Records include dialogue ID, prompt and text hashes, model identifiers,
UTC timestamp, raw response text, parsed prediction, validity, and usage.
The initial eight-worker GPT invocation hit the gateway concurrency limit
and stopped; the run resumed with three workers, retaining saved outputs.
Unpersisted requests from that failed invocation may have incurred extra
usage not present in the prediction totals. The prompt was unchanged.

The slow default-reasoning GPT run was stopped. After a development-only latency and
schema check, the formal GPT run restarted from scratch with fixed `low`
reasoning. The partial default-reasoning run is excluded from main metrics;
no test-score-based model or prompt selection was performed.

Prompts and settings must be frozen before the test run. A prompt or model
change requires a separate result file. A successful response does not
establish that a call was covered by a free quota.

From the repository root, run a single development example:

```sh
python3 submission/code/run_human_llm.py --provider gpt
python3 submission/code/run_human_llm.py --provider qwen
python3 submission/code/run_human_llm.py --provider deepseek
```

The default split is the training/development pool and the default limit is
one. Set `--split test --limit 634` only for the fixed final evaluation.
Completed records are reused when rerunning the same protocol.

Credentials are loaded from local configuration outside this repository.
See `code/llm_clients.py` for configuration lookup. Credentials are never
stored in prediction records.

## Negative Controls and Uncertainty

`robustness.json` is a post-hoc analysis of these saved predictions; it
does not select prompts, fit parameters, change thresholds, or call APIs.
Dataset and prediction hashes identify its inputs.

The no-pressure test partition contains 130 dialogues. Human request/advice
labels identify 125 of these, and rational-persuasion labels identify 45;
their union has 126 unique dialogues. All six systems have valid binary
outputs. The file supplies counts, false-positive rates, 95% Wilson
intervals, and false-positive IDs for each individual group, their union,
and all negatives. Strategy tags are proxies for ordinary influence, not
independent action-request annotations. The analysis does not rule out
lexical or scenario shortcuts.

Intensity and strategy intervals use 634 dialogues. Dimension intervals
use both the common 295-dialogue valid-profile subset and the full test
set for the five methods with 634 valid profiles. Full-test exact rating
accuracy uses 634 dialogues for every method, treating invalid ratings as
incorrect. Within each group, 1,000 bootstrap draws with seed 2026 resample
whole dialogues, preserving within-dialogue label dependence and pairing
methods on identical draws. Percentile intervals are pointwise 95%, with
no multiple-comparison adjustment. Differences are model minus word or
character TF-IDF on paired samples; lower MAE and higher F1/QWK/accuracy
are better. Qwen is not compared on 295 dialogues against a baseline on 634.

Mean QWK is undefined when any dimension's QWK is undefined. Its intervals
exclude such draws and report the count; paired differences use only draws
defined for both methods. These conditional intervals, particularly on
the sparse 295-dialogue subset, require caution. No undefined score is
replaced with zero. MAE, strategy-F1, and exact accuracy use all draws.
Intervals estimate test-dialogue sampling uncertainty conditional on
fixed predictions, not training-seed or API-generation variability.

Run from the supplement directory:

```sh
python code/analyze_human_robustness.py
python code/write_robustness_tables.py
```
