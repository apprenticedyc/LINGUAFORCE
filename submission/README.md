# LINGUAFORCE: Anonymous Supplement

[中文](README_zh.md)

This supplement accompanies *LINGUAFORCE: A Human-Annotated Benchmark for
Pressure and Influence Strategies in Dialogue*.

ARR provides the data and software as separate archives. Extract both archives
into the same parent directory; they share a `submission/` root. Shared README
and permissions files are identical. The paper is uploaded separately as a PDF.

## Contents

- `data/human/`: 4,066 annotated dialogues, statistics, fixed split IDs.
- `data/datasheet.md`: construction, annotation, scope, permissions.
- `data/annotation_guide.md`: English translation of the instructions.
- `data/annotation_guide_zh.md`: original Chinese instructions.
- `code/`: validation, training, inference, scoring, and tests.
- `results/human_llm/`: three formal 634-dialogue API runs and protocol.
- `results/human_supervised/`: word, character, and constant predictions.
- `results/qwen3_4b/`: matched original/LoRA responses, protocol, and scores.
- `SHA256SUMS_DATA` and `SHA256SUMS_CODE`: per-archive checksums.

The repository additionally contains a `paper/` release copy and combined
`SHA256SUMS`; these are not required inside the two upload archives.

The split is 3,088 training, 344 validation, and 634 test dialogues. All four
targets are human annotations; test labels are not inference-time inputs.
Each dialogue has one annotation; inter-annotator agreement was not measured.

## Reproduce Scores

Run from this directory with Python 3.9 or later. The recorded environment
used Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, and scikit-learn 1.6.1.

```sh
python -m pip install -r requirements.txt
python code/validate_release.py
python -m unittest discover -s code -p 'test_*.py'
python code/score_human_llm.py
python code/bootstrap_human_pressure.py
python code/write_human_result_tables.py
python code/analyze_human_robustness.py
python code/write_robustness_tables.py
python code/score_qwen4b.py
```

These commands use saved predictions: no API access or GPU is needed.
The scorer verifies all 634 test IDs. Derived result summaries are overwritten.

The robustness analysis reports request/advice and rational-persuasion
negative controls, their deduplicated union, and all negatives. It also
computes pointwise 95% paired bootstrap intervals for intensity, strategy,
and dimension metrics. See `results/human_llm/README.md` for subsets and
undefined-QWK handling. Analyses do not refit models or change thresholds.

## Retrain CPU Baselines

```sh
python code/train_human_baselines.py
python code/score_human_llm.py
python code/bootstrap_human_pressure.py
python code/write_human_result_tables.py
```

This overwrites supervised predictions. Selection uses training and validation
data only, seed 2026. See `results/human_supervised/README.md`.
The script reconstructs the published split; the release validator checks
it against the protocol hash.

## Optional API Generation

API generation may incur charges. Provider aliases and implementations can
change; fresh generations need not reproduce saved responses.
See `code/README.md` for external configuration.

```sh
python code/run_human_llm.py --provider qwen --split training --limit 1
python code/run_human_llm.py --provider qwen --split test --limit 634
```

Substitute `gpt` or `deepseek` for other providers. The runner resumes
existing outputs; move the corresponding output aside before a fresh run,
preserving the supplied predictions. Never tune on test labels.

## Interpretation and Use

Report coverage with valid-only scores. Qwen has 295 valid complete dimension
profiles; compare dimension MAE/QWK on the released common subset.
Out-of-range LLM ratings are not clipped. Full-test exact rating accuracy
counts invalid outputs as wrong. These cross-model baseline comparisons do
not measure the effect of fine-tuning one model.

A separate Qwen3-4B diagnostic compares the original and LoRA-tuned model
using identical test requests with `/no_think`. See `results/qwen3_4b/README.md`.
Its saved-response scores and paired intervals can be reproduced offline;
the managed training backend and adapter weights are not included.

See `TERMS.md` for permissions. Raw workbooks, author identities, credentials,
and private service URLs are excluded.
