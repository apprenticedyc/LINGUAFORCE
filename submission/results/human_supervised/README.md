# Supervised baselines on human labels

The fixed seed is 2026. The original 3,432-dialogue development pool is
split into 3,088 training and 344 validation dialogues, stratified by human
intensity. The 634-dialogue test partition is unchanged. Identifiers are
saved in `data/human/splits.json`.

Word TF-IDF uses 1--2-grams; character TF-IDF uses 3--5-grams. Both use
50,000 maximum features, minimum document frequency 2, and sublinear term
frequency. Vectorizers see only training dialogue text.

Pressure uses logistic regression with C selected from 0.1, 1, and 10 by
validation AUROC. Validation F1 selects a threshold from 0.10 to 0.90 in
steps of 0.01, with ties closest to 0.5. Strategy labels use independent
logistic regressions with a shared C and threshold selected by validation
macro-F1 (C from the same grid; threshold 0.3, 0.4, 0.5, or 0.6).

Intensity and each dimension use ridge regression with alpha selected
independently from 0.1, 1, and 10 by validation MAE. Regression outputs are
clipped to the target range and rounded to integers. The constant baseline
uses training-majority binary/strategy labels and training-median ordinal
labels. No human test dimensions are used as input features.

These are single-seed results. All predictions and selected settings are
saved. Test scores do not select models or thresholds. Run from the root:

```sh
python submission/code/train_human_baselines.py
python submission/code/score_human_llm.py
python submission/code/bootstrap_human_pressure.py
```

Scoring requires the complete three-model test outputs. Report coverage
with valid-output metrics; the common-subset file provides aligned
comparisons when model output validity differs.
