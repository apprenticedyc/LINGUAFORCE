# Experiments

## Core Scripts

- `release_dataset.py`: package the released JSONL files.
- `verify_paper.py`: recompute the main reported metrics.
- `make_figs_release.py`: regenerate the paper figures.
- `extract_dims.py`, `extract_dims_full.py`, `extract_dims_hairfree.py`:
  run the seven-dimension parser.
- `zero_shot_direct.py`, `eval_direct.py`: evaluate the direct baseline.
- `transfer_data_prep.py`, `eval_transfer.py`: prepare and evaluate transfer.
- `run_t2_rq3.py`, `annotate_types.py`, `finetune_t2.py`: T2 and RQ3.
- `ablation_linear_readout.py`, `tgb_aggregation.py`: dimension and aggregation
  ablations.
- `extract_turn_dims.py`, `extract_prompt_ablation.py`,
  `eval_prompt_ablation.py`: implementation ablations.
- `compute_iaa.py`, `compute_d5_iaa.py`, `audit_iaa.py`: agreement analysis.

Generated predictions and intermediate results are kept in `output/` and
should not be edited by hand.

## Recommended Order

1. Inspect the release data and annotation protocol.
2. Run `python experiments/verify_paper.py`.
3. Run evaluation or ablation scripts and write results to `output/`.

## Reproducibility Rules

- Keep raw annotation workbooks unchanged.
- Record any derived workbook under a clearly named subdirectory.
- Do not commit credentials or third-party raw datasets.
- Report the script, input release, and metric definition with every result.
