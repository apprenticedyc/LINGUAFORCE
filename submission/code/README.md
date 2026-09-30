# Analysis Code

The scripts in this directory use paths relative to the submission package:

- `../data/`: released JSONL files.
- `../results/`: outputs used by the reported evaluation.
- `../paper/`: figures and paper source.

## Main scripts

- `verify_paper.py`: recompute the main release metrics.
- `make_figs_release.py`: regenerate the paper figures.
- `ablation_linear_readout.py`: run the dimension-readout ablation.
- `run_t2_rq3.py`: reproduce strategy and transfer analyses.
- `eval_transfer.py`: evaluate cross-domain transfer outputs.
- `compute_iaa.py` and `audit_iaa.py`: inspect agreement inputs supplied by
  the user.

The package contains the released outputs needed for deterministic checks.
Provider calls, credentials, private workbooks, and local configuration files
are intentionally excluded.
