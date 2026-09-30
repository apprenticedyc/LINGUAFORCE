# LINGUAFORCE Submission Package

This directory is the clean paper-and-reproduction package for the
LINGUAFORCE submission.

## Layout

- `paper/`: LaTeX source, PDF, bibliography, and figures.
- `data/`: the released JSONL files, data README, and datasheet.
- `code/`: deterministic evaluation, plotting, ablation, T2/RQ3, and
  agreement-analysis scripts.
- `results/`: model outputs needed to reproduce the reported tables and
  metrics.
- `CITATION.cff`: citation metadata for the release.
- `requirements.txt`: Python dependencies for the analysis scripts.

## Reproduction

Run from this directory:

```bash
python code/verify_paper.py
python code/make_figs_release.py
python code/ablation_linear_readout.py
python code/run_t2_rq3.py
```

Compile the paper from `paper/` with a LaTeX engine that supports the
IEEEtran class:

```bash
cd paper
tectonic -X compile main.tex
```

The parser calls and any API credentials are intentionally not included.
The released result files are included so that the deterministic evaluation
does not require an external provider or a secret key.

## Data and licensing

The data provenance, de-identification, intended use, and licensing
constraints are documented in `data/README.md` and `data/datasheet.md`.
Those documents must be reviewed together with the license of the source
corpus before public redistribution. Raw upstream data and private annotation
workbooks are not included in this package.

## Submission checklist

See `PUBLIC_RELEASE_CHECKLIST.md`. In particular, complete the license and
data-redistribution review, confirm the final human-agreement artifact used
for the reported statistics, and fill in the author contact information
required by the target venue.
