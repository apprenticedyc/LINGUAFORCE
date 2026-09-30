# LINGUAFORCE Release Package

This directory contains the paper, released data, evaluation outputs, and
reproduction code for LINGUAFORCE.

## Layout

- `paper/`: LaTeX source, PDF, bibliography, and figures.
- `data/`: the two JSONL data releases, data documentation, and datasheet.
- `code/`: deterministic evaluation, plotting, ablation, transfer, and
  agreement-analysis scripts.
- `results/`: outputs used to produce the reported metrics.
- `CITATION.cff`: citation metadata.
- `requirements.txt`: Python dependencies for the analysis scripts.

## Reproduction

Run from this directory:

```bash
python3 code/verify_paper.py
python3 code/make_figs_release.py
python3 code/ablation_linear_readout.py
python3 code/run_t2_rq3.py
```

Compile the paper from `paper/` with a LaTeX engine that supports the
IEEEtran class:

```bash
cd paper
tectonic -X compile main.tex
```

The package includes the outputs needed to verify the reported results. Model
provider calls and credentials are not included.

## Data and use

The release contains de-identified English dialogue text and derived
annotations for research use. The data documentation records the schema,
intended uses, privacy scope, and applicable license conditions. Review those
conditions before redistributing the text or derived files.

## Release checks

See `PUBLIC_RELEASE_CHECKLIST.md` for the remaining venue, license, and
reproducibility checks.
