# LINGUAFORCE Release Package

[中文说明](README_zh.md) | [Paper PDF](paper/main.pdf) | [Repository overview](../README.md)

This directory contains the release package for LINGUAFORCE: paper files,
data, reported experiment outputs, and analysis code.

## Contents

```text
paper/              Paper PDF, source, bibliography, and figures
data/               JSONL releases, README, and datasheet
code/               Evaluation and analysis scripts
results/            Outputs used by the reported experiments
requirements.txt    Python dependencies for the analysis scripts
CITATION.cff        Machine-readable citation metadata
SHA256SUMS          Checksums for the release files
```

## Reproduce the reported results

The following commands use only the files included in this package. They do
not require API credentials or GPU access.

```bash
python3 -m pip install -r requirements.txt

python3 code/verify_paper.py
python3 code/eval_transfer.py
python3 code/ablation_linear_readout.py
python3 code/run_t2_rq3.py
```

To regenerate the release figures:

```bash
python3 code/make_figs_release.py
```

The result files make the reported metrics independently checkable without
repeating provider calls. Provider credentials, private annotation workbooks,
and local configuration files are excluded.

## Data and labels

The data files contain reference pressure labels and model-derived intensity,
dimension, and strategy fields. See [`data/README.md`](data/README.md) for the
schema and [`data/datasheet.md`](data/datasheet.md) for intended use, privacy
scope, and limitations.

## Paper files

The `paper/` directory contains the paper PDF, LaTeX source, bibliography, and
figures. These files are provided for reading and citation; paper compilation
is not part of the experiment reproduction workflow.

## Release status

Before public redistribution, finalize the applicable license, attribution
notice, author metadata, and clean-checkout verification described in
[`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md).
