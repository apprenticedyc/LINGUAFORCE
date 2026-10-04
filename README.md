# LINGUAFORCE

**Benchmarking Agentive Force in Dialogues via Unified Measurement Dimensions**

[中文说明](README_zh.md) | [Paper PDF](submission/paper/main.pdf) | [Release package](submission/)

LINGUAFORCE is a benchmark for measuring how language pressures, constrains,
or steers a listener's decision in English multi-turn dialogue. It combines a
seven-dimension measurement space, a fifteen-type strategy taxonomy,
dialogue-level reference labels, model-derived profiles, and deterministic
evaluation outputs.

## Benchmark

| Property | Value |
|---|---|
| Language | English |
| Dialogues | 4,066 |
| Training split | 3,432 dialogues |
| Held-out split | 634 dialogues |
| Measurement dimensions | 7 |
| Strategy types | 15 types in 4 families |
| Intensity | Ordinal scale from 0 to 5 |
| Input | De-identified multi-turn dialogue text |

The seven dimensions are directive force, option constraint, normative
pressure, emotional pressure, deceptiveness, toxicity, and explicitness.

## Tasks

- **T1: Pressure detection.** Predict whether a dialogue contains
  listener-directed pressure.
- **T2: Strategy recognition.** Predict the applicable strategy types.
- **T3: Intensity prediction.** Predict dialogue-level intensity on a 0--5
  ordinal scale.
- **Cross-domain transfer.** Evaluate the dimension profile on related public
  dialogue resources.

## Reported results

The release contains the outputs needed to verify these metrics without
calling an external model provider.

| Evaluation split | Binary AUC | Intensity Spearman rho |
|---|---:|---:|
| Training split | 0.852 | 0.651 |
| Held-out split | 0.831 | 0.665 |

Cross-domain AUROC is 0.742 on MentalManip, 0.706 on MultiManip, 0.729 on
TalkDown, and 0.587 on ToxiChat.

## Repository

```text
.
├── README.md
├── README_zh.md
└── submission/
    ├── paper/       # Paper PDF, source, bibliography, and figures
    ├── data/        # JSONL releases and data documentation
    ├── code/        # Evaluation and analysis scripts
    ├── results/     # Outputs used by the reported experiments
    ├── requirements.txt
    ├── CITATION.cff
    └── SHA256SUMS
```

The release package has its own [English README](submission/README.md) and
[Chinese README](submission/README_zh.md).

## Reproduction

The commands below reproduce the deterministic metrics from the included data
and result files. They do not require API credentials or GPU access.

```bash
cd submission
python3 -m pip install -r requirements.txt

# Main metrics and cross-domain results
python3 code/verify_paper.py
python3 code/eval_transfer.py

# CPU-only analyses reported in the paper
python3 code/ablation_linear_readout.py
python3 code/run_t2_rq3.py
```

To regenerate the release figures:

```bash
python3 code/make_figs_release.py
```

The included JSONL/JSON files are the outputs used for the reported results.
Provider calls, credentials, private annotation workbooks, and local
configuration files are not part of the release.

## Data

See [`submission/data/README.md`](submission/data/README.md) for the schema,
split statistics, and field definitions. The reference labels used as
evaluation targets are kept distinct from model-derived intensity, dimension,
and strategy fields. The [datasheet](submission/data/datasheet.md) documents
intended use, privacy scope, and known limitations.

## Citation

```bibtex
@misc{linguaforce2026,
  title  = {LINGUAFORCE: Benchmarking Agentive Force in Dialogues via Unified Measurement Dimensions},
  year   = {2026},
  note   = {Research release}
}
```

Machine-readable citation metadata is available in
[`submission/CITATION.cff`](submission/CITATION.cff).

## Responsible use

The dialogues may contain threats, insults, deception, or other distressing
language. The release is intended for research on dialogue analysis and
detection. Do not use it for surveillance, consequential decisions about
individuals, or automated moderation without human review.

## License

The applicable text and dataset license, together with the required
attribution notice, must be finalized before public redistribution. See
[`submission/PUBLIC_RELEASE_CHECKLIST.md`](submission/PUBLIC_RELEASE_CHECKLIST.md).
