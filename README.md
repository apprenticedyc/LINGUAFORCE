# LINGUAFORCE

**A Human-Annotated Benchmark for Pressure and Influence Strategies in Dialogue**

[中文](README_zh.md) | [Paper](paper_acl/main.pdf) | [Data](submission/data/human/) | [Reproduction](submission/README.md)

LINGUAFORCE contains 4,066 English dialogues with human judgments of
pressure presence, intensity, influence strategies, and seven dimensions.
The dialogues were manually collected, organized, and written in English
around common everyday interaction scenarios.

## Dataset

| Property | Value |
|---|---|
| Training / validation / test | 3,088 / 344 / 634 |
| Pressure / no pressure | 3,144 / 922 |
| Pressure presence | Binary |
| Overall intensity | Ordinal, 0–5 |
| Dimensions | Seven ordinal ratings, 0–3 each |
| Strategies | Fifteen non-exclusive binary labels |

Twenty computer science graduate students proficient in reading English
read the shared guidelines and independently annotated their assigned
batches. Participation was voluntary and unpaid. Each dialogue was labeled
by one annotator; inter-annotator agreement was not measured.

## Baselines

| Method | Setting | Pressure AUROC | Pressure F1 |
|---|---|---:|---:|
| GPT-6 Astra | Zero-shot | 0.947 | 0.959 |
| Qwen3.8-Max | Zero-shot | 0.958 | 0.953 |
| DeepSeek-Flash | Zero-shot | 0.954 | 0.950 |
| Word TF-IDF | Supervised | 0.949 | 0.942 |
| Character TF-IDF | Supervised | 0.948 | 0.943 |

All methods use the same 634 test dialogues. Full results cover four tasks.
Qwen has valid complete dimension profiles for 295 dialogues; the supplement
reports coverage, aligned-subset metrics, and exact rating accuracy counting
invalid outputs as incorrect. These cross-model comparisons are accompanied
by a separate matched Qwen3-4B LoRA diagnostic. That single run yields little
change across tasks and a small increase in mean dimension QWK (0.305 to
0.312); saved responses, settings, and paired intervals are included.

## Repository

- `paper_acl/`: editable ACL review manuscript and compiled PDF.
- `submission/`: self-contained anonymous data, code, results, and paper.
- `annotation_package/`: local annotation workbooks and import materials;
  not included in the anonymous supplement.

See the [datasheet](submission/data/datasheet.md) for construction and scope,
and the [schema](submission/data/human/README.md) for field definitions.

## Reproduction

Python 3.9 or later. No GPU or API access is needed to score saved predictions.

```sh
python -m pip install -r submission/requirements.txt
python submission/code/validate_release.py
python submission/code/score_human_llm.py
python submission/code/bootstrap_human_pressure.py
python submission/code/analyze_human_robustness.py
python submission/code/write_robustness_tables.py
python submission/code/score_qwen4b.py
```

CPU retraining and optional API generation are documented in the
[submission README](submission/README.md). Dialogues may contain threats,
insults, and deception. Labels do not establish actual compliance or harm.
See [permissions](submission/TERMS.md).

The supplement also reports false positives on 126 no-pressure test
dialogues containing requests/advice or rational persuasion, and paired
bootstrap intervals for intensity, strategy, and dimension scores.
These diagnostics use fixed predictions; they do not measure variation
across training seeds or API runs.
