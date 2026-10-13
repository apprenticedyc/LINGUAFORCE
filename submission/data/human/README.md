# Human Annotations

4,066 dialogues were each labeled once by one of 20 computer science
graduate students proficient in English dialogue. All read shared
instructions and worked independently. Participation was voluntary and
unpaid. No inter-annotator agreement was measured.

## Schema

`annotations.jsonl` contains one object per dialogue:

| Field | Meaning |
|---|---|
| `id` | Globally unique partition-prefixed ID |
| `source_id` | Numeric source ID, not globally unique |
| `source_split` | `training` development pool or `test` |
| `utterances` | Ordered English turns |
| `binary` | Human pressure judgment, 0/1 |
| `intensity` | Human overall intensity, 0–5 |
| `dimensions` | Human D1–D7 ratings, each 0–3 |
| `strategies` | Fifteen full-name strategy keys, each 0/1 |
| `notes` | Optional comment as supplied |
| `annotation_file` | Anonymous batch filename |

Dimensions: D1 directive force; D2 option constraint; D3 normative pressure;
D4 emotional pressure; D5 deceptiveness; D6 toxicity; D7 explicitness.
Dimension keys are separate from strategy keys with D1–D4 prefixes.

`statistics.json` describes the complete release and development/test pools.
`splits.json` provides actual 3,088 training, 344 validation, and 634 test IDs.
Do not train on all `source_split=training` records: these include validation.
The validation split uses seed 2026 and intensity stratification.

Import matched numeric IDs and full text. Required labels are complete and
in range, without binary/intensity contradictions or normalized exact
duplicates. These are structural checks, not evidence of human agreement.
Run `python code/validate_release.py` from the supplement directory.
Source workbooks remain local and are not needed to reproduce scores.
