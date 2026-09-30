# LINGUAFORCE Dataset (Release v1)

LINGUAFORCE contains 4,066 de-identified English multi-turn dialogues. Each
dialogue has a seven-dimension agentive-force profile, a strategy taxonomy,
and intensity labels produced by a frozen instruction-tuned model at
temperature 0.

## Files

- `linguaforce_full.jsonl`: 3,432 dialogues for training.
- `linguaforce_first_release.jsonl`: 634 disjoint dialogues for held-out
  evaluation.
- `linguaforce_full_types.jsonl`: strategy labels for the training split.
- `linguaforce_first_release_types.jsonl`: strategy labels for the held-out
  split.

## Schema

Each line is one JSON object:

```json
{
  "dialogue_id": 4,
  "utterances": ["speaker turn 1", "speaker turn 2", "..."],
  "gold_binary": 1,
  "gold_multi": 4,
  "intensity": 4,
  "dims": {
    "D1": {"score": 0.6, "level": 2},
    "D2": {"score": 0.7, "level": 3},
    "D3": {"score": 0.5, "level": 2},
    "D4": {"score": 0.8, "level": 3},
    "D5": {"score": 0.2, "level": 1},
    "D6": {"score": 0.7, "level": 3},
    "D7": {"score": 0.8, "level": 3}
  }
}
```

`gold_binary` is the binary reference label used for evaluation. `gold_multi`
is the ordinal reference intensity on a 0--5 scale. `intensity` is the
released overall intensity prediction. Each dimension has a continuous
`score` in [0, 1] and a four-level `level`:

```text
0 = None, 1 = Low, 2 = Moderate, 3 = High
```

## Statistics

| Split | Dialogues | Positive reference label | Negative reference label |
|---|---:|---:|---:|
| Training | 3,432 | 1,862 (54.2%) | 1,570 (45.8%) |
| Held-out | 634 | 332 (52.4%) | 302 (47.6%) |

Reference intensity counts in the training split for levels 0--5 are
`986/267/317/337/621/904`.

## Annotation and license

The released dimension and strategy labels were generated with a frozen
instruction-tuned model using the protocol described in the paper. The paper
also reports agreement on a stratified sample independently labeled by two
annotators without access to model outputs.

The dialogue text is de-identified and distributed for research use. Users
must review the applicable license and attribution requirements before
redistribution. No private annotation workbooks, provider credentials, or
local configuration files are included.

## Reproduce

Run from the package root:

```bash
python3 code/verify_paper.py
python3 code/make_figs_release.py
```
