# LINGUAFORCE Dataset (Release v1)

A dataset of 4,066 English multi-turn dialogues annotated with a seven-dimension
"agentive force" profile. The source corpus is the moral-coercion benchmark
from the COERCION project; we keep its binary coercion and 0--5 intensity
labels and re-annotate every dialogue under a unified psychological-dimension
scheme with a frozen instruction-tuned LLM at temperature 0.

## Files
- `linguaforce_full.jsonl` -- 3,432 dialogues (training partition of the source corpus)
- `linguaforce_first_release.jsonl` -- 634 disjoint held-out dialogues

## Schema (per line, one JSON object)
```json
{
  "dialogue_id": 4,
  "utterances": ["speaker turn 1", "speaker turn 2", "..."],
  "gold_binary": 1,          // source binary coercion label (0/1)
  "gold_multi": 4,           // source coercion intensity (0-5, 0 = none)
  "intensity": 4,            // model-predicted overall intensity (0-5)
  "dims": {
    "D1": {"score": 0.6, "level": 2},  // Directive force
    "D2": {"score": 0.7, "level": 3},  // Option constraint
    "D3": {"score": 0.5, "level": 2},  // Normative pressure
    "D4": {"score": 0.8, "level": 3},  // Emotional pressure
    "D5": {"score": 0.2, "level": 1},  // Deceptiveness
    "D6": {"score": 0.7, "level": 3},  // Toxicity
    "D7": {"score": 0.8, "level": 3}   // Explicitness
  }
}
```
- `score` is a continuous 0--1 value; `level` maps 0.0->0, (0,0.33]->1, (0.33,0.66]->2, (0.66,1.0]->3 (None/Low/Moderate/High).

## Statistics
| Set | Dialogues | Coercive | Non-coercive |
|-----|-----------|----------|--------------|
| Full | 3,432 | 1,862 (54.2%) | 1,570 (45.8%) |
| First release | 634 | 332 (52.4%) | 302 (47.6%) |

Gold intensity (Full): levels 0-5 = 986/267/317/337/621/904.

## Provenance & License
Built from the COERCION moral-coercion corpus (see `references.bib` entry in the
paper). Dialogues are de-identified everyday scenarios; no personal information
is released. Annotations were produced by an instruction-tuned LLM at
temperature 0 and validated against the source gold labels; a human agreement
study is planned. Please cite the companion paper; check the source corpus
license before redistribution.

## Reproduce
```bash
# 1) rebuild the release from source + annotation outputs
python experiments/release_dataset.py
# 2) recompute every number in the paper
python experiments/verify_paper.py
```
