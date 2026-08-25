# LINGUAFORCE: Benchmarking Agentive Force of Discourse

A unified benchmark for the **agentive force of discourse** in real-world
multi-turn dialogues: how language drives, guides, or restricts a
listener's behavior — from polite requests to threats, guilt-tripping,
peer pressure, and covert manipulation.

This repository contains the LINGUAFORCE paper (LaTeX, IEEE conference
format), the released dataset and annotation materials, and all
experiment code, together with the underlying COERCION codebase
(moral-coercion dimension decomposition) that this work builds on.

## Layout

```
COERCION/                    prior codebase & analyses (source data in inputters/data)
experiments/                 annotation, evaluation, ablation, and IAA scripts + outputs
linguistic_agency_paper/     the paper (main.tex), figures, dataset, datasheet
  data/linguaforce_full.jsonl            full release (n=3,432)
  data/linguaforce_first_release.jsonl   held-out consistency set (n=634)
  data/linguaforce_*_types.jsonl         15-type multi-label labels
  data/spotcheck/                        human spot-check materials
```

## Dataset

- 3,432 dialogues annotated with dialogue-level intensity (0–5) and seven
  psychological dimensions (directive force, option constraint, normative
  pressure, emotional pressure, deceptiveness, toxicity, explicitness).
- 15 manipulation types under 4 families (rational persuasion, authority,
  coercion, deception).
- Annotation is LLM-assisted, verified by human spot-checks (first-author
  120-dialogue check; two-annotator agreement study on 150 dialogues).

See `linguistic_agency_paper/datasheet.md` and
`linguistic_agency_paper/data/README.md` for details.

## Paper

- Source: `linguistic_agency_paper/main.tex`
- Compiled: `linguistic_agency_paper/main.pdf`

## Notes

- Raw third-party corpora (e.g., ToxiGen, TalkDown, Suicide Detection) are
  **not** redistributed here; see their original releases for access.
- API keys are never committed; copy `api_config.json` locally as needed.
