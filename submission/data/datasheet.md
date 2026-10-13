# LINGUAFORCE Datasheet

## Purpose

LINGUAFORCE evaluates pressure presence, intensity, strategies, and seven
dimensions in English dialogue. Ordinary requests and reasoned persuasion
provide contrasting examples. Labels concern text, not actual compliance
or psychological harm.

## Dialogue Construction

The dialogues were manually collected, organized, and written in English
around common everyday interaction scenarios. They are not presented as
verbatim real-world chat logs. This is a scenario-based collection, not a
probability sample of conversation. Collection dates, proportions of
collected versus authored examples, and scenario-level provenance are not
recorded in released rows. Text construction and annotation are separate.

## Composition

There are 4,066 dialogues: 3,144 pressure and 922 no pressure.
The 3,432-dialogue development pool is divided into 3,088 training and 344
validation records with seed 2026, stratified by intensity. The fixed test
set contains 634 dialogues (504 pressure, 130 no pressure).
All four targets are human labels: binary pressure (0/1), intensity (0–5),
seven dimensions (0–3), and fifteen non-exclusive binary strategies.
The schema and split IDs are in `human/`.

## Annotation and Quality

Twenty computer science graduate students proficient in English dialogue
read the shared guidelines and independently labeled twenty non-overlapping
batches without consulting others' answers. Each dialogue received one
annotation. Participation was voluntary and unpaid. The supplied worksheets
did not contain model predictions. Original Chinese instructions and their
English translation are included.

No independent duplicate annotation or adjudication was performed.
Inter-annotator agreement is unavailable. Structural checks verify coverage,
ranges, normalized exact duplicates, and binary/intensity consistency.
They do not establish label correctness.

## Evaluation

Four tasks use the fixed test set. Three zero-shot language models are
compared with supervised word/character TF-IDF and constant baselines.
Prompts, settings, raw predictions, scoring code, per-label results, and
paired pressure bootstrap intervals are supplied. Test labels are targets,
not inference-time features. A separate matched Qwen3-4B LoRA diagnostic
uses the same test requests before and after training. Its saved responses,
available settings, validity counts, and paired intervals are released;
managed-service training code and adapter weights are not distributed.

## Privacy and Permissions

The creators state that the dialogue texts may be released. References to
people in everyday scenarios are not verified accounts of identifiable
individuals. No institutional ethics approval, formal participant-consent
procedure, or independently certified anonymization is asserted.
The supplement excludes annotator identities, workbook metadata, credentials,
and author-identifying repository links. Automated identifier checks do
not certify removal of all contextual identifiers.

## Intended Use and Limitations

Intended use is research on dialogue understanding and model evaluation.
Content may include threats, deception, insults, and distressing language.
It is not appropriate for diagnosis, surveillance, or unattended decisions
about individuals. The data cover English scenarios and an imbalanced,
predefined inventory. Exact-duplicate checks do not rule out paraphrases or
shared templates. Single annotations introduce unmeasured variability.

## Distribution

The creators authorize release. This is an anonymous review distribution;
a specific public reuse license has not been selected. See `../TERMS.md`.
