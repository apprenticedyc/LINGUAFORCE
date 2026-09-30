# Datasheet: LINGUAFORCE

## Motivation

- **Purpose:** LINGUAFORCE measures how language can pressure, constrain, or
  steer a listener's decision in everyday multi-turn dialogue.
- **Scope:** The release covers English dialogue, seven measurement
  dimensions, a fifteen-type strategy taxonomy, and 0--5 intensity labels.
- **Creators and funding:** The dataset was prepared by the paper authors for
  academic research. No external funding was used.

## Composition

- **Instances:** 4,066 de-identified English dialogues, each containing
  speaker turns and dialogue-level annotations.
- **Splits:** 3,432 training dialogues and 634 disjoint held-out dialogues.
- **Labels:** A binary reference label, a 0--5 reference intensity, a released
  overall intensity prediction, seven continuous dimension scores with
  four-level discretizations, and multi-label strategy annotations.
- **Dimensions:** Directive force, option constraint, normative pressure,
  emotional pressure, deceptiveness, toxicity, and explicitness.
- **Modalities:** Text only. No audio, images, user profiles, or identifying
  metadata are included.
- **Sampling:** The release is a convenience sample of everyday dialogue and
  is not intended to represent every language, culture, or social setting.

## Labeling and quality checks

- **Label production:** A frozen instruction-tuned model at temperature 0
  follows a three-stage decision procedure: identify a listener-directed
  action or decision, assign applicable strategy types, and assign intensity
  and dimension scores. Outputs are constrained to a JSON schema.
- **Human agreement:** A stratified sample of 150 dialogues was labeled
  independently by two annotators without access to model outputs. The paper
  reports agreement for the discrete labels.
- **Quality checks:** The release reports correlations with the reference
  binary and intensity labels, classification AUC, monotonicity tests,
  dimension-space correlations, ablations, and cross-domain transfer.
- **Known limitations:** The dimension and strategy labels may contain model
  bias. Strategy frequencies are imbalanced, and many dialogues contain more
  than one strategy.

## Data handling and privacy

- **Text:** The dialogues are de-identified English text describing everyday
  scenarios. No real-person identifiers are intentionally included.
- **Privacy:** Do not use the release for surveillance, automated decisions
  about individuals, or moderation without human review.
- **Sensitive content:** Some dialogues contain threats, insults, deception,
  or other distressing language. Users should provide appropriate review and
  opt-out procedures in human studies.

## Intended use

The release is intended for research on computational pragmatics, language
influence, dialogue analysis, model evaluation, and detection of harmful
interaction patterns. It is not intended to generate or optimize manipulative
messages.

## Distribution and maintenance

- **License:** Research use only. Review the applicable license and required
  attribution before redistribution.
- **Package:** The repository includes the paper, data, derived results, and
  deterministic analysis scripts. Provider credentials and private workbooks
  are excluded.
- **Maintenance:** Future corrections or extensions should be released as
  versioned updates with a corresponding changelog.
