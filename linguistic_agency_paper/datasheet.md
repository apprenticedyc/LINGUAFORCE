# Datasheet: LINGUAFORCE

For questions about this datasheet, contact the dataset authors.

## Motivation
- **For what purpose was the dataset created?** LINGUAFORCE studies *agentive force* in everyday multi-turn dialogues: the psychological pressure an utterance exerts on a listener. It provides a unified seven-dimension annotation scheme (directive force, option constraint, normative pressure, emotional pressure, deceptiveness, toxicity, explicitness), a fifteen-type strategy taxonomy in four families, and a 0--5 intensity label.
- **Who created it and who funded it?** Created by the authors of this paper for academic research. No external funding was used.

## Composition
- **What do the instances represent?** Each instance is a multi-turn English dialogue (6--8 turns) between two speakers, labeled at the dialogue level with: a binary agentive-pressure flag, a 0--5 intensity score, seven continuous 0--1 dimension scores (each with a companion None/Low/Moderate/High level), and a multi-label set of strategies from a 15-type taxonomy. Where available, the original binary coercion label and 0--5 coercion intensity from the source corpus are retained for validation.
- **How many instances are there?** Full release: 3,432 dialogues (1,862 coercive / 1,570 non-coercive). First-release held-out set: 634 dialogues (332 / 302). The two partitions are disjoint.
- **Does the dataset contain all possible instances?** No. It is a convenience sample of everyday conversational dialogues from the source corpus described below; it is not a random sample of any broader population.
- **What data does each instance consist of?** Raw dialogue text, speaker turns, and the annotations described above. No audio, images, or metadata are included.
- **Is there a label or target associated with each instance?** Yes (see above). Labels are produced by an instruction-tuned LLM at temperature 0 following a three-level decision tree, with a human spot-check of 120 dialogues by the first author and consistency validation against the source corpus labels.
- **Are there recommended data splits?** The full release is intended as the training partition; the 634-dialogue set is the held-out evaluation partition.
- **Are there any errors, sources of noise, or redundancies in the dataset?** Dimension and type labels are LLM-generated and inherit systematic annotation bias; types are highly imbalanced (e.g., A3 n=29, D2 n=7) and 82% of dialogues are multi-label. Redundant fields are removed.
- **Is the dataset self-contained?** Yes, all required text and labels are included in the released files.
- **Does the dataset contain data that might be considered confidential?** No. The source corpus was already de-identified by prior work; this release adds no personally identifying information.
- **Does the dataset relate to people?** The dialogues depict fictional or anonymized everyday scenarios; no real individuals are identified.

## Collection Process
- **How was the data associated with each instance acquired?** The dialogues are taken from the source corpus introduced by prior work (COERCION), a real-world corpus of everyday multi-turn dialogues collected under that project's terms of service and de-identification process. This paper re-annotates the existing text; no new data collection was performed.
- **Who was involved?** The prior corpus authors collected raw text; the current authors designed the annotation scheme and ran the annotation pipeline.
- **Over what timeframe?** Annotation and validation were performed during the preparation of this paper.

## Preprocessing / Cleaning / Labeling
- **What was done to clean the data?** Dialogues with malformed or empty turns were removed; redundant metadata fields were dropped. The retained fields are dialogue text plus the annotations described above.
- **What was the raw text language?** English.
- **How was the label defined and produced?** An instruction-tuned LLM follows a three-level decision tree: (1) does the discourse target a listener action/decision? (2) which strategies are present? (3) what are the intensity and seven dimension scores? Output is constrained to a JSON schema, temperature 0. A random spot-check of 120 dialogues was re-annotated by the first author; agreement statistics are reported in the paper. Reliability is additionally validated via consistency with the source corpus binary/intensity labels.
- **Is the software used to preprocess/clean/label available?** Yes, the annotation scripts and prompts are released with the code repository.

## Uses
- **Has the dataset been used for any tasks already?** It is used in this paper for three tasks: T1 manipulation detection, T2 fifteen-way multi-label type recognition, T3 intensity prediction, plus zero-shot cross-domain transfer to MentalManip, MultiManip, TalkDown, and ToxicChat.
- **What are the intended uses?** Research on manipulative and persuasive language, computational pragmatics, and LLM safety evaluation.
- **What are the (potential) misuse risks?** The framework and taxonomy could in principle be used to craft more effective manipulative language. We release the data for research only and do not provide generation recipes; the primary intended use is *detection* and *analysis*.
- **Are there tasks for which the dataset should not be used?** It should not be used for automated moderation of real users without human review, for surveillance, or to make consequential decisions about individuals.

## Distribution
- **Will the dataset be distributed to third parties outside the entity?** Yes, released publicly for research use under a research-only license.
- **How will the dataset be distributed?** Via the project repository (code + data + annotation prompts + reproduction scripts).
- **When will the dataset be released?** With the camera-ready of this paper.
- **Are there any legal restrictions?** Research-only license; the source corpus's terms of service are respected.

## Maintenance
- **Who will support/host/maintain the dataset?** The authors.
- **Is there an erratum mechanism?** Dataset versions are tagged; corrections are released as new versions with a changelog.
- **Will the dataset be updated?** Periodically, e.g., to add additional languages or domains; updates are versioned.

## Additional Notes (added by authors)
- Culture and language scope: the current release is English, everyday scenarios. The seven-dimension scheme is argued from pragmatics and persuasion theory but its cross-lingual/cross-cultural validity requires future study.
- Annotation is LLM-generated; human inter-annotator agreement on a larger subsample is planned follow-up work.