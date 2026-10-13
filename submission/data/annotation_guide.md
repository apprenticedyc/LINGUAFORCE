# LINGUAFORCE Annotation Instructions

English translation of the Chinese instructions supplied to annotators.
The original is `annotation_guide_zh.md`.

## Task

Judge negative pressure from the dialogue alone. Pressure is an attempt to
push the listener toward an action or make alternative actions harder to
choose. Ordinary requests, advice, factual statements, and reasoned
persuasion do not automatically count as negative pressure. Read the whole
exchange; do not consult model answers, earlier labels, or another person's
annotations.

## Fields

1. `binary`: pressure absent (0) or present (1).
2. `intensity`: none (0), very slight (1), weak (2), moderate (3), strong (4),
   or very strong (5).
3. `D1`–`D7`: rate each dimension from 0 to 3.
4. Strategy columns `A1`–`D4`: mark each applicable strategy 1, otherwise 0.
5. `notes`: optional explanation of uncertainty or ambiguity.

Intensity describes overall pressure, not the number of strategies.
One strategy may exert strong pressure; several may exert little pressure.

## Dimensions

Use absent (0), low (1), moderate (2), or high (3) for each dimension.

| Dimension | Judgment |
|---|---|
| D1 Directive force | Strength of the push toward the target action. |
| D2 Option constraint | Reduction of acceptable alternatives; ultimatums, forced choices, and threatened consequences often increase this. |
| D3 Normative pressure | Use of duty, morality, identity, authority, group norms, or indebtedness as leverage. |
| D4 Emotional pressure | Use of guilt, shame, fear, or pity to obtain compliance. |
| D5 Deceptiveness | Misleading facts, memory, consensus, or promises supported by context; an unverified statement alone is insufficient. |
| D6 Toxicity | Insults, denigration, profanity, or clear hostility. |
| D7 Explicitness | Directness of the pressure or requested action; a stated demand is more direct than a hint. |

Several dimensions can be high together. For example, "Everyone is
disappointed in you; you must stay" can express normative and emotional
pressure, directive force, and explicitness. D7 describes expression, not
overall pressure intensity.

## Strategies

| Code | Strategy | Judgment |
|---|---|---|
| A1 | Request/Advice | An ordinary, transparent request or advice, usually a low-pressure contrast. |
| A2 | Directive/Order | A direct command or demand for compliance. |
| A3 | Rational Persuasion | Reasons, evidence, or expert opinions supporting an action. |
| B1 | Promise/Inducement | Benefits, rewards, or promises offered in exchange for action. |
| B2 | Threat/Warning | Harm, punishment, or adverse consequences used to obtain action. |
| C1 | Moral Appeal/Guilt | Conscience, moral evaluation, or guilt invoked to obtain action. |
| C2 | Obligation/Duty | Duties associated with a role or identity used to obtain action. |
| C3 | Value Judgement/Shaming | Negative labels such as selfish or heartless used to induce compliance. |
| C4 | Authority | Rank, expertise, identity, or status used to demand compliance. |
| C5 | Conformity/Social Proof | Others' conduct or agreement used as social leverage. |
| C6 | Reciprocity/Debt | Past help or favors used to demand repayment through action. |
| D1 | Emotional Manipulation | Feigned vulnerability, exaggerated distress, or emotional weaknesses used to change decisions. |
| D2 | Deception/Gaslighting | Distorted facts, denial of memories, or misleading narratives. |
| D3 | Logical Trap/False Dilemma | False alternatives or other reasoning traps that restrict choice. |
| D4 | Verbal Abuse/Toxicity | Insults, denigration, or hostility used to obtain compliance. |

Multiple strategies may apply. A1 and A3 are not inherently negative
pressure; use the complete context for the binary judgment. If no strategy
applies, set all strategy columns to zero. Strategy D1–D4 are distinct from
the seven dimension ratings.

## Workflow

1. Read the entire dialogue and assign binary presence and intensity.
2. Rate the seven dimensions.
3. Select applicable strategies.
4. If uncertain, retain your judgment and explain in notes without viewing
   earlier answers.

The twenty files are non-overlapping parts of 4,066 dialogues, not repeated
copies. Estimating agreement would require independent second annotations
on a sample. No such second annotations are part of this release.
