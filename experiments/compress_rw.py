# -*- coding: utf-8 -*-
import io, re
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()

# fix dangling tab:rq1 ref
s = s.replace(r'(Tables~\ref{tab:rq1} and~\ref{tab:full})', r'(Table~\ref{tab:full})')

# replace Related Work block
i = s.find(r'\section{Related Work}')
j = s.find(r'\section{Framework', i)
assert i != -1 and j != -1
new_rw = r'''\section{Related Work}
\subsection{Datasets of Manipulation and Persuasion}
Communication research has long studied compliance-gaining: Marwell and
Schmitt~\cite{marwell1967} extracted sixteen strategies and
Wiseman and Schenck-Hamlin~\cite{wiseman1981} fourteen, later organized by
Cialdini~\cite{cialdini1984} into reciprocity, commitment, social proof,
authority, liking, and scarcity. In NLP, SemEval-2023 Task~3 detects
persuasion techniques in online news~\cite{piskorski2023}, and recent
datasets target covert manipulation in dialogues: MentalManip,
MultiManip~\cite{mentalmanip,multimanip}, and ReaMent~\cite{reament}. These
resources are valuable but narrow: they either target monological news text
or focus on covert and pathological strategies, omit benign and transparent
controls (requests, advice, rational persuasion), and use incompatible label
spaces. Toxicity detection~\cite{toxicchat2023}, condescension
detection~\cite{talkdown2023}, and microaggression work~\cite{breitfeller2019}
cover the abusive end of our spectrum but treat hostility as the phenomenon
rather than as a channel of listener-directed pressure. \LNGF{} covers the
full spectrum from transparent requests to covert abuse and makes all types
comparable through one seven-dimension scale.

\subsection{Moral Coercion and Moral Foundations}
The most direct prior evidence for dimensional analysis of coercion comes
from a 4{,}700-dialogue study annotating coercion intensity (0--5) and four
psychological dimensions---obligation, constraint, value judgement,
toxicity---used as structural anchors by a two-stage pipeline~\cite{coercion}.
It shows that moral coercion overlaps ordinary dialogue in embedding space
and that dimension features are partially independent. \LNGF{} keeps the
dimension-as-anchor view but generalizes it to \afd{}: fifteen types in four
families on seven dimensions, with the existing annotations serving as one
validation target. Moral foundations theory~\cite{graham2009} (care,
fairness, loyalty, authority, sanctity, liberty) grounds our normative
family~C: C1 (moral appeal) typically loads on care and loyalty, while C4
(authority) loads on authority and liberty, sharpening the boundary between
normative pressure and pure threats ($D_2$).

\subsection{Agentive Force in LLM Safety Evaluation}
Beyond detection, \afd{} matters for LLM safety. Current evaluations measure
explicit harms such as toxicity, bias, and refusal correctness but rarely
score manipulative \emph{styles}---guilt-heavy advice, authority-laden
justifications, false-dilemma framings---even though they can steer users
toward harmful decisions in health, finance, and politics. \LNGF{}'s
seven-dimension profile provides a continuous, interpretable measurement
applicable to generated text (e.g., scoring $D_2$ or $D_3$ with the same
rubric used for human dialogue), making the benchmark a dual-purpose
instrument: a detection benchmark for human conversation and a safety rubric
for auditing model behavior. In linguistic anthropology,
Ahearn~\cite{ahearn2001} uses ``linguistic agency'' for the speaker's
socioculturally mediated capacity to act; to avoid conflation with our
listener-directed notion, we use \emph{\afd{}} (equivalently, perlocutionary
pressure) throughout.

'''
s = s[:i] + new_rw + s[j:]
io.open(p, 'w', encoding='utf-8').write(s)
print('Related Work compressed, len', len(s))