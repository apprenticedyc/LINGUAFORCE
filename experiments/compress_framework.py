# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()
reps = []

# 1) Definition and Inclusion Criterion
old = r'''We define the \emph{\afd{}} of an utterance or dialogue as its perlocutionary
capacity to drive, guide, or restrict a listener's behavior and decision
space. Formally, for an utterance $u$ addressed to a listener $L$ in context
$C$, let $\mathcal{A}(u,C)$ be the set of actions that $L$ considers
acceptable after processing $u$, and let $\mathcal{A}^*(C)$ be the
speaker-preferred subset. The agentive force of $u$ is the pressure it exerts
to shift $\mathcal{A}(u,C)$ toward $\mathcal{A}^*(C)$, along one or more of
the seven channels defined below. Three properties are important. First,
\emph{success is not required}: perlocutionary force exists even when the
listener resists. Second, \emph{force is compositional}: an utterance can
combine several channels (e.g., guilt plus social proof). Third, \emph{force
is scalar}: strategies differ in intensity, not only in kind.'''
new = r'''We define the \emph{\afd{}} of an utterance or dialogue as its perlocutionary
capacity to drive, guide, or restrict a listener's behavior and decision
space: the pressure an utterance exerts to shift the listener's acceptable
action set toward the speaker's preferred subset, along one or more of seven
channels. Three properties matter. \emph{Success is not required}:
perlocutionary force exists even when the listener resists. \emph{Force is
compositional}: an utterance can combine channels (e.g., guilt plus social
proof). \emph{Force is scalar}: strategies differ in intensity, not only in
kind.'''
assert old in s, 'def1'; s = s.replace(old, new); reps.append('def')

old = r'''The inclusion criterion follows directly: a discourse is \emph{agency-bearing}
if and only if it (i)~targets a recognizable action or decision of the
listener, (ii)~employs at least one strategy in our taxonomy, and (iii)~aims
to drive or restrict behavior. Pure information statements and pure emotional
venting that do not direct behavior are excluded and serve as negative
examples. This criterion makes the negative class meaningful---a common
weakness in manipulation benchmarks that sample only positive examples.'''
new = r'''A discourse is \emph{agency-bearing} iff it (i)~targets a recognizable action
or decision of the listener, (ii)~employs at least one strategy in our
taxonomy, and (iii)~aims to drive or restrict behavior. Pure information and
emotional venting are excluded and serve as negative examples, making the
negative class meaningful---a common weakness in manipulation benchmarks.'''
assert old in s, 'def2'; s = s.replace(old, new); reps.append('def2')

# 2) Three-Layer Architecture paragraph
old = r'''Figure~\ref{fig:framework} shows the architecture. Layer~1 is a functional
taxonomy: \emph{what} the speaker is doing (fifteen strategy types in four
families). Layer~2 is a measurement layer: \emph{how much} pressure is
applied along each of seven psychological dimensions. Layer~3 is the
linguistic-realization layer: \emph{how} the pressure is realized in surface
form. The three layers are jointly annotated in \LNGF{}, which allows models
to reason at the intended abstraction level and enables error analysis that
is impossible with binary labels alone.'''
new = r'''Figure~\ref{fig:framework} shows the architecture: Layer~1 is a functional
taxonomy (\emph{what} the speaker does; fifteen types in four families),
Layer~2 measures \emph{how much} pressure along seven psychological
dimensions, and Layer~3 links both to surface realizations (\emph{how}).
Jointly annotating the three layers lets models reason at the intended
abstraction level and enables error analysis impossible with binary labels
alone.'''
assert old in s, 'arch'; s = s.replace(old, new); reps.append('arch')

# 3) Strategy Taxonomy paragraph
old = r'''Table~\ref{tab:taxonomy} presents the taxonomy. Four families organize the
strategies by their dominant pressure channel. Family~A (transparent)
contains low-pressure, intention-transparent strategies and provides the
behavioral baseline of the spectrum. Family~B (exchange) motivates behavior
through anticipated benefit or harm. Family~C (social-normative) exploits
norms, roles, and group identity; it contains the moral-coercion strategies
previously studied as a single task and expands them into a family of six
types. Family~D (covert) contains implicit or deceptive strategies that
operate below the listener's awareness. The taxonomy is compatible with prior
coercion annotations: obligation types ($C_2$) and constraint types ($B_2$,
$D_3$) recover the obligation and constraint channels, moral appeal, shaming,
and authority ($C_1,C_3,C_4$) cover the value-judgement channel, and verbal
abuse ($D_4$) covers the toxicity channel. The remaining types extend the
label space with strategies not covered by prior coercion work.'''
new = r'''Table~\ref{tab:taxonomy} presents the taxonomy. Four families organize
strategies by dominant pressure channel: A (transparent) provides the
behavioral baseline; B (exchange) motivates through anticipated benefit or
harm; C (social-normative) exploits norms, roles, and identity, expanding the
previously single moral-coercion task into six types; D (covert) contains
implicit or deceptive strategies. The taxonomy is compatible with prior
coercion annotations: $C_2$ and $B_2$/$D_3$ recover obligation and
constraint, $C_1,C_3,C_4$ cover value judgement, and $D_4$ covers toxicity;
the remaining types extend the label space.'''
assert old in s, 'taxo'; s = s.replace(old, new); reps.append('taxo')

# 4) Universal Psychological Dimensions: merge two paragraphs
old = r'''Table~\ref{tab:dims} defines seven dimensions measured on a continuous
0--1 scale (with a companion four-level discrete label), at both turn and
dialogue level. The inventory is designed to be \emph{complete with respect
to the taxonomy}: every strategy in Table~\ref{tab:taxonomy} loads on at
least one dimension, and the seven dimensions are intended as the
minimal set that separates all fifteen types. It is also
\emph{compatible with prior coercion annotations} by construction:
$D_2 \supseteq \mathrm{CS}$, $D_3 \supseteq \mathrm{OB}\cup\mathrm{VJ}$
(normative pressure merges the two moral channels), and
$D_6 \supseteq \mathrm{TX}$.
Consequently, dialogues annotated with these channels can be embedded into
the seven-dimensional space without loss of information, which is what makes
the annotation-consistency checks in Section~\ref{sec:exp} well-defined.'''
new = r'''Table~\ref{tab:dims} defines seven dimensions on a continuous 0--1 scale
(with a companion four-level discrete label), at both turn and dialogue
level. The inventory is \emph{complete with respect to the taxonomy} (every
strategy loads on at least one dimension; the seven are intended as the
minimal separating set) and \emph{compatible with prior coercion annotations}
by construction: $D_2 \supseteq \mathrm{CS}$, $D_3 \supseteq
\mathrm{OB}\cup\mathrm{VJ}$, and $D_6 \supseteq \mathrm{TX}$. Dialogues
annotated with those channels therefore embed into the seven-dimensional
space without loss, making the consistency checks in
Section~\ref{sec:exp} well-defined.'''
assert old in s, 'dims1'; s = s.replace(old, new); reps.append('dims1')

old = r'''Because the inventory is compatible with prior coercion annotations by
construction (Table~\ref{tab:dims}), the reused dialogues double as a
validation set for the new annotation scheme: dialogues labeled as morally
coercive should receive high $D_3$ and low-to-moderate $D_7$. We also expect
each strategy family to load primarily on a small set of dimensions---$D_3$
for family~C, $D_2$/$D_4$ for family~B, and $D_5$/$D_6$ for family~D---which
we examine through per-family dimension profiles and a clustering analysis
(Section~\ref{sec:dataset}). These analyses serve as quality checks on the
annotations and as descriptive evidence for the structure of the new
benchmark.'''
new = r'''The reused dialogues therefore double as a validation set: morally coercive
dialogues should receive high $D_3$ and low-to-moderate $D_7$. We also expect
each family to load on a small set of dimensions---$D_3$ for family~C,
$D_2$/$D_4$ for B, $D_5$/$D_6$ for D---examined via per-family profiles and
clustering (Section~\ref{sec:dataset}) as quality checks and descriptive
evidence.'''
assert old in s, 'dims2'; s = s.replace(old, new); reps.append('dims2')

# 5) Linguistic Realization Layer paragraph
old = r'''Table~\ref{tab:cues} links each psychological dimension to its typical
surface realizations. The layer serves two purposes. First, it makes the
annotation operational: annotators use the cue checklist to justify dimension
scores, which improves agreement. Second, it supports error analysis: when a
model misclassifies a dialogue, we can trace whether the failure stems from
missing cues, cue ambiguity, or long-range context that surface forms cannot
capture. The realization layer also connects \LNGF{} to explainability:
dimension scores extracted by an upstream parser (Section~\ref{sec:exp})
are directly verbalizable as cue-based rationales.'''
new = r'''Table~\ref{tab:cues} links each dimension to typical surface realizations.
The layer makes annotation operational (annotators justify scores from a cue
checklist, improving agreement), supports error analysis (failures trace to
missing cues, cue ambiguity, or long-range context), and connects \LNGF{} to
explainability: upstream-parser dimension scores (Section~\ref{sec:exp}) are
directly verbalizable as cue-based rationales.'''
assert old in s, 'cues'; s = s.replace(old, new); reps.append('cues')

# 6) Worked Annotation Example paragraph
old = r'''Table~\ref{tab:example} shows a two-turn dialogue and its gold annotation to
make the protocol concrete. The dialogue mixes two strategies---moral appeal
(C1) and obligation (C2)---and therefore receives a multi-label type
annotation; its intensity is high, and the dimension profile is dominated by
$D_3$ with secondary $D_1$. The example illustrates the two properties that
make the task hard: the language is polite (no $D_6$) and the pressure is
implicit (low $D_7$), exactly the pattern that defeats lexical toxicity
classifiers.'''
new = r'''Table~\ref{tab:example} shows a worked annotation. The dialogue mixes moral
appeal (C1) and obligation (C2), hence receives a multi-label type
annotation; intensity is high and the profile is dominated by $D_3$ with
secondary $D_1$. The example illustrates why the task is hard: the language
is polite (no $D_6$) and the pressure implicit (low $D_7$)---the pattern that
defeats lexical toxicity classifiers.'''
assert old in s, 'example'; s = s.replace(old, new); reps.append('example')

# 7) Dimension Structure and Interdependence
old = r'''The seven dimensions are not assumed independent; their correlation
structure is itself an empirical object. We state the expected structure
explicitly so it can be falsified. (i)~$D_1$ (directive force) should
correlate positively with $D_2$ (option constraint): the harder an utterance
pushes, the more it tends to narrow options; the transparent family
(A1) is the predicted exception, with high $D_1$ but low $D_2$.
(ii)~$D_3$ (normative pressure) and $D_4$ (emotional pressure) should
co-occur in family~C---guilt and moral appeal are usually emotionally
loaded---but should separate in family~B, where threats are affectively
cold. (iii)~$D_5$ (deceptiveness) should be roughly orthogonal to the other
dimensions, since gaslighting can occur in both polite and hostile
registers; a strong correlation with $D_6$ or $D_7$ would indicate an
annotation bias rather than a property of the phenomenon. (iv)~$D_7$
(explicitness) should correlate negatively with $D_5$: covert strategies are
typically indirect. These predictions are tested in the statistical
analysis plan (Section~\ref{sec:dataset}) via correlation matrices, factor
analysis, and per-family conditional correlations; any deviation is reported
and used to revise the guideline rather than silently discarded.'''
new = r'''The dimensions are not assumed independent; their correlation structure is
an empirical object that we state explicitly so it can be falsified.
(i)~$D_1$ should correlate with $D_2$ (the harder the push, the narrower the
options), with transparent A1 as the predicted exception. (ii)~$D_3$ and
$D_4$ should co-occur in family~C (guilt and moral appeal are emotionally
loaded) but separate in family~B (threats are affectively cold).
(iii)~$D_5$ should be roughly orthogonal to the others---gaslighting occurs
in both polite and hostile registers---so a strong correlation with $D_6$ or
$D_7$ would indicate annotation bias. (iv)~$D_7$ should correlate negatively
with $D_5$: covert strategies are typically indirect. The statistical
analysis (Section~\ref{sec:dataset}) tests these via correlation matrices
and per-family profiles; deviations are reported and used to revise the
guideline rather than discarded.'''
assert old in s, 'struct'; s = s.replace(old, new); reps.append('struct')

io.open(p, 'w', encoding='utf-8').write(s)
print('Framework compressed:', reps)