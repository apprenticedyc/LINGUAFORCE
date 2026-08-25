# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()
reps = []

# 1) pipeline intro paragraph
old = r'''The pipeline is evaluated in a zero-shot setting: a frozen instruction-tuned
LLM (DeepSeek) acts as the upstream dimension parser, and a direct judge that
skips the profile serves as the control. Prompt variants include zero-shot,
chain-of-thought~\cite{wei2022}, and self-reflection, following the
self-perception approach of MultiManip~\cite{multimanip}. Table~\ref{tab:models}
summarizes the configuration.'''
new = r'''The pipeline is evaluated zero-shot: a frozen instruction-tuned LLM
(DeepSeek) parses each dialogue into the profile, and a direct judge that
skips the profile serves as control. Prompt variants include zero-shot,
chain-of-thought~\cite{wei2022}, and self-reflection, following the
self-perception approach of MultiManip~\cite{multimanip}
(Table~\ref{tab:models}).'''
assert old in s, 'pipe'; s = s.replace(old, new); reps.append('pipe')

# 2) Experimental Setup
old = r'''All evaluation in this release is zero-shot: a frozen instruction-tuned LLM
(DeepSeek) parses each dialogue into the seven-dimension profile and a
dialogue-level intensity at temperature~$0$ for reproducibility, and T1 and
T3 are read off the profile directly, with no fine-tuning and no in-domain
training data. We compare against a direct zero-shot judge that skips the
profile (dashed path in Fig.~\ref{fig:pipeline}). Metrics are binary AUC and
best F1 over the intensity threshold for T1, and Spearman, Pearson, QWK, and
accuracy-within-$\pm1$ for T3. Cross-domain transfer uses the same frozen
parser on four public corpora (Section~\ref{sec:exp}). The parser runs
through a public API at low cost; no GPU cluster is required.'''
new = r'''All evaluation is zero-shot: a frozen DeepSeek parses each dialogue into the
seven-dimension profile and a dialogue-level intensity at temperature~$0$;
T1 and T3 are read off the profile directly, with no fine-tuning. The control
is a direct zero-shot judge that skips the profile (dashed path in
Fig.~\ref{fig:pipeline}). Metrics are binary AUC and best F1 over the
intensity threshold for T1, and Spearman, QWK, and accuracy-within-$\pm1$
for T3. Cross-domain transfer uses the same frozen parser on four public
corpora.'''
assert old in s, 'setup'; s = s.replace(old, new); reps.append('setup')

# 3) RQ1
old = r'''\textbf{RQ1 (annotation consistency):} Do the seven dimensions align with
existing coercion annotations on the shared dialogues? Protocol: on the
634-dialogue first release and the full 3{,}432-dialogue release
(Sections~\ref{sec:firstrelease}--\ref{sec:fullscale}), report per-dimension
correlation with the existing binary coercion label and the 0--5 intensity
label, together with the AUC of each dimension for separating coercive from
non-coercive dialogues. High consistency shows that the new annotation
scheme reliably covers previously studied phenomena.'''
new = r'''\textbf{RQ1 (annotation consistency):} Do the seven dimensions align with
existing coercion annotations? On the 634-dialogue and full 3{,}432-dialogue
releases (Sections~\ref{sec:firstrelease}--\ref{sec:fullscale}), report
per-dimension correlation with the binary coercion label and 0--5 intensity
label, plus each dimension's AUC for separating coercive from non-coercive
dialogues.'''
assert old in s, 'rq1'; s = s.replace(old, new); reps.append('rq1')

# 4) RQ2
old = r'''\textbf{RQ2 (dimension structure):} Do the seven dimensions carry
distinct, complementary signal? Protocol: report mean dimension scores and
per-dimension AUC for separating coercive from non-coercive dialogues, and
examine the inter-dimension correlation matrix for expected couplings (e.g.,
$D_3$--$D_4$) and unexpected ones (e.g., $D_5$--$D_6$). Per-family and per-type
analyses on the type-labeled split are reported in
Section~\ref{sec:results}.'''
new = r'''\textbf{RQ2 (dimension structure):} Do the seven dimensions carry distinct,
complementary signal? Report per-dimension scores and AUCs, examine the
inter-dimension correlation matrix for expected couplings (e.g., $D_3$--$D_4$)
and unexpected ones (e.g., $D_5$--$D_6$), and analyze per-family profiles
(Section~\ref{sec:results}).'''
assert old in s, 'rq2'; s = s.replace(old, new); reps.append('rq2')

# 5) RQ3
old = r'''\textbf{RQ3 (zero-shot generalization):} Can dimension conditioning enable
detection of unseen manipulation types? The primary evidence is cross-domain
zero-shot transfer to corpora whose manipulation types are never seen during
training (Section~\ref{sec:exp}): AUROC 0.742 on MentalManip, 0.706 on
MultiManip, and 0.729 on TalkDown. We additionally ran the within-taxonomy
leave-one-type-out protocol on the type-labeled held-out split: a model
trained only on the other fourteen types cannot separate a held-out type as
a novel cluster in the seven-dimension space (mean novelty AUROC 0.54, near
chance), consistent with the framework's design in which types are graded on
shared continuous dimensions rather than forming disjoint clusters.
Crucially, the dimension-conditioned pipeline still flags dialogues of unseen
types as manipulative: predicted intensity separates dialogues containing
any strategy from benign dialogues with AUROC 0.963. The framework therefore
transfers to unseen manipulation types as a pressure signal, while assigning
the specific type label remains an open challenge.'''
new = r'''\textbf{RQ3 (zero-shot generalization):} Can dimension conditioning detect
unseen manipulation types? Cross-domain transfer to corpora whose types are
never seen during training (Section~\ref{sec:exp}) yields AUROC 0.742 on
MentalManip, 0.706 on MultiManip, and 0.729 on TalkDown. Within-taxonomy
leave-one-type-out shows a held-out type is not a separable novel cluster
(mean novelty AUROC 0.54, near chance), consistent with types being graded on
shared continuous dimensions rather than disjoint clusters. Crucially, the
pipeline still flags unseen types as manipulative: predicted intensity
separates any-strategy from benign dialogues with AUROC 0.963. The framework
thus transfers as a pressure signal, while assigning the specific type label
remains open.'''
assert old in s, 'rq3'; s = s.replace(old, new); reps.append('rq3')

# 6) RQ4
old = r'''\textbf{RQ4 (engineering conclusions):} How do aggregation mode and
prompt design interact with the dimension-conditioned pipeline? We run two
slices of the grid with the same linear readout as
Table~\ref{tab:ablation} (trained on the full release, evaluated on the
held-out split). \emph{Aggregation} (Table~\ref{tab:agg}): global-only
(G), turn-mean (T), and their concatenation (B). Turn-mean pooling
collapses toward chance (binary AUC 0.525) because manipulative pressure
is concentrated in a minority of turns and averaging dilutes it; the
global profile (G) recovers the signal, and concatenation (B) performs
best (binary AUC 0.859, intensity Spearman 0.679, QWK 0.680).
\emph{Prompt variants} (Table~\ref{tab:prompt}): zero-shot,
chain-of-thought, and self-reflection yield comparable downstream
performance (AUC 0.83--0.84), showing the pipeline is robust to prompt
design; CoT marginally helps T1 and self-reflection T3. The remaining
grid dimensions (parser-model choice and ICC fidelity) are left to
future work.'''
new = r'''\textbf{RQ4 (engineering conclusions):} How do aggregation mode and prompt
design interact with the pipeline? Using the same linear readout as
Table~\ref{tab:ablation} (train full release, held-out evaluation):
\emph{aggregation} (Table~\ref{tab:agg}) compares global-only (G),
turn-mean (T), and their concatenation (B). Turn-mean pooling collapses
toward chance (binary AUC 0.525) because pressure is concentrated in a
minority of turns and averaging dilutes it; B performs best (AUC 0.859,
Spearman 0.679, QWK 0.680). \emph{Prompt variants} (Table~\ref{tab:prompt})
are comparable (AUC 0.83--0.84), showing robustness to prompt design; CoT
marginally helps T1 and self-reflection T3. Remaining grid dimensions
(parser-model choice, ICC fidelity) are left to future work.'''
assert old in s, 'rq4'; s = s.replace(old, new); reps.append('rq4')

# 7) Cross-Domain Evaluation
old = r'''To test transfer and distinctiveness, we evaluate the zero-shot pipeline on
existing resources with compatible labels: MentalManip~\cite{mentalmanip} and
MultiManip~\cite{multimanip} for psychological manipulation,
TalkDown~\cite{talkdown2023} for patronizing language, and
ToxicChat~\cite{toxicchat2023} for generic toxicity. We report each corpus's
standard metric under its own decision rule and the threshold-free AUROC,
which is directly comparable across label schemes. ReaMent~\cite{reament}
and SemEval-2023 Task~3~\cite{piskorski2023} are left as future work. We use
the confusion across these corpora to identify which constructs are shared
with agency and which are distinct.'''
new = r'''To test transfer and distinctiveness, we evaluate the zero-shot pipeline on
existing resources with compatible labels: MentalManip~\cite{mentalmanip} and
MultiManip~\cite{multimanip} (psychological manipulation),
TalkDown~\cite{talkdown2023} (patronizing language), and
ToxicChat~\cite{toxicchat2023} (toxicity). We report each corpus's standard
metric under its own decision rule plus threshold-free AUROC, and use the
cross-corpus confusion to identify which constructs are shared with agency
and which are distinct. ReaMent~\cite{reament} and SemEval-2023
Task~3~\cite{piskorski2023} are left as future work.'''
assert old in s, 'xdom'; s = s.replace(old, new); reps.append('xdom')

# 8) Dimension Contribution paragraph
old = r'''In this release, the contribution of each dimension is measured by its
consistency with the gold labels: per-dimension Spearman correlation and AUC
against the existing binary and 0--5 intensity labels
(Table~\ref{tab:full}). Dimensions with weak consistency
(e.g., toxicity, $D_6$) are weaker anchors for the coercion decision, while
strong ones (e.g., option constraint, $D_2$; emotional pressure, $D_4$)
carry most of the signal. To quantify each dimension's marginal contribution, we fit a
lightweight linear readout over the seven dimension scores---logistic
regression for the binary decision and linear regression for the 0--5
intensity---trained on the full release and evaluated on the held-out
split. The linear readout reaches binary AUC 0.855 and intensity
Spearman 0.674 on the held-out set, slightly above the LLM's
dimension-conditioned readout on the same split (AUC 0.831, Spearman
0.665); the annotated profiles alone therefore carry the
decision-relevant signal and support cheap downstream models without an
LLM. Table~\ref{tab:ablation} reports leave-one-dimension-out results:
removing toxicity ($D_6$) degrades detection most (AUC drop of 0.008),
removing normative pressure ($D_3$) degrades intensity most (Spearman
drop of 0.021), and option constraint ($D_2$) also contributes. Notably,
$D_6$ has the weakest univariate correlation yet its removal hurts
detection most, indicating it supplies non-redundant information (e.g.,
separating hostile-but-non-coercive dialogues). Emotional pressure
($D_4$) is largely redundant with $D_3$ in the linear readout, and the
remaining dimensions have small marginal effects, consistent with the
inter-dimension correlations of Figure~\ref{fig:dim-corr-full}.'''
new = r'''Each dimension's contribution is measured by consistency with the gold
labels (per-dimension Spearman and AUC, Table~\ref{tab:full}) and by a
marginal-contribution analysis: a linear readout over the seven scores
(logistic for binary, linear for intensity), trained on the full release and
evaluated on the held-out split, reaches binary AUC 0.855 and intensity
Spearman 0.674---slightly above the LLM's dimension-conditioned readout
(AUC 0.831, Spearman 0.665)---so the profiles carry the decision-relevant
signal without an LLM. Table~\ref{tab:ablation} reports leave-one-dimension-out
results: removing toxicity ($D_6$) degrades detection most (AUC drop 0.008)
and normative pressure ($D_3$) degrades intensity most (Spearman drop 0.021).
Notably, $D_6$ has the weakest univariate correlation yet its removal hurts
detection most, indicating non-redundant information (e.g., separating
hostile-but-non-coercive dialogues); $D_4$ is largely redundant with $D_3$,
and the remaining dimensions have small marginal effects, consistent with
Figure~\ref{fig:dim-corr-full}.'''
assert old in s, 'dimcont'; s = s.replace(old, new); reps.append('dimcont')

io.open(p, 'w', encoding='utf-8').write(s)
print('Experimental compressed:', reps)