# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()
reps = []

# Abstract: replace by boundaries
i0 = s.find(r'\begin{abstract}')
i1 = s.find(r'\end{abstract}')
assert i0 != -1 and i1 != -1
new_abs = r'''\begin{abstract}
Discourse that drives, guides, or restricts a listener's behavior---orders,
threats, inducements, guilt-tripping, peer pressure, gaslighting, and verbal
abuse---shares a common perlocutionary mechanism, yet existing benchmarks
study each strategy in isolation with incompatible labels and scales. We
propose \LNGF{}, a unified benchmark for the \afd{} in real-world multi-turn
dialogues, built on a three-layer framework: (i) a strategy taxonomy of four
families and fifteen types integrating compliance-gaining and speech-act
theory; (ii) seven universal psychological dimensions grounded in pragmatics
and persuasion research, which specialize to the obligation, constraint,
value-judgement, and toxicity channels of prior moral-coercion work; and
(iii) a linguistic-realization layer linking dimensions to surface cues.
We release 3{,}432 re-annotated dialogues with intensity and seven-dimension
labels; reliability is verified through consistency with existing labels
and a two-annotator agreement study. We define three benchmark tasks, a
two-stage dimension-aware pipeline, and cross-domain evaluation on
MentalManip, MultiManip, TalkDown, and ToxicChat, supporting zero-shot
recognition of unseen manipulation types.
\end{abstract}
'''
s = s[:i0] + new_abs + s[i1 + len(r'\end{abstract}'):]
reps.append('abstract')

# Design Principles
old = r'''Four principles guide the construction. (i)~\emph{Real-world multi-turn
dialogues}: the source corpus contains authentic multi-turn interaction
rather than scripted dialogue, which lacks the fragmentation, topic drift,
and pragmatic embedding of real conversation. (ii)~\emph{Full-spectrum
coverage}: the corpus spans coercive and non-coercive dialogue across the
full 0--5 intensity range, so benign controls are always present and
degenerate models that classify any request as manipulation are prevented.
(iii)~\emph{Comparable measurement}: every instance receives the same
seven-dimension annotation, so all dialogues are comparable on one scale.
(iv)~\emph{Hard negatives}: polite coercive utterances, sarcastic requests,
and superficially kind manipulation are naturally present in the corpus and
stress-test lexical shortcuts.'''
new = r'''Four principles guide construction. (i)~\emph{Real-world multi-turn
dialogues}: authentic interaction rather than scripted text, which lacks the
fragmentation and pragmatic embedding of real conversation.
(ii)~\emph{Full-spectrum coverage}: coercive and non-coercive dialogue across
the 0--5 range, so benign controls prevent degenerate models that classify
any request as manipulation. (iii)~\emph{Comparable measurement}: every
instance receives the same seven-dimension annotation on one scale.
(iv)~\emph{Hard negatives}: polite coercion, sarcastic requests, and
superficially kind manipulation stress-test lexical shortcuts.'''
assert old in s, 'principles'; s = s.replace(old, new); reps.append('principles')

# Sources and Sampling
old = r'''\LNGF{} is built on a real-world corpus of everyday multi-turn dialogues
collected by prior work~\cite{coercion}. We take the corpus's
dialogues as raw text, retain their existing binary coercion and 0--5
intensity labels for validation, and re-annotate every dialogue under
our unified scheme
with the seven dimensions and a dialogue-level intensity. The full release
comprises the 3{,}432-dialogue training partition; a disjoint 634-dialogue
held-out partition is released as a first consistency-check set
(Section~\ref{sec:firstrelease}), and its statistics are reported alongside
the full set. The dialogues are already de-identified and cover everyday
scenarios, so the negative class (non-coercive dialogue) and the intensity
tail arise naturally; no additional collection is required for this
release.'''
new = r'''\LNGF{} re-annotates the real-world corpus of everyday multi-turn dialogues
collected by prior work~\cite{coercion}, retaining its binary coercion and
0--5 intensity labels for validation. The full release is the 3{,}432-
dialogue training partition; a disjoint 634-dialogue held-out partition is
released as a consistency-check set (Section~\ref{sec:firstrelease}) and its
statistics are reported alongside. Dialogues are already de-identified, so
the negative class and the intensity tail arise naturally without new
collection.'''
assert old in s, 'sources'; s = s.replace(old, new); reps.append('sources')

# Annotation Decision Tree and Negative Cases
old = r'''To keep the negative class principled, annotation follows the decision tree
in Figure~\ref{fig:tree}. The first question filters out the majority of
ordinary conversation; the second assigns strategy labels; the third assigns
intensity and the seven dimension scores. Negative examples are annotated at
the same granularity as positives so that the dimension space is defined over
all dialogues, not only manipulative ones---this is what allows the
dimension space to represent the full spectrum, including non-manipulative
dialogue. Typical negative cases include pure information exchange (``The
meeting is at three''), emotional venting with no behavioral target (``I am
completely exhausted''), and questions that seek information rather than
action (``What time is the deadline?''). Borderline cases, such as advice
that is also a subtle request, are resolved by the multi-label rule: assign
every applicable type, and let the dimension scores capture the dominant
channel.'''
new = r'''To keep the negative class principled, annotation follows the decision tree
in Figure~\ref{fig:tree}: the first question filters out ordinary
conversation, the second assigns strategy labels, and the third assigns
intensity and dimension scores. Negative examples are annotated at the same
granularity as positives so the dimension space covers the full spectrum.
Typical negatives include pure information exchange, emotional venting with
no behavioral target, and information-seeking questions. Borderline cases
(e.g., advice that is also a subtle request) are resolved by the multi-label
rule: assign every applicable type and let dimension scores capture the
dominant channel.'''
assert old in s, 'tree'; s = s.replace(old, new); reps.append('tree')

io.open(p, 'w', encoding='utf-8').write(s)
print('compressed:', reps)