# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()

# ---------- 1) Replace RQ4 paragraph ----------
old_rq4 = r'''\textbf{RQ4 (engineering conclusions):} How do dimension-model matching,
aggregation mode (T/G/B), and prompt complexity interact? Protocol: grid over
$\{$parser model$\}\times\{$aggregation mode$\}\times\{$prompt variant$\}$,
reporting downstream T1--T3 metrics and dimension-extraction fidelity (ICC
against gold annotations). This grid requires supervised downstream models
and is deferred to future work; the zero-shot results in this release use a
single frozen parser with global (dialogue-level) aggregation.'''
new_rq4 = r'''\textbf{RQ4 (engineering conclusions):} How do aggregation mode and
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
assert old_rq4 in s, 'RQ4 block not found'
s = s.replace(old_rq4, new_rq4)

# ---------- 2) Insert aggregation + prompt tables before Cross-Domain ----------
anchor = r'\subsection{Cross-Domain Evaluation}'
tables = r'''\begin{table}[t]
\centering
\caption{Aggregation ablation of the dimension-conditioned pipeline
(linear readout; train full release, held-out split). T mean-pools the
per-turn profiles over turns; B concatenates the turn-mean and global
profiles.}
\label{tab:agg}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Aggregation} & \textbf{Binary AUC} & \textbf{Intensity $\rho$} & \textbf{QWK}\\
\midrule
G (global only) & 0.855 & 0.674 & 0.667\\
T (turn mean) & 0.525 & 0.040 & 0.066\\
B (turn+global) & \textbf{0.859} & \textbf{0.679} & \textbf{0.680}\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Prompt ablation of the upstream dimension parser (held-out
split, $n{=}634$). All variants use the same downstream linear readout.}
\label{tab:prompt}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Prompt variant} & \textbf{T1 AUC} & \textbf{T3 Spearman} & \textbf{T3 QWK}\\
\midrule
Zero-shot & 0.832 & 0.666 & 0.623\\
Chain-of-thought & 0.839 & 0.639 & 0.613\\
Self-reflection & 0.838 & 0.670 & 0.630\\
\bottomrule
\end{tabular}
\end{table}

'''
assert anchor in s
s = s.replace(anchor, tables + anchor, 1)

# ---------- 3) Insert dimension-space subsection before Results on the Full Release ----------
anchor2 = r'\subsection{Results on the Full Release}'
viz = r'''\subsection{Dimension-Space Structure}
\label{sec:viz}
To verify that the seven dimensions carve distinct, interpretable regions
for the four strategy families, Figure~\ref{fig:tsne} embeds the 3{,}432
dialogue profiles in two dimensions (t-SNE), colored by the dominant
family, together with the 226 dialogues for which no strategy was detected
(benign). Families occupy distinguishable regions, and benign dialogues
cluster at the low-pressure side. A logistic classifier over the seven raw
dimensions separates the five groups with 0.640 accuracy (chance 0.2) in
five-fold cross-validation, and each dimension alone separates benign from
manipulative dialogues with AUC 0.73--0.94. Figure~\ref{fig:famheat}
reports each family's mean dimension profile: the social-normative family
peaks on normative and emotional pressure ($D_3$, $D_4$), the exchange
family on option constraint and toxicity ($D_2$, $D_6$), the covert family
on emotional pressure ($D_4$), and benign dialogues are uniformly low
across all dimensions.

\begin{figure*}[t]
\centering
\includegraphics[width=0.70\textwidth]{figs/fig5_tsne_family.png}
\caption{t-SNE embedding of the seven-dimension profiles ($n{=}3{,}432$),
colored by dominant strategy family; the 226 no-strategy dialogues form the
benign group.}
\label{fig:tsne}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.78\textwidth]{figs/fig6_family_dims_heatmap.png}
\caption{Mean dimension score per strategy family. Social-normative and
covert families concentrate pressure on normative/emotional dimensions;
benign dialogues are uniformly low.}
\label{fig:famheat}
\end{figure*}

'''
assert anchor2 in s
s = s.replace(anchor2, viz + anchor2, 1)

io.open(p, 'w', encoding='utf-8').write(s)
print('main.tex integrated, len', len(s))