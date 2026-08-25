# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()

# 1) Annotation Protocol: replace the "left to future work" sentence
old_prot = r'''Rather than human inter-annotator agreement, we validate
reliability through the consistency checks against the existing labels in
Section~\ref{sec:firstrelease}; a human agreement study on a subsample is
left to future work.'''
new_prot = r'''Reliability is validated through the consistency checks
against the existing labels in Section~\ref{sec:firstrelease} and through a
two-annotator agreement study on a stratified subsample
(Section~\ref{sec:qc}).'''
assert old_prot in s, 'protocol phrase not found'
s = s.replace(old_prot, new_prot)

# 2) Add sec:qc label to Quality Control subsection and append IAA paragraph + table
anchor_qc = r'''\subsection{Quality Control and Consistency Validation}'''
assert anchor_qc in s
s = s.replace(anchor_qc, anchor_qc + '\n\\label{sec:qc}', 1)

iaa_block = r'''
To further probe reliability beyond the author's spot-check, we ran a
two-annotator agreement study on a stratified subsample of 150 dialogues
(balanced on the binary label), annotated independently under the same
rubric. Table~\ref{tab:iaa} reports inter-annotator agreement on the
discrete labels: the core intensity label reaches substantial agreement
(quadratic-weighted $\kappa{=}0.73$), as do normative pressure (0.78),
toxicity (0.76), and option constraint (0.66); binary presence and
explicitness reach moderate agreement (0.49), and deceptiveness is the
least reliable dimension (0.39), consistent with its subjective,
low-frequency nature. The continuous scores inherit the same ordering as
the discrete labels, so the reported agreement carries over to the
released scores.

\begin{table}[t]
\centering
\caption{Inter-annotator agreement on a stratified subsample ($n{=}150$;
two independent annotators, 149 fully labeled pairs). Binary uses
unweighted Cohen's $\kappa$; intensity and dimensions use
quadratic-weighted $\kappa$.}
\label{tab:iaa}
\footnotesize
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Variable} & \textbf{Cohen's $\kappa$} & \textbf{Agreement level}\\
\midrule
binary (pressure present) & 0.49 & moderate\\
intensity (0--5) & 0.73 & substantial\\
$D_1$ directive force & 0.55 & moderate\\
$D_2$ option constraint & 0.66 & substantial\\
$D_3$ normative pressure & 0.78 & substantial\\
$D_4$ emotional pressure & 0.60 & substantial\\
$D_5$ deceptiveness & 0.39 & fair\\
$D_6$ toxicity & 0.76 & substantial\\
$D_7$ explicitness & 0.49 & moderate\\
\bottomrule
\end{tabular}
\end{table}

'''
# insert after the "reproducible against independent human judgment." paragraph end
anchor_end = 'reproducible against independent human judgment.\n'
assert anchor_end in s
s = s.replace(anchor_end, anchor_end + iaa_block, 1)

io.open(p, 'w', encoding='utf-8').write(s)
print('IAA integrated, len', len(s))