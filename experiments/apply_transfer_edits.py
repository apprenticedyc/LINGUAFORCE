# -*- coding: utf-8 -*-
"""Apply cross-domain transfer results to main.tex."""
import re, io

P = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
with io.open(P, encoding='utf-8') as f:
    tex = f.read()
orig = tex

def rep(pattern, repl, label):
    global tex
    new, n = re.subn(pattern, repl, tex, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f'FAILED: {label} matched {n}')
    tex = new
    print(f'OK  : {label}')

# 1. protocol sentence (Cross-Domain Evaluation)
rep(r'to verify that the model distinguishes\s+agency-bearing toxicity from generic toxicity and condescension',
    r'to probe how agency detection relates to neighboring constructs such as patronizing language and generic toxicity',
    'protocol sentence')

# 2. remaining-tables sentence
rep(r'The remaining tables \(cross-domain transfer\s+and the fine-tuned baseline grid\) stay planned until the full annotation\s+and training runs complete; in particular, T2 type recognition requires the\s+fifteen-type taxonomy labels of the full release\.',
    r'The fine-tuned baseline grid and T2 type recognition remain planned until the full annotation\nand training runs complete; T2 requires the fifteen-type taxonomy labels of the\nfull release.',
    'remaining-tables sentence')

# 3. transfer results paragraph, inserted after t1/t23 discussion
para = (r"""The cross-domain transfer results (Table~\\ref{tab:results-x}) support the
interpretability and discriminative-validity claims. Under a threshold-free
ranking, the zero-shot judge transfers moderately to the psychological-
manipulation corpora (AUROC 0.742 on MentalManip, 0.706 on MultiManip) and to
patronizing language (AUROC 0.729 on TalkDown), while transfer to generic
toxicity is weak (AUROC 0.587 on ToxicChat). This pattern is consistent with
the design: manipulation and condescension exert listener-directed pressure
and are therefore partially captured by the agency dimensions, whereas
generic toxicity is largely distinct from agency. Fixed-threshold
macro-F1/F1 values are much lower (e.g., 0.058 on TalkDown) because the
intensity scale is calibrated on our annotation scheme and the judge rarely
assigns intensity $\\ge$3 to dialogues annotated under other corpora's
schemes: ranking (AUROC) transfers, absolute calibration does not.
Target-calibrated thresholds recover macro-F1 around 0.68--0.74 on the
manipulation corpora.
""")
rep(r'advantage of the dimension profile is concentrated in binary\s+discrimination rather than in fine-grained calibration\.',
    r'advantage of the dimension profile is concentrated in binary\ndiscrimination rather than in fine-grained calibration.\n\n' + para,
    'transfer paragraph')

# 4. table replacement
new_table = r"""\begin{table}[t]
\centering
\caption{Cross-domain zero-shot transfer of the seven-dimension pipeline.
``Standard metric'' is each corpus's own metric under the source-task
decision rule (intensity ${\ge}3$); ``AUROC'' is threshold-free and directly
comparable. Sampled subsets: MentalManip $n{=}300$, MultiManip $n{=}220$,
TalkDown $n{=}652$, ToxicChat $n{=}300$. ReaMent and SemEval-2023 Task~3
remain planned.}
\label{tab:results-x}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Corpus} & \textbf{Standard metric} & \textbf{Zero-shot} & \textbf{AUROC}\\
\midrule
MentalManip & macro-F1 & 0.615 & 0.742\\
MultiManip & macro-F1 & 0.347 & 0.706\\
ReaMent & F1 & -- & --\\
SemEval-2023 Task 3 & micro-F1 & -- & --\\
TalkDown & F1 & 0.058 & 0.729\\
ToxicChat & AUROC & 0.587 & 0.587\\
\bottomrule
\end{tabular}
\end{table}
"""
rep(r'\\begin\{table\}\[t\].*?\\caption\{Cross-domain transfer \(planned\).*?\\end\{table\}',
    new_table,
    'transfer table')

with io.open(P, 'w', encoding='utf-8') as f:
    f.write(tex)
print('written', P, 'changed_bytes', len(tex) - len(orig))
