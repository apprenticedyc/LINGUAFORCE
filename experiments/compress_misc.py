# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()
reps = []

# Limitations
old = r'''Several limitations should be acknowledged. First, the seven dimensions are
a design choice; a finer or coarser decomposition is possible, and the
dimension structure may be sensitive to it. We mitigate this by reporting
consistency against the existing coercion labels at both the first-release
and full-scale level. Second, cross-platform variation in norms and
politeness conventions may limit the transferability of both annotations and
models; the cross-domain evaluation and a planned cross-lingual extension
bound this risk. Third, annotation of covert strategies such as gaslighting
is inherently subjective; because the current annotation is LLM-produced, we
rely on consistency with the source corpus's gold labels as the reliability
check, and we flag that a human agreement study is still needed. Fourth, the
dataset inherits the sampling biases of the source corpus and of the
platforms it was drawn from; we document the provenance in the data
statement.
Finally, the benchmark measures the \emph{exertion} of agentive force, not
its \emph{effect}: a coercive dialogue that the listener successfully resists
is still annotated as high-pressure, which is the correct label for
detection but should not be read as evidence of harm.'''
new = r'''Several limitations should be acknowledged. First, the seven dimensions are
a design choice; a finer or coarser decomposition is possible. We mitigate
this by reporting consistency against existing coercion labels at both
first-release and full-scale level. Second, cross-platform variation in
norms and politeness may limit transferability; the cross-domain evaluation
and a planned cross-lingual extension bound this risk. Third, annotation of
covert strategies such as gaslighting is inherently subjective; our
two-annotator study (Section~\ref{sec:qc}) shows deceptiveness ($D_5$) is
the least reliable dimension, which we report rather than hide. Fourth, the
dataset inherits the sampling biases of the source corpus, documented in the
data statement. Finally, the benchmark measures the \emph{exertion} of
agentive force, not its \emph{effect}: a coercive dialogue the listener
resists is still annotated as high-pressure, correct for detection but not
evidence of harm.'''
assert old in s, 'limits'; s = s.replace(old, new); reps.append('limits')

# Discussion
old = r'''Three outcomes would matter beyond the specific numbers. First, if the
dimension profiles are reliable, they provide a continuous, interpretable
measurement of how language exerts pressure---covering transparent requests,
exchange, normative pressure, and covert manipulation under one rubric---and
locate each strategy family precisely in this space. Second, if each family
exhibits a distinct dimension profile, system designers could select parsers
and aggregation modes from a small set of validated configurations rather
than re-tuning per task. Third, the seven-dimension profile offers an
interpretable audit trail: a detection system can justify ``this dialogue is
coercive' with concrete dimension evidence (e.g., high $D_3$, low $D_7$),
which is the kind of accountability that binary toxicity scores cannot
provide. These implications motivate releasing not only the labels but also
the gold dimension profiles and the parser outputs as part of the benchmark
package.'''
new = r'''Three outcomes matter beyond the numbers. First, reliable dimension profiles
provide a continuous, interpretable measurement of how language exerts
pressure---from transparent requests to covert manipulation under one
rubric---and locate each strategy family precisely in this space. Second,
distinct per-family profiles let system designers select parsers and
aggregation modes from a small set of validated configurations rather than
re-tuning per task. Third, the profile offers an interpretable audit trail:
a detection system can justify a ``coercive'' verdict with concrete dimension
evidence (e.g., high $D_3$, low $D_7$), the accountability that binary
toxicity scores cannot provide. These implications motivate releasing the
gold dimension profiles and parser outputs with the benchmark.'''
assert old in s, 'disc'; s = s.replace(old, new); reps.append('disc')

# heatmap: figure* -> figure single column
old = r'''\begin{figure*}[t]
\centering
\includegraphics[width=0.68\textwidth]{figs/fig6_family_dims_heatmap.png}
\caption{Mean dimension score per strategy family. Social-normative and
covert families concentrate pressure on normative/emotional dimensions;
benign dialogues are uniformly low.}
\label{fig:famheat}
\end{figure*}'''
new = r'''\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figs/fig6_family_dims_heatmap.png}
\caption{Mean dimension score per strategy family. Social-normative and
covert families concentrate pressure on normative/emotional dimensions;
benign dialogues are uniformly low.}
\label{fig:famheat}
\end{figure}'''
assert old in s, 'famheat'; s = s.replace(old, new); reps.append('famheat')

# t-SNE: shrink
s = s.replace(r'\includegraphics[width=0.62\textwidth]{figs/fig5_tsne_family.png}',
              r'\includegraphics[width=0.56\textwidth]{figs/fig5_tsne_family.png}')
reps.append('tsne')

io.open(p, 'w', encoding='utf-8').write(s)
print('done:', reps)