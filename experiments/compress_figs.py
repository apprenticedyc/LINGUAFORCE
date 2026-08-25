# -*- coding: utf-8 -*-
import io, re
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()
drop = ['fig1_dims_vs_gold.png', 'fig2_boxplot_by_label.png',
        'fig3_intensity_monotonic.png', 'fig4_dim_corr.png']

# collect non-star figure environment boundaries in order
begins = [m.start() for m in re.finditer(r'\\begin\{figure\}(?!\*)', s)]
ends   = [m.start() for m in re.finditer(r'\\end\{figure\}(?!\*)', s)]
print('begin count:', len(begins), 'end count:', len(ends))
assert len(begins) == len(ends), 'figure env mismatch'

remove = []
for bi, ei in zip(begins, ends):
    block = s[bi:ei + len(r'\end{figure}')]
    if any(d in block for d in drop):
        remove.append((bi, ei + len(r'\end{figure}')))

# remove from end to start
for bi, ei in reversed(remove):
    s = s[:bi] + s[ei:]
print('removed figure blocks:', len(remove))

for old, new in [(r'\ref{fig:dim-gold}', r'\ref{fig:dim-gold-full}'),
                 (r'\ref{fig:box}', r'\ref{fig:box-full}'),
                 (r'\ref{fig:mono}', r'\ref{fig:mono-full}'),
                 (r'\ref{fig:dim-corr}', r'\ref{fig:dim-corr-full}')]:
    n = s.count(old)
    s = s.replace(old, new)
    print('rename %r x%d' % (old, n))

for kw in ['Quality Control', 'First Release', 'Statistical Analyses', 'fig:tree', 'Annotation Decision Tree']:
    print('keep %-28s count=%d' % (kw, s.count(kw)))
io.open(p, 'w', encoding='utf-8').write(s)
print('done len', len(s))