# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\linguistic_agency_paper\main.tex'
s = io.open(p, encoding='utf-8').read()

# 1) shrink single-column figures
for name in ['fig1_dims_vs_gold.png','fig2_boxplot_by_label.png','fig3_intensity_monotonic.png',
             'fig4_dim_corr.png','fig1_full_dims_vs_gold.png','fig2_full_boxplot_by_label.png',
             'fig3_full_intensity_monotonic.png','fig4_full_dim_corr.png']:
    s = s.replace(r'\includegraphics[width=0.92\linewidth]{figs/%s}' % name,
                  r'\includegraphics[width=0.78\linewidth]{figs/%s}' % name)

# 2) shrink double-column figures
s = s.replace(r'\includegraphics[width=0.70\textwidth]{figs/fig5_tsne_family.png}',
              r'\includegraphics[width=0.62\textwidth]{figs/fig5_tsne_family.png}')
s = s.replace(r'\includegraphics[width=0.78\textwidth]{figs/fig6_family_dims_heatmap.png}',
              r'\includegraphics[width=0.68\textwidth]{figs/fig6_family_dims_heatmap.png}')

# 3) large tables: taxonomy / example / cues -> scriptsize
#    (do it by replacing \footnotesize right after those table environments)
import re
# find all \footnotesize occurrences and the nearest caption; switch large ones
count = 0
for m in re.finditer(r'(\\caption\{[^}]{0,120}(?:taxonomy|worked annotation|surface cues)[^}]*\})', s):
    # find the table env start before caption, switch its footnotesize
    start = s.rfind(r'\begin{table', 0, m.start())
    end = s.find(r'\end{table', m.end())
    if start == -1 or end == -1:
        continue
    block = s[start:end]
    if r'\footnotesize' in block:
        s = s[:start] + block.replace(r'\footnotesize', r'\scriptsize', 1) + s[end:]
        count += 1
print('large tables switched:', count)

io.open(p, 'w', encoding='utf-8').write(s)
print('mechanical compression done')