# -*- coding: utf-8 -*-
"""Generate LINGUAFORCE descriptive figures from dims_test_clean.jsonl."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

BASE = r'E:\PythonCode\Paper\experiments'
OUT = os.path.join(BASE, 'output', 'figs')
os.makedirs(OUT, exist_ok=True)

rows = []
with open(os.path.join(BASE, 'output', 'dims_test_clean.jsonl'), encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

DIMS = ['D1','D2','D3','D4','D5','D6','D7']
DIMLAB = {
 'D1':'Directive\nForce','D2':'Option\nConstraint','D3':'Normative\nPressure',
 'D4':'Emotional\nPressure','D5':'Deceptiveness','D6':'Toxicity','D7':'Explicitness'}
gold_b = np.array([r['gold_binary'] for r in rows], dtype=float)
gold_m = np.array([r['gold_multi'] for r in rows], dtype=float)
inten  = np.array([float(r['intensity']) for r in rows])
S = {d: np.array([r['dims'][d]['score'] for r in rows], dtype=float) for d in DIMS}

# ---- Fig 1: dims vs gold correlation heatmap ----
fig, ax = plt.subplots(figsize=(6,4.2))
targets = {'Binary\ncoercion': gold_b, 'Intensity\n0-5': gold_m}
mat = np.zeros((len(DIMS), len(targets)))
for i,d in enumerate(DIMS):
    for j,(name,t) in enumerate(targets.items()):
        mat[i,j] = spearmanr(S[d], t)[0]
im = ax.imshow(mat, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
ax.set_xticks(range(len(targets))); ax.set_xticklabels(list(targets.keys()))
ax.set_yticks(range(len(DIMS))); ax.set_yticklabels([DIMLAB[d] for d in DIMS], fontsize=9)
for i in range(len(DIMS)):
    for j in range(len(targets)):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', fontsize=10,
                color='black')
ax.set_title('Seven dimensions vs. COERCION labels (Spearman)', fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig1_dims_vs_gold.png'), dpi=160); plt.close(fig)

# ---- Fig 2: boxplot positive vs negative per dim ----
fig, axes = plt.subplots(2,4, figsize=(13,6))
axes = axes.flatten()
for k,d in enumerate(DIMS):
    ax = axes[k]
    pos = S[d][gold_b==1]; neg = S[d][gold_b==0]
    ax.boxplot([neg, pos], labels=['neg','pos'], widths=0.6)
    ax.set_title(DIMLAB[d].replace('\n',' '), fontsize=10)
    ax.set_ylim(0,1.05)
    ax.tick_params(labelsize=8)
axes[-1].axis('off')
fig.suptitle('Dimension scores by coercion label (test, n=634)', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig2_boxplot_by_label.png'), dpi=160); plt.close(fig)

# ---- Fig 3: pred intensity vs gold intensity ----
fig, ax = plt.subplots(figsize=(6,4.5))
gold_levels = sorted(set(gold_m.astype(int)))
means = [inten[gold_m.astype(int)==g].mean() for g in gold_levels]
ax.plot(gold_levels, means, 'o-', color='tab:blue')
ax.set_xlabel('COERCION gold intensity (0-5)'); ax.set_ylabel('Mean predicted intensity')
ax.set_xticks(gold_levels)
rho = spearmanr(inten, gold_m)[0]
ax.set_title(f'Predicted vs. gold intensity (Spearman={rho:.3f})', fontsize=12)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig3_intensity_monotonic.png'), dpi=160); plt.close(fig)

# ---- Fig 4: dim x dim correlation heatmap ----
fig, ax = plt.subplots(figsize=(6,5))
M = np.zeros((7,7))
for i,a in enumerate(DIMS):
    for j,b in enumerate(DIMS):
        M[i,j] = spearmanr(S[a], S[b])[0]
im = ax.imshow(M, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xticklabels([DIMLAB[d] for d in DIMS], fontsize=8)
ax.set_yticklabels([DIMLAB[d] for d in DIMS], fontsize=8)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{M[i,j]:.2f}", ha='center', va='center', fontsize=8)
ax.set_title('Inter-dimension correlation (Spearman)', fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig4_dim_corr.png'), dpi=160); plt.close(fig)

print('figures saved to', OUT)
for f in sorted(os.listdir(OUT)):
    print(' -', f, os.path.getsize(os.path.join(OUT,f)))