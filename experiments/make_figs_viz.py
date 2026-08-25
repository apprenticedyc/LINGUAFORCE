# -*- coding: utf-8 -*-
"""LINGUAFORCE dimension-space visualization (zero API cost).
1) t-SNE of the 7-dim dialogue profiles colored by strategy family (+ benign).
2) Family x dimension mean-score heatmap.
3) Quantitative separability: 5-fold CV one-vs-rest logistic over 7 dims.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = r'E:\PythonCode\Paper'
FULL = os.path.join(ROOT, 'linguistic_agency_paper', 'data', 'linguaforce_full.jsonl')
TYPES = os.path.join(ROOT, 'experiments', 'output', 'types_full.jsonl')
OUT = os.path.join(ROOT, 'experiments', 'output', 'figs')
os.makedirs(OUT, exist_ok=True)

DIMS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
DIMLAB = ['Directive\nForce', 'Option\nConstraint', 'Normative\nPressure',
          'Emotional\nPressure', 'Deceptiveness', 'Toxicity', 'Explicitness']
FAMILY = {'A': 'A. Transparent', 'B': 'B. Exchange',
          'C': 'C. Social-normative', 'D': 'D. Covert', 'N': 'N. Benign'}
FAM_COLOR = {'A': '#4C72B0', 'B': '#55A868', 'C': '#C44E52', 'D': '#8172B2', 'N': '#B0B0B0'}

rows = [json.loads(l) for l in open(FULL, encoding='utf-8') if l.strip()]
types = [json.loads(l) for l in open(TYPES, encoding='utf-8') if l.strip()]
by_id = {r['dialogue_id']: r for r in rows}
for t in types:
    by_id[t['dialogue_id']]['types'] = t['types']

X = np.array([[r['dims'][d]['score'] for d in DIMS] for r in rows], dtype=float)
main = [r['types'][0][0] if r['types'] else 'N' for r in rows]
Y = np.array([FAMILY[m] for m in main])
cnt = {k: sum(1 for m in main if m == k) for k in 'ABCDN'}
print('n =', len(rows), ' counts:', cnt)

# ---- 1) t-SNE ----
ts = TSNE(n_components=2, perplexity=30, init='pca', random_state=42, n_jobs=1)
Z = ts.fit_transform(X)
fig, ax = plt.subplots(figsize=(7.4, 5.8))
for fam in 'ABCDN':
    m = np.array([x == fam for x in main])
    ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.55, color=FAM_COLOR[fam],
               label=f'{FAMILY[fam]} (n={cnt[fam]})', edgecolors='none')
ax.set_title('Seven-dimension space by strategy family (t-SNE, n=3,432)', fontsize=12)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(markerscale=2.5, fontsize=9, loc='best')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_tsne_family.png'), dpi=200); plt.close(fig)
print('saved fig_tsne_family.png')

# ---- 2) Family x dimension heatmap ----
mat = np.zeros((5, 7))
for i, fam in enumerate('ABCDN'):
    m = np.array([x == fam for x in main])
    mat[i] = X[m].mean(axis=0)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=0.9)
ax.set_xticks(range(7)); ax.set_xticklabels(DIMLAB, fontsize=9)
ax.set_yticks(range(5)); ax.set_yticklabels([FAMILY[f] for f in 'ABCDN'], fontsize=9)
for i in range(5):
    for j in range(7):
        ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=9,
                color='white' if mat[i,j] < 0.55 else 'black')
ax.set_title('Mean dimension score by strategy family', fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_family_dims_heatmap.png'), dpi=200); plt.close(fig)
print('saved fig_family_dims_heatmap.png')
np.set_printoptions(precision=2, suppress=True)
print('family x dim matrix:\n', mat)

# ---- 3) Separability: 5-fold CV logistic (5 classes incl. benign) ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(LogisticRegression(max_iter=3000), X, main, cv=skf)
acc = accuracy_score(main, pred)
mf1 = f1_score(main, pred, average='macro')
print(f'5-class separability (A/B/C/D/N): acc={acc:.4f} macro-F1={mf1:.4f} (chance acc ~0.2)')

# 4-family only (exclude benign)
nb = [i for i, m in enumerate(main) if m != 'N']
Xb = X[nb]; mb = [main[i] for i in nb]
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred2 = cross_val_predict(LogisticRegression(max_iter=3000), Xb, mb, cv=skf2)
print(f'4-family separability: acc={accuracy_score(mb, pred2):.4f} macro-F1={f1_score(mb, pred2, average="macro"):.4f} (chance ~0.25)')

# benign vs manipulative discrimination per dim + full
yb = np.array([0 if m == 'N' else 1 for m in main])
for i, d in enumerate(DIMS):
    print(f'  dim {d} AUC(benign vs manip) = {roc_auc_score(yb, X[:,i]):.3f}')