# -*- coding: utf-8 -*-
"""Regenerate all LINGUAFORCE figures from the released dataset files.

Reads linguistic_agency_paper/data/linguaforce_{full,first_release}.jsonl and
writes figures into linguistic_agency_paper/figs/ (matching main.tex paths).
Run:  python experiments/make_figs_release.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "linguistic_agency_paper", "data")
FIG = os.path.join(ROOT, "linguistic_agency_paper", "figs")
os.makedirs(FIG, exist_ok=True)

DIMS = ["D1","D2","D3","D4","D5","D6","D7"]
DIMLAB = {"D1":"Directive\nForce","D2":"Option\nConstraint","D3":"Normative\nPressure",
          "D4":"Emotional\nPressure","D5":"Deceptiveness","D6":"Toxicity","D7":"Explicitness"}

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def render(rows, tag, title_suffix):
    gold_b = np.array([r["gold_binary"] for r in rows], dtype=float)
    gold_m = np.array([r["gold_multi"] for r in rows], dtype=float)
    inten  = np.array([float(r["intensity"]) for r in rows])
    S = {d: np.array([r["dims"][d]["score"] for r in rows], dtype=float) for d in DIMS}
    suf = "" if tag == "first" else "_full"

    # Fig 1: dims vs gold correlation
    fig, ax = plt.subplots(figsize=(6,4.2))
    targets = {"Binary\ncoercion": gold_b, "Intensity\n0-5": gold_m}
    mat = np.zeros((len(DIMS), len(targets)))
    for i,d in enumerate(DIMS):
        for j,(name,t) in enumerate(targets.items()):
            mat[i,j] = spearmanr(S[d], t)[0]
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    ax.set_xticks(range(len(targets))); ax.set_xticklabels(list(targets.keys()))
    ax.set_yticks(range(len(DIMS))); ax.set_yticklabels([DIMLAB[d] for d in DIMS], fontsize=9)
    for i in range(len(DIMS)):
        for j in range(len(targets)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title(f"Seven dimensions vs. coercion labels (Spearman){title_suffix}", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig1_dims_vs_gold{suf}.png"), dpi=160); plt.close(fig)

    # Fig 2: boxplot pos vs neg
    fig, axes = plt.subplots(2,4, figsize=(13,6)); axes = axes.flatten()
    for k,d in enumerate(DIMS):
        ax = axes[k]
        pos = S[d][gold_b==1]; neg = S[d][gold_b==0]
        ax.boxplot([neg, pos], labels=["neg","pos"], widths=0.6)
        ax.set_title(DIMLAB[d].replace("\n"," "), fontsize=10)
        ax.set_ylim(0,1.05); ax.tick_params(labelsize=8)
    axes[-1].axis("off")
    fig.suptitle(f"Dimension scores by coercion label ({title_suffix.strip()})", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig2_boxplot_by_label{suf}.png"), dpi=160); plt.close(fig)

    # Fig 3: pred vs gold intensity
    fig, ax = plt.subplots(figsize=(6,4.5))
    gold_levels = sorted(set(gold_m.astype(int)))
    means = [inten[gold_m.astype(int)==g].mean() for g in gold_levels]
    ax.plot(gold_levels, means, "o-", color="tab:blue")
    ax.set_xlabel("Coercion gold intensity (0-5)"); ax.set_ylabel("Mean predicted intensity")
    ax.set_xticks(gold_levels)
    rho = spearmanr(inten, gold_m)[0]
    ax.set_title(f"Predicted vs. gold intensity (Spearman={rho:.3f}){title_suffix}", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig3_intensity_monotonic{suf}.png"), dpi=160); plt.close(fig)

    # Fig 4: dim x dim correlation
    fig, ax = plt.subplots(figsize=(6,5))
    M = np.zeros((7,7))
    for i,a in enumerate(DIMS):
        for j,b in enumerate(DIMS):
            M[i,j] = spearmanr(S[a], S[b])[0]
    im = ax.imshow(M, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels([DIMLAB[d] for d in DIMS], fontsize=8)
    ax.set_yticklabels([DIMLAB[d] for d in DIMS], fontsize=8)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Inter-dimension correlation (Spearman){title_suffix}", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig4_dim_corr{suf}.png"), dpi=160); plt.close(fig)
    print(f"[{tag}] figures written to {FIG}")

if __name__ == "__main__":
    render(load(os.path.join(DATA, "linguaforce_first_release.jsonl")), "first", " (first release, n=634)")
    render(load(os.path.join(DATA, "linguaforce_full.jsonl")), "full", " (full release, n=3432)")
    print("done")
