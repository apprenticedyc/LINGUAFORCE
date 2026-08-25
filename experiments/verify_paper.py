# -*- coding: utf-8 -*-
"""Recompute every number claimed in main.tex from the released dataset.

Run:  python experiments/verify_paper.py
All metric logic mirrors the paper exactly:
  - QWK uses squared weights (i-j)^2  (true quadratic weighted kappa)
  - T1: AUC over predicted intensity; best F1 over threshold 1..5
  - T3: Spearman/Pearson/QWK/acc-within-+-1/exact on predicted intensity
Cross-domain numbers are read from experiments/output/transfer_dims.jsonl.
"""
import json, math, os
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FULL = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_full.jsonl")
FIRST = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_first_release.jsonl")
TRANSFER = os.path.join(ROOT, "experiments", "output", "transfer_dims.jsonl")

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i]); r = [0.0]*n; i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]: j += 1
            avg = (i + j - 1) / 2.0
            for k in range(i, j): r[order[k]] = avg
            i = j
        return r
    rx, ry = rank(xs), rank(ys); mx = sum(rx)/n; my = sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = math.sqrt(sum((rx[i]-mx)**2 for i in range(n))); dy = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0.0

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx = math.sqrt(sum((xs[i]-mx)**2 for i in range(n))); dy = math.sqrt(sum((ys[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0.0

def qwk(y, p, ncls=6):
    mat = [[0]*ncls for _ in range(ncls)]
    for a, b in zip(y, p): mat[a][b] += 1
    hr = [sum(mat[i][j] for j in range(ncls)) for i in range(ncls)]
    hc = [sum(mat[i][j] for i in range(ncls)) for j in range(ncls)]
    total = sum(hr)
    e = sum((hr[i]*hc[j]/total)*((i-j)**2) for i in range(ncls) for j in range(ncls))
    o = sum(mat[i][j]*((i-j)**2) for i in range(ncls) for j in range(ncls))
    return 1 - o/e if e else 1.0

def auc(scores, labels):
    pos = sorted([s for s, l in zip(scores, labels) if l == 1])
    neg = sorted([s for s, l in zip(scores, labels) if l == 0])
    np_, ng = len(pos), len(neg)
    if np_ == 0 or ng == 0: return float("nan")
    return sum(sum(1 for t in neg if t < s) + 0.5*sum(1 for t in neg if t == s) for s in pos)/(np_*ng)

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return (2*p*r/(p+r) if p+r else 0.0)

def r3(x):
    """round to 3 decimals, half-up (0.2165 -> 0.217, 0.6915 -> 0.692)."""
    return round(x + 1e-9, 3)


def report(recs, name):
    gb = [r["gold_binary"] for r in recs]
    gm = [r["gold_multi"] for r in recs]
    pi = [int(r["intensity"]) for r in recs]
    n = len(recs)
    a = auc(pi, gb)
    best = (0, 0.0, 0.0)
    for t in range(1, 6):
        pb = [1 if x >= t else 0 for x in pi]
        tp = sum(1 for g, q in zip(gb, pb) if g == 1 and q == 1)
        fp = sum(1 for g, q in zip(gb, pb) if g == 0 and q == 1)
        fn = sum(1 for g, q in zip(gb, pb) if g == 1 and q == 0)
        ff = prf(tp, fp, fn); aa = (tp + (n-fp-fn-tp))/n
        if ff > best[1]: best = (t, ff, aa)
    print(f"== {name} n={n} ==")
    print(f"  pos={sum(gb)} neg={n-sum(gb)}  gold_multi={dict(sorted(Counter(gm).items()))}")
    print(f"  T1  AUC={r3(a):.3f}  bestF1={r3(best[1]):.3f}@thr{best[0]}  acc={r3(best[2]):.3f}")
    print(f"  T3  Spearman={r3(spearman(pi, gm)):.3f} Pearson={r3(pearson(pi, gm)):.3f} "
          f"QWK={r3(qwk(gm, pi)):.3f} +-1={r3(sum(1 for g,p in zip(gm,pi) if abs(g-p)<=1)/n):.3f} "
          f"exact={r3(sum(1 for g,p in zip(gm,pi) if g==p)/n):.3f}")
    print("  per-dim (mean, rho_bin, rho_multi, AUC):")
    for k in ["D1","D2","D3","D4","D5","D6","D7"]:
        sc = [r["dims"][k]["score"] for r in recs]
        print(f"    {k}: {r3(sum(sc)/n):.3f} {r3(spearman(sc,gb)):.3f} {r3(spearman(sc,gm)):.3f} {r3(auc(sc,gb)):.3f}")

def cross_domain():
    rows = [r for r in load(TRANSFER)]
    print("== Cross-domain (AUROC / std metric @ thr=3) ==")
    for prefix, name, kind in [("mm","MentalManip","macro-F1"),("multi","MultiManip","macro-F1"),
                               ("td","TalkDown","F1"),("tx","ToxiChat","AUROC")]:
        g = [r for r in rows if r["dialogue_id"].split("-")[0] == prefix]
        y = [r["gold_binary"] for r in g]; s = [float(r["intensity"]) for r in g]
        pb = [1 if x >= 3 else 0 for x in s]
        tp = sum(1 for gg, pp in zip(y, pb) if gg == 1 and pp == 1)
        fp = sum(1 for gg, pp in zip(y, pb) if gg == 0 and pp == 1)
        fn = sum(1 for gg, pp in zip(y, pb) if gg == 1 and pp == 0)
        tn = len(g) - tp - fp - fn
        f1 = prf(tp, fp, fn)
        if kind == "macro-F1":
            f0 = prf(tn, fn, fp); std = (f0 + f1)/2
        elif kind == "AUROC":
            std = auc(s, y)
        else:
            std = f1
        print(f"  {name} n={len(g)}: AUROC={r3(auc(s,y)):.3f} {kind}={r3(std):.3f}")

if __name__ == "__main__":
    report(load(FULL), "FULL release")
    report(load(FIRST), "FIRST release")
    cross_domain()
