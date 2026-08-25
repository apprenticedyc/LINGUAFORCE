# -*- coding: utf-8 -*-
"""Dimension-ablation study with a lightweight linear readout (no LLM calls).

Trains logistic regression (binary) / linear regression (intensity) on the
seven dimension scores from the FULL release and evaluates on the HELD-OUT
(634) split. Reports leave-one-dimension-out results. This closes the
"ablation deferred to future work" gap in the paper at zero API cost.

Run:  python experiments/ablation_linear_readout.py
"""
import json, math, os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FULL = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_full.jsonl")
FIRST = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_first_release.jsonl")
DIMS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def feats(rec):
    return [rec["dims"][d]["score"] for d in DIMS]

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

def main():
    train = load(FULL); test = load(FIRST)
    X = [feats(r) for r in train]; Xt = [feats(r) for r in test]
    yb = [r["gold_binary"] for r in train]; ybt = [r["gold_binary"] for r in test]
    yi = [r["gold_multi"] for r in train]; yit = [r["gold_multi"] for r in test]

    lr = LogisticRegression(max_iter=2000); lr.fit(X, yb)
    a0 = roc_auc_score(ybt, lr.predict_proba(Xt)[:, 1])
    reg = LinearRegression(); reg.fit(X, yi)
    s0 = spearman(list(reg.predict(Xt)), yit)
    print("full-7dims  binary AUC=%.4f  intensity Spearman=%.4f" % (a0, s0))

    for idx, d in enumerate(DIMS):
        keep = [k for k in range(7) if k != idx]
        Xr = [[x[k] for k in keep] for x in X]; Xrt = [[x[k] for k in keep] for x in Xt]
        lr2 = LogisticRegression(max_iter=2000); lr2.fit(Xr, yb)
        a = roc_auc_score(ybt, lr2.predict_proba(Xrt)[:, 1])
        reg2 = LinearRegression(); reg2.fit(Xr, yi)
        s = spearman(list(reg2.predict(Xrt)), yit)
        print(f"  LO-{d}: AUC={a:.4f} (d{a0-a:+.4f})  Spearman={s:.4f} (d{s0-s:+.4f})")

    print("LLM readout on the same 634 split (from verify_paper.py): AUC 0.831 / Spearman 0.665")

if __name__ == "__main__":
    main()
