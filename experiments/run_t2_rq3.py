# -*- coding: utf-8 -*-
"""T2 (15-type multi-label recognition) + RQ3 (leave-one-type-out).

Type labels are LLM-annotated for the full release (3,432) and the held-out
split (634). A lightweight one-vs-rest logistic readout over the seven
dimension scores is used (no LLM calls in the readout).

T2-A: train on FULL 3,432 -> test on held-out 634 (macro/micro F1, 15-type and family).
T2-B: five-fold cross-validation on the full 3,432.
RQ3: (a) leave-one-type-out novelty AUROC (held-out type never in training);
     (b) predicted-intensity separation of "any strategy present" (AUROC).

Run:  python experiments/run_t2_rq3.py
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FULL = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_full.jsonl")
FIRST = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_first_release.jsonl")
TFULL = os.path.join(ROOT, "experiments", "output", "types_full.jsonl")
T634 = os.path.join(ROOT, "experiments", "output", "types_634.jsonl")
DIMS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
TYPE_LIST = ["A1","A2","A3","B1","B2","C1","C2","C3","C4","C5","C6","D1","D2","D3","D4"]
FAMS = {"A":["A1","A2","A3"],"B":["B1","B2"],"C":["C1","C2","C3","C4","C5","C6"],"D":["D1","D2","D3","D4"]}

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def build(rows, typ):
    X = np.array([[r["dims"][d]["score"] for d in DIMS] for r in rows])
    Y15 = np.array([[1 if t in typ[r["dialogue_id"]] else 0 for t in TYPE_LIST] for r in rows])
    Yfam = np.array([[1 if any(t in FAMS[f] for t in typ[r["dialogue_id"]]) else 0 for f in "ABCD"] for r in rows])
    return X, Y15, Yfam

def macro_micro_f1(y_true, y_pred):
    f = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[2]
    macro = float(np.mean([fi for fi, s in zip(f, y_true.sum(axis=0)) if s > 0]))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    micro = (2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0
    return macro, micro

def best_f1(Y, P):
    best = None
    for t in [0.3, 0.4, 0.5, 0.6]:
        mac, mic = macro_micro_f1(Y, (P >= t).astype(int))
        if best is None or (mac + mic) > (best[0] + best[1]):
            best = (mac, mic, t)
    return best

def main():
    rows_full = load(FULL); rows_634 = load(FIRST)
    tf = {r["dialogue_id"]: r["types"] for r in load(TFULL)}
    t6 = {r["dialogue_id"]: r["types"] for r in load(T634)}
    Xf, Yf15, Yff = build(rows_full, tf)
    X6, Y615, Y6f = build(rows_634, t6)

    print("== T2-A: train full(3432) -> test held-out(634) ==")
    for name, Ytr, Yte in [("15 types", Yf15, Y615), ("4 families", Yff, Y6f)]:
        P = np.zeros((len(X6), Ytr.shape[1]))
        for j in range(Ytr.shape[1]):
            if Ytr[:, j].sum() == 0:
                continue
            c = LogisticRegression(max_iter=2000).fit(Xf, Ytr[:, j])
            P[:, j] = c.predict_proba(X6)[:, 1]
        mac, mic, thr = best_f1(Yte, P)
        print(f"  {name}: macro-F1={mac:.3f} micro-F1={mic:.3f} @thr={thr:.1f}")

    print("== T2-B: 5-fold CV on full(3432) ==")
    for name, Y in [("15 types", Yf15), ("4 families", Yff)]:
        strata = np.array([1 if y.sum() > 0 else 0 for y in Y])
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        P = np.zeros_like(Y, dtype=float)
        for tr, te in skf.split(Xf, strata):
            for j in range(Y.shape[1]):
                if Y[tr, j].sum() == 0:
                    continue
                c = LogisticRegression(max_iter=2000).fit(Xf[tr], Y[tr, j])
                P[te, j] = c.predict_proba(Xf[te])[:, 1]
        mac, mic, thr = best_f1(Y, P)
        print(f"  {name}: macro-F1={mac:.3f} micro-F1={mic:.3f} @thr={thr:.1f}")

    print("== RQ3 (leave-one-type-out, held-out split) ==")
    vals = []
    for li, t in enumerate(TYPE_LIST):
        n_pos = int(Y615[:, li].sum())
        if n_pos < 30:
            continue
        tr = Y615[:, li] == 0
        score = np.zeros(len(X6))
        for ui in range(Y615.shape[1]):
            if ui == li or Y615[tr, ui].sum() == 0:
                continue
            c = LogisticRegression(max_iter=2000).fit(X6[tr], Y615[tr, ui])
            score = np.maximum(score, c.predict_proba(X6)[:, 1])
        vals.append(roc_auc_score(Y615[:, li], score))
    print(f"  novelty AUROC (types>=30): mean={sum(vals)/len(vals):.3f} (n={len(vals)})")
    inten = np.array([r["intensity"] for r in rows_634])
    has = np.array([1 if t6[r["dialogue_id"]] else 0 for r in rows_634])
    print(f"  predicted-intensity separates 'any strategy present': AUROC={roc_auc_score(has, inten):.3f}")

if __name__ == "__main__":
    main()
