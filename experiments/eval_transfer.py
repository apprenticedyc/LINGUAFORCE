# -*- coding: utf-8 -*-
"""Evaluate zero-shot cross-domain transfer from transfer_dims.jsonl.
Metrics per corpus (source-task standard metric): MentalManip/MultiManip macro-F1,
TalkDown F1, ToxicChat AUROC. Fixed source threshold thr=3 for F1-type; AUROC threshold-free.
Also reports best-threshold F1 for context.
"""
import json, math
from collections import Counter

DIMS = r'E:\PythonCode\Paper\experiments\output\transfer_dims.jsonl'

def auc(scores, labels):
    pos = sorted([s for s, l in zip(scores, labels) if l == 1])
    neg = sorted([s for s, l in zip(scores, labels) if l == 0])
    npos, nneg = len(pos), len(neg)
    if npos == 0 or nneg == 0:
        return float('nan')
    return sum(sum(1 for t in neg if t < s) + 0.5 * sum(1 for t in neg if t == s) for s in pos) / (npos * nneg)

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def corpus_of(did):
    return {'mm': 'MentalManip', 'multi': 'MultiManip', 'td': 'TalkDown', 'tx': 'ToxiChat'}[did.split('-')[0]]

rows = []
with open(DIMS, encoding='utf-8') as f:
    for l in f:
        if l.strip():
            rows.append(json.loads(l))

groups = {}
for r in rows:
    c = corpus_of(r['dialogue_id'])
    groups.setdefault(c, []).append(r)

ORDER = ['MentalManip', 'MultiManip', 'TalkDown', 'ToxiChat']
for c in ORDER:
    g = groups[c]
    y = [r['gold_binary'] for r in g]
    s = [float(r['intensity']) for r in g]
    n = len(g)
    print(f"===== {c} (n={n}, pos={sum(y)}) =====")
    print(f"  AUROC(int) = {auc(s, y):.3f}")
    # fixed threshold 3
    for thr in (3, 4):
        pb = [1 if x >= thr else 0 for x in s]
        tp = sum(1 for gg, pp in zip(y, pb) if gg == 1 and pp == 1)
        fp = sum(1 for gg, pp in zip(y, pb) if gg == 0 and pp == 1)
        fn = sum(1 for gg, pp in zip(y, pb) if gg == 1 and pp == 0)
        tn = n - tp - fp - fn
        p1, r1, f1 = prf(tp, fp, fn)
        # class 0 f1
        p0, r0, f0 = prf(tn, fn, fp)
        mf = (f0 + f1) / 2
        acc = (tp + tn) / n
        print(f"  thr={thr}: F1(1)={f1:.3f} macroF1={mf:.3f} Acc={acc:.3f} (tp={tp},fp={fp},fn={fn})")
    # best threshold 1..5
    best = (0, 0.0)
    for t in range(1, 6):
        pb = [1 if x >= t else 0 for x in s]
        tp = sum(1 for gg, pp in zip(y, pb) if gg == 1 and pp == 1)
        fp = sum(1 for gg, pp in zip(y, pb) if gg == 0 and pp == 1)
        fn = sum(1 for gg, pp in zip(y, pb) if gg == 1 and pp == 0)
        tn = n - tp - fp - fn
        p1, r1, f1 = prf(tp, fp, fn)
        p0, r0, f0 = prf(tn, fn, fp)
        mf = (f0 + f1) / 2
        if c in ('TalkDown',):
            if f1 > best[1]: best = (t, f1)
        else:
            if mf > best[1]: best = (t, mf)
    print(f"  BEST thr={best[0]} metric={best[1]:.3f}")
    print(f"  pred>=3 dist: {Counter(1 if x >= 3 else 0 for x in s)}")
