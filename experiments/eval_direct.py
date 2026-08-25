# -*- coding: utf-8 -*-
"""Evaluate zero-shot direct detection vs gold labels and vs 7-dim conditioned."""
import json, math
from collections import Counter

DIRECT = r'E:\PythonCode\Paper\experiments\output\direct_test.jsonl'

def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0]*n
        i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            avg = (i + j - 1) / 2.0
            for k in range(i, j):
                r[order[k]] = avg
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0.0

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx = math.sqrt(sum((xs[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((ys[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0.0

def qwk(y, p, ncls=6):
    # y,p are lists of ints 0..5
    mat = [[0]*ncls for _ in range(ncls)]
    for a,b in zip(y,p):
        mat[a][b] += 1
    def hist(m):
        return [sum(m[i][j] for i in range(ncls)) for j in range(ncls)]
    def histr(m):
        return [sum(m[i][j] for j in range(ncls)) for i in range(ncls)]
    hr, hc = histr(mat), hist(mat)
    total = sum(hr)
    # weighted kappa (linear)
    oe = sum((hr[i]*hc[j])/total for i in range(ncls) for j in range(ncls))
    denom = sum(abs(i-j) for i in range(ncls) for j in range(ncls))
    e_expected = sum((hr[i]*hc[j]/total)*abs(i-j) for i in range(ncls) for j in range(ncls))
    o_expected = sum(mat[i][j]*abs(i-j) for i in range(ncls) for j in range(ncls))
    if e_expected == 0:
        return 1.0
    return 1 - o_expected/e_expected

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f

def auc(scores, labels):
    pos = sorted([s for s,l in zip(scores,labels) if l==1])
    neg = sorted([s for s,l in zip(scores,labels) if l==0])
    npos, nneg = len(pos), len(neg)
    if npos==0 or nneg==0:
        return float('nan')
    inv = 0.0
    for s in pos:
        inv += sum(1 for t in neg if t < s) + 0.5*sum(1 for t in neg if t==s)
    return inv/(npos*nneg)

rows = []
with open(DIRECT, encoding='utf-8') as f:
    for l in f:
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        rows.append(r)

print(f"n={len(rows)}")
gold_bin = [r['gold_binary'] for r in rows]
gold_multi = [r['gold_multi'] for r in rows]
pred_bin = [int(r['is_moral_coercion']) for r in rows]
pred_int = [int(r['intensity']) for r in rows]
print("gold bin dist:", Counter(gold_bin))
print("pred bin dist:", Counter(pred_bin))

# binary metrics
tp = sum(1 for g,p in zip(gold_bin,pred_bin) if g==1 and p==1)
fp = sum(1 for g,p in zip(gold_bin,pred_bin) if g==0 and p==1)
fn = sum(1 for g,p in zip(gold_bin,pred_bin) if g==1 and p==0)
tn = sum(1 for g,p in zip(gold_bin,pred_bin) if g==0 and p==0)
acc = (tp+tn)/len(rows)
p, r, f = prf(tp, fp, fn)
print(f"\n[Direct binary @0.5] Acc={acc:.3f} P={p:.3f} R={r:.3f} F1={f:.3f} (tp={tp} fp={fp} fn={fn} tn={tn})")

# AUC using intensity as score
a = auc(pred_int, gold_bin)
print(f"[Direct intensity->binary] AUC={a:.3f}")

# best-F1 over thresholds on intensity
best = (0, 0.0)
for t in range(1, 6):
    pb = [1 if x >= t else 0 for x in pred_int]
    tp2 = sum(1 for g,q in zip(gold_bin,pb) if g==1 and q==1)
    fp2 = sum(1 for g,q in zip(gold_bin,pb) if g==0 and q==1)
    fn2 = sum(1 for g,q in zip(gold_bin,pb) if g==1 and q==0)
    _,_,ff = prf(tp2,fp2,fn2)
    aa = (tp2 + (len(rows)-fp2-fn2-tp2))/len(rows)
    print(f"  thr={t}: F1={ff:.3f} Acc={aa:.3f}")
    if ff > best[1]: best = (t, ff)
print(f"  best thr={best[0]} F1={best[1]:.3f}")

# intensity correlation
print(f"\n[Intensity] Spearman={spearman(pred_int, gold_multi):.3f} Pearson={pearson(pred_int, gold_multi):.3f}")
print(f"QWK={qwk(gold_multi, pred_int):.3f}")
exact = sum(1 for g,p in zip(gold_multi,pred_int) if g==p)/len(rows)
plus1 = sum(1 for g,p in zip(gold_multi,pred_int) if abs(g-p)<=1)/len(rows)
print(f"AccExact={exact:.3f} Acc±1={plus1:.3f}")

# gold intensity mean by pred intensity
grp = {}
for g,pr in zip(gold_multi,pred_int):
    grp.setdefault(pr, []).append(g)
print("\ngold mean by pred intensity:", {k: round(sum(v)/len(v),2) for k,v in sorted(grp.items())})

# confusion style: among gold=0 (negative), how many pred positive
neg_ids = [r['dialogue_id'] for r in rows if r['gold_binary']==0]
neg_pos_pred = [r for r in rows if r['gold_binary']==0 and r['is_moral_coercion']==1]
print(f"\nnegatives n={len(neg_ids)}, false-positive rate (pred=1 among gold=0)={len(neg_pos_pred)/max(len(neg_ids),1):.3f}")
