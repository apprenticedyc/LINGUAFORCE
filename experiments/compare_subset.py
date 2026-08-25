# -*- coding: utf-8 -*-
"""Compare 7-dim conditioned vs direct on the SAME 150-sample subset."""
import json, math
from collections import Counter

CLEAN = r'E:\PythonCode\Paper\experiments\output\dims_test_clean.jsonl'
DIRECT = r'E:\PythonCode\Paper\experiments\output\direct_test.jsonl'

def spearman(xs, ys):
    n = len(xs)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0]*n; i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]: j += 1
            avg = (i + j - 1) / 2.0
            for k in range(i, j): r[order[k]] = avg
            i = j
        return r
    rx, ry = rank(xs), rank(ys)
    mx = sum(rx)/n; my = sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0.0

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx = math.sqrt(sum((xs[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((ys[i]-my)**2 for i in range(n)))
    return num/(dx*dy) if dx*dy else 0.0

def qwk(y, p, ncls=6):
    mat = [[0]*ncls for _ in range(ncls)]
    for a,b in zip(y,p): mat[a][b] += 1
    hr = [sum(mat[i][j] for j in range(ncls)) for i in range(ncls)]
    hc = [sum(mat[i][j] for i in range(ncls)) for j in range(ncls)]
    total = sum(hr)
    e_expected = sum((hr[i]*hc[j]/total)*abs(i-j) for i in range(ncls) for j in range(ncls))
    o_expected = sum(mat[i][j]*abs(i-j) for i in range(ncls) for j in range(ncls))
    return 1 - o_expected/e_expected if e_expected else 1.0

def auc(scores, labels):
    pos = sorted([s for s,l in zip(scores,labels) if l==1])
    neg = sorted([s for s,l in zip(scores,labels) if l==0])
    npos, nneg = len(pos), len(neg)
    if npos==0 or nneg==0: return float('nan')
    return sum(sum(1 for t in neg if t < s) + 0.5*sum(1 for t in neg if t==s) for s in pos)/(npos*nneg)

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f

# direct
direct = {}
with open(DIRECT, encoding='utf-8') as f:
    for l in f:
        l = l.strip()
        if not l: continue
        r = json.loads(l); direct[r['dialogue_id']] = r
ids = list(direct.keys())
print(f"subset ids = {len(ids)}")

# 7-dim conditioned from clean
clean = {}
with open(CLEAN, encoding='utf-8') as f:
    for l in f:
        l = l.strip()
        if not l: continue
        r = json.loads(l); clean[r['dialogue_id']] = r

def metrics(pred_int, gold_multi, gold_bin, name):
    print(f"\n===== {name} =====")
    print(f"AUC(int->bin)={auc(pred_int, gold_bin):.3f}")
    best = (0,0.0,0.0)
    for t in range(1,6):
        pb = [1 if x>=t else 0 for x in pred_int]
        tp = sum(1 for g,q in zip(gold_bin,pb) if g==1 and q==1)
        fp = sum(1 for g,q in zip(gold_bin,pb) if g==0 and q==1)
        fn = sum(1 for g,q in zip(gold_bin,pb) if g==1 and q==0)
        _,_,ff = prf(tp,fp,fn)
        aa = (tp + (len(pb)-fp-fn-tp))/len(pb)
        if ff > best[1]: best = (t,ff,aa)
        print(f"  thr={t}: F1={ff:.3f} Acc={aa:.3f}")
    print(f"  best thr={best[0]} F1={best[1]:.3f} Acc={best[2]:.3f}")
    print(f"Spearman={spearman(pred_int, gold_multi):.3f} Pearson={pearson(pred_int, gold_multi):.3f} QWK={qwk(gold_multi, pred_int):.3f}")
    print(f"AccExact={sum(1 for g,p in zip(gold_multi,pred_int) if g==p)/len(pred_int):.3f} Acc±1={sum(1 for g,p in zip(gold_multi,pred_int) if abs(g-p)<=1)/len(pred_int):.3f}")

# direct subset
d_gold_bin = [direct[i]['gold_binary'] for i in ids]
d_gold_multi = [direct[i]['gold_multi'] for i in ids]
d_pred_int = [int(direct[i]['intensity']) for i in ids]
metrics(d_pred_int, d_gold_multi, d_gold_bin, "DIRECT zero-shot (n=150)")

# 7-dim conditioned subset
c_pred_int = [int(clean[i]['intensity']) for i in ids]
metrics(c_pred_int, d_gold_multi, d_gold_bin, "7-DIM conditioned (n=150)")

# check D4 max as alternative conditioned intensity? print D1..D7 AUC on subset
print("\nPer-dim AUC on subset:")
for d in ['D1','D2','D3','D4','D5','D6','D7']:
    sc = [clean[i]['dims'][d]['score'] for i in ids]
    print(f"  {d} AUC={auc(sc, d_gold_bin):.3f}")
