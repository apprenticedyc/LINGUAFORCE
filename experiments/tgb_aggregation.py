# -*- coding: utf-8 -*-
"""T/G/B aggregation ablation (CPU-only, no API cost).

Aggregation modes over the seven-dimension agentive profile:
  G = global (dialogue-level) profile only        -> 7 dims
  T = turn-level profiles, mean-pooled            -> 7 dims
  B = both, concatenated                          -> 14 dims
Downstream: logistic regression for T1 (binary), linear regression for T3
(intensity). Protocol matches ablation_linear_readout.py: train FULL(3432),
evaluate FIRST(634). Prints a comparison table for the paper.

Run: python experiments/tgb_aggregation.py
"""
import json, os, math
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, f1_score

ROOT = r'E:\PythonCode\Paper'
FULL = os.path.join(ROOT, 'linguistic_agency_paper', 'data', 'linguaforce_full.jsonl')
FIRST = os.path.join(ROOT, 'linguistic_agency_paper', 'data', 'linguaforce_first_release.jsonl')
TURN = os.path.join(ROOT, 'experiments', 'output', 'turn_dims_full.jsonl')
DIMS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def dims_vec(d):
    return [d[k]['score'] for k in DIMS]

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

def qwk(y_true, y_pred):
    # quadratic weighted kappa for ordinal 0-5
    def hist(v):
        h = [0.0]*6
        for x in v: h[int(round(x))] += 1
        return h
    n = len(y_true)
    h1, h2 = hist(y_true), hist([max(0,min(5,int(round(x)))) for x in y_pred])
    num = den = 0.0
    for i in range(6):
        for j in range(6):
            w = (i-j)**2
            o = sum(1 for a,b in zip(y_true,y_pred) if int(round(a))==i and int(round(b))==j)
            e = h1[i]*h2[j]/n
            num += w*o; den += w*e
    return 1.0 - num/den if den else 0.0

def main():
    full = load(FULL); first = load(FIRST); turns = load(TURN)
    # group turn dims by dialogue_id
    tmap = {}
    for r in turns:
        tmap.setdefault(r['dialogue_id'], []).append(dims_vec(r['dims']))
    def t_feats(rec):
        v = tmap.get(rec['dialogue_id'])
        if not v: return None
        return [sum(c)/len(c) for c in zip(*v)]
    def g_feats(rec):
        return dims_vec(rec['dims'])

    # prepare train/test feature sets
    sets = {}
    for name, gf, tf in [('train', g_feats, t_feats), ('test', g_feats, t_feats)]:
        pass
    tr, te = full, first
    Xtr_g = [g_feats(r) for r in tr]
    Xtr_t = [t_feats(r) for r in tr]
    Xte_g = [g_feats(r) for r in te]
    Xte_t = [t_feats(r) for r in te]
    # drop rows missing turn dims (should be none after full annotation)
    def pack(xs, ys):
        keep = [i for i, x in enumerate(xs) if x is not None]
        return [xs[i] for i in keep], [ys[i] for i in keep]
    ytr_b = [r['gold_binary'] for r in tr]; yte_b = [r['gold_binary'] for r in te]
    ytr_i = [r['gold_multi'] for r in tr]; yte_i = [r['gold_multi'] for r in te]
    Xtr_t, ytr_b2 = pack(Xtr_t, ytr_b); _, ytr_i2 = pack(Xtr_t, ytr_i)
    Xte_t, yte_b2 = pack(Xte_t, yte_b); _, yte_i2 = pack(Xte_t, yte_i)

    rows = []
    # G: global only
    lr = LogisticRegression(max_iter=3000).fit(Xtr_g, ytr_b)
    auc_g = roc_auc_score(yte_b, lr.predict_proba(Xte_g)[:, 1])
    reg = LinearRegression().fit(Xtr_g, ytr_i)
    sp_g = spearman(list(reg.predict(Xte_g)), yte_i)
    q_g = qwk(yte_i, list(reg.predict(Xte_g)))
    rows.append(('G (dialogue)', len(Xtr_g[0]), auc_g, sp_g, q_g))

    # T: turn mean
    lr = LogisticRegression(max_iter=3000).fit(Xtr_t, ytr_b2)
    auc_t = roc_auc_score(yte_b2, lr.predict_proba(Xte_t)[:, 1])
    reg = LinearRegression().fit(Xtr_t, ytr_i2)
    sp_t = spearman(list(reg.predict(Xte_t)), yte_i2)
    q_t = qwk(yte_i2, list(reg.predict(Xte_t)))
    rows.append(('T (turn mean)', len(Xtr_t[0]), auc_t, sp_t, q_t))

    # B: concat
    Xb_tr = [a + b for a, b in zip(Xtr_g, Xtr_t)]
    Xb_te = [a + b for a, b in zip(Xte_g, Xte_t)]
    lr = LogisticRegression(max_iter=3000).fit(Xb_tr, ytr_b2)
    auc_b = roc_auc_score(yte_b2, lr.predict_proba(Xb_te)[:, 1])
    reg = LinearRegression().fit(Xb_tr, ytr_i2)
    sp_b = spearman(list(reg.predict(Xb_te)), yte_i2)
    q_b = qwk(yte_i2, list(reg.predict(Xb_te)))
    rows.append(('B (turn+global)', len(Xb_tr[0]), auc_b, sp_b, q_b))

    print(f'{"agg":<18}{"feats":>6}{"T1 AUC":>9}{"T3 Spearman":>13}{"T3 QWK":>9}')
    for name, nf, a, s, q in rows:
        print(f'{name:<18}{nf:>6}{a:>9.4f}{s:>13.4f}{q:>9.4f}')
    print(f'\nbaseline (direct LLM judge, from verify_paper.py): T1 AUC 0.831 / T3 Spearman 0.665')

if __name__ == '__main__':
    main()