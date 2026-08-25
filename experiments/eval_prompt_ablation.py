# -*- coding: utf-8 -*-
"""Prompt ablation evaluation: zero-shot vs CoT vs self-reflection.
T1: AUC + best F1 over intensity threshold 1..5; T3: Spearman/QWK.
Zero-shot = existing dialogue-level parser output (dims_test_clean.jsonl).
"""
import json, os, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_paper as vp

BASE = r'E:\PythonCode\Paper\experiments\output'
ZERO = os.path.join(BASE, 'dims_test_clean.jsonl')
COT = os.path.join(BASE, 'prompt_cot_v2.jsonl')
SELF = os.path.join(BASE, 'prompt_selfref_v2.jsonl')

def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def metrics(recs, name):
    gb = [r['gold_binary'] for r in recs]
    gm = [r['gold_multi'] for r in recs]
    pred = [float(r['intensity']) for r in recs]
    a = vp.auc(pred, gb)
    best = (0, 0)
    for thr in range(1, 6):
        tp = sum(1 for p, g in zip(pred, gb) if p >= thr and g == 1)
        fp = sum(1 for p, g in zip(pred, gb) if p >= thr and g == 0)
        fn = sum(1 for p, g in zip(pred, gb) if p < thr and g == 1)
        f = vp.prf(tp, fp, fn)
        if f > best[1]:
            best = (thr, f)
    sp = vp.spearman(pred, gm)
    q = vp.qwk(gm, [int(round(x)) for x in pred])
    print(f'{name:<14} n={len(recs)} T1 AUC={a:.4f} bestF1={best[1]:.4f}@thr{best[0]} | T3 Spearman={sp:.4f} QWK={q:.4f}')
    return dict(name=name, auc=a, f1=best[1], thr=best[0], sp=sp, qwk=q)

zero = load(ZERO)
# zero file may lack gold fields? check and join by id
first = vp.load(os.path.join(os.path.dirname(BASE), '..', 'linguistic_agency_paper', 'data', 'linguaforce_first_release.jsonl'))
gmap = {r['dialogue_id']: r for r in first}
for r in zero:
    g = gmap.get(r['dialogue_id'])
    if g:
        r.setdefault('gold_binary', g['gold_binary'])
        r.setdefault('gold_multi', g['gold_multi'])
zero = [r for r in zero if 'gold_binary' in r]
print('== Prompt ablation on FIRST(634) ==')
metrics(zero, 'zero-shot')
metrics(load(COT), 'CoT')
metrics(load(SELF), 'self-reflection')