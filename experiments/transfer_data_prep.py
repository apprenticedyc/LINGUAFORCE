# -*- coding: utf-8 -*-
"""Prepare cross-domain transfer input (unified COERCION-style utterances).
Output: experiments/transfer_data/transfer_input.jsonl
Records: dialogue_id (prefixed), utterances (no speaker prefix), dialog_binary_label, corpus.
"""
import csv, json, os, re, random
from collections import Counter

BASE = r'E:\PythonCode\Paper'
C_DATA = os.path.join(BASE, 'COERCION', 'inputters', 'c_data')
OUT_DIR = os.path.join(BASE, 'experiments', 'transfer_data')
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'transfer_input.jsonl')
SEED = 42

def strip_speaker(text):
    return re.sub(r'^\s*(?:Person\s*[A-Z0-9]+\s*:)\s*', '', text).strip()

recs = []

# ---------- MentalManip (con) ----------
mm_rows = []
with open(os.path.join(C_DATA, 'MentalManip', 'mentalmanip_con.csv'), encoding='utf-8-sig', newline='') as f:
    for r in csv.DictReader(f):
        utts = [strip_speaker(u) for u in r['dialogue'].split('\n') if u.strip()]
        if not utts:
            continue
        mm_rows.append({'did': f"mm-{r['id'].strip()}", 'utts': utts, 'label': int(r['manipulative'])})
print('MentalManip parsed', len(mm_rows), Counter(r['label'] for r in mm_rows))
rng = random.Random(SEED)
pos = [r for r in mm_rows if r['label'] == 1]; neg = [r for r in mm_rows if r['label'] == 0]
rng.shuffle(pos); rng.shuffle(neg)
for r in (pos[:150] + neg[:150]):
    recs.append({'dialogue_id': r['did'], 'utterances': r['utts'],
                 'dialog_binary_label': r['label'], 'corpus': 'MentalManip'})
print('MentalManip sampled 300')

# ---------- MultiManip ----------
m_rows = []
with open(os.path.join(C_DATA, 'MultiManip', 'MultiManip dataset.csv'), encoding='utf-8-sig', newline='') as f:
    for i, r in enumerate(csv.DictReader(f)):
        conv = r['Conversation']
        utts = [strip_speaker(u) for u in conv.split('\n') if u.strip()]
        if not utts:
            continue
        tech = r['Manipulation Technique']
        toks = [t.strip() for t in tech.split(',')]
        label = 1 if any(t and t.lower() != 'non-manip' for t in toks) else 0
        m_rows.append({'did': f"multi-{i}", 'utts': utts, 'label': label})
print('MultiManip parsed', len(m_rows), Counter(r['label'] for r in m_rows))
for r in m_rows:
    recs.append({'dialogue_id': r['did'], 'utterances': r['utts'],
                 'dialog_binary_label': r['label'], 'corpus': 'MultiManip'})

# ---------- TalkDown (balanced_test) ----------
td_rows = []
with open(os.path.join(C_DATA, 'TalkDown', 'balanced_test.jsonl'), encoding='utf-8') as f:
    for i, l in enumerate(f):
        if not l.strip():
            continue
        r = json.loads(l)
        utts = [r['post'].strip(), r['reply'].strip()]
        utts = [u for u in utts if u]
        if not utts:
            continue
        td_rows.append({'did': f"td-{i}", 'utts': utts, 'label': int(bool(r['label']))})
print('TalkDown parsed', len(td_rows), Counter(r['label'] for r in td_rows))
for r in td_rows:
    recs.append({'dialogue_id': r['did'], 'utterances': r['utts'],
                 'dialog_binary_label': r['label'], 'corpus': 'TalkDown'})

# ---------- ToxicChat (test) ----------
tx_rows = []
with open(os.path.join(C_DATA, 'ToxiChat', 'test.jsonl'), encoding='utf-8') as f:
    for i, l in enumerate(f):
        if not l.strip():
            continue
        r = json.loads(l)
        utts = [t.get('text','').strip() for t in r.get('reddit_thread', [])]
        fr = r.get('final_gpt3_response') or {}
        if fr.get('text'):
            utts.append(fr['text'].strip())
        utts = [u for u in utts if u]
        if not utts:
            continue
        label = 1 if fr.get('offense_label') == 'Offensive' else 0
        tx_rows.append({'did': f"tx-{i}", 'utts': utts, 'label': label})
print('ToxiChat parsed', len(tx_rows), Counter(r['label'] for r in tx_rows))
for r in tx_rows:
    recs.append({'dialogue_id': r['did'], 'utterances': r['utts'],
                 'dialog_binary_label': r['label'], 'corpus': 'ToxiChat'})

with open(OUT, 'w', encoding='utf-8') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('TOTAL', len(recs), '->', OUT)
print(Counter(r['corpus'] for r in recs))
