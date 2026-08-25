# -*- coding: utf-8 -*-
"""Package the released LINGUAFORCE dataset from source + annotation outputs.

Outputs (into linguistic_agency_paper/data/):
  linguaforce_full.jsonl          -- 3,432 dialogues, train partition
  linguaforce_first_release.jsonl --   634 dialogues, held-out partition

Each record: dialogue_id, utterances (list of str),
gold_binary (0/1), gold_multi (0-5), intensity (predicted 0-5),
dims = {D1..D7: {score: 0-1 float, level: 0-3 int}}.
"""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_DATA = os.path.join(ROOT, "linguistic_agency_paper", "data")
COERCION_TRAIN = os.path.join(ROOT, "COERCION", "inputters", "data", "train.jsonl")
COERCION_TEST = os.path.join(ROOT, "COERCION", "inputters", "data", "test.jsonl")
DIMS_FULL = os.path.join(ROOT, "experiments", "output", "train_dims.jsonl")
DIMS_FIRST = os.path.join(ROOT, "experiments", "output", "dims_test_clean.jsonl")

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def merge(text_records, dim_records, tag):
    dmap = {r["dialogue_id"]: r for r in dim_records}
    out = []
    for t in text_records:
        d = dmap[t["dialogue_id"]]
        if d["gold_binary"] != t["dialog_binary_label"] or d["gold_multi"] != t["dialog_multi_label"]:
            raise SystemExit(f"[FATAL] gold mismatch id={t['dialogue_id']}")
        out.append({
            "dialogue_id": t["dialogue_id"],
            "utterances": t["utterances"],
            "gold_binary": t["dialog_binary_label"],
            "gold_multi": t["dialog_multi_label"],
            "intensity": d["intensity"],
            "dims": d["dims"],
        })
    ids = [r["dialogue_id"] for r in out]
    assert len(ids) == len(set(ids)), tag + " duplicate ids"
    print(f"[{tag}] merged={len(out)}")
    return out

def write(recs, name):
    os.makedirs(PAPER_DATA, exist_ok=True)
    path = os.path.join(PAPER_DATA, name)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] {name}: {len(recs)} records -> {path}")

if __name__ == "__main__":
    write(merge(load(COERCION_TRAIN), load(DIMS_FULL), "FULL"), "linguaforce_full.jsonl")
    write(merge(load(COERCION_TEST), load(DIMS_FIRST), "FIRST"), "linguaforce_first_release.jsonl")
