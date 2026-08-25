# -*- coding: utf-8 -*-
"""Compute human spot-check agreement for LINGUAFORCE (7-dim focus).

Inputs (in linguistic_agency_paper/data/spotcheck/):
  spotcheck_annotate.xlsx   - annotator fills cols D/E/F:
      D = overall agency strength 0-5
      E = primary agency dimension D1-D7
      F = noticeable pressure present? 0/1
  spotcheck_reference.csv   - LLM 7-dim scores + derived aggregates (llm_agg_agency,
                              llm_argmax_dim, llm_presence = max score >= 0.5)

Agreement vs the NEW dimension annotations (not the old coercion labels):
  - D vs llm_agg_agency (max D-score * 5): Spearman / Pearson / quadratic QWK / +-1 / exact
  - E vs llm_argmax_dim: exact-match rate, top-2 rate
  - F vs llm_presence: Cohen's kappa + % agreement
Blank rows are skipped.
"""
import csv, math, os
import openpyxl

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SPOT = os.path.join(ROOT, "linguistic_agency_paper", "data", "spotcheck")
ANNOT = os.path.join(SPOT, "spotcheck_annotate.xlsx")
REF = os.path.join(SPOT, "spotcheck_reference.csv")

def spearman(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
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
    for a, b in zip(y, p): mat[int(a)][int(b)] += 1
    hr = [sum(mat[i][j] for j in range(ncls)) for i in range(ncls)]
    hc = [sum(mat[i][j] for i in range(ncls)) for j in range(ncls)]
    total = sum(hr)
    e = sum((hr[i]*hc[j]/total)*((i-j)**2) for i in range(ncls) for j in range(ncls))
    o = sum(mat[i][j]*((i-j)**2) for i in range(ncls) for j in range(ncls))
    return 1 - o/e if e else 1.0

def cohen_kappa(a, b):
    n = len(a)
    classes = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(classes)}
    mat = [[0]*len(classes) for _ in classes]
    for x, y in zip(a, b): mat[idx[x]][idx[y]] += 1
    row = [sum(r) for r in mat]; col = [sum(mat[i][j] for i in range(len(classes))) for j in range(len(classes))]
    po = sum(mat[i][i] for i in range(len(classes)))/n
    pe = sum(row[i]*col[i] for i in range(len(classes)))/(n*n)
    return (po-pe)/(1-pe) if pe != 1 else 1.0, po

def load_reference():
    ref = {}
    with open(REF, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ref[int(row["dialogue_id"])] = row
    return ref

def load_annotations():
    wb = openpyxl.load_workbook(ANNOT)
    ws = wb.active
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if len(r) < 8: continue
        seq, did, content, agen, prim, pres, second, note = r[:8]
        if did is None: continue
        rows.append({"did": int(did), "agen": agen, "prim": prim, "pres": pres,
                     "second": (second or "").strip()})
    return rows

def main():
    ref = load_reference()
    anns = load_annotations()
    used = [a for a in anns if a["agen"] not in (None, "") or a["prim"] not in (None, "")
            or a["pres"] not in (None, "")]
    if not used:
        print("No annotations filled yet. Open the xlsx, fill cols D/E/F, save, then rerun.")
        return

    agen_a, agen_b = [], []
    prim_a, prim_b = [], []
    pres_a, pres_b = [], []
    for a in used:
        row = ref.get(a["did"])
        if row is None: continue
        if a["agen"] not in (None, ""):
            agen_a.append(float(a["agen"])); agen_b.append(float(row["llm_agg_agency"]))
        if a["prim"] not in (None, "") and str(a["prim"]).strip().upper().startswith("D"):
            prim_a.append(str(a["prim"]).strip().upper()); prim_b.append(str(row["llm_argmax_dim"]).strip().upper())
        if a["pres"] not in (None, ""):
            pres_a.append(int(float(a["pres"]))); pres_b.append(int(row["llm_presence"]))

    print(f"filled: {len(used)} / {len(anns)} rows")
    if len(agen_a) >= 2:
        p1 = sum(1 for x, y in zip(agen_a, agen_b) if abs(x-y) <= 1)/len(agen_a)
        ex = sum(1 for x, y in zip(agen_a, agen_b) if x == y)/len(agen_a)
        print(f"overall agency (human vs LLM aggregate): n={len(agen_a)}  "
              f"Spearman={spearman(agen_a,agen_b):.3f}  Pearson={pearson(agen_a,agen_b):.3f}  "
              f"QWK={qwk([int(i) for i in agen_a],[int(i) for i in agen_b]):.3f}  +-1={p1:.3f}  exact={ex:.3f}")
    if prim_a:
        ex = sum(1 for x, y in zip(prim_a, prim_b) if x == y)/len(prim_a)
        print(f"primary dimension (human vs LLM argmax): n={len(prim_a)}  exact-match={ex:.3f}")
    if pres_a:
        k, po = cohen_kappa(pres_a, pres_b)
        print(f"presence binary (human vs LLM max>=0.5): n={len(pres_a)}  Cohen's kappa={k:.3f}  agreement={po:.3f}")

if __name__ == "__main__":
    main()
