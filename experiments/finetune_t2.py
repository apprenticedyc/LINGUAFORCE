# -*- coding: utf-8 -*-
"""T2 supervised fine-tuning baseline (RoBERTa-base, multi-label 15-type).

Protocols (mirror the linear-readout results in run_t2_rq3.py):
  heldout: train on full release (n=3,432) -> eval on type-labeled held-out (n=634)
  cv5:     5-fold cross-validation over the full release

Reports macro/micro F1 at the best threshold over {0.3,0.4,0.5,0.6} for both
15 types and 4 families. Run with the pytorch2.3.1 env (torch+CUDA+transformers).

Usage:
  set HF_ENDPOINT=https://hf-mirror.com
  python experiments/finetune_t2.py --mode heldout --epochs 3
  python experiments/finetune_t2.py --mode cv5 --epochs 3
"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (RobertaForSequenceClassification, RobertaTokenizer,
                          AdamW, get_linear_schedule_with_warmup)
from sklearn.metrics import precision_recall_fscore_support

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FULL = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_full.jsonl")
FIRST = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_first_release.jsonl")
TFULL = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_full_types.jsonl")
T634 = os.path.join(ROOT, "linguistic_agency_paper", "data", "linguaforce_first_release_types.jsonl")
OUT = os.path.join(ROOT, "experiments", "output")
TYPE_LIST = ["A1","A2","A3","B1","B2","C1","C2","C3","C4","C5","C6","D1","D2","D3","D4"]
FAMS = {"A":["A1","A2","A3"],"B":["B1","B2"],"C":["C1","C2","C3","C4","C5","C6"],"D":["D1","D2","D3","D4"]}
FAM_ORDER = ["A","B","C","D"]
MAX_LEN = 512
THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def log_progress(path, msg):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
            f.flush()
    except Exception as e:
        print("WARN progress write failed: %s" % e, flush=True)


def text_of(row):
    return " ".join(row["utterances"])


def multi_hot(types):
    y = np.zeros(len(TYPE_LIST), dtype=np.int64)
    for t in types:
        if t in TYPE_LIST:
            y[TYPE_LIST.index(t)] = 1
    return y


class T2Dataset(Dataset):
    def __init__(self, rows, type_map, tokenizer):
        self.x = []
        self.y = []
        for r in rows:
            self.x.append(text_of(r))
            self.y.append(multi_hot(type_map[r["dialogue_id"]]))
        self.tok = tokenizer
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        enc = self.tok(self.x[i], max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(self.y[i], dtype=torch.float),
        }


def macro_micro_f1(y_true, y_pred):
    f = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[2]
    macro = float(np.mean([fi for fi, s in zip(f, y_true.sum(axis=0)) if s > 0]))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    micro = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return macro, micro


def family_matrix(Y15):
    n = Y15.shape[0]
    Yf = np.zeros((n, len(FAM_ORDER)), dtype=np.int64)
    for fi, fam in enumerate(FAM_ORDER):
        cols = [TYPE_LIST.index(t) for t in FAMS[fam]]
        Yf[:, fi] = (Y15[:, cols].sum(axis=1) > 0).astype(np.int64)
    return Yf


def best_f1(Y, P):
    best = None
    for t in THRESHOLDS:
        mac, mic = macro_micro_f1(Y, (P >= t).astype(int))
        if best is None or (mac + mic) > (best[0] + best[1]):
            best = (mac, mic, t)
    return best


def predict(model, loader, device):
    model.eval()
    Ps = []
    with torch.no_grad():
        for b in loader:
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            Ps.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.vstack(Ps)


def run_training(train_rows, train_types, test_rows, test_types, tokenizer, args, tag=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress = getattr(args, "progress", None)
    step_interval = getattr(args, "step_interval", 200)
    if progress:
        log_progress(progress, "[%s] start train=%d test=%d device=%s fp16=%s" % (tag, len(train_rows), len(test_rows), device, bool(args.fp16)))
    train_ds = T2Dataset(train_rows, train_types, tokenizer)
    test_ds = T2Dataset(test_rows, test_types, tokenizer)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(TYPE_LIST))
    model.to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    warmup = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    model.train()
    global_step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        epoch_loss = 0.0
        for b in loader:
            inp = {"input_ids": b["input_ids"].to(device),
                   "attention_mask": b["attention_mask"].to(device)}
            labels = b["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(**inp).logits
                loss = bce(logits.float(), labels) / args.grad_accum
            scaler.scale(loss).backward()
            if (global_step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            epoch_loss += loss.item() * args.grad_accum
            global_step += 1
            if args.verbose and global_step % 200 == 0:
                el = time.time() - t0
                print(f"  [{tag}] step {global_step}/{total_steps} loss {loss.item()*args.grad_accum:.4f} ({el:.0f}s)", flush=True)
            if progress and global_step % step_interval == 0:
                log_progress(progress, "[%s] step %d/%d loss %.4f" % (tag, global_step, total_steps, loss.item() * args.grad_accum))
        if progress:
            log_progress(progress, "[%s] epoch %d/%d avg_loss %.4f" % (tag, ep + 1, args.epochs, epoch_loss / len(loader)))

    P = predict(model, test_loader, device)
    Y15 = np.vstack([multi_hot(test_types[r["dialogue_id"]]) for r in test_rows])
    r15 = best_f1(Y15, P)
    Yf = family_matrix(Y15)
    Pf = family_matrix_from_probs(P)
    rf = best_f1(Yf, Pf)
    if progress:
        log_progress(progress, "[%s] eval-done types macro=%.4f micro=%.4f thr=%.1f | fam macro=%.4f micro=%.4f thr=%.1f" % (tag, r15[0], r15[1], r15[2], rf[0], rf[1], rf[2]))
    del model
    torch.cuda.empty_cache()
    return {"types": {"macro": r15[0], "micro": r15[1], "thr": r15[2]},
            "families": {"macro": rf[0], "micro": rf[1], "thr": rf[2]}}


def family_matrix_from_probs(P15):
    Pf = np.zeros((P15.shape[0], len(FAM_ORDER)), dtype=np.float64)
    for fi, fam in enumerate(FAM_ORDER):
        cols = [TYPE_LIST.index(t) for t in FAMS[fam]]
        Pf[:, fi] = P15[:, cols].max(axis=1)
    return Pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heldout", "cv5"], default="heldout")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", action="store_true", default=False, help="disable fp16 autocast (more stable on some GPUs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test limit on train rows")
    ap.add_argument("--progress", type=str, default=os.path.join(ROOT, "experiments", "logs", "t2_progress.log"))
    ap.add_argument("--step_interval", type=int, default=100, help="log a step line every N batches")
    ap.add_argument("--out", type=str, default="", help="output json path (default: experiments/output/t2_finetune_<mode>.json)")
    args = ap.parse_args()
    if args.no_fp16:
        args.fp16 = False
    os.makedirs(os.path.dirname(args.progress), exist_ok=True)
    log_progress(args.progress, "[MAIN] start mode=%s epochs=%d batch=%d grad_accum=%d fp16=%s device=%s limit=%d" % (
        args.mode, args.epochs, args.batch_size, args.grad_accum, bool(args.fp16),
        "cuda" if torch.cuda.is_available() else "cpu", args.limit))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    rows_full = load(FULL)
    rows_634 = load(FIRST)
    if args.limit > 0:
        rows_full = rows_full[:args.limit]
        rows_634 = rows_634[:args.limit]
    tfull = {r["dialogue_id"]: r["types"] for r in load(TFULL)}
    t634 = {r["dialogue_id"]: r["types"] for r in load(T634)}

    results = {"mode": args.mode, "model": "roberta-base", "epochs": args.epochs,
               "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed}
    if args.mode == "heldout":
        r = run_training(rows_full, tfull, rows_634, t634, tokenizer, args, "heldout")
        results["heldout"] = r
        print("== T2 supervised baseline (train full -> held-out 634) ==")
        print(f"  15 types: macro-F1 {r['types']['macro']:.3f} micro-F1 {r['types']['micro']:.3f} @thr={r['types']['thr']}")
        print(f"  4 families: macro-F1 {r['families']['macro']:.3f} micro-F1 {r['families']['micro']:.3f} @thr={r['families']['thr']}")
    else:
        from sklearn.model_selection import StratifiedKFold
        strata = np.array([1 if tfull[r["dialogue_id"]] else 0 for r in rows_full])
        skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
        agg = {"types": [], "families": []}
        for fold, (tr, te) in enumerate(skf.split(rows_full, strata)):
            tr_rows = [rows_full[i] for i in tr]; te_rows = [rows_full[i] for i in te]
            log_progress(args.progress, "[MAIN] ===== fold %d/5 start (train %d, test %d) =====" % (fold + 1, len(tr_rows), len(te_rows)))
            t_fold = time.time()
            r = run_training(tr_rows, tfull, te_rows, tfull, tokenizer, args, "cv5-f%d" % (fold + 1))
            agg["types"].append(r["types"]); agg["families"].append(r["families"])
            log_progress(args.progress, "[MAIN] fold %d/5 done in %.0fs | types %.4f/%.4f | fam %.4f/%.4f" % (fold + 1, time.time() - t_fold, r["types"]["macro"], r["types"]["micro"], r["families"]["macro"], r["families"]["micro"]))
            print(f"  fold {fold+1}: types macro/micro {r['types']['macro']:.3f}/{r['types']['micro']:.3f} | fam {r['families']['macro']:.3f}/{r['families']['micro']:.3f}", flush=True)
        results["cv5"] = {"types": {"macro": float(np.mean([x["macro"] for x in agg["types"]])),
                                    "micro": float(np.mean([x["micro"] for x in agg["types"]]))},
                          "families": {"macro": float(np.mean([x["macro"] for x in agg["families"]])),
                                       "micro": float(np.mean([x["micro"] for x in agg["families"]]))}}
        log_progress(args.progress, "[MAIN] CV5 COMPLETE types %.4f/%.4f fam %.4f/%.4f" % (results["cv5"]["types"]["macro"], results["cv5"]["types"]["micro"], results["cv5"]["families"]["macro"], results["cv5"]["families"]["micro"]))
        print("== T2 supervised baseline (5-fold CV) ==")
        print(f"  15 types: macro-F1 {results['cv5']['types']['macro']:.3f} micro-F1 {results['cv5']['types']['micro']:.3f}")
        print(f"  4 families: macro-F1 {results['cv5']['families']['macro']:.3f} micro-F1 {results['cv5']['families']['micro']:.3f}")

    os.makedirs(OUT, exist_ok=True)
    fn = args.out if args.out else os.path.join(OUT, f"t2_finetune_{args.mode}.json")
    json.dump(results, open(fn, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log_progress(args.progress, "[MAIN] SAVED %s" % fn)
    print("saved:", fn)


if __name__ == "__main__":
    main()
