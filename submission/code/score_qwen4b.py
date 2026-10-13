"""Reproduce the matched Qwen3-4B comparison from saved final responses."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error

from run_human_llm import STRATEGIES
from score_human_llm import ordinal, qwk

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/qwen3_4b"
DIMS = [f"D{i}" for i in range(1, 8)]
TASKS = ("pressure", "intensity", "strategies", "dimensions")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_answer(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        raise ValueError("Unclosed thinking block")
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fence:
        text = fence.group(1)
    value = json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError("Final answer is not an object")
    labels = value.get("strategies")
    alias = "D3_Logical_False_Dilemma"
    canonical = "D3_Logical_Trap_False_Dilemma"
    if isinstance(labels, dict) and alias in labels and canonical not in labels:
        labels[canonical] = labels.pop(alias)
    return value


def valid(task, prediction):
    if task == "pressure":
        return ordinal(prediction.get("binary"), 1)
    if task == "intensity":
        return ordinal(prediction.get("intensity"), 5)
    keys, upper = (STRATEGIES, 1) if task == "strategies" else (DIMS, 3)
    value = prediction.get(task)
    return (isinstance(value, dict) and set(value) == set(keys)
            and all(ordinal(value[k], upper) for k in keys))


def array(task, ids, records):
    if task in ("pressure", "intensity"):
        key = "binary" if task == "pressure" else task
        return np.array([records[i][key] for i in ids])
    keys = STRATEGIES if task == "strategies" else DIMS
    return np.array([[records[i][task][k] for k in keys] for i in ids])


def metrics(task, y, p):
    if task == "pressure":
        return {"f1": float(f1_score(y, p, zero_division=0))}
    if task == "strategies":
        return {a + "_f1": float(f1_score(y, p, average=a, zero_division=0))
                for a in ("micro", "macro")}
    kappas = ([qwk(y, p, 5)] if task == "intensity" else
              [qwk(y[:, j], p[:, j], 3) for j in range(7)])
    return {"mae": float(mean_absolute_error(y, p)),
            "qwk": float(np.mean(kappas)) if all(k is not None for k in kappas) else None}


def compare(task, y, a, b, rng, repetitions):
    before, after = metrics(task, y, a), metrics(task, y, b)
    samples = {k: [] for k in before}
    for _ in range(repetitions):
        take = rng.integers(len(y), size=len(y))
        ma, mb = metrics(task, y[take], a[take]), metrics(task, y[take], b[take])
        for key in samples:
            if ma[key] is not None and mb[key] is not None:
                samples[key].append(mb[key] - ma[key])
    return {
        "base": before, "sft": after,
        "differences": {
            k: {"sft_minus_base": after[k] - before[k]
                if after[k] is not None and before[k] is not None else None,
                "ci95": np.quantile(v, [0.025, 0.975]).tolist() if v else None,
                "defined_bootstrap_n": len(v)} for k, v in samples.items()}}


def main():
    protocol = json.loads((RESULTS / "protocol.json").read_text())
    source = ROOT / "data/human/annotations.jsonl"
    split_path = ROOT / "data/human/splits.json"
    for key, path in (("annotation_sha256", source), ("split_sha256", split_path),
                      ("prompt_sha256", RESULTS / "prompt.txt")):
        if digest(path) != protocol[key]:
            raise ValueError("Changed input: " + key)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    all_gold = {r["id"]: r for r in rows}
    splits = json.loads(split_path.read_text())
    ids = set(splits["test"])
    if (len(all_gold) != len(rows) or len(ids) != 634 or
            ids & (set(splits["train"]) | set(splits["validation"]))):
        raise ValueError("Invalid split or duplicate data IDs")
    gold = {i: all_gold[i] for i in ids}
    predictions, coverage = {}, {}
    for name in ("base", "sft"):
        path = RESULTS / f"{name}_test.jsonl"
        if digest(path) != protocol["prediction_sha256"][name]:
            raise ValueError("Changed saved responses: " + name)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if len(records) != len(ids) or {r["id"] for r in records} != ids:
            raise ValueError("Missing, duplicate, or unexpected prediction IDs")
        parsed, failures = {}, []
        for record in records:
            uid = record["id"]
            expected = hashlib.sha256(json.dumps(
                gold[uid]["utterances"], ensure_ascii=False).encode()).hexdigest()
            if record["dialogue_sha256"] != expected:
                raise ValueError("Mismatched dialogue: " + uid)
            try:
                parsed[uid] = parse_answer(record["text"])
            except (ValueError, TypeError) as exc:
                parsed[uid] = {}
                failures.append({"id": uid, "error": str(exc)})
        predictions[name] = parsed
        coverage[name] = {
            "returned_n": len(records), "json_valid_n": len(records) - len(failures),
            "parse_failures": failures,
            "task_valid_n": {t: sum(valid(t, p) for p in parsed.values()) for t in TASKS},
            "invalid_task_ids": {t: sorted(i for i, p in parsed.items() if not valid(t, p))
                                 for t in TASKS}}
    seed, repetitions = 2026, 1000
    rng = np.random.default_rng(seed)
    comparisons = {}
    for task in TASKS:
        common = sorted(i for i in ids if all(valid(task, predictions[n][i])
                                             for n in predictions))
        if not common:
            raise ValueError("No common valid responses for " + task)
        y = array(task, common, gold)
        a, b = [array(task, common, predictions[n]) for n in ("base", "sft")]
        result = compare(task, y, a, b, rng, repetitions)
        comparisons[task] = {"n": len(common), "ids": common, **result}
        print(task, json.dumps({"n": len(common), **result}), flush=True)
    result = {"bootstrap": {"seed": seed, "repetitions": repetitions,
                            "task_order": list(TASKS), "paired": True},
              "coverage": coverage, "comparisons": comparisons}
    (RESULTS / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
