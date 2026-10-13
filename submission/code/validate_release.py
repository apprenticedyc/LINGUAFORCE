"""Check the released human data and prediction coverage without API calls."""
import collections
import hashlib
import json
from pathlib import Path
from run_human_llm import PROMPT, STRATEGIES

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    data_path = ROOT / "data/human/annotations.jsonl"
    rows = read_rows(data_path)
    by_id = {r["id"]: r for r in rows}
    require(len(rows) == len(by_id) == 4066, "Wrong row count or duplicate IDs")
    normalized = set()
    for r in rows:
        require(r["source_split"] in ("training", "test"), "Unknown source split")
        require(r["id"] == f'{r["source_split"]}_{r["source_id"]}', "ID mismatch")
        require(isinstance(r["utterances"], list) and r["utterances"] and
                all(isinstance(t, str) and t.strip() for t in r["utterances"]),
                f'Invalid dialogue: {r["id"]}')
        require(set(r["dimensions"]) == {f"D{i}" for i in range(1, 8)}, "Dimension keys")
        require(set(r["strategies"]) == set(STRATEGIES), "Strategy keys")
        values = [(r["binary"], 1), (r["intensity"], 5)]
        values += [(v, 3) for v in r["dimensions"].values()]
        values += [(v, 1) for v in r["strategies"].values()]
        require(all(type(v) is int and 0 <= v <= upper for v, upper in values), "Label range")
        require((r["binary"] == 0) == (r["intensity"] == 0), "Presence/intensity conflict")
        text = " ".join(" ".join(r["utterances"]).lower().split())
        require(text not in normalized, "Normalized duplicate dialogue")
        normalized.add(text)
    require(sum(r["binary"] for r in rows) == 3144, "Pressure count mismatch")
    require(len({r["annotation_file"] for r in rows}) == 20, "Batch count mismatch")

    split_path = ROOT / "data/human/splits.json"
    splits = json.loads(split_path.read_text())
    expected = {"train": 3088, "validation": 344, "test": 634}
    require(set(splits) == set(expected), "Split keys")
    seen = set()
    for name, n in expected.items():
        ids = splits[name]
        require(len(ids) == len(set(ids)) == n, f"Wrong {name} size")
        require(not seen.intersection(ids), "Overlapping partitions")
        require(set(ids) <= set(by_id), "Unknown split IDs")
        seen.update(ids)
        source = "test" if name == "test" else "training"
        require(all(by_id[i]["source_split"] == source for i in ids), "Membership mismatch")
    require(seen == set(by_id), "Incomplete split coverage")
    stats = json.loads((ROOT / "data/human/statistics.json").read_text())
    for name in ("training", "test", "all"):
        subset = [r for r in rows if name == "all" or r["source_split"] == name]
        saved = stats["splits"][name]
        require(saved["n"] == len(subset), "Statistics count mismatch")
        for field in ("binary", "intensity"):
            counts = dict(collections.Counter(str(r[field]) for r in subset))
            require(saved[field] == counts, f"Statistics {field} mismatch")
        for key in STRATEGIES:
            require(saved["strategy_counts"][key] == sum(r["strategies"][key] for r in subset),
                    "Strategy statistics mismatch")

    protocol = json.loads((ROOT / "results/human_llm/protocol.json").read_text())
    for key, path in (("annotation_sha256", data_path), ("split_sha256", split_path)):
        require(hashlib.sha256(path.read_bytes()).hexdigest() == protocol[key], f"Changed {key}")
    require(hashlib.sha256(PROMPT.encode()).hexdigest() == protocol["prompt_sha256"], "Prompt changed")
    require((ROOT / "results/human_llm/prompt.txt").read_text() == PROMPT, "Prompt file mismatch")
    for folder, models in (("human_llm", ("gpt", "qwen", "deepseek")),
                           ("human_supervised", ("word", "char", "constant"))):
        for model in models:
            predictions = read_rows(ROOT / f"results/{folder}/{model}_test.jsonl")
            require(len(predictions) == 634 and {p["id"] for p in predictions} == set(splits["test"]),
                    f"Incomplete or duplicate predictions: {model}")
    print("PASS: 4,066 labels; 20 batches; 3,088/344/634 disjoint splits; hashes; six complete runs.")


if __name__ == "__main__":
    main()
