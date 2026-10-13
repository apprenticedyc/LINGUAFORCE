"""Run resumable zero-shot predictions. Only dialogue text enters the prompt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_clients import complete, configuration

ROOT = Path(__file__).resolve().parents[1]
PROMPT = """Annotate negative pressure in this complete English dialogue.
Negative pressure pushes a listener toward an action by making refusal costly,
unacceptable, or difficult. Ordinary requests, advice, factual explanations,
and reasoned persuasion do not automatically qualify. Consider context, not
keywords. Resistance does not imply absence of pressure. Do not infer actual
compliance, harm, or unsupported deception.

Return one JSON object with these fields:
binary: integer 0 (absent) or 1 (present).
pressure_score: number from 0 to 1 indicating confidence pressure is present.
intensity: integer 0 none, 1 very slight, 2 weak, 3 moderate, 4 strong, 5 very strong.
dimensions: object with D1 through D7, each integer 0 absent, 1 low, 2 moderate, 3 high.
D1: strength of the push toward a target action.
D2: restriction of the listener's alternatives.
D3: duty, morality, authority, roles, or group norms used as leverage.
D4: guilt, shame, fear, or pity used as leverage.
D5: misleading facts, memory, consensus, or promises supported by context.
An unverified statement or ordinary future promise alone is insufficient.
D6: insults, hostility, or verbal abuse.
D7: directness of the requested action or pressure, not overall intensity.
strategies: object containing all 15 keys below, each 0 absent or 1 present.
Multiple strategies may apply; ordinary strategies do not imply binary=1.
A1_Request_Advice: open request or advice.
A2_Directive_Order: direct instruction expecting compliance.
A3_Rational_Persuasion: reasons or evidence supporting action.
B1_Promise_Inducement: benefit, reward, or promise offered for action.
B2_Threat_Warning: adverse consequences used to obtain action.
C1_Moral_Appeal_Guilt: conscience or guilt invoked to obtain action.
C2_Obligation_Duty: role or duty invoked to obtain action.
C3_Value_Judgement_Shaming: negative character judgments used as pressure.
C4_Authority: rank, expertise, or status used to demand compliance.
C5_Conformity_Social_Proof: others' agreement or behavior used as leverage.
C6_Reciprocity_Debt: past favors used to demand repayment through action.
D1_Emotional_Manipulation: emotional vulnerability exploited to change decisions.
D2_Deception_Gaslighting: contextual distortion of facts or memory.
D3_Logical_Trap_False_Dilemma: misleading reasoning or false restriction of alternatives.
D4_Verbal_Abuse_Toxicity: insults or hostility used to obtain compliance.
All-zero strategies are allowed. Output no explanation or markdown.

Dialogue turns (ordered JSON array):
"""
STRATEGIES = re.findall(r"^([A-D][1-6]_\w+):", PROMPT, re.M)


def validate(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    prediction = json.loads(text)
    def integer(value, upper):
        if type(value) is not int or not 0 <= value <= upper:
            raise ValueError("Invalid ordinal or binary value")
    integer(prediction["binary"], 1)
    integer(prediction["intensity"], 5)
    score = prediction["pressure_score"]
    if type(score) not in (int, float) or not 0 <= score <= 1:
        raise ValueError("Invalid pressure score")
    for field, keys, upper in [("dimensions", [f"D{i}" for i in range(1,8)], 3),
                               ("strategies", STRATEGIES, 1)]:
        if set(prediction[field]) != set(keys):
            raise ValueError(f"Invalid {field} keys")
        for v in prediction[field].values():
            integer(v, upper)
    return prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["gpt", "qwen", "deepseek"], required=True)
    ap.add_argument("--split", choices=["training", "test"], default="training")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.limit < 1:
        ap.error("Use an explicit positive limit")
    source = ROOT / "data/human/annotations.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = sorted((r for r in rows if r["source_split"] == args.split), key=lambda r:r["id"])[:args.limit]
    cfg = configuration(args.provider)
    digest = hashlib.sha256(PROMPT.encode()).hexdigest()
    out = ROOT / "results/human_llm"
    out.mkdir(exist_ok=True)
    path = out / f"{args.provider}_{args.split}.jsonl"
    previous = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    for item in previous:
        if item["prompt_sha256"] != digest or item["requested_model"] != cfg["model"]:
            raise ValueError("Existing predictions use a different protocol; use a separate output file")
    done = {r["id"] for r in previous}
    def predict(row):
        prompt = PROMPT + json.dumps(row["utterances"], ensure_ascii=False)
        failures = []
        for attempt in range(4):
            try:
                response = complete(args.provider, prompt, 1600)
                break
            except Exception as exc:
                failures.append(str(exc)[:900])
                if attempt == 3:
                    return {"id": row["id"], "provider": args.provider,
                            "requested_model":cfg["model"], "prompt_sha256":digest,
                            "valid":False,"transport_errors":failures}
                time.sleep(5 * (attempt + 1))
        result = {"id": row["id"], "provider": args.provider,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "prompt_sha256": digest,
                  "dialogue_sha256": hashlib.sha256(json.dumps(row["utterances"],ensure_ascii=False).encode()).hexdigest(),
                  **response}
        result["transport_errors"] = failures
        try:
            result["prediction"] = validate(response["text"])
            result["valid"] = True
        except (ValueError, KeyError, TypeError) as exc:
            result["valid"] = False
            result["validation_error"] = type(exc).__name__
        return result
    pending = [row for row in rows if row["id"] not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with path.open("a") as handle:
            futures = [pool.submit(predict, row) for row in pending]
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False)+"\n")
                handle.flush()
                if index % 20 == 0 or not result["valid"] or index == len(pending):
                    print(json.dumps({"completed":len(done)+index,"provider":args.provider,
                                      "valid":result["valid"]}),flush=True)
    print(f"Predictions: {path}")


if __name__ == "__main__":
    main()
