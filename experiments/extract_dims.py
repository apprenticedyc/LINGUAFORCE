# -*- coding: utf-8 -*-
"""
LINGUAFORCE 7-dimension extraction (concurrent).
Reads API config from E:/PythonCode/Paper/api_config.json
Usage:
    python extract_dims.py --input <data.jsonl> --out <out.jsonl> [--max N] [--workers 6] [--model MODEL]
Resume: skips dialogue_ids already present in --out. Concurrent with ThreadPoolExecutor.
"""
import os, json, sys, time, argparse, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = r'E:\PythonCode\Paper'
CONFIG = os.path.join(BASE, 'api_config.json')
DEFAULT_MODEL = 'deepseek-v4-flash'
_write_lock = threading.Lock()

def load_config():
    cfg = {}
    with open(CONFIG, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            cfg[k.strip().lower()] = v.strip()
    api_key = cfg.get('api_key') or os.environ.get('DEEPSEEK_API_KEY', '')
    model = os.environ.get('DEEPSEEK_MODEL') or DEFAULT_MODEL
    base_url = cfg.get('base_url (openai)') or cfg.get('base_url') or 'https://api.deepseek.com'
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    return api_key, model, base_url

SEVEN_DIMS = [
    ('D1', 'Directive Force', 'strength with which the utterance drives a target action (command, order, request-to-act)'),
    ('D2', 'Option Constraint', 'degree to which the listener\'s options are narrowed (ultimatum, no-alternative, threat)'),
    ('D3', 'Normative Pressure', 'pressure via duty, morality, norms, roles (moral appeal, guilt, obligation, shaming)'),
    ('D4', 'Emotional Pressure', 'use of guilt, shame, fear, pity as leverage (emotional blackmail, self-pity)'),
    ('D5', 'Deceptiveness', 'degree of factual or epistemic distortion (gaslighting, false claims, memory disputes)'),
    ('D6', 'Toxicity', 'explicit hostility and abuse (insults, disparagement, profanity)'),
    ('D7', 'Explicitness', 'transparency vs indirectness of the pressure (direct vs hint/innuendo)'),
]

SYSTEM_PROMPT = (
    'You are an expert annotator of linguistic manipulation. You will be given a multi-turn dialogue. '
    'Rate the dialogue on seven psychological dimensions. For each dimension output: '
    'score (0.0-1.0 continuous) and level (0=None,1=Low,2=Moderate,3=High). '
    'Also output a 0-5 intensity of overall manipulative pressure (0=none). '
    'Respond with STRICT JSON only, no extra text:\n'
    '{\n'
    '  "dims": {"D1": {"score": 0.0, "level": 0}, ..., "D7": {...}},\n'
    '  "intensity": 0\n'
    '}\n'
)

def build_dialogue_text(utterances):
    lines = []
    for u in utterances:
        u = u.strip()
        if u:
            lines.append(f"Utterance{len(lines)+1}: {u}")
    return "\n".join(lines)

def build_user_prompt(utterances):
    dim_defs = "\n".join(f"- {code} ({name}): {desc}" for code, name, desc in SEVEN_DIMS)
    text = build_dialogue_text(utterances)
    return (
        "Dimension definitions:\n" + dim_defs +
        "\n\nDialogue:\n" + text +
        "\n\nRate the dialogue on the seven dimensions and output JSON as specified."
    )

def call_api(api_key, model, base_url, sys_prompt, user_prompt, max_retries=3):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    url = base_url + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            if not content:
                raise RuntimeError("empty content (reasoning consumed budget?)")
            return content, usage
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt)
    raise last_err

def parse_json_response(content):
    s = content.strip()
    if s.startswith('```'):
        s = s.strip('`')
        if s.startswith('json'):
            s = s[4:]
    start = s.find('{')
    end = s.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(s[start:end + 1])

def process_one(rec, api_key, model, base_url, out_path, prog):
    did = rec['dialogue_id']
    user_prompt = build_user_prompt(rec['utterances'])
    content, usage = call_api(api_key, model, base_url, SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_response(content)
    out_row = {
        "dialogue_id": did,
        "dims": parsed.get("dims", {}),
        "intensity": parsed.get("intensity", None),
        "gold_binary": rec.get("dialog_binary_label"),
        "gold_multi": rec.get("dialog_multi_label"),
        "usage": usage,
    }
    with _write_lock:
        with open(out_path, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        prog['done'] += 1
        n = prog['done']; total = prog['total']
        tin = usage.get('prompt_tokens', 0); tout = usage.get('completion_tokens', 0)
        prog['in'] += tin; prog['out'] += tout
        print(f"[{n}/{total}] id={did} in={tin} out={tout} D3={parsed.get('dims',{}).get('D3',{}).get('score')} int={parsed.get('intensity')}", flush=True)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=0, help='0 = all')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--model', default=None)
    args = ap.parse_args()

    api_key, cfg_model, base_url = load_config()
    model = args.model or cfg_model
    if not api_key:
        print("[ERROR] api_key not found"); sys.exit(1)
    print(f"[*] model={model} base_url={base_url}")
    print(f"[*] input={args.input} out={args.out} max={args.max} workers={args.workers}")

    with open(args.input, encoding='utf-8') as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.max > 0:
        records = records[:args.max]

    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            for l in f:
                if l.strip():
                    try:
                        done_ids.add(json.loads(l)['dialogue_id'])
                    except Exception:
                        pass
    to_do = [r for r in records if r['dialogue_id'] not in done_ids]
    print(f"[*] total={len(records)} done={len(done_ids)} to_do={len(to_do)}")

    if not to_do:
        print("[*] nothing to do"); return

    prog = {'done': 0, 'total': len(to_do), 'in': 0, 'out': 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, r, api_key, model, base_url, args.out, prog): r for r in to_do}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                did = futures[fut]['dialogue_id']
                print(f"[!] id={did} FAILED: {e}", flush=True)
    dt = time.time() - t0
    print(f"[*] done {prog['done']} in {dt:.0f}s ({dt/max(prog['done'],1):.1f}s/req). "
          f"total_in={prog['in']} total_out={prog['out']}")

if __name__ == '__main__':
    main()