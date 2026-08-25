# -*- coding: utf-8 -*-
"""Zero-shot DIRECT detection baseline (no 7-dim conditioning).
Sample balanced pos/neg from dims_test_clean.jsonl, ask model directly:
"is this moral coercion (0/1) + intensity (0-5)". Concurrent, resumable.
Usage: python zero_shot_direct.py [--max N] [--workers 6]   (--max 0 = ALL records)
"""
import os, json, sys, time, argparse, threading, random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = r'E:\PythonCode\Paper'
CONFIG = os.path.join(BASE, 'api_config.json')
SRC = os.path.join(BASE, 'experiments', 'output', 'dims_test_clean.jsonl')
OUT = os.path.join(BASE, 'experiments', 'output', 'direct_test.jsonl')
COERCION_DATA = os.path.join(BASE, 'COERCION', 'inputters', 'data')
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
    model = os.environ.get('DEEPSEEK_MODEL') or 'deepseek-v4-flash'
    base_url = cfg.get('base_url (openai)') or cfg.get('base_url') or 'https://api.deepseek.com'
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    return api_key, model, base_url

def build_utterance_lookup():
    lookup = {}
    for name in ('train.jsonl', 'valid.jsonl', 'test.jsonl'):
        path = os.path.join(COERCION_DATA, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if 'utterances' in r:
                    lookup[r['dialogue_id']] = r['utterances']
    return lookup

SYSTEM_PROMPT = (
    "You are an expert annotator of moral coercion (moral blackmail) in dialogue. "
    "Moral coercion = speech that pressures the listener to act by invoking duty, "
    "guilt, morality, norms, obligations, or emotional leverage (e.g. 'If you don't do it, "
    "you're a bad person / you owe me / how could you leave me'). "
    "For each dialogue output STRICT JSON only, no extra text:\n"
    "{\n"
    '  "is_moral_coercion": 0,\n'
    '  "intensity": 0\n'
    "}\n"
    'Where is_moral_coercion is 0 (no) or 1 (yes), and intensity is 0-5 '
    "(0=none, 1-2=weak, 3=moderate, 4-5=strong)."
)

def build_dialogue_text(utterances):
    lines = []
    for u in utterances:
        u = u.strip()
        if u:
            lines.append(f"Utterance{len(lines)+1}: {u}")
    return "\n".join(lines)

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
        url, data=json.dumps(payload).encode('utf-8'),
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
                raise RuntimeError("empty content")
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
    start = s.find('{'); end = s.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(s[start:end + 1])

def process_one(rec, api_key, model, base_url, prog):
    did = rec['dialogue_id']
    text = build_dialogue_text(rec['utterances'])
    user_prompt = "Dialogue:\n" + text + "\n\nOutput JSON as specified."
    content, usage = call_api(api_key, model, base_url, SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_response(content)
    out_row = {
        "dialogue_id": did,
        "is_moral_coercion": parsed.get("is_moral_coercion"),
        "intensity": parsed.get("intensity"),
        "gold_binary": rec.get("gold_binary"),
        "gold_multi": rec.get("gold_multi"),
        "usage": usage,
    }
    with _write_lock:
        with open(OUT, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        prog['done'] += 1
        n = prog['done']; total = prog['total']
        tin = usage.get('prompt_tokens', 0); tout = usage.get('completion_tokens', 0)
        prog['in'] += tin; prog['out'] += tout
        print(f"[{n}/{total}] id={did} pred={out_row['is_moral_coercion']}/{out_row['intensity']} gold={out_row['gold_binary']}/{out_row['gold_multi']}", flush=True)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=0, help='0 = ALL records; else balanced sample of N')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    api_key, model, base_url = load_config()
    if not api_key:
        print("[ERROR] api_key not found"); sys.exit(1)
    print(f"[*] model={model} base_url={base_url} workers={args.workers} max={args.max}")

    lookup = build_utterance_lookup()
    print(f"[*] utterance lookup size={len(lookup)}")

    with open(SRC, encoding='utf-8') as f:
        records = []
        missing = 0
        for l in f:
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            if r['dialogue_id'] in lookup:
                r['utterances'] = lookup[r['dialogue_id']]
                records.append(r)
            else:
                missing += 1
        print(f"[*] loaded={len(records)} missing_utterances={missing}")

    if args.max > 0:
        rng = random.Random(args.seed)
        pos = [r for r in records if r.get('gold_binary') == 1]
        neg = [r for r in records if r.get('gold_binary') == 0]
        n = args.max // 2
        rng.shuffle(pos); rng.shuffle(neg)
        sample = (pos[:n] + neg[:n])[:args.max]
        rng.shuffle(sample)
    else:
        sample = records
    print(f"[*] sample={len(sample)} ({sum(1 for r in sample if r['gold_binary']==1)} pos / {sum(1 for r in sample if r['gold_binary']==0)} neg)")

    done_ids = set()
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as f:
            for l in f:
                if l.strip():
                    try: done_ids.add(json.loads(l)['dialogue_id'])
                    except Exception: pass
    to_do = [r for r in sample if r['dialogue_id'] not in done_ids]
    print(f"[*] done={len(done_ids)} to_do={len(to_do)}")

    if not to_do:
        print("[*] nothing to do"); return

    prog = {'done': 0, 'total': len(to_do), 'in': 0, 'out': 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, r, api_key, model, base_url, prog): r for r in to_do}
        for fut in as_completed(futures):
            try: fut.result()
            except Exception as e:
                did = futures[fut]['dialogue_id']
                print(f"[!] id={did} FAILED: {e}", flush=True)
    dt = time.time() - t0
    print(f"[*] done {prog['done']} in {dt:.0f}s ({dt/max(prog['done'],1):.1f}s/req). "
          f"total_in={prog['in']} total_out={prog['out']} total={prog['in']+prog['out']}")

if __name__ == '__main__':
    main()
