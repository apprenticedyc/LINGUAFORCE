# -*- coding: utf-8 -*-
"""Turn-level 7-dimension extraction (concurrent, resume-safe).
Annotates EACH utterance separately with the seven dimensions only
(no intensity) to support T/G/B aggregation ablation. Cheaper than the
dialogue-level call because output is shorter.

Reads API config from E:/PythonCode/Paper/api_config.json
Usage:
    python extract_turn_dims.py --input <union.jsonl> --out <out.jsonl> [--workers 8] [--max N]
Resume: skips (dialogue_id, turn_idx) already present in --out.
Stops after FUSE consecutive failures (budget exhaustion / API down);
re-running the same command resumes from the last good checkpoint.
"""
import os, json, sys, time, argparse, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = r'E:\PythonCode\Paper'
CONFIG = os.path.join(BASE, 'api_config.json')
DEFAULT_MODEL = 'deepseek-v4-flash'
_write_lock = threading.Lock()
FUSE_DEFAULT = 12

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
    'You are an expert annotator of linguistic manipulation. You will be given a single utterance. '
    'Rate the utterance on seven psychological dimensions. For each dimension output: '
    'score (0.0-1.0 continuous) and level (0=None,1=Low,2=Moderate,3=High). '
    'Respond with STRICT JSON only, no extra text:\n'
    '{\n'
    '  "dims": {"D1": {"score": 0.0, "level": 0}, ..., "D7": {...}}\n'
    '}\n'
)

def build_user_prompt(utterance):
    dim_defs = "\n".join(f"- {code} ({name}): {desc}" for code, name, desc in SEVEN_DIMS)
    return (
        "Dimension definitions:\n" + dim_defs +
        "\n\nUtterance:\n" + utterance.strip() +
        "\n\nRate the utterance on the seven dimensions and output JSON as specified."
    )

def call_api(api_key, model, base_url, sys_prompt, user_prompt, max_retries=3):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
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
    start = s.find('{')
    end = s.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(s[start:end + 1])

def load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding='utf-8'):
            line = line.strip()
            if line:
                rec = json.loads(line)
                done.add((rec['dialogue_id'], rec['turn_idx']))
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max', type=int, default=0, help='0 = all')
    ap.add_argument('--fuse', type=int, default=FUSE_DEFAULT)
    args = ap.parse_args()

    api_key, cfg_model, base_url = load_config()
    model = cfg_model
    if not api_key:
        print('[ERROR] api_key not found'); sys.exit(1)

    done = load_done(args.out)
    tasks = []
    for line in open(args.input, encoding='utf-8'):
        if not line.strip():
            continue
        rec = json.loads(line)
        did = rec['dialogue_id']
        for ti, u in enumerate(rec['utterances']):
            u = u.strip()
            if not u:
                continue
            if (did, ti) in done:
                continue
            tasks.append((did, ti, u))
    if args.max:
        tasks = tasks[:args.max]
    print(f'[*] model={model} base_url={base_url}')
    print(f'[*] workers={args.workers} pending={len(tasks)} already_done={len(done)}')
    if not tasks:
        print('[.] nothing to do'); return

    start = time.time()
    ok = fail = 0
    fuse = 0
    progress = {'n': 0}

    def work(item):
        did, ti, u = item
        content, usage = call_api(api_key, model, base_url, SYSTEM_PROMPT, build_user_prompt(u))
        parsed = parse_json_response(content)
        return did, ti, u, parsed, usage

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        try:
            for fut in as_completed(futs):
                try:
                    did, ti, u, parsed, usage = fut.result()
                    rec = {'dialogue_id': did, 'turn_idx': ti, 'utterance': u,
                           'dims': parsed.get('dims', {}), 'usage': usage}
                    with _write_lock:
                        with open(args.out, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    ok += 1; fuse = 0
                except Exception as e:
                    fail += 1; fuse += 1
                    print(f'[!] fail={fail} fuse={fuse} err={e}')
                    if fuse >= args.fuse:
                        print('[X] circuit breaker: stopping (resume later).')
                        for f2 in futs:
                            f2.cancel()
                        break
                progress['n'] += 1
                if progress['n'] % 200 == 0:
                    el = time.time() - start
                    rate = progress['n'] / el
                    eta = (len(tasks) - progress['n']) / rate if rate else 0
                    print(f'[.] {progress["n"]}/{len(tasks)} ok={ok} fail={fail} '
                          f'rate={rate:.1f}/s eta={eta/60:.0f}min')
        except KeyboardInterrupt:
            print('[!] interrupted; resume later.')
    el = time.time() - start
    print(f'[done] ok={ok} fail={fail} elapsed={el/60:.1f}min')
    print(f'[hint] re-run the same command to resume any unfinished turns.')

if __name__ == '__main__':
    main()