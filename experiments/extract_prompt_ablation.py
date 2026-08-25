# -*- coding: utf-8 -*-
"""Prompt ablation for the upstream dimension parser (T1/T3 zero-shot).
Variants: zero-shot (reuse existing dialogue-level dims), chain-of-thought,
self-reflection. Runs on the FIRST(634) held-out set, dialogue-level.

Usage:
    python extract_prompt_ablation.py --variant cot --out out.jsonl [--max N] [--workers 8]
"""
import os, sys, json, time, argparse, threading
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor, as_completed
import extract_dims as ed

FUSE_DEFAULT = 8
SYSTEM_BASE = ed.SYSTEM_PROMPT

PROMPT_COT = (
    "First reason step by step about the dialogue's pressure tactics "
    "in plain text (max 3 short sentences). Then output ONLY the final JSON object:\n"
    '{"dims": {"D1": {"score": 0.0, "level": 0}, ..., "D7": {...}}, "intensity": 0}\n'
)
PROMPT_SELFREF = (
    "First draft a judgment in plain text, then critically re-examine it "
    "for the opposite interpretation (is pressure present/absent, is any "
    "dimension mis-scored?), then output ONLY the FINAL JSON object:\n"
    '{"dims": {"D1": {"score": 0.0, "level": 0}, ..., "D7": {...}}, "intensity": 0}\n'
)
FORMAT_RULES = (
    " Dimension scores MUST be floats in [0.0, 1.0]; levels integers in {0,1,2,3}; "
    "intensity an integer in {0,1,2,3,4,5}. No extra JSON fields."
)

def build_user_prompt_variant(utterances, variant):
    dim_defs = "\n".join(f"- {code} ({name}): {desc}" for code, name, desc in ed.SEVEN_DIMS)
    text = ed.build_dialogue_text(utterances)
    base = (
        "Dimension definitions:\n" + dim_defs +
        "\n\nDialogue:\n" + text +
        "\n\nRate the dialogue on the seven dimensions and output JSON as specified."
    )
    if variant == 'cot':
        return base + "\n\nInstructions:\n" + PROMPT_COT
    if variant == 'selfref':
        return base + "\n\nInstructions:\n" + PROMPT_SELFREF
    return base

def call_api_variant(api_key, model, base_url, variant, user_prompt, max_retries=3):
    if variant == 'cot':
        sys_prompt = SYSTEM_BASE + (
            ' You may reason step by step in plain text, but your reply MUST end '
            'with exactly one JSON object.' + FORMAT_RULES)
    elif variant == 'selfref':
        sys_prompt = SYSTEM_BASE + (
            ' Draft and self-critique in plain text, then end with exactly one '
            'FINAL JSON object.' + FORMAT_RULES)
    else:
        sys_prompt = SYSTEM_BASE
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": 0.0,
        "max_tokens": 800,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    url = base_url + "/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST")
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['cot', 'selfref'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--fuse', type=int, default=FUSE_DEFAULT)
    args = ap.parse_args()

    api_key, model, base_url = ed.load_config()
    if not api_key:
        print('[ERROR] no api key'); sys.exit(1)
    FIRST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                         'linguistic_agency_paper', 'data', 'linguaforce_first_release.jsonl')
    recs = [json.loads(l) for l in open(FIRST, encoding='utf-8') if l.strip()]
    if args.max:
        recs = recs[:args.max]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding='utf-8'):
            line = line.strip()
            if line:
                done.add(json.loads(line)['dialogue_id'])
    todo = [r for r in recs if r['dialogue_id'] not in done]
    print(f'[*] variant={args.variant} pending={len(todo)} done={len(done)}', flush=True)

    lock = threading.Lock()
    ok = fail = fuse = 0
    start = time.time()
    def work(rec):
        content, usage = call_api_variant(api_key, model, base_url, args.variant,
                                          build_user_prompt_variant(rec['utterances'], args.variant))
        parsed = ed.parse_json_response(content)
        return rec, parsed, usage

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        try:
            for fut in as_completed(futs):
                try:
                    rec, parsed, usage = fut.result()
                    out = {'dialogue_id': rec['dialogue_id'],
                           'gold_binary': rec['gold_binary'],
                           'gold_multi': rec['gold_multi'],
                           'dims': parsed.get('dims', {}),
                           'intensity': parsed.get('intensity', 0),
                           'usage': usage}
                    with lock:
                        with open(args.out, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(out, ensure_ascii=False) + '\n')
                    ok += 1; fuse = 0
                except Exception as e:
                    fail += 1; fuse += 1
                    print(f'[!] fail={fail} fuse={fuse} err={e}', flush=True)
                    if fuse >= args.fuse:
                        print('[X] breaker', flush=True)
                        for f2 in futs:
                            f2.cancel()
                        break
        except KeyboardInterrupt:
            print('[!] interrupted', flush=True)
    el = time.time() - start
    print(f'[done] ok={ok} fail={fail} elapsed={el/60:.1f}min', flush=True)

if __name__ == '__main__':
    main()