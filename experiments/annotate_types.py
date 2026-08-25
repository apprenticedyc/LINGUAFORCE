# -*- coding: utf-8 -*-
"""
LINGUAFORCE 15-type taxonomy annotation (concurrent, resumable).
Reads API config from E:/PythonCode/Paper/api_config.json
Usage:
    python annotate_types.py --input <data.jsonl> --out <out.jsonl> [--max N] [--workers 8]
Skips dialogue_ids already present in --out.
"""
import os, json, sys, time, argparse, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = r'E:\PythonCode\Paper'
CONFIG = os.path.join(BASE, 'api_config.json')
_write_lock = threading.Lock()

TYPES = [
    ('A1', 'Request/Advice', 'Low-pressure, intention-transparent proposal. "Could you revise this report by Friday?"'),
    ('A2', 'Directive/Order', 'Direct instruction with expected compliance. "Send the report before you leave."'),
    ('A3', 'Rational Persuasion', 'Argument, evidence, expert opinion. "The data show 30% lower cost; adopt it."'),
    ('B1', 'Promise/Inducement', 'Benefit-driven motivation. "Help me this once and I will return the favor."'),
    ('B2', 'Threat/Warning', 'Harm-driven motivation. "If you do not sign, you will face consequences."'),
    ('C1', 'Moral Appeal/Guilt', 'Appeals to conscience or moral duty. "A dutiful son would never let his mother suffer."'),
    ('C2', 'Obligation/Duty', 'Invokes role responsibilities. "You are the team leader; it is your duty."'),
    ('C3', 'Value Judgement/Shaming', 'Moral labeling of the target. "You are so selfish; you never think of others."'),
    ('C4', 'Authority', 'Appeals to status, rank, seniority. "The boss decided; why keep arguing?"'),
    ('C5', 'Conformity/Social Proof', 'Appeals to group norms. "Everyone is doing it; why are you special?"'),
    ('C6', 'Reciprocity/Debt', 'Invokes past favors. "I helped you last time; do not refuse this."'),
    ('D1', 'Emotional Manipulation', 'Feigned weakness, pity, or fear to steer behavior. "If you leave too, I do not know what to do."'),
    ('D2', 'Deception/Gaslighting', 'Distorts facts or memory. "I never said that; you misremember."'),
    ('D3', 'Logical Trap/False Dilemma', 'Shrinks the option space. "Either you follow me or we are done."'),
    ('D4', 'Verbal Abuse/Toxicity', 'Explicit hostility and disparagement. "You are useless; you cannot even do this."'),
]

SYSTEM_PROMPT = (
    'You are an expert annotator of linguistic manipulation strategies. '
    'Given a multi-turn dialogue, identify ALL strategy types present (any speaker may '
    'use pressure strategies), using the taxonomy below. This is a MULTI-LABEL task: '
    'a dialogue can contain more than one type. Only include a type if there is clear '
    'evidence in the dialogue. Respond with STRICT JSON only, no extra text:\n'
    '{"types": ["A1", "C2"]}\n'
    'Use {"types": []} if no strategy applies.\n\n'
    'Taxonomy:\n' +
    '\n'.join(f'- {code} {name}: {desc}' for code, name, desc in TYPES)
)

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
    model = os.environ.get('DEEPSEEK_MODEL') or cfg.get('model') or 'deepseek-v4-flash'
    base_url = cfg.get('base_url (openai)') or cfg.get('base_url') or 'https://api.deepseek.com'
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    return api_key, model, base_url

def build_dialogue_text(utterances):
    return "\n".join(f"Utterance{i+1}: {u.strip()}" for i, u in enumerate(utterances) if u.strip())

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
        raise ValueError("no JSON object found")
    return json.loads(s[start:end + 1])

def process_one(rec, api_key, model, base_url, out_path, prog):
    did = rec['dialogue_id']
    user_prompt = build_dialogue_text(rec['utterances'])
    content, usage = call_api(api_key, model, base_url, SYSTEM_PROMPT, user_prompt)
    parsed = parse_json_response(content)
    types = parsed.get('types', [])
    if not isinstance(types, list):
        types = [types]
    out_row = {"dialogue_id": did, "types": types, "usage": usage}
    with _write_lock:
        with open(out_path, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        prog['done'] += 1
        print(f"[{prog['done']}/{prog['total']}] id={did} types={types}", flush=True)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    api_key, model, base_url = load_config()
    if not api_key:
        print("[ERROR] api_key not found"); sys.exit(1)
    print(f"[*] model={model} base_url={base_url} workers={args.workers}")

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
    todo = [r for r in records if r['dialogue_id'] not in done_ids]
    print(f"[*] total={len(records)} done={len(done_ids)} todo={len(todo)}")

    prog = {'done': 0, 'total': len(todo)}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, r, api_key, model, base_url, args.out, prog): r for r in todo}
        for fut in as_completed(futures):
            fut.result()
    print("[*] done")

if __name__ == '__main__':
    main()
