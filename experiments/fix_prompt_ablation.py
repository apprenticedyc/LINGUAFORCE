# -*- coding: utf-8 -*-
"""Re-annotate out-of-schema entries from prompt ablation with a strict
format-constraint suffix, then merge back (cheap: ~67 calls)."""
import os, sys, json, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor, as_completed
import extract_dims as ed
import extract_prompt_ablation as epa

BASE = r'E:\PythonCode\Paper\experiments\output'
STRICT_SUFFIX = (
    "\n\nIMPORTANT FORMAT RULES:\n"
    "- Each dimension score MUST be a float in [0.0, 1.0].\n"
    "- Each dimension level MUST be an integer in {0, 1, 2, 3}.\n"
    "- intensity MUST be an integer in {0, 1, 2, 3, 4, 5}.\n"
    "- Output ONLY ONE JSON object, no extra text."
)

def is_bad(rec):
    it = rec.get('intensity')
    if not isinstance(it, (int, float)) or isinstance(it, bool) or not (0 <= it <= 5):
        return True
    for d in ed.SEVEN_DIMS:
        c = rec['dims'].get(d[0])
        if not c or not isinstance(c.get('score'), (int, float)) or not (0 <= c['score'] <= 1):
            return True
        if c.get('level') not in (0, 1, 2, 3):
            return True
    return False

def main():
    api_key, model, base_url = ed.load_config()
    jobs = []
    for variant, out in [('cot', 'prompt_cot_full.jsonl'), ('selfref', 'prompt_selfref_full.jsonl')]:
        path = os.path.join(BASE, out)
        lines = open(path, encoding='utf-8').readlines()
        bad = []
        for line in lines:
            r = json.loads(line)
            if is_bad(r):
                bad.append(r['dialogue_id'])
        print(f'{variant}: {len(bad)} bad entries', flush=True)
        jobs.append((variant, out, path, bad))
    total_bad = sum(len(j[3]) for j in jobs)
    if total_bad == 0:
        print('nothing to fix'); return

    first = {r['dialogue_id']: r for r in
             ed.load(os.path.join(BASE, '..', '..', 'linguistic_agency_paper', 'data', 'linguaforce_first_release.jsonl'))}
    lock = threading.Lock()
    ok = fail = 0
    start = time.time()

    def work(variant, rec):
        user = epa.build_user_prompt_variant(rec['utterances'], variant) + STRICT_SUFFIX
        content, usage = epa.call_api_variant(api_key, model, base_url, variant, user)
        parsed = ed.parse_json_response(content)
        return rec, parsed, usage

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        for variant, out, path, bad_ids in jobs:
            for did in bad_ids:
                rec = first[did]
                futs[ex.submit(work, variant, rec)] = (variant, out, path, did)
        for fut in as_completed(futs):
            variant, out, path, did = futs[fut]
            try:
                rec, parsed, usage = fut.result()
                outrec = {'dialogue_id': did,
                          'gold_binary': rec['gold_binary'],
                          'gold_multi': rec['gold_multi'],
                          'dims': parsed.get('dims', {}),
                          'intensity': parsed.get('intensity', 0),
                          'usage': usage}
                if is_bad(outrec):
                    print(f'[!] still bad after strict re-run: {did}', flush=True)
                with lock:
                    lines = open(path, encoding='utf-8').readlines()
                    kept = [l for l in lines if json.loads(l)['dialogue_id'] != did]
                    kept.append(json.dumps(outrec, ensure_ascii=False) + '\n')
                    with open(path, 'w', encoding='utf-8') as f:
                        f.writelines(kept)
                ok += 1
            except Exception as e:
                fail += 1
                print(f'[!] fix fail {did}: {e}', flush=True)
    print(f'[done] fixed ok={ok} fail={fail} elapsed={time.time()-start:.0f}s', flush=True)

if __name__ == '__main__':
    main()