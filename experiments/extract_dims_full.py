# -*- coding: utf-8 -*-
"""
LINGUAFORCE full 7-dim extraction with batched submit + circuit breaker.
Reuses extract_dims helpers. Resume-safe: skips ids already in --out.
Stops after FUSE consecutive failures (budget exhaustion / API down),
then re-running the same command resumes from the last good checkpoint.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor, as_completed
import extract_dims as ed

FUSE_DEFAULT = 12   # consecutive failures before we stop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=0, help='0 = all')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--model', default=None)
    ap.add_argument('--fuse', type=int, default=FUSE_DEFAULT)
    ap.add_argument('--batchsize', type=int, default=0, help='0 = workers*2')
    args = ap.parse_args()

    api_key, cfg_model, base_url = ed.load_config()
    model = args.model or cfg_model
    if not api_key:
        print("[ERROR] api_key not found"); sys.exit(1)
    print(f"[*] model={model} base_url={base_url}")
    print(f"[*] input={args.input} out={args.out} max={args.max} workers={args.workers} fuse={args.fuse}")

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

    batch = args.batchsize or args.workers * 2
    prog = {'done': 0, 'total': len(to_do), 'in': 0, 'out': 0}
    t0 = time.time()
    consec_fail = 0
    stop = False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i in range(0, len(to_do), batch):
            if stop:
                break
            chunk = to_do[i:i+batch]
            futures = {ex.submit(ed.process_one, r, api_key, model, base_url, args.out, prog): r for r in chunk}
            for fut in as_completed(futures):
                try:
                    fut.result()
                    consec_fail = 0
                except Exception as e:
                    did = futures[fut]['dialogue_id']
                    print(f"[!] id={did} FAILED: {e}", flush=True)
                    consec_fail += 1
                    if consec_fail >= args.fuse:
                        print(f"[FUSE] {consec_fail} consecutive failures -> stopping. Re-run to resume.", flush=True)
                        stop = True
                        for f in futures:
                            f.cancel()
                        break
            if stop:
                break
            dt = time.time() - t0
            print(f"[batch] {min(i+batch, len(to_do))}/{len(to_do)} done_batch={prog['done']} "
                  f"elapsed={dt:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"[*] finished with {prog['done']}/{len(to_do)} in {dt:.0f}s "
          f"({dt/max(prog['done'],1):.1f}s/req). total_in={prog['in']} total_out={prog['out']}")
    if prog['done'] < len(to_do):
        print("[*] NOT complete - re-run same command to resume.")
        sys.exit(2)

if __name__ == '__main__':
    main()
