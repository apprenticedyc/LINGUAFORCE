# -*- coding: utf-8 -*-
"""Run both v2 prompt-ablation variants sequentially in background."""
import subprocess, sys, os, time
PY = sys.executable
base = r'E:\PythonCode\Paper'
cmds = [
    [PY, 'experiments/extract_prompt_ablation.py', '--variant', 'cot', '--out', 'experiments/output/prompt_cot_v2.jsonl', '--workers', '8'],
    [PY, 'experiments/extract_prompt_ablation.py', '--variant', 'selfref', '--out', 'experiments/output/prompt_selfref_v2.jsonl', '--workers', '8'],
]
for i, cmd in enumerate(cmds, 1):
    print(f'[{time.strftime("%H:%M:%S")}] start cmd {i}: {cmd[2]}', flush=True)
    r = subprocess.run(cmd, cwd=base)
    print(f'[{time.strftime("%H:%M:%S")}] cmd {i} done rc={r.returncode}', flush=True)
print('ALL DONE', flush=True)