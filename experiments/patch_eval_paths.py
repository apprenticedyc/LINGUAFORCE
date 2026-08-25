# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\experiments\eval_prompt_ablation.py'
s = io.open(p, encoding='utf-8').read()
s = s.replace("COT = os.path.join(BASE, 'prompt_cot_full.jsonl')", "COT = os.path.join(BASE, 'prompt_cot_v2.jsonl')")
s = s.replace("SELF = os.path.join(BASE, 'prompt_selfref_full.jsonl')", "SELF = os.path.join(BASE, 'prompt_selfref_v2.jsonl')")
io.open(p, 'w', encoding='utf-8').write(s)
print('paths updated')