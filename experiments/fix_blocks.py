# -*- coding: utf-8 -*-
import io, re
p = r'E:\PythonCode\Paper\experiments\extract_prompt_ablation.py'
s = io.open(p, encoding='utf-8').read()

pat = re.compile(r"PROMPT_COT = \(.*?^\)", re.M | re.S)
new_cot = "PROMPT_COT = (\n" \
    "    \"First reason step by step about the dialogue's pressure tactics \"\n" \
    "    \"in plain text (max 3 short sentences). Then output ONLY the final JSON object:\\n\"\n" \
    "    '{\"dims\": {\"D1\": {\"score\": 0.0, \"level\": 0}, ..., \"D7\": {...}}, \"intensity\": 0}\\n'\n" \
    ")"
s, n1 = pat.subn(new_cot, s)

pat2 = re.compile(r"PROMPT_SELFREF = \(.*?^\)", re.M | re.S)
new_self = "PROMPT_SELFREF = (\n" \
    "    \"First draft a judgment in plain text, then critically re-examine it \"\n" \
    "    \"for the opposite interpretation (is pressure present/absent, is any \"\n" \
    "    \"dimension mis-scored?), then output ONLY the FINAL JSON object:\\n\"\n" \
    "    '{\"dims\": {\"D1\": {\"score\": 0.0, \"level\": 0}, ..., \"D7\": {...}}, \"intensity\": 0}\\n'\n" \
    ")"
s, n2 = pat2.subn(new_self, s)

io.open(p, 'w', encoding='utf-8').write(s)
print('blocks replaced: cot=%d self=%d' % (n1, n2))