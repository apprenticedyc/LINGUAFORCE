# -*- coding: utf-8 -*-
import io, re
p = r'E:\PythonCode\Paper\experiments\extract_prompt_ablation.py'
s = io.open(p, encoding='utf-8').read()

# Replace whole PROMPT_COT block
pat_cot = re.compile(r"PROMPT_COT = \(.*?^\)", re.M | re.S)
new_cot = '''PROMPT_COT = (
    "First reason step by step about the dialogue's pressure tactics "
    "in plain text (max 3 short sentences). Then output ONLY the final JSON object:\n"
    '{"dims": {"D1": {"score": 0.0, "level": 0}, ..., "D7": {...}}, "intensity": 0}\n'
)'''
s, n1 = pat_cot.subn(new_cot, s)

# Replace whole PROMPT_SELFREF block
pat_self = re.compile(r"PROMPT_SELFREF = \(.*?^\)", re.M | re.S)
new_self = '''PROMPT_SELFREF = (
    "First draft a judgment in plain text, then critically re-examine it "
    "for the opposite interpretation (is pressure present/absent, is any "
    "dimension mis-scored?), then output ONLY the FINAL JSON object:\n"
    '{"dims": {"D1": {"score": 0.0, "level": 0}, ..., "D7": {...}}, "intensity": 0}\n'
)'''
s, n2 = pat_self.subn(new_self, s)

old_sys = "If using reasoning fields, keep them brief and factual."
new_sys = (" Dimension scores MUST be floats in [0.0, 1.0]; levels integers in {0,1,2,3}; "
           "intensity an integer in {0,1,2,3,4,5}. No extra JSON fields.")
n3 = s.count(old_sys)
s = s.replace(old_sys, new_sys)
io.open(p, 'w', encoding='utf-8').write(s)
print('replaced: cot=%d self=%d sys=%d' % (n1, n2, n3))