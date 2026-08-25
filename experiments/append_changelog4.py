# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\CHANGELOG.md'
add = """
### 人工 IAA 材料（2026-08-21，零成本，待人工执行）
- 目录：`experiments/data/iaa/`
  - `annotator_A.xlsx` / `annotator_B.xlsx`：150 条（75 正 / 75 负，seed=2026 分层抽样），每条填 binary + intensity + D1-D7 level（0-3），不含 gold 防引导。
  - `iaa_guidelines.md`：中文标注指南（三步判断流程 + 维度锚点 + 示例）。
  - `iaa_sample_150.json`：抽样带 gold（仅研究者使用）。
  - `compute_iaa.py`：人工-人工 Cohen's κ（binary 未加权，intensity/D1-D7 二次加权）+ 人工-LLM 一致性。
- 自检：同源 LLM 数据填两份模板 → 所有 κ=1.0（公式验证通过）。
- 待办：让 1-2 位同学各标 150 条（约 4-6 小时/人），结果发回后运行 `python experiments/compute_iaa.py`，把 κ 填进论文。
"""
io.open(p, 'a', encoding='utf-8').write(add)
print('CHANGELOG updated')