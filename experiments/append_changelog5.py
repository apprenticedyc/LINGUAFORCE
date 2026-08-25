# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\CHANGELOG.md'
add = """
### 人工 IAA 完成并写入论文（2026-08-22，零成本）
- 两份完成标注来自 Downloads：annotator_A_completed.xlsx（150 条全填）/ annotator_B_completed.xlsx（缺 1 条整行，149 对可用），值均为整数、无越界。
- 人工-人工一致性（compute_iaa.py）：
  - binary κ=0.491（moderate）；intensity QWK=0.730（substantial）；D3=0.781、D6=0.760、D2=0.659、D4=0.596、D1=0.552、D7=0.494、D5=0.390（最低，欺骗性主观/低频）。
- 人工-LLM 一致性（备用，未写进论文正文）：binary 0.400、intensity 0.520、D6 0.561、D3 0.518、D1 0.327、D4 0.266、D7 0.254、D2 0.203、D5 0.023 —— LLM 与人工存在系统偏差，可作 limitation 讨论。
- main.tex 更新（脚本 integrate_iaa.py）：Annotation Protocol 的 "left to future work" 改为引用 sec:qc；Quality Control 加 \label{sec:qc} 并追加双人 IAA 段落 + 表 tab:iaa（binary/intensity/D1-D7 共 9 行）。
- 编译验证通过，main.pdf 14 页。
"""
io.open(p, 'a', encoding='utf-8').write(add)
print('CHANGELOG updated')