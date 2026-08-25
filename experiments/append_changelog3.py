# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\CHANGELOG.md'
add = """
### 论文整合（2026-08-21，零成本）
- `main.tex` 更新（脚本：`experiments/integrate_paper.py`）：
  1. RQ4 从 "deferred to future work" 改为报告两轮结果（聚合消融 + 提示词消融），引用 `tab:agg` / `tab:prompt`。
  2. 新增 `tab:agg`（T/G/B：G AUC 0.855 / T 0.525 / B 0.859 最优）与 `tab:prompt`（zero-shot 0.832 / CoT 0.839 / SelfRef 0.838）。
  3. 新增 `\subsection{Dimension-Space Structure}`（sec:viz）：t-SNE 图 `fig5_tsne_family.png`、家族×维度热图 `fig6_family_dims_heatmap.png`，配 5 类可分 acc 0.640、良性 vs 操纵单维 AUC 0.73-0.94。
- 编译验证：Tectonic 通过，`main.pdf` 13→14 页，10 张图，全部新内容渲染正确。
- 图表文件复制到 `linguistic_agency_paper/figs/`（fig5/fig6）。
"""
io.open(p, 'a', encoding='utf-8').write(add)
print('CHANGELOG updated')