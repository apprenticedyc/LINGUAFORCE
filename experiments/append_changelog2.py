# -*- coding: utf-8 -*-
import io
p = r'E:\PythonCode\Paper\CHANGELOG.md'
add = """
### 提示词消融（已完成，v2 全量 1.76 元）
- 脚本：`experiments/extract_prompt_ablation.py`（--variant cot/selfref）、`experiments/eval_prompt_ablation.py`。
- 数据：`experiments/output/prompt_cot_v2.jsonl`、`prompt_selfref_v2.jsonl`（各 634 条，格式遵循 100%）。
- 结果（FIRST 634，与 zero-shot 即 dims_test_clean 对比）：
  - zero-shot: T1 AUC 0.8316 / F1 0.7951 | T3 Spearman 0.6664 / QWK 0.6231
  - CoT: T1 AUC 0.8394 / F1 0.8047 | T3 Spearman 0.6392 / QWK 0.6130
  - SelfRef: T1 AUC 0.8375 / F1 0.7941 | T3 Spearman 0.6699 / QWK 0.6301
- 结论（写论文用）：三变体性能相近（AUC 0.83-0.84），pipeline 对提示词鲁棒；CoT 略升 T1、SelfRef 略升 T3。
- 坑：v1 提示词让模型把量表输出成 0-10（CoT 862 个 score 越界）→ 已废弃（浪费 2.84 元）；v2 改为「推理在文末 JSON 外 + 强格式约束」，格式遵循 100%。
- 中断记录：23:48 触发 HTTP 402 欠费熔断（cot 停在 520/634）；用户充值后断点续跑补齐，SelfRef 从头跑，无重复计费。
### 费用总账（截至 2026-08-21 凌晨，全部闲时价）
- turn 级标注 24,853 次：18.95 元
- 提示词消融 v1（废弃）：2.84 元
- 提示词消融 v2：1.76 元
- 合计约 23.6 元（另含各 warmup 不足 0.1 元）
"""
io.open(p, 'a', encoding='utf-8').write(add)
print('CHANGELOG updated')