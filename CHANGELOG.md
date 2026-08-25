# LINGUAFORCE 论文：决策与变更记录（长期跟踪）

> 用途：沉淀本论文自 2026-08 起的全部关键决策、改动与数字，供长期跟踪、复现、交接使用。
> 论文：`linguistic_agency_paper/main.tex`（IEEE 会议格式，当前 12 页，编译通过）
> 实验代码：`experiments/`；发布数据：`linguistic_agency_paper/data/`
> 维护规则：每次改动在「变更记录」追加一条（日期 + 内容 + 理由）。

## 0. 一句话定位
独立构建「语言能动性（linguistic agency）」七维数据集 **LINGUAFORCE**（3,432 全量 + 634 held-out）。
师兄的 COERCION 只是素材来源；论文**不声称是 COERCION 续作、不证明其落后**，核心叙事是「我建了一个新数据集」。

## 1. 核心决策
- **框架**：七维（约束/毒性等）刻画操纵性语言，道德胁迫只是其中一维；15 类操纵策略多标签 + 4 大家族（A 理性说服 / B 权威 / C 胁迫 / D 欺骗类）。
- **叙事**：不写 "Building on COERCION"；不把师兄工作当"要超越/证明落后"的对象。
- **标注**：DeepSeek `deepseek-v4-flash`、thinking 禁用、并发 8、max_tokens 2000、key 在 `api_config.json`；15 类全量 3,432 已标注并发布。
- **诚实底线**：LLM 标注**不冒充**专业人工标注；论文写「第一作者人工标注 120 条」+ 抽检一致性。
- **成本/硬件**：GPU 仅 6GB（RTX 3060 Laptop）；重活优先 fp32 或租卡。
- **文件布局**：论文在 `linguistic_agency_paper/`，实验代码在 `experiments/`，互不混放。

## 2. 变更记录（倒序）
### 2026-08-17 — T2 5 折 CV（RoBERTa 基线）完成
- 结果：15 类 macro/micro **0.540/0.734**，4 家族 **0.767/0.815**（`experiments/output/t2_finetune_cv5.json`）。
- 论文：`tab:rq2` 两处 `--` 填入数字；删除临时脚注；正文新增"结果对 train/test 划分不敏感"表述；重编译 12 页验证通过。
- 脚本：`finetune_t2.py` 新增实时进度日志（`--progress`，每折起点/step/epoch/完成 append 落盘）、`--out`（输出路径）、`--no_fp16`（稳定性开关）。
- 新增 `experiments/RENTAL_GPU.md`（云 GPU 复现说明）。
- 首次 fp16 运行 4.75h 后证实为 GPU 卡死（见 §5），终止并改 fp32 重跑，约 3.5h 完成。

### 2026-08-16 — T2 全量 + RoBERTa held-out 基线
- 15 类标注扩到全量 3,432 条；发布 `data/linguaforce_full_types.jsonl`（634 版 `linguaforce_first_release_types.jsonl`）。
- RoBERTa-base held-out：15 类 0.544/0.731，4 家族 0.780/0.820。
- `tab:rq2` 改为四行双协议表格 + RoBERTa 列；同步修正 T2 定义、交叉引用、结论 future work。

### 2026-08-15 — 数据发布与实验闭环
- 发布数据集（`release_dataset.py`）：binary/multi/intensity/七维/类型字段齐全。
- 实验补齐：T1/T2/T3、RQ3、消融、跨域、人工抽检（120 条）数字全部入论文。
- 参照模型建议：不把 LLM 标注说成人工标注；论文声明第一作者人工标注 120 条。

## 3. 关键数字速查（与论文一一对应）
| 项 | 数字 |
|----|------|
| LLM 读出器（634） | AUC 0.831 / Spearman 0.665 |
| 线性读出器消融（634） | AUC 0.855 / Spearman 0.674 |
| T1 全量检测 | AUC 0.852 / best F1 0.817 |
| T3 强度 | Spearman 0.651 / QWK 0.595 |
| RQ3 新颖性 | AUROC 0.543 |
| RQ3 强度分离 | AUROC 0.963 |
| 跨域 | MentalManip 0.742 / MultiManip 0.706 / TalkDown 0.729 / ToxicChat 0.587 |
| 人工抽检（120） | 强度 Spearman 0.79 / 主机制维度 top-2 81.2% |
| T2 线性读出器 held-out | 15 类 0.480/0.668；4 家族 0.772/0.827 |
| T2 线性读出器 5 折 CV | 15 类 0.477/0.669；4 家族 0.752/0.820 |
| RoBERTa held-out | 15 类 0.544/0.731；4 家族 0.780/0.820 |
| **RoBERTa 5 折 CV** | **15 类 0.540/0.734；4 家族 0.767/0.815** |
| 15 类全量分布 | A1=1125 A2=844 A3=29 B1=244 B2=495 C1=1118 C2=732 C3=1690 C4=23 C5=693 C6=928 D1=972 D2=7 D3=326 D4=993；空 226 |

## 4. 复现命令（Windows PowerShell）
```powershell
D:\Anaconda3\python.exe experiments\release_dataset.py          # 生成发布数据
D:\Anaconda3\python.exe experiments\make_figs_release.py        # 重新生成图表
D:\Anaconda3\python.exe experiments\verify_paper.py             # 校验论文全部数字
D:\Anaconda3\python.exe experiments\run_t2_rq3.py               # T2 多标签 + RQ3 留一类型
D:\Anaconda3\envs\pytorch2.3.1\python.exe experiments\finetune_t2.py --mode heldout --batch_size 4 --grad_accum 2
D:\Anaconda3\envs\pytorch2.3.1\python.exe experiments\finetune_t2.py --mode cv5 --batch_size 2 --grad_accum 2 --no_fp16
# 编译论文：
cd E:\PythonCode\Paper\linguistic_agency_paper; $env:PATH='E:\Program_Files\texlive\bin\windows;'+$env:PATH
pdflatex.exe -interaction=nonstopmode main.tex   # 跑两遍
```

## 5. 已知坑与注意事项（重要）
- **6GB 笔记本 GPU + fp16 GradScaler** 在满载 86°C 时会偶发 CUDA 卡死（Windows TDR）。症状：GPU 100% 但进度停滞，py-spy 显示卡在 `scaler.step() → .item()`。规避：`--no_fp16`（fp32 稳定、结果一致）或租卡（见 `experiments/RENTAL_GPU.md`）。
- **PowerShell stdin 管道会损坏中文**：含中文的 Python 脚本用 `[System.IO.File]::WriteAllText(path, text, UTF8)` 写脚本文件再执行，不要用 heredoc 直接管道。
- **`Get-Content` 读 UTF-8 中文文件会乱码**：用 Python `io.open(..., encoding="utf-8")` 读。
- **`Start-Process` 的 stdout/stderr 重定向可能卡住不写文件**：长任务用脚本内 `--progress` 直接 append 写盘。
- **shell 命令超时会被杀掉子进程**：长任务必须 `Start-Process -WindowStyle Hidden` 脱离启动。
- **`Remove-Item -Force` 会被策略拦截**：删除用不带 `-Force` 的 `Remove-Item -LiteralPath ...`。

## 6. 投稿前待办（未完成）
- [ ] 作者信息：`main.tex` 中 `First Author, Second Author, Third Author` 需替换为真实作者（含单位/邮箱）。
- [ ] `references.bib` 中 `coercion` 为占位（`Anonymous, Under review`）→ 换成师兄论文真实题目/作者/年份。
- [ ] 数据许可与 provenance：确认 COERCION 是否允许二次发布，`data/README.md` 补来源说明。
- [ ] 投稿目标（会议/期刊 + 页数限制，当前 12 页可能要精简到 8-10 页）。
- [ ] （可选，审稿建议）伦理章节/IRB、Datasheet、开源代码整理。
## 7. 2026-08-20 低成本补强（CCF-B 缺口）
### T/G/B 聚合消融（已完成，全闲时价 18.95 元 / 24,853 次 turn 级 API）
- 新增脚本：xperiments/extract_turn_dims.py（turn 级 7 维标注，并发 8、禁思考、断点续跑）、xperiments/tgb_aggregation.py（CPU 聚合评估）。
- 数据：xperiments/output/turn_dims_full.jsonl（union FULL+FIRST 共 3,502 条对话、24,853 个非空 turn）。
- 协议：train FULL(3432) / test FIRST(634)，逻辑回归(T1)+线性回归(T3)。结果：
  - G（对话级 7 维）：T1 AUC 0.8554 | T3 Spearman 0.6743 | QWK 0.6670
  - T（turn 平均 7 维）：T1 AUC 0.5250 | T3 Spearman 0.0395 | QWK 0.0658
  - B（turn+global 14 维）：T1 AUC 0.8591 | T3 Spearman 0.6787 | QWK 0.6798  ← 最优
- 结论（写论文用）：简单 mean-pool 会稀释稀疏操纵信号（多数 turn 中性），T 远差于 G/B；B 拼接互补信息最好 → 回答 RQ4 聚合消融。
### 维度空间可视化（零成本）
- 新增 xperiments/make_figs_viz.py：t-SNE 按 4 家族+良性着色、家族×维度热图 → xperiments/output/figs/fig_tsne_family.png、ig_family_dims_heatmap.png。
- 量化：5 类(A/B/C/D/良性)线性可分 acc=0.640（随机 0.2）；良性 vs 操纵单维 AUC 0.727–0.938。
### 伦理 + Datasheet（零成本）
- main.tex Ethics 段扩充为 6 段（来源合规/IRB 豁免/标注者保护/双用途/文化局限/维护），并修 bug：'We three aggregation modes' → 'We compare three aggregation modes'。
- 新增 linguistic_agency_paper/datasheet.md（完整数据卡）。
- 新增 linguistic_agency_paper/references_extra.bib 放 Gebru2018（main.tex 的 bibliography 改为 {references,references_extra}），Tectonic 编译验证通过（13 页，[17] 引用正确）。
- 坑：
eferences.bib ACL 缺 DYC666 权限（BUILTIN\\Users 只读），沙箱无法写/删/改 ACL → 用 references_extra.bib 绕过；用户需手动在资源管理器给文件加完全控制。

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

### 论文整合（2026-08-21，零成本）
- `main.tex` 更新（脚本：`experiments/integrate_paper.py`）：
  1. RQ4 从 "deferred to future work" 改为报告两轮结果（聚合消融 + 提示词消融），引用 `tab:agg` / `tab:prompt`。
  2. 新增 `tab:agg`（T/G/B：G AUC 0.855 / T 0.525 / B 0.859 最优）与 `tab:prompt`（zero-shot 0.832 / CoT 0.839 / SelfRef 0.838）。
  3. 新增 `\subsection{Dimension-Space Structure}`（sec:viz）：t-SNE 图 `fig5_tsne_family.png`、家族×维度热图 `fig6_family_dims_heatmap.png`，配 5 类可分 acc 0.640、良性 vs 操纵单维 AUC 0.73-0.94。
- 编译验证：Tectonic 通过，`main.pdf` 13→14 页，10 张图，全部新内容渲染正确。
- 图表文件复制到 `linguistic_agency_paper/figs/`（fig5/fig6）。

### 人工 IAA 材料（2026-08-21，零成本，待人工执行）
- 目录：`experiments/data/iaa/`
  - `annotator_A.xlsx` / `annotator_B.xlsx`：150 条（75 正 / 75 负，seed=2026 分层抽样），每条填 binary + intensity + D1-D7 level（0-3），不含 gold 防引导。
  - `iaa_guidelines.md`：中文标注指南（三步判断流程 + 维度锚点 + 示例）。
  - `iaa_sample_150.json`：抽样带 gold（仅研究者使用）。
  - `compute_iaa.py`：人工-人工 Cohen's κ（binary 未加权，intensity/D1-D7 二次加权）+ 人工-LLM 一致性。
- 自检：同源 LLM 数据填两份模板 → 所有 κ=1.0（公式验证通过）。
- 待办：让 1-2 位同学各标 150 条（约 4-6 小时/人），结果发回后运行 `python experiments/compute_iaa.py`，把 κ 填进论文。

### 人工 IAA 完成并写入论文（2026-08-22，零成本）
- 两份完成标注来自 Downloads：annotator_A_completed.xlsx（150 条全填）/ annotator_B_completed.xlsx（缺 1 条整行，149 对可用），值均为整数、无越界。
- 人工-人工一致性（compute_iaa.py）：
  - binary κ=0.491（moderate）；intensity QWK=0.730（substantial）；D3=0.781、D6=0.760、D2=0.659、D4=0.596、D1=0.552、D7=0.494、D5=0.390（最低，欺骗性主观/低频）。
- 人工-LLM 一致性（备用，未写进论文正文）：binary 0.400、intensity 0.520、D6 0.561、D3 0.518、D1 0.327、D4 0.266、D7 0.254、D2 0.203、D5 0.023 —— LLM 与人工存在系统偏差，可作 limitation 讨论。
- main.tex 更新（脚本 integrate_iaa.py）：Annotation Protocol 的 "left to future work" 改为引用 sec:qc；Quality Control 加 \label{sec:qc} 并追加双人 IAA 段落 + 表 tab:iaa（binary/intensity/D1-D7 共 9 行）。
- 编译验证通过，main.pdf 14 页。

### 归档（2026-08-22）
- `main.tex.bak_iaa`：含 IAA 双人一致性研究 + 全部低成本补强（T/G/B、提示词消融、t-SNE、伦理/Datasheet）的 main.tex 快照。
- `main_iaa_20260822.pdf`：对应 PDF（14 页）。
- 后续页数压缩等修改可在该快照基础上进行，便于回退对比。
