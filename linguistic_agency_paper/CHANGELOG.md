# LINGUAFORCE 论文：决策与变更记录（长期跟踪）

> 用途：沉淀本论文自 2026-08 起的全部关键决策、改动与数字，供长期跟踪、复现、交接使用。
> 论文：`linguistic_agency_paper/main.tex`（IEEE 会议格式，当前 11 页，编译通过）
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
### 2026-08-22 — 压缩至 11 页（与师兄论文体量相当）
- 按顺序执行 7 个压缩脚本（figs/mech/rw/framework/exp/dataset/misc）：删除 4 张 first-release 重复图、图宽 0.92→0.78、RW 841→437 词、Framework/Experimental/Dataset/Limitations 段落精简。
- 进一步压页：浮动体间距收紧（textfloatsep/floatsep/intextsep≈6pt）、单栏图 0.78→0.70、fig5 0.56→0.50、fig6 0.98→0.85、参考文献 \\small + itemsep 1pt。
- 编译 11 页通过；30 个 \\ref / 15 个 \\cite 全部解析；16 表 + 9 图 + 15 条参考文献完整。备份：main.tex.bak_compact11。

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