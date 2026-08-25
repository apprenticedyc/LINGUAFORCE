# LINGUAFORCE 数据集论文 — Release v1 状态（2026-08-16）

> 论文 `linguistic_agency_paper/main.tex` 为 IEEE 会议格式，当前 12 页，已编译通过
> 发布数据在 `linguistic_agency_paper/data/`，与论文数字一一对应
> 复现方式：运行 `experiments/` 下脚本即可（见第 4 节）

## 1. 已完成事项
- [x] 全量数据 3,432 条 + 634 held-out 已用 DeepSeek 完成 7 维标注，无需 GPU
- [x] 数据集打包：binary / multi / intensity / 七维打分与分级字段齐全
- [x] 论文数字全量可复现：`python experiments/verify_paper.py`
      T1 AUC 0.852 / F1 0.817、T3 Spearman 0.651 / QWK 0.595、跨域 AUROC 0.742/0.706/0.729/0.587
- [x] 发布数据生成：`python experiments/release_dataset.py`
      产出 `data/linguaforce_full.jsonl`（3432 条）与 `data/linguaforce_first_release.jsonl`（634 条）
- [x] 全部图表按发布数据重生成：`python experiments/make_figs_release.py`
- [x] 15 类标注全量化：全量 3,432 条 LLM 多标签标注，发布到 `data/linguaforce_full_types.jsonl`（另 634 版 `data/linguaforce_first_release_types.jsonl`）
- [x] T2 多标签识别（线性读出器，7 维，全量口径，脚本 run_t2_rq3.py）：
      ① Held-out（全量 3,432 训练 → 634 测试）：15 类 macro/micro 0.480/0.668，4 家族 0.772/0.827
      ② 5 折 CV（全量 3,432）：15 类 0.477/0.669，4 家族 0.752/0.820（表 tab:rq2）
- [x] T2 微调基线（RoBERTa-base，表 tab:rq2，脚本 `finetune_t2.py`）：
      ① Held-out：15 类 0.544/0.731，4 家族 0.780/0.820
      ② 5 折 CV：15 类 0.540/0.734，4 家族 0.767/0.815
- [x] RQ3 留一类型零样本：新颖性 AUROC 0.543（类型为共享维度、非独立簇），
      强度分离“是否有策略”AUROC 0.963；跨域零样本为主证据（0.742/0.706/0.729/0.587）
- [x] 消融实验：7 维线性读出器 AUC 0.855 / Spearman 0.674（表 tab:ablation），
      留一维中 D6（毒性）/D3（规范压力）最敏感
- [x] 人工抽检复核：第一作者人工标注 120 条，强度 Spearman 0.79、主机制维度 top-2 81.2%
- [x] 参考文献 24 篇，无 `coercion` 占位条目

## 2. 投稿前待办
- [ ] 作者信息：`main.tex` 中 `First Author, Second Author, Third Author` 需替换为真实作者（含单位/邮箱）
- [ ] COERCION 数据来源：`references.bib` 中需加 `coercion` 真实引用
      （当前为 `Anonymous, Under review`，投稿前替换为师兄论文的题目+作者+年份/公开版本）
- [ ] 数据集许可与授权：确认 COERCION 是否允许二次发布，`data/README.md` 的 provenance 处补充说明
- [ ] 定投稿目标（会议/期刊 + 页数限制，当前 12 页可能需要精简到 8-10 页）

## 3. 后续可扩展方向（非投稿必需）
- 多模型鲁棒性（第二个解析器，如方舟 Qwen）
- RQ4 工程消融（T/G/B 聚合、提示词变体）

## 4. 复现命令
```bash
python experiments/release_dataset.py         # 生成发布数据
python experiments/make_figs_release.py       # 重新生成全部图表
python experiments/verify_paper.py            # 校验论文全部数字
python experiments/ablation_linear_readout.py # 7 维线性读出器 + 留一维消融
python experiments/run_t2_rq3.py              # T2 多标签 + RQ3 留一类型
python experiments/annotate_types.py ...      # 重新生成 15 类标注（需 API）
```
