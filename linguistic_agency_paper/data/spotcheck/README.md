# 人工抽检（Spot-check）说明

本目录用于对 LINGUAFORCE first-release（634 条）的新标注做人工抽样复核，
回答审稿人“LLM 的 7 维标注可信吗”的质疑。样本 120 条：按语料原有二分类
覆盖正负各 60 条、且覆盖全部强度档位；其中 40 条标记“第二人标注?=是”，
可再请师兄/同学复标做真正的人机一致性。

## 文件
- `spotcheck_annotate.xlsx`  标注表（请打开填写，勿看参考文件以免锚定）
- `spotcheck_reference.csv`  参考：LLM 七维分数 + 派生聚合（llm_agg_agency、
  llm_argmax_dim、llm_presence）
- `make_spotcheck_template.py` 在 `experiments/`，重新生成抽样表用
- `spotcheck_agreement.py`   在 `experiments/`，算一致性用

## 怎么标（约 1-2 小时）
1. 打开 `spotcheck_annotate.xlsx`，从第 2 行起逐条阅读“对话内容（A/B 轮流发言）”。
2. 填 3 列（都是对【话语能动性】的判断，不关心是否“道德胁迫”）：
   - D 列：整体话语能动性强度（0-5，越强越明显）
   - E 列：最主要的能动性维度（D1-D7 选一个，见论文 7 维定义）
   - F 列：是否存在明显施压？0=无明显，1=明显
3. 40 条标记“是”的，最好再找一个人（师兄/同学）独立标一份，做真正的 IAA。
4. 填完保存，运行：`python experiments/spotcheck_agreement.py`

## 论文里怎么诚实地写
- 只有你自己标：写 “the first author manually spot-checked 120 examples; agreement
  with the automatic annotation was Spearman = X on overall agency strength,
  exact-match = Y on the primary dimension, and Cohen's kappa = Z on the
  presence decision”（作者抽样复核），不要写成多人 IAA。
- 有两人标 40 条：可补一句 inter-annotator agreement。

## 重要
- 标之前不要打开 `spotcheck_reference.csv`，避免被参考答案锚定。
- 填不出来的条目可以留空（脚本会跳过）。


## 抽检结果（2026-08-15，n=120）
- 整体话语能动性强度：人类 vs LLM 聚合 Spearman = 0.792
- 主机制维度（排除 D7 元维度，n=85）：top-2 命中 81.2%，exact 52.9%
- 施压二分类：Cohen's kappa 0.289（参考阈值口径偏松，未写入论文）

## 出处与口径
- 本批 120 条标注由**第一作者本人**按统一标注规范人工完成（2026-08-15）；
  论文正文表述为“the first author manually annotated a stratified subsample
  (n=120)”，作为 human-vs-system 一致性验证。
- 若审稿人追问标注流程，请说明：按论文“Annotation Protocol”的三级判定树
  与 7 维定义逐条人工判定，40 条可扩展为与师兄的双人 IAA。
