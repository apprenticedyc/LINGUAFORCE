# LINGUAFORCE

**对话压力与影响策略的人工标注评测数据集**

[English](README.md) | [论文](paper_acl/main.pdf) | [数据](submission/data/human/) | [复现](submission/README_zh.md)

4,066条英文对话由人工围绕常见日常交流场景收集、整理和编写，原始语言为英语，不作为真实聊天记录样本来描述。

## 数据

| 项目 | 内容 |
|---|---|
| 训练／验证／测试 | 3,088／344／634条 |
| 有压力／无压力 | 3,144／922条 |
| 是否施压 | 0或1 |
| 总体强度 | 0–5 |
| 七个维度 | 各0–3 |
| 十五种策略 | 可多选，各0或1 |

20名能熟练阅读英文对话的计算机专业研究生阅读统一说明后，独立标注各自批次，没有参考他人的结果。参与自愿、无报酬。每条对话由一人标注，未进行重复标注或一致性测量。

## 实验

| 方法 | 设置 | 压力AUROC | 压力F1 |
|---|---|---:|---:|
| GPT-6 Astra | 零样本 | 0.947 | 0.959 |
| Qwen3.8-Max | 零样本 | 0.958 | 0.953 |
| DeepSeek-Flash | 零样本 | 0.954 | 0.950 |
| 词TF-IDF | 有监督 | 0.949 | 0.942 |
| 字符TF-IDF | 有监督 | 0.948 | 0.943 |

完整结果覆盖四项任务，并包含常量基线。Qwen3.8-Max完整有效的七维评分为295条，因此同时提供有效数量、共同子集评价和无效输出计错的准确率。此外提供Qwen3-4B的LoRA微调前后对照：单次训练在各任务上变化较小，平均维度QWK从0.305增至0.312。预测、配置与配对置信区间均随材料提供。

## 目录

- `paper_acl/`：当前ACL匿名论文源码和PDF。
- `submission/`：独立可用的匿名数据、代码、结果和论文。
- `annotation_package/`：本地原始标注表和导入材料，不进入匿名附件。

数据构建见[数据说明](submission/data/datasheet.md)，字段见[格式说明](submission/data/human/README.md)。

## 复现

Python 3.9或以上版本；以下命令不调用API、不需要GPU：

```sh
python -m pip install -r submission/requirements.txt
python submission/code/validate_release.py
python submission/code/score_human_llm.py
python submission/code/bootstrap_human_pressure.py
python submission/code/analyze_human_robustness.py
python submission/code/write_robustness_tables.py
python submission/code/score_qwen4b.py
```

CPU训练和API配置见[复现说明](submission/README_zh.md)。数据含威胁、辱骂等内容；标签不代表实际心理伤害或行为后果。作者允许发布数据，但未指定标准开源许可证，见[使用说明](submission/TERMS.md)。

附件另报告126条含请求／建议或理性说服的无压力测试对话上的误报率，以及强度、策略和七维指标的配对bootstrap区间。这些分析复用固定预测，不代表多次训练或API运行的稳定性。
