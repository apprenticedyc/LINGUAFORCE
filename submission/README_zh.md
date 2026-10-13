# LINGUAFORCE：匿名补充材料

[English](README.md)

本附件包含当前论文、4,066条人工标注对话、固定划分及四项任务的实验结果。训练／验证／测试分别为3,088／344／634条。每条对话由一人标注，未测量标注者间一致性。

ARR的数据包与代码包分别上传。将两个ZIP解压到同一个目录，合并其中的 `submission/` 文件夹；重叠的README和权限文件内容相同。论文PDF另行上传。

## 内容

- `data/human/`：对话、人工标签、统计与划分。
- `data/datasheet.md`：构建、标注和适用范围。
- `data/annotation_guide*.md`：中文原始说明及英文译文。
- `code/`：验证、训练、推理、评分和测试。
- `results/`：三种API大模型、有监督基线以及Qwen3-4B微调前后结果。
- `SHA256SUMS_DATA`、`SHA256SUMS_CODE`：两个附件各自的文件校验值。

仓库中的 `paper/` 和总校验文件 `SHA256SUMS` 是本地发布副本，不包含在两个上传附件内。

## 复现

在本目录使用Python 3.9或以上版本运行，不需要API或GPU：

```sh
python -m pip install -r requirements.txt
python code/validate_release.py
python -m unittest discover -s code -p 'test_*.py'
python code/score_human_llm.py
python code/bootstrap_human_pressure.py
python code/write_human_result_tables.py
python code/analyze_human_robustness.py
python code/write_robustness_tables.py
python code/score_qwen4b.py
```

CPU重新训练先运行 `python code/train_human_baselines.py`，再运行评分等命令。这会覆盖有监督预测。API配置见英文README和代码说明；已有输出会复用，重新调用可能收费。

Qwen3.8-Max完整有效的七维输出只有295条，维度分数须结合覆盖率和共同子集比较。无效输出不裁剪；全测试集准确率将其计错。跨模型基线比较不等于同一模型微调前后的比较。

另有Qwen3-4B原始模型与LoRA微调模型的对照，使用相同测试请求和 `/no_think` 指令。预测、配置与配对区间见 `results/qwen3_4b/`，可以离线复算；附件不包含平台训练后端或适配器权重。

使用权限见 `TERMS.md`。附件不包含原始工作簿、作者信息、密钥或内部地址。

`robustness.json`记录请求／建议和理性说服负例的误报数量、比例、Wilson区间及样本ID，并给出强度、策略、七维和全测试集维度准确率的配对bootstrap区间。重叠负例只计一次；维度比较明确区分完整有效样本与共同子集。少数重采样中的QWK不可定义，区间排除这些抽样并报告数量，不把它们记成零。分析不重新训练或调整阈值，不能替代多次运行。
