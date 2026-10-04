# LINGUAFORCE

**基于统一测量维度的对话能动效应基准**

[English README](README.md) | [论文 PDF](submission/paper/main.pdf) | [发布包](submission/)

LINGUAFORCE 用于衡量英语多轮对话中的语言如何对听者施加压力、限制选择或引导决策。项目包含七维测量空间、十五类策略体系、对话级参考标签、模型生成的维度表示，以及可确定性复现的实验结果。

## 数据集概况

| 项目 | 内容 |
|---|---|
| 语言 | 英语 |
| 对话数量 | 4,066 |
| 训练集 | 3,432 条 |
| 留出集 | 634 条 |
| 测量维度 | 7 个 |
| 策略类型 | 4 个家族共 15 类 |
| 强度标签 | 0--5 有序等级 |
| 输入形式 | 去标识化多轮对话文本 |

七个维度分别是指令性力量、选项约束、规范压力、情绪压力、欺骗性、攻击性和显性程度。

## 任务

- **T1：压力检测。** 判断对话是否包含面向听者的压力。
- **T2：策略识别。** 预测对话中适用的策略类型。
- **T3：强度预测。** 预测对话级别的 0--5 强度。
- **跨域迁移。** 在相关公开对话资源上评估维度表示的迁移能力。

## 主要结果

发布包包含论文结果所需的输出文件，不需要调用外部模型服务即可验证下列指标。

| 评测划分 | 二分类 AUC | 强度 Spearman rho |
|---|---:|---:|
| 训练集 | 0.852 | 0.651 |
| 留出集 | 0.831 | 0.665 |

跨域 AUROC 分别为：MentalManip 0.742、MultiManip 0.706、TalkDown 0.729、ToxiChat 0.587。

## 仓库结构

```text
.
├── README.md       # 英文说明
├── README_zh.md    # 中文说明
└── submission/
    ├── paper/      # 论文 PDF、源文件、参考文献和图片
    ├── data/       # JSONL 数据和数据说明
    ├── code/       # 评测和分析脚本
    ├── results/    # 论文实验使用的结果文件
    ├── requirements.txt
    ├── CITATION.cff
    └── SHA256SUMS
```

发布包中也提供独立的[英文说明](submission/README.md)和[中文说明](submission/README_zh.md)。

## 复现评测结果

以下命令用于复现发布包中的确定性指标，不需要 API 密钥或 GPU。

```bash
cd submission
python3 -m pip install -r requirements.txt

# 主实验和跨域结果
python3 code/verify_paper.py
python3 code/eval_transfer.py

# 论文中的 CPU 实验
python3 code/ablation_linear_readout.py
python3 code/run_t2_rq3.py
```

重新生成论文图片：

```bash
python3 code/make_figs_release.py
```

发布包包含论文结果对应的 JSONL/JSON 输出文件，不包含模型服务调用、密钥、私人标注工作簿或本地配置文件。

## 数据说明

数据字段、划分统计和使用方式见 [`submission/data/README.md`](submission/data/README.md)。作为评测目标的参考标签与模型生成的强度、维度和策略字段分别保存。[datasheet](submission/data/datasheet.md) 记录了数据用途、隐私范围和已知限制。

## 引用

```bibtex
@misc{linguaforce2026,
  title  = {LINGUAFORCE: Benchmarking Agentive Force in Dialogues via Unified Measurement Dimensions},
  year   = {2026},
  note   = {Research release}
}
```

机器可读的引用信息见 [`submission/CITATION.cff`](submission/CITATION.cff)。

## 规范使用

部分对话包含威胁、辱骂、欺骗或其他令人不适的语言。项目用于对话分析和检测研究，不应被用于监控、对个人作出重要决策，或在没有人工复核的情况下进行自动化审核。

## 许可证

公开再分发前，需要确定适用的数据和文本许可证，并补充必要的署名说明。具体待办见 [`submission/PUBLIC_RELEASE_CHECKLIST.md`](submission/PUBLIC_RELEASE_CHECKLIST.md)。
