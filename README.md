# LINGUAFORCE

**Benchmarking Agentive Force in Dialogues via Unified Measurement Dimensions**

LINGUAFORCE is a benchmark for measuring how language can pressure, constrain,
or steer a listener's decision in English multi-turn dialogue. The release
combines a seven-dimension measurement space, a fifteen-type strategy
taxonomy, dialogue-level intensity labels, and reproducible evaluation outputs.

LINGUAFORCE 是一个用于分析英语多轮对话中语言如何施加压力、限制选择或引导听者决策的基准数据集。项目提供七维测量空间、十五类策略分类体系、对话级强度标签，以及可复现的实验结果。

## Overview | 项目概览

| Item | Details |
|---|---|
| Language / 语言 | English / 英语 |
| Dialogues / 对话数 | 4,066 |
| Training split / 训练集 | 3,432 dialogues |
| Held-out split / 留出集 | 634 dialogues |
| Dimensions / 测量维度 | 7 |
| Strategy types / 策略类型 | 15 types in 4 families / 4 个家族共 15 类 |
| Intensity / 强度 | 0--5 ordinal scale / 0--5 有序等级 |
| Main modalities / 数据形式 | De-identified multi-turn text / 去标识化多轮文本 |

The seven dimensions are directive force, option constraint, normative
pressure, emotional pressure, deceptiveness, toxicity, and explicitness.

七个维度分别是指令性力量、选项约束、规范压力、情绪压力、欺骗性、攻击性和显性程度。

## Tasks and evaluation | 任务与评测

The release supports three main tasks and a transfer evaluation:

项目包含三个主要任务和一项迁移评测：

1. **T1: Binary detection**: predict whether a dialogue contains
   listener-directed pressure.
2. **T2: Strategy recognition**: predict the applicable strategy types.
3. **T3: Intensity prediction**: estimate the dialogue-level intensity on a
   0--5 scale.
4. **Transfer evaluation**: test whether the dimension representation transfers
   to related dialogue resources.

1. **T1：二分类检测**：判断对话是否包含面向听者的压力。
2. **T2：策略识别**：预测对话中出现的策略类型。
3. **T3：强度预测**：预测对话级别的 0--5 强度。
4. **迁移评测**：检验维度表示能否迁移到相关对话数据。

## Reported results | 主要结果

The deterministic verification script reports the following headline metrics:

验证脚本复现的主要指标如下：

| Evaluation | Binary AUC | Intensity Spearman |
|---|---:|---:|
| Training split | 0.852 | 0.651 |
| Held-out split | 0.831 | 0.665 |

Cross-domain AUROC is 0.742 on MentalManip, 0.706 on MultiManip, 0.729 on
TalkDown, and 0.587 on ToxiChat.

跨领域 AUROC 分别为：MentalManip 0.742、MultiManip 0.706、TalkDown 0.729、ToxiChat 0.587。

## Repository structure | 仓库结构

```text
.
├── README.md
└── submission/
    ├── paper/       # Paper source, PDF, bibliography, and figures
    ├── data/        # JSONL releases and data documentation
    ├── code/        # Evaluation and reproduction scripts
    ├── results/     # Released outputs used by the paper
    ├── requirements.txt
    ├── CITATION.cff
    └── SHA256SUMS
```

```text
.
├── README.md       # 仓库总说明
└── submission/
    ├── paper/       # 论文源文件、PDF、参考文献和图片
    ├── data/        # JSONL 数据文件和数据说明
    ├── code/        # 评测与复现实验脚本
    ├── results/     # 论文使用的已发布结果
    ├── requirements.txt
    ├── CITATION.cff
    └── SHA256SUMS
```

## Reproduction | 复现方法

Run the commands from `submission/`:

在 `submission/` 目录下运行：

```bash
python3 code/verify_paper.py
python3 code/make_figs_release.py
python3 code/ablation_linear_readout.py
python3 code/run_t2_rq3.py
```

To compile the paper:

编译论文：

```bash
cd paper
tectonic -X compile main.tex
```

The package includes the released outputs needed for deterministic checks.
External model-provider calls, credentials, and private workbooks are not
included.

发布包中包含复现确定性评测所需的结果文件，不包含外部模型服务调用、密钥或私人标注工作簿。

## Data and responsible use | 数据与规范使用

The data consists of de-identified English dialogue text and derived
annotations for research use. Some dialogues contain threats, insults,
deception, or other distressing language. The release should not be used for
surveillance, consequential decisions about individuals, or automated
moderation without human review.

数据由去标识化英语对话文本及其派生标注组成，仅供研究使用。部分对话包含威胁、辱骂、欺骗或其他令人不适的语言内容。不得将其用于监控、对个人作出重要决策，或在没有人工复核的情况下进行自动化审核。

Review the documentation in [`submission/data/`](submission/data/) and the
release checklist before redistribution. The applicable license and required
attribution notice must be finalized before public release.

重新发布前请阅读 [`submission/data/`](submission/data/) 中的数据说明和发布清单，并确认适用许可证及必要的署名信息。

## Citation | 引用

```bibtex
@misc{linguaforce2026,
  title  = {LINGUAFORCE: Benchmarking Agentive Force in Dialogues via Unified Measurement Dimensions},
  year   = {2026},
  note   = {Research release}
}
```

The machine-readable citation record is available at
[`submission/CITATION.cff`](submission/CITATION.cff).

机器可读的引用信息见 [`submission/CITATION.cff`](submission/CITATION.cff)。

## Contact | 联系方式

See the author and venue metadata in the paper source and `CITATION.cff`.

作者和投稿 venue 信息请以论文源文件和 `CITATION.cff` 为准。
