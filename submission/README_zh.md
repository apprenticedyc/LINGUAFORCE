# LINGUAFORCE 发布包

[English README](README.md) | [论文 PDF](paper/main.pdf) | [仓库总览](../README_zh.md)

本目录是 LINGUAFORCE 的发布包，包含论文文件、数据、论文实验使用的结果文件和分析代码。

## 目录内容

```text
paper/              论文 PDF、源文件、参考文献和图片
data/               JSONL 数据、README 和 datasheet
code/               评测与分析脚本
results/            论文实验使用的结果文件
requirements.txt    分析脚本所需的 Python 依赖
CITATION.cff        机器可读的引用信息
SHA256SUMS          发布文件校验值
```

## 复现论文结果

以下命令只使用本发布包中的文件，不需要 API 密钥或 GPU。

```bash
python3 -m pip install -r requirements.txt

python3 code/verify_paper.py
python3 code/eval_transfer.py
python3 code/ablation_linear_readout.py
python3 code/run_t2_rq3.py
```

重新生成论文图片：

```bash
python3 code/make_figs_release.py
```

结果文件用于在不重新调用外部模型服务的情况下核验论文指标。发布包不包含模型服务密钥、私人标注工作簿或本地配置文件。

## 数据和标签

数据文件包含参考压力标签，以及模型生成的强度、维度和策略字段。字段定义见 [`data/README.md`](data/README.md)，数据用途、隐私范围和限制见 [`data/datasheet.md`](data/datasheet.md)。

## 论文文件

`paper/` 包含论文 PDF、LaTeX 源文件、参考文献和图片，用于阅读和引用。论文编译不属于实验复现流程。

## 发布前检查

公开再分发前，需要完成许可证、署名说明、作者信息和 clean checkout 验证。检查清单见 [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md)。
