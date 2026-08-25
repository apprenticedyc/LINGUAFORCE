# 租算力跑 T2 5-fold CV（RoBERTa 微调）

本地 6GB 显卡因热降频要跑 4 小时以上，租一台云 GPU 可在 15~30 分钟内跑完。

## 1. 要上传的文件（保持相对路径）
只需 2 个路径的东西：

- 脚本: `experiments/finetune_t2.py`
- 数据（4 个文件）:
  - `linguistic_agency_paper/data/linguaforce_full.jsonl`
  - `linguistic_agency_paper/data/linguaforce_full_types.jsonl`
  - `linguistic_agency_paper/data/linguaforce_first_release.jsonl`
  - `linguistic_agency_paper/data/linguaforce_first_release_types.jsonl`

上传后保持目录结构：
```
<项目根>/
  experiments/finetune_t2.py
  linguistic_agency_paper/data/xxx.jsonl   (4 个)
```

## 2. 环境（租机一般自带，缺啥装啥）
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.33.3 scikit-learn numpy
```
（torch 版本 2.x 均可；脚本已兼容。）

## 3. roberta-base 模型下载
- 国内租机：`export HF_ENDPOINT=https://hf-mirror.com`（或 Windows: `$env:HF_ENDPOINT=...`）
- 国外租机：直接可下载，无需设置。
首次会自动下载到缓存，之后秒加载。

## 4. 运行命令
按显存选 batch（越大越快）：
- 24GB（4090/A5000 等）:
  `python experiments/finetune_t2.py --mode cv5 --epochs 3 --batch_size 32 --grad_accum 1 --progress t2_cv5_rental.log --out t2_finetune_cv5_rental.json`
- 16GB:
  `... --batch_size 16 --grad_accum 1 ...`
- 12GB:
  `... --batch_size 8 --grad_accum 1 ...`
- 8GB:
  `... --batch_size 4 --grad_accum 2 ...`

预期耗时：4090 约 10~20 分钟；A100 更快。

## 5. 看进度
进度实时写入 `t2_progress.log`（默认）或 `--progress` 指定的文件：
```
tail -f t2_cv5_rental.log
```
能看到：MAIN start → fold 1/5 start → step 200/2061 → epoch → fold 1/5 done → … → CV5 COMPLETE。

## 6. 回传结果
只需两个文件：
- `t2_finetune_cv5_rental.json`（最终结果，填表用）
- `t2_cv5_rental.log`（进度+每折指标，论文复现描述可引用）

## 7. 参考数字（线性读出器，用于核对量级）
15 types: held-out 0.480/0.668, 5-fold CV 0.477/0.669
4 families: held-out 0.772/0.827, 5-fold CV 0.752/0.820
RoBERTa held-out: 15 types 0.544/0.731, 4 families 0.780/0.820
CV 结果应接近 held-out（±0.03 以内），若差异大先检查代码/数据是否传全。