# 不微调 No-Finetuned
=================RoBERTa-LARGE-GOEMOTIONS=======================
[TEST FINISHED]
  > 样本总数  : 634
  > Accuracy  : 45.27% (阈值 0.5)       
  > MAE       : 0.2847
  > Spearman  : -0.1769
================================================================


==================RoBERTa-Moral-Emotion-ENG======================
[TEST FINISHED]
  > 样本总数  : 634
  > Accuracy  : 53.9432% (阈值: 0.5)
  > MAE       : 0.2816
  > Spearman  : 0.1364
=================================================================


=================MoralFoundationsClassifier=======================
[TEST FINISHED]
  > 样本总数  : 634
  > Accuracy  : 47.7918% (阈值: 0.5)
  > MAE       : 0.2953
  > Spearman  : -0.1705
==================================================================


# 微调 Finetuned
## MSELOSS
=================RoBERTa-LARGE-GOEMOTIONS=======================
[Epoch 1] Valid MAE: 0.0797, Spearman: 0.8952, Accuracy: 88.6435%
[Epoch 2] Valid MAE: 0.1411, Spearman: 0.8313, Accuracy: 78.2334%


==================== 最终测试结果Epoch 1 ====================
Spearman 相关系数: 0.8882
MAE 平均绝对误差: 0.0852
最终分类准确率:   88.6435%
==================================================
================================================================



==================RoBERTa-Moral-Emotion-ENG-o======================
[Epoch 1] Valid MAE: 0.1226, Spearman: 0.8405, ACC: 0.8833
[Epoch 2] Valid MAE: 0.1301, Spearman: 0.8509, ACC: 0.8139
[Epoch 3] Valid MAE: 0.1239, Spearman: 0.8651, ACC: 0.8644
[Epoch 4] Valid MAE: 0.0922, Spearman: 0.9106, ACC: 0.8912
[Epoch 5] Valid MAE: 0.0687, Spearman: 0.9080, ACC: 0.9132
=================================================================


==================RoBERTa-Moral-Emotion-ENG======================
[Validation] MAE: 0.1401, Spearman: 0.7570, ACC: 0.8644
[Validation] MAE: 0.1272, Spearman: 0.7758, ACC: 0.8628
[Validation] MAE: 0.1181, Spearman: 0.8095, ACC: 0.8675
[Validation] MAE: 0.1273, Spearman: 0.7953, ACC: 0.8502
[Validation] MAE: 0.1301, Spearman: 0.7848, ACC: 0.8644

==================== 最终测试 (Test Set) ====================
Spearman: 0.8177
MAE:      0.1237
Accuracy: 85.9621%
=============================================================
=================================================================



=================MoralFoundationsClassifier-o=======================
[Epoch 1] Valid MAE: 0.1957, Spearman: 0.6298, ACC: 72.0820%
[Epoch 2] Valid MAE: 0.1673, Spearman: 0.7242, ACC: 79.4953%
[Epoch 3] Valid MAE: 0.1692, Spearman: 0.7062, ACC: 75.3943%
[Epoch 4] Valid MAE: 0.1605, Spearman: 0.7459, ACC: 77.9180%
[Epoch 5] Valid MAE: 0.1328, Spearman: 0.7817, ACC: 82.0189%

==================== 最终测试 (Test Set) ====================
最终测试结果 (用于论文汇报):
Spearman 相关系数: 0.7876
MAE 平均绝对误差: 0.1375
分类准确率 (Acc):  82.0189%
==================================================




=================MoralFoundationsClassifier=======================
[Epoch 1] Valid MAE: 0.1846, Spearman: 0.5964, ACC: 79.6530%
[Epoch 2] Valid MAE: 0.1775, Spearman: 0.6219, ACC: 78.2334%
[Epoch 3] Valid MAE: 0.1667, Spearman: 0.6728, ACC: 82.8076%
[Epoch 4] Valid MAE: 0.1661, Spearman: 0.7129, ACC: 81.3880%
[Epoch 5] Valid MAE: 0.1514, Spearman: 0.7397, ACC: 80.5994%

==================== 最终测试 (Test Set) ====================
最终测试结果 (用于论文汇报):
Spearman 相关系数: 0.6263
MAE 平均绝对误差: 0.1848
分类准确率 (Acc):  77.4448%
==================================================

==================================================================


==================deberta-v3-large-emotion======================
[Epoch 1 Validation] MAE: 0.1771, Spearman: 0.7694, ACC: 0.8360
[Epoch 2 Validation] MAE: 0.1159, Spearman: 0.8272, ACC: 0.8738
[Epoch 3 Validation] MAE: 0.1052, Spearman: 0.8406, ACC: 0.8849
[Epoch 4 Validation] MAE: 0.0935, Spearman: 0.8425, ACC: 0.8849
[Epoch 5 Validation] MAE: 0.0892, Spearman: 0.8544, ACC: 0.9022

==================== 最终测试 (Test Set) ====================
最终测试结果 (DeBERTa-v3-Large):
Spearman: 0.7494
MAE:      0.1874
Accuracy: 84.0694%
=============================================================

==================== 最终测试 (Test Set) ====================
最终测试结果 (DeBERTa-v3-Large):
Spearman: 0.8231
MAE:      0.1026
Accuracy: 86.9085%
=================================================================

## LLM
===========================DeepSeekR1=============================
[ZH TEST FINISHED]
  > 样本总数  : 634
  > Spearman  : 0.4789
  > MAE       : 0.2308
  > Accuracy  : 0.7050 (Threshold: 0.5)

[EN TEST FINISHED]
  > 样本总数  : 634
  > Spearman  : 0.5141
  > MAE       : 0.2374
  > Accuracy  : 0.7098 (Threshold: 0.5)



  样本总数: 634
  MAE: 0.1397
  Spearman Correlation: 0.7824
  Accuracy (Threshold 0.5): 0.8770
==================================================================

===========================GPT-4o=============================
[ZH TEST FINISHED]
  > 样本总数  : 634
  > Spearman  : 0.
  > MAE       : 0.
  > Accuracy  : 0. (Threshold: 0.5)

[EN TEST FINISHED]
  > 样本总数: 634
  > MAE: 0.2418
  > Spearman Correlation: 0.4781
  > Accuracy (Threshold 0.5): 0.6767



  样本总数: 634
  MAE: 0.0893
  Spearman Correlation: 0.8116
  Accuracy (Threshold 0.5): 0.9227


  样本总数: 634
  MAE: 0.1671
  Spearman Correlation: 0.6698
  Accuracy (Threshold 0.5): 0.8281
==================================================================

===========================llama-3.3-70b-instruct=============================
[ZH TEST FINISHED]
  > 样本总数  : 634
  > Spearman  : 0.
  > MAE       : 0.
  > Accuracy  : 0. (Threshold: 0.5)

[EN TEST FINISHED]
  > 样本总数: 634
  > MAE: 0.2507
  > Spearman Correlation: 0.4512
  > Accuracy (Threshold 0.5): 0.6719



  样本总数: 634
  MAE: 0.0603
  Spearman Correlation: 0.8640
  Accuracy (Threshold 0.5): 0.9574

  样本总数: 634
  MAE: 0.1747
  Spearman Correlation: 0.6504
  Accuracy (Threshold 0.5): 0.8202
==================================================================


