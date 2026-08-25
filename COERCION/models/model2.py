import torch
import torch.nn as nn
from transformers import RobertaModel

class MoralCoercionModel(nn.Module):
    def __init__(self, model_name='roberta-base', num_fine_labels=6):
        super(MoralCoercionModel, self).__init__()
        # 加载基础的 RoBERTa 模型
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # 针对 CLS token 输出的 Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 精细理解层：将 RoBERTa 的隐藏状态映射到 6 个细分类别
        self.fine_classifier = nn.Linear(self.roberta.config.hidden_size, num_fine_labels)

    def forward(self, input_ids, attention_mask):
        # 1. 提取文本特征
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取 CLS token 的表征 (pooler_output 已经经过了一个带 Tanh 激活的线性层)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 2. 精细分类 (6分类 Logits)
        # 形状: [batch_size, 6]
        fine_logits = self.fine_classifier(pooled_output)

        # 3. 转化为概率分布
        fine_probs = torch.softmax(fine_logits, dim=-1)

        # 4. 概率聚合 -> 二分类
        # 非道德绑架 (标签 0): 包含细分类别 0, 1
        prob_binary_0 = fine_probs[:, 0:2].sum(dim=-1, keepdim=True)
        
        # 存在道德绑架 (标签 1): 包含细分类别 3, 4, 5
        prob_binary_1 = fine_probs[:, 3:6].sum(dim=-1, keepdim=True)

        # 拼接为二分类概率分布, 形状: [batch_size, 2]
        binary_probs = torch.cat([prob_binary_0, prob_binary_1], dim=-1)

        # 5. 转回 Logits (为了兼容 CrossEntropyLoss)
        # 加上极小值 1e-9 防止 log(0) 导致梯度爆炸
        binary_logits = torch.log(binary_probs + 1e-9)

        # 返回两组预测结果，以便在训练时同时计算损失
        return fine_logits, binary_logits