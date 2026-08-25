import torch
import torch.nn as nn
from transformers import RobertaModel

class MoralCoercionModel(nn.Module):
    def __init__(self, model_name='roberta-base', num_fine_labels=6):
        super(MoralCoercionModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # --- 核心修改：双分支解耦分类头 ---
        # 1. 专门负责 6 分类的精细头
        self.fine_classifier = nn.Linear(self.roberta.config.hidden_size, num_fine_labels)
        
        # 2. 专门负责二分类的粗粒度头
        self.binary_classifier = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)

        # 两个分类头各自独立计算，互不干扰
        fine_logits = self.fine_classifier(pooled_output)
        binary_logits = self.binary_classifier(pooled_output)

        return fine_logits, binary_logits