import torch
import torch.nn as nn

class JointCoercionLoss(nn.Module):
    def __init__(self, alpha=0.5, multi_class_weights=None, binary_class_weights=None):
        """
        道德绑架联合损失函数 (Joint Loss)
        结合细粒度 6 分类损失和二分类损失，并支持类别权重以缓解数据不平衡。
        
        参数:
        :param alpha: 多分类损失的权重，范围 [0, 1]。二分类权重为 (1 - alpha)。
                      alpha 越大，模型越侧重于学习细粒度特征。
        :param multi_class_weights: 6分类的类别权重张量 (Tensor)，形状 [6]。
        :param binary_class_weights: 二分类的类别权重张量 (Tensor)，形状 [2]。
        """
        super(JointCoercionLoss, self).__init__()
        self.alpha = alpha
        
        # 定义 6 分类交叉熵损失，支持传入权重
        self.multi_criterion = nn.CrossEntropyLoss(weight=multi_class_weights)
        
        # 定义 2 分类交叉熵损失，支持传入权重
        self.binary_criterion = nn.CrossEntropyLoss(weight=binary_class_weights)

    def forward(self, multi_logits, binary_logits, multi_labels, binary_labels):
        """
        前向传播，计算联合损失。
        
        参数:
        :param multi_logits: 模型的 6分类 Logits 输出，形状 [batch_size, 6]
        :param binary_logits: 模型的 2分类 Logits 输出，形状 [batch_size, 2]
        :param multi_labels: 真实的 6分类标签，形状 [batch_size]
        :param binary_labels: 真实的 2分类标签，形状 [batch_size]
        
        返回:
        :return: loss_total (总损失), loss_multi (多分类损失), loss_binary (二分类损失)
        """
        # 1. 计算细粒度 6 分类损失
        loss_multi = self.multi_criterion(multi_logits, multi_labels)
        # loss_multi = 0
        
        # 2. 计算二分类损失
        loss_binary = self.binary_criterion(binary_logits, binary_labels)
        
        # 3. 按照公式进行加权求和
        loss_total = self.alpha * loss_multi + (1.0 - self.alpha) * loss_binary
        
        return loss_total, loss_multi, loss_binary