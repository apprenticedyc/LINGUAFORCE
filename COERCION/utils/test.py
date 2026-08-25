import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from utils.loss import JointCoercionLoss

def test_model(model, args, dataloader, device="cuda"):
    """
    模型测试函数
    在测试集上评估模型，并输出详细的分类报告和混淆矩阵
    """
    model.eval()

    # 初始化损失计算器
    alpha = getattr(args, 'alpha', 0.5)
    criterion = JointCoercionLoss(alpha=alpha).to(device)

    total_loss = 0.0
    
    # 容器：收集所有预测和真实标签
    all_binary_preds = []
    all_binary_labels = []
    all_multi_preds = []
    all_multi_labels = []

    progress_bar = tqdm(dataloader, desc="[TESTING]", leave=False, ascii=True)

    # 关闭梯度计算，节省显存并加速
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            multi_labels = batch['multi_label'].to(device)
            binary_labels = batch['binary_label'].to(device)

            # 前向传播
            fine_logits, binary_logits = model(input_ids, attention_mask)
            
            # 计算损失
            loss_total, loss_multi, loss_binary = criterion(
                fine_logits, binary_logits, multi_labels, binary_labels
            )
            total_loss += loss_total.item()

            # 获取预测结果
            binary_preds = torch.argmax(binary_logits, dim=-1).cpu().numpy()
            multi_preds = torch.argmax(fine_logits, dim=-1).cpu().numpy()
            
            all_binary_preds.extend(binary_preds)
            all_binary_labels.extend(binary_labels.cpu().numpy())
            
            all_multi_preds.extend(multi_preds)
            all_multi_labels.extend(multi_labels.cpu().numpy())

    # --- 计算核心评估指标 ---
    avg_loss = total_loss / len(dataloader)
    
    # 二分类指标
    b_acc = accuracy_score(all_binary_labels, all_binary_preds)
    b_prec = precision_score(all_binary_labels, all_binary_preds, pos_label=1, zero_division=0)
    b_rec = recall_score(all_binary_labels, all_binary_preds, pos_label=1, zero_division=0)
    b_f1 = f1_score(all_binary_labels, all_binary_preds, pos_label=1, zero_division=0)
    
    # 六分类准确率
    m_acc = accuracy_score(all_multi_labels, all_multi_preds)

    # --- 打印详细的测试报告 ---
    print("\n" + "="*50)
    print(" " * 15 + "TESTING REPORT")
    print("="*50)
    
    # 1. 打印二分类详细报告
    print("\n[Binary Classification (0: Normal, 1: Coercion)]")
    print(classification_report(all_binary_labels, all_binary_preds, digits=4, zero_division=0))
    
    print("\n[Binary Confusion Matrix]")
    # 混淆矩阵格式：
    # [[TN, FP],
    #  [FN, TP]]
    cm_binary = confusion_matrix(all_binary_labels, all_binary_preds)
    print(cm_binary)

    # 2. 打印六分类详细报告
    # 对于分布不均的数据（如中间标签 1, 2, 3 较少），查看这部分的 recall 和 f1 尤为重要
    print("\n[Fine-grained 6-Class Classification]")
    target_names = [
        "0 (None)", "1 (Slight Suspicion)",  
        "2 (Slight Bias)", "3 (Obvious Hint)", 
        "4 (Egocentric)", "5 (Extreme)"
    ]
    # 注意：如果测试集中缺少某些类别，classification_report 会报错或警告，所以传入实际存在的 labels
    unique_labels = sorted(list(set(all_multi_labels)))
    print(classification_report(
        all_multi_labels, 
        all_multi_preds, 
        labels=unique_labels,
        target_names=[target_names[i] for i in unique_labels],
        digits=4, 
        zero_division=0
    ))

    # --- 组装返回值 ---
    return {
        'loss': avg_loss,
        'binary_acc': b_acc,
        'binary_precision': b_prec,
        'binary_recall': b_rec,
        'binary_f1': b_f1,
        'multi_acc': m_acc,
        'all_binary_preds': all_binary_preds, # 返回预测值方便后续做 Bad Case 分析
        'all_binary_labels': all_binary_labels
    }