import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.loss import JointCoercionLoss

def train_or_eval_model(model, args, epoch, dataloader, optimizer=None, scheduler=None, mode="train", device="cuda"):
    """
    训练或验证/评估模型的核心循环
    """
    if mode == "train":
        model.train()
    else:
        model.eval()

    # 初始化损失计算器 (获取命令行传入的 alpha 权重，默认 0.5)
    alpha = getattr(args, 'alpha', 0.5)
    # alpha = getattr(args, 'alpha', 0)
    criterion = JointCoercionLoss(alpha=alpha).to(device)

    total_loss = 0.0
    
    # 容器：用于在整个 Epoch 结束后统一计算全局指标
    all_binary_preds = []
    all_binary_labels = []
    all_multi_preds = []
    all_multi_labels = []

    # 使用 tqdm 显示进度条
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [{mode.upper()}]", leave=False, ascii=True)

    for batch in progress_bar:
        # 1. 数据迁移到 GPU/CPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        multi_labels = batch['multi_label'].to(device)
        binary_labels = batch['binary_label'].to(device)

        # 训练模式下清空梯度
        if mode == "train":
            optimizer.zero_grad()

        # 2. 前向传播与损失计算 (验证模式下关闭梯度图计算以节省显存)
        with torch.set_grad_enabled(mode == "train"):
            fine_logits, binary_logits = model(input_ids, attention_mask)
            
            loss_total, loss_multi, loss_binary = criterion(
                fine_logits, binary_logits, multi_labels, binary_labels
            )

            # 3. 反向传播与参数更新
            if mode == "train":
                loss_total.backward()
                # 梯度裁剪：防止 Transformer 模型训练初期梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step() # 更新学习率

        total_loss += loss_total.item()

        # 4. 收集当前 batch 的预测结果
        # 使用 argmax 获取概率最大的类别索引
        binary_preds = torch.argmax(binary_logits, dim=-1).detach().cpu().numpy()
        multi_preds = torch.argmax(fine_logits, dim=-1).detach().cpu().numpy()
        
        all_binary_preds.extend(binary_preds)
        all_binary_labels.extend(binary_labels.cpu().numpy())
        
        all_multi_preds.extend(multi_preds)
        all_multi_labels.extend(multi_labels.cpu().numpy())
        
        # 更新进度条显示的实时 Loss
        progress_bar.set_postfix({'loss': f"{loss_total.item():.4f}"})

    # --- 5. 计算 Epoch 级别的核心评估指标 ---
    avg_loss = total_loss / len(dataloader)
    
    # 【核心修改点】计算你需要的四个二分类参数
    # pos_label=1 意味着我们将“存在道德绑架 (标签1)”视为正例
    # zero_division=0 防止在训练初期模型输出单一类别时除以零报错
    b_acc = accuracy_score(all_binary_labels, all_binary_preds)
    b_prec = precision_score(all_binary_labels, all_binary_preds, pos_label=1, zero_division=0)
    b_rec = recall_score(all_binary_labels, all_binary_preds, pos_label=1, zero_division=0)
    b_f1 = f1_score(all_binary_labels, all_binary_preds, pos_label=1, zero_division=0)
    
    # 顺便记录一下 6 分类的简单准确率，用于观察精细理解层的学习进度
    m_acc = accuracy_score(all_multi_labels, all_multi_preds)

    # 打印该 Epoch 的总结数据
    print(f"[{mode.upper()}] Loss: {avg_loss:.4f} | "
          f"Bin_Acc: {b_acc:.4f} | Bin_Prec: {b_prec:.4f} | Bin_Rec: {b_rec:.4f} | Bin_F1: {b_f1:.4f} | "
          f"Multi_Acc: {m_acc:.4f}")

    # 6. 将所有指标打包返回，供 train.py 记录到 history 字典中
    return {
        'loss': avg_loss,
        'binary_acc': b_acc,
        'binary_precision': b_prec,
        'binary_recall': b_rec,
        'binary_f1': b_f1,
        'multi_acc': m_acc
    }