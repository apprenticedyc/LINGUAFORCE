import argparse
import os
import json
from datetime import datetime
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# 导入自定义模块 (请确保路径和类名与你的项目结构一致)
from models.model import MoralCoercionModel
from inputters.dataloader import get_MoralCoercion_loaders
from utils.common import seed_everything, save_training_curves
from utils.train_eval import train_or_eval_model
from utils.test import test_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument('--train_data_path', type=str, default='inputters/data/train.jsonl')
    parser.add_argument('--valid_data_path', type=str, default='inputters/data/valid.jsonl')
    parser.add_argument('--test_data_path', type=str, default='inputters/data/test.jsonl')
    parser.add_argument('--save_path', type=str, default='save/')
    
    # 训练超参
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5) 
    
    # 联合损失超参
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for multi-class loss (0 to 1)')
    
    # 实验配置
    parser.add_argument('--pretrained_model', type=str, default='D:/model/roberta-base') 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--is_test', action='store_true')
    args = parser.parse_args()

    # 确保保存路径存在
    os.makedirs(args.save_path, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # 1. 准备 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # 2. 加载数据与模型
    train_loader, valid_loader, test_loader = get_MoralCoercion_loaders(args, tokenizer)
    model = MoralCoercionModel(model_name=args.pretrained_model, num_fine_labels=6).to(device)

    # 3. 优化器设计 (Standard NLP weight decay exclusion)
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # 4. 训练阶段
    if args.is_train:
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * args.warmup_rate), 
            num_training_steps=total_steps
        )
        
        best_val_binary_f1 = -1.0 
        patience_counter = 0

        # 初始化用于记录核心指标的字典
        history = {
            "train_loss": [], "val_loss": [],
            "train_multi_acc": [], "val_multi_acc": [],
            "train_binary_acc": [], "val_binary_acc": [],
            "train_binary_prec": [], "val_binary_prec": [],
            "train_binary_rec": [], "val_binary_rec": [],
            "train_binary_f1": [], "val_binary_f1": []
        }

        for epoch in range(args.epochs):
            print(f"\n========== Epoch {epoch+1}/{args.epochs} ==========")
            
            # 训练与验证
            train_res = train_or_eval_model(model, args, epoch, train_loader, optimizer, scheduler, mode="train", device=device)
            val_res = train_or_eval_model(model, args, epoch, valid_loader, mode="eval", device=device)

            # 记录每一轮的 Loss 和 Multi-Acc
            history["train_loss"].append(train_res['loss'])
            history["val_loss"].append(val_res['loss'])
            history["train_multi_acc"].append(train_res['multi_acc'])
            history["val_multi_acc"].append(val_res['multi_acc'])

            # 记录二分类的 4 个核心指标
            history["train_binary_acc"].append(train_res['binary_acc'])
            history["val_binary_acc"].append(val_res['binary_acc'])
            history["train_binary_prec"].append(train_res['binary_precision'])
            history["val_binary_prec"].append(val_res['binary_precision'])
            history["train_binary_rec"].append(train_res['binary_recall'])
            history["val_binary_rec"].append(val_res['binary_recall'])
            history["train_binary_f1"].append(train_res['binary_f1'])
            history["val_binary_f1"].append(val_res['binary_f1'])

            # 基于二分类 F1-Score 的早停与模型保存
            current_f1 = val_res['binary_f1']
            if current_f1 > best_val_binary_f1:
                best_val_binary_f1 = current_f1
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.save_path, f"{epoch}_best_mcc_model.pt"))
                print(f"[*] Best Model Updated. Val Binary F1: {best_val_binary_f1:.4f}")
            else:
                patience_counter += 1
                print(f"[*] No improvement. Patience: {patience_counter}/{args.patience}")
                torch.save(model.state_dict(), os.path.join(args.save_path, f"{epoch}_not_best_mcc_model.pt"))
            
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        # 训练结束后，保存指标数据与图表
        current_time = datetime.now().strftime("%m%d_%H%M")
        
        # 1. 保存折线图
        save_name_img = os.path.join(args.save_path, f"results_{current_time}.png")
        save_training_curves(history, save_name_img)
        print(f"[*] Training curves saved to {save_name_img}")

        # 2. 将指标字典保存为 JSON
        save_name_json = os.path.join(args.save_path, f"metrics_{current_time}.json")
        with open(save_name_json, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        print(f"[*] Metrics history saved to {save_name_json}")

    # 5. 测试阶段
    if args.is_test:
        model.load_state_dict(torch.load(os.path.join(args.save_path, "3_not_best_mcc_model.pt")))
        test_res = test_model(model, args, test_loader, device=device)
        print("\n" + "="*50)
        print(f"TEST METRICS:")
        print(f"Total Loss: {test_res['loss']:.4f}")
        print(f"Multi-class (6) Acc: {test_res['multi_acc']:.4f}")
        print(f"Binary-class (2) -> Acc: {test_res['binary_acc']:.4f} | Prec: {test_res['binary_precision']:.4f} | Rec: {test_res['binary_recall']:.4f} | F1: {test_res['binary_f1']:.4f}")
        print("="*50)