# 文件: train_eval.py
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

import nltk
from nltk.corpus import wordnet, stopwords
from utils.constants import WORD_PAIRS as word_pairs
stop_words = stopwords.words("english")
# try:
#     stop_words = stopwords.words("english")
# except LookupError:
#     # 下载停用词资源  
#     nltk.download('stopwords') 
#     stop_words = stopwords.words("english")



# 随机数种子
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

        

def save_training_curves(history, save_path):
    """
    绘制并保存分类任务的训练曲线 (包含 Loss, Acc, Precision, Recall, F1)
    
    :param history: train.py 中记录的指标字典
    :param save_path: 图片保存路径
    """
    # 设置图表风格（如果你的环境中没有 seaborn，可以注释掉这行，不影响运行）
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.grid(True)

    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建 2x2 的画布，设置较高的分辨率以满足论文发表要求
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle('Moral Coercion Detection Training Metrics', fontsize=16, fontweight='bold', y=0.98)

    # --- 1. Top-Left: Training & Validation Loss ---
    axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='tab:blue', linewidth=2)
    axs[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color='tab:red', linestyle='--', linewidth=2)
    axs[0, 0].set_title('Joint Loss (Multi + Binary)')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # --- 2. Top-Right: Binary F1-Score ---
    axs[0, 1].plot(epochs, history['train_binary_f1'], label='Train F1', color='tab:blue', linewidth=2)
    axs[0, 1].plot(epochs, history['val_binary_f1'], label='Val F1', color='tab:orange', linestyle='--', linewidth=2)
    axs[0, 1].set_title('Binary F1-Score')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('F1')
    axs[0, 1].legend()

    # --- 3. Bottom-Left: Multi-class vs Binary Validation Accuracy ---
    axs[1, 0].plot(epochs, history['val_multi_acc'], label='Val Multi-class Acc (6)', color='tab:green', linewidth=2)
    axs[1, 0].plot(epochs, history['val_binary_acc'], label='Val Binary Acc (2)', color='tab:purple', linewidth=2)
    axs[1, 0].set_title('Validation Accuracy Comparison')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()

    # --- 4. Bottom-Right: Validation Precision vs Recall ---
    # 这个子图非常关键，用于观察模型是否为了提高召回率而牺牲了准确率
    axs[1, 1].plot(epochs, history['val_binary_prec'], label='Val Precision', color='tab:cyan', linewidth=2)
    axs[1, 1].plot(epochs, history['val_binary_rec'], label='Val Recall', color='tab:brown', linestyle='--', linewidth=2)
    axs[1, 1].set_title('Validation Precision vs Recall (Binary)')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].legend()

    # 调整子图间距并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 同时保存为 PNG 和 PDF（PDF 格式是矢量图，在 LaTeX 编译时放大不会失真）
    plt.savefig(save_path)
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path)
    plt.close()