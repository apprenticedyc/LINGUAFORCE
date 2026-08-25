import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# -------------------- 配置 --------------------
DATA_PATH = "inputters/data/test.jsonl"                          # 你的 jsonl 文件
MODEL_NAME = "D:/model/roberta-large-goemotions"               # 实际模型名称，请确认
SEEDS = [42, 520, 2026]
TEST_SIZE = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 加载情感模型和 tokenizer --------------------
print(f"加载情感模型 {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# 获取输出类别数（GoEmotions 通常为 28）
NUM_EMOTIONS = model.config.num_labels
print(f"模型输出维度（情感类别数）: {NUM_EMOTIONS}")

# GoEmotions 为多标签分类，使用 sigmoid
USE_SIGMOID = True

# -------------------- 读取数据集 --------------------
def load_dataset(path):
    texts = []
    binary_labels = []
    multi_labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            combined = " ".join(item["utterances"])
            texts.append(combined)
            binary_labels.append(item["dialog_binary_label"])
            multi_labels.append(item["dialog_multi_label"])
    return texts, np.array(binary_labels), np.array(multi_labels)

print(f"从 {DATA_PATH} 加载数据集...")
texts, y_binary, y_multi = load_dataset(DATA_PATH)
print(f"共加载 {len(texts)} 条对话。")

# -------------------- 用情感模型提取特征 --------------------
def extract_features(texts, batch_size=16):
    """返回每个文本的情感概率向量（维度 = NUM_EMOTIONS）"""
    features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                           padding=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if USE_SIGMOID:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
        features.append(probs)
    return np.vstack(features)

print("使用 RoBERTa-LARGE-GOEMOTIONS 提取特征...")
X = extract_features(texts)
print(f"特征提取完成，特征形状：{X.shape}")  # 应为 (样本数, 28)

# -------------------- 评估函数 --------------------
def evaluate(y_true_bin, y_pred_bin, y_true_multi, y_pred_multi):
    acc_bin = accuracy_score(y_true_bin, y_pred_bin)
    prec_bin = precision_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
    rec_bin = recall_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
    f1_bin = f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
    
    acc_multi = accuracy_score(y_true_multi, y_pred_multi)
    prec_multi = precision_score(y_true_multi, y_pred_multi, average='macro', zero_division=0)
    rec_multi = recall_score(y_true_multi, y_pred_multi, average='macro', zero_division=0)
    f1_multi = f1_score(y_true_multi, y_pred_multi, average='macro', zero_division=0)
    
    return {
        'binary': (acc_bin, prec_bin, rec_bin, f1_bin),
        'multi': (acc_multi, prec_multi, rec_multi, f1_multi)
    }

# -------------------- 训练和评估循环 --------------------
results_bin = []
results_multi = []

for seed in SEEDS:
    print(f"\n========== 种子 {seed} ==========")
    
    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
        X, y_binary, y_multi, test_size=TEST_SIZE, random_state=seed, stratify=y_binary
    )
    
    # 二分类逻辑回归
    clf_bin = LogisticRegression(max_iter=1000, random_state=seed)
    clf_bin.fit(X_train, y_bin_train)
    y_bin_pred = clf_bin.predict(X_test)
    
    # 六分类逻辑回归（多分类）
    clf_multi = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=seed)
    clf_multi.fit(X_train, y_multi_train)
    y_multi_pred = clf_multi.predict(X_test)
    
    metrics = evaluate(y_bin_test, y_bin_pred, y_multi_test, y_multi_pred)
    results_bin.append(metrics['binary'])
    results_multi.append(metrics['multi'])
    
    print(f"二分类 - Acc: {metrics['binary'][0]:.4f}, Prec: {metrics['binary'][1]:.4f}, Rec: {metrics['binary'][2]:.4f}, F1: {metrics['binary'][3]:.4f}")
    print(f"六分类 - Acc: {metrics['multi'][0]:.4f}, Prec: {metrics['multi'][1]:.4f}, Rec: {metrics['multi'][2]:.4f}, F1: {metrics['multi'][3]:.4f}")

# -------------------- 均值 ± 标准差 --------------------
def mean_std_str(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1)
    return [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean, std)]

bin_means_std = mean_std_str(results_bin)
multi_means_std = mean_std_str(results_multi)

print("\n==================== 最终结果 (均值 ± 标准差) ====================")
print(f"二分类 - Acc: {bin_means_std[0]}, Prec: {bin_means_std[1]}, Rec: {bin_means_std[2]}, F1: {bin_means_std[3]}")
print(f"六分类 - Acc: {multi_means_std[0]}, Prec: {multi_means_std[1]}, Rec: {multi_means_std[2]}, F1: {multi_means_std[3]}")