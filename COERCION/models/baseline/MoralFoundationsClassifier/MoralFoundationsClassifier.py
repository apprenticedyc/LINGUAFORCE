import json
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# -------------------- 配置 --------------------
DATA_PATH = "inputters/data/test.jsonl"          # 你的 jsonl 文件路径
SEEDS = [42, 520, 2026]               # 三个随机种子
TEST_SIZE = 0.3                       # 测试集比例（可调整）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签名称（来自模型）
LABEL_NAMES = [
    "care_virtue", "care_vice",
    "fairness_virtue", "fairness_vice",
    "loyalty_virtue", "loyalty_vice",
    "authority_virtue", "authority_vice",
    "sanctity_virtue", "sanctity_vice"
]

# -------------------- 加载模型和 tokenizer --------------------
print("加载 MoralFoundationsClassifier 模型...")
model_path = "D:/model/MoralFoundationsClassifier"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.to(DEVICE)
model.eval()
print("模型加载完成。\n")

# -------------------- 读取数据集 --------------------
def load_dataset(path):
    """读取 jsonl 文件，返回文本列表、二分类标签、六分类标签"""
    texts = []
    binary_labels = []
    multi_labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 将 utterances 合并为一个文本
            combined = " ".join(item["utterances"])
            texts.append(combined)
            binary_labels.append(item["dialog_binary_label"])
            multi_labels.append(item["dialog_multi_label"])
    return texts, np.array(binary_labels), np.array(multi_labels)

print(f"从 {DATA_PATH} 加载数据集...")
texts, y_binary, y_multi = load_dataset(DATA_PATH)
print(f"共加载 {len(texts)} 条对话。")

# -------------------- 用模型提取特征 --------------------
def extract_features(texts, batch_size=16):
    """将文本列表输入模型，返回每个文本的 10 维道德基础概率"""
    features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                           padding=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
        features.append(probs)
    return np.vstack(features)

print("使用道德基础分类器提取特征...")
X = extract_features(texts)
print(f"特征提取完成，特征形状：{X.shape}")

# -------------------- 评估函数 --------------------
def evaluate(y_true_bin, y_pred_bin, y_true_multi, y_pred_multi):
    """计算二分类和六分类的各项指标"""
    # 二分类指标
    acc_bin = accuracy_score(y_true_bin, y_pred_bin)
    prec_bin = precision_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
    rec_bin = recall_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
    f1_bin = f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
    
    # 六分类指标（使用 macro average）
    acc_multi = accuracy_score(y_true_multi, y_pred_multi)
    prec_multi = precision_score(y_true_multi, y_pred_multi, average='macro', zero_division=0)
    rec_multi = recall_score(y_true_multi, y_pred_multi, average='macro', zero_division=0)
    f1_multi = f1_score(y_true_multi, y_pred_multi, average='macro', zero_division=0)
    
    return {
        'binary': (acc_bin, prec_bin, rec_bin, f1_bin),
        'multi': (acc_multi, prec_multi, rec_multi, f1_multi)
    }

# -------------------- 训练和评估循环 --------------------
results_bin = []  # 每个种子存储 (acc, prec, rec, f1)
results_multi = []

for seed in SEEDS:
    print(f"\n========== 种子 {seed} ==========")
    
    # 划分训练集和测试集
    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
        X, y_binary, y_multi, test_size=TEST_SIZE, random_state=seed, stratify=y_binary
    )
    
    # ---------- 二分类逻辑回归 ----------
    clf_bin = LogisticRegression(max_iter=1000, random_state=seed)
    clf_bin.fit(X_train, y_bin_train)
    y_bin_pred = clf_bin.predict(X_test)
    
    # ---------- 六分类逻辑回归 ----------
    # 六分类标签是 0-5 的整数，可直接用多分类逻辑回归
    clf_multi = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=seed)
    clf_multi.fit(X_train, y_multi_train)
    y_multi_pred = clf_multi.predict(X_test)
    
    # 评估
    metrics = evaluate(y_bin_test, y_bin_pred, y_multi_test, y_multi_pred)
    results_bin.append(metrics['binary'])
    results_multi.append(metrics['multi'])
    
    print(f"二分类 - Acc: {metrics['binary'][0]:.4f}, Prec: {metrics['binary'][1]:.4f}, Rec: {metrics['binary'][2]:.4f}, F1: {metrics['binary'][3]:.4f}")
    print(f"六分类 - Acc: {metrics['multi'][0]:.4f}, Prec: {metrics['multi'][1]:.4f}, Rec: {metrics['multi'][2]:.4f}, F1: {metrics['multi'][3]:.4f}")

# -------------------- 计算均值 ± 标准差 --------------------
def mean_std_str(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=1)  # 样本标准差
    return [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean, std)]

bin_means_std = mean_std_str(results_bin)
multi_means_std = mean_std_str(results_multi)

print("\n==================== 最终结果 (均值 ± 标准差) ====================")
print(f"二分类 - Acc: {bin_means_std[0]}, Prec: {bin_means_std[1]}, Rec: {bin_means_std[2]}, F1: {bin_means_std[3]}")
print(f"六分类 - Acc: {multi_means_std[0]}, Prec: {multi_means_std[1]}, Rec: {multi_means_std[2]}, F1: {multi_means_std[3]}")