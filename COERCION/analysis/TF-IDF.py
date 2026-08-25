import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# 1. 依然使用极简停用词表（保留极其重要的代词和情态动词）
CUSTOM_STOP_WORDS = [
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'am', 'are', 'was', 'were',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'it', 'that', 'this'
]

def load_data(file_path):
    positive_texts, negative_texts = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text = " ".join(data['utterances']).lower()
            if data['dialog_binary_label'] == 1:
                positive_texts.append(text)
            else:
                negative_texts.append(text)
    return positive_texts, negative_texts

def plot_log_odds(pos_texts, neg_texts, top_k=20):
    """
    计算对数胜率并绘制红蓝对抗图
    top_k: 分别展示最偏向正/负样本的前 N 个特征
    """
    # 2. 提取特征，我们同时看 1-gram 到 3-gram
    # min_df=3 过滤掉极其生僻的词组，防止它们因为偶然出现而获得畸高权重
    vec = CountVectorizer(ngram_range=(1, 3), stop_words=CUSTOM_STOP_WORDS, min_df=3)
    
    # 拟合并转换数据
    X_pos = vec.fit_transform(pos_texts)
    X_neg = vec.transform(neg_texts)
    
    vocab = vec.get_feature_names_out()
    
    # 3. 计算频次 (加入 Laplace 平滑 +1，避免除以 0 的错误)
    pos_counts = np.asarray(X_pos.sum(axis=0)).flatten() + 1
    neg_counts = np.asarray(X_neg.sum(axis=0)).flatten() + 1
    
    # 计算总词频
    pos_total = pos_counts.sum()
    neg_total = neg_counts.sum()
    
    # 4. 计算 Log-Odds Ratio (对数胜率)
    pos_rate = pos_counts / pos_total
    neg_rate = neg_counts / neg_total
    log_odds = np.log(pos_rate / neg_rate)
    
    # 5. 整理数据为 DataFrame
    df = pd.DataFrame({'feature': vocab, 'log_odds': log_odds})
    
    # 获取最具代表性的 Top K 正特征 (道德绑架) 和负特征 (正常)
    top_pos = df.sort_values(by='log_odds', ascending=False).head(top_k)
    top_neg = df.sort_values(by='log_odds', ascending=True).head(top_k)
    
    # 合并数据用于画图 (按照 log_odds 从小到大排序，方便横向条形图展示)
    plot_df = pd.concat([top_pos, top_neg]).sort_values(by='log_odds', ascending=True)
    
    # 6. 画图: 经典的红蓝对比条形图
    plt.figure(figsize=(12, 10))
    # 负分（正常）用蓝色，正分（绑架）用红色
    colors = ['cornflowerblue' if x < 0 else 'salmon' for x in plot_df['log_odds']]
    
    # 绘制水平条形图
    bars = plt.barh(plot_df['feature'], plot_df['log_odds'], color=colors, edgecolor='black', alpha=0.8)
    
    # 添加一条中轴线
    plt.axvline(0, color='black', linewidth=1.5)
    
    # 美化图表
    plt.title('Most Distinctive N-grams (Moral Coercion vs. Normal Request)', fontsize=16, pad=20)
    plt.xlabel('Log-Odds Ratio ( > 0: Guilt-trip, < 0: Normal )', fontsize=12)
    plt.yticks(fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# ================= 运行主程序 =================
# 假设你的文件名为 'dataset.jsonl'
file_path = 'inputters/data/train.jsonl' 
print("正在计算 Log-Odds 差异...")
pos_texts, neg_texts = load_data(file_path)
plot_log_odds(pos_texts, neg_texts, top_k=20)