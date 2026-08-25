import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# 1. 自定义极简停用词表（保留人称代词和情态动词）
# 我们只去除最基础的冠词、连词和无意义的介词
CUSTOM_STOP_WORDS = [
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'am', 'are', 'was', 'were', 
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'it', 'that', 'this'
]

def load_and_split_data(file_path):
    """
    读取 JSONL 文件，并将正负样本的文本分开
    """
    positive_texts = []
    negative_texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            # 将多轮对话拼接成一段完整的文本
            # 如果你想细粒度到每句话，这里可以改为 data['utterances'] 的循环
            combined_text = " ".join(data['utterances']).lower()
            
            # 1 代表正样本（存在道德绑架），0 代表负样本
            if data['dialog_binary_label'] == 1:
                positive_texts.append(combined_text)
            else:
                negative_texts.append(combined_text)
                
    return positive_texts, negative_texts

def get_top_ngrams(corpus, n_gram_range, top_k=15):
    """
    使用 CountVectorizer 提取频率最高的 N-gram
    """
    # ngram_range=(2,2) 表示只提取 Bi-gram，(3,3) 表示 Tri-gram
    vec = CountVectorizer(ngram_range=n_gram_range, stop_words=CUSTOM_STOP_WORDS).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    return words_freq[:top_k]

def plot_ngram_comparison(pos_ngrams, neg_ngrams, title):
    """
    将正负样本的 N-gram 结果画成对比柱状图
    """
    pos_df = pd.DataFrame(pos_ngrams, columns=['N-gram', 'Frequency'])
    neg_df = pd.DataFrame(neg_ngrams, columns=['N-gram', 'Frequency'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 画正样本
    axes[0].barh(pos_df['N-gram'][::-1], pos_df['Frequency'][::-1], color='salmon')
    axes[0].set_title(f'Top {title} in Positive (Guilt-trip)')
    axes[0].set_xlabel('Frequency')
    
    # 画负样本
    axes[1].barh(neg_df['N-gram'][::-1], neg_df['Frequency'][::-1], color='skyblue')
    axes[1].set_title(f'Top {title} in Negative (Normal)')
    axes[1].set_xlabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# ================= 运行主程序 =================

# 假设你的文件名为 'dataset.jsonl'
file_path = 'inputters/data/train.jsonl' 

# 1. 加载数据
print("正在加载数据...")
pos_texts, neg_texts = load_and_split_data(file_path)
print(f"正样本数量: {len(pos_texts)} | 负样本数量: {len(neg_texts)}")

# 2. 提取并可视化 Bi-gram (双词组合)
print("\n正在分析 Bi-grams...")
pos_bigrams = get_top_ngrams(pos_texts, (2, 2))
neg_bigrams = get_top_ngrams(neg_texts, (2, 2))
# 取消下面的注释即可显示图表
# plot_ngram_comparison(pos_bigrams, neg_bigrams, "Bi-grams")
print("正样本 Top 5 Bi-grams:", pos_bigrams[:20])
print("负样本 Top 5 Bi-grams:", neg_bigrams[:20])

# 3. 提取并可视化 Tri-gram (三词组合)
print("\n正在分析 Tri-grams...")
pos_trigrams = get_top_ngrams(pos_texts, (3, 3))
neg_trigrams = get_top_ngrams(neg_texts, (3, 3))
# 取消下面的注释即可显示图表
# plot_ngram_comparison(pos_trigrams, neg_trigrams, "Tri-grams")
print("正样本 Top 5 Tri-grams:", pos_trigrams[:20])
print("负样本 Top 5 Tri-grams:", neg_trigrams[:20])