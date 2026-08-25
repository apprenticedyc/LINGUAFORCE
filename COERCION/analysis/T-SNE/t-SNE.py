import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# 1. 加载数据 (假设文件名为 data.jsonl)
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 2. 提取特征嵌入
def get_embeddings(texts):
    print("Extracting embeddings...")
    model = SentenceTransformer('D:/model/all-MiniLM-L12-v2')
    return model.encode(texts, show_progress_bar=True)

# 3. 执行降维与高级感绘图
def plot_tsne(df, embeddings):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, init='pca', random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    df['tsne_1'] = tsne_results[:, 0]
    df['tsne_2'] = tsne_results[:, 1]
    
    # 将 label 转换为分类描述
    df['label_type'] = df['dialog_label'].apply(lambda x: 'Moral Coercion' if x >= 0.5 else 'Not Moral Coercion')
    
    # ================= 开始高级感绘图 =================
    
    # 【审美升级 1】与你的柱状图保持系列感的高级配色
    COLOR_MC = '#C0392B'      # 深铁锈红 (Coercion)
    COLOR_NORMAL = '#5D6D7E'  # 灰蓝色 (Not Moral Coercion)
    
    # 清除默认样式，使用干净的白底
    sns.set_style("white")

    # 【关键动作】：在 seaborn 设置完之后，立刻把字体抢回来！
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Linux Libertine O', 'Linux Libertine', 'LinLibertine']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 【审美升级 2】细腻的散点质感：
    # s=25 缩小点径显得更精致；alpha=0.6 叠加产生密度感；edgecolor=None 彻底去掉点的杂色边缘
    sns.scatterplot(
        data=df, x='tsne_1', y='tsne_2', hue='label_type',
        palette={'Moral Coercion': COLOR_MC, 'Not Moral Coercion': COLOR_NORMAL},
        alpha=0.6, s=25, edgecolor=None, zorder=3
    )
    
    # 【审美升级 3】拆除牢笼：去掉顶部和右侧的边框，淡化左侧和底部边框
    sns.despine(top=True, right=True)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # 隐藏坐标轴凸出的小刻度线，更清爽
    ax.tick_params(axis='both', length=0)
    
    # 【审美升级 4】优雅的网格线：压在散点下方 (zorder=0)
    ax.grid(True, linestyle='--', alpha=0.4, color='#999999', zorder=0)
    
    # 【审美升级 5】专业且比例协调的字体大小 (摒弃了之前粗暴的字体30)
    plt.xlabel('t-SNE Dimension 1', fontsize=30, fontweight='medium', color='#333333', labelpad=10)
    plt.ylabel('t-SNE Dimension 2', fontsize=30, fontweight='medium', color='#333333', labelpad=10)
    plt.xticks(fontsize=25, color='#333333')
    plt.yticks(fontsize=25, color='#333333')
    
    # 【审美升级 6】高级图例排版：移出主画板，置于顶部居中，去掉边框，放大图例点(markerscale)
    plt.legend(title=None, 
            loc='upper right',           # 1. 告诉系统图例本身的锚点是其“右上角”
            bbox_to_anchor=(1.0, 1.0),   # 2. 将该锚点精准钉在子图坐标系的绝对右上角 (1.0, 1.0)
            ncol=2, 
            fontsize=20, 
            frameon=False, 
            markerscale=2.5)
    
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs('analysis/T-SNE', exist_ok=True)
    
    # 同时保存高分辨率 PNG 和 PDF
    plt.savefig('analysis/T-SNE/tsne_elegant3.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig('analysis/T-SNE/tsne_elegant3.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("绝美 t-SNE 图表已保存！")
    plt.show()

# ================= 执行流程 =================
if __name__ == "__main__":
    try:
        df = load_data('inputters/data/data.jsonl')
        
        def format_dialogue(utterance_list):
            """将对话列表转为带有角色前缀的单一字符串"""
            formatted = []
            for i, utt in enumerate(utterance_list):
                speaker = "Person1" if i % 2 == 0 else "Person2"
                # formatted.append(f"{speaker}: {utt}")
                formatted.append(f"{utt}")
            return " ".join(formatted)
            
        # 1. 生成带角色的拼接文本
        df['formatted_text'] = df['utterances'].apply(format_dialogue)
        
        # 2. 获取向量
        embeddings = get_embeddings(df['formatted_text'].tolist())
        
        # 3. 绘图
        plot_tsne(df, embeddings)
        
    except FileNotFoundError:
        print("找不到文件，请检查 'inputters/data/data.jsonl' 路径是否正确。")