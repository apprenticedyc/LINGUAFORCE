# import json
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib as mpl
# import os

# # ==========================================
# # 【新增核心配置】全局修改 matplotlib 字体为 Linux Libertine
# # ==========================================
# # 告诉 matplotlib 使用衬线字体
# mpl.rcParams['font.family'] = 'serif'
# # 优先寻找系统中的 Linux Libertine (不同系统命名可能有细微差别，提供多个备选)
# mpl.rcParams['font.serif'] = ['Linux Libertine O', 'Linux Libertine', 'LinLibertine']

# # 【极度重要】确保 PDF 导出时嵌入矢量字体 (Type 42)，防止学术会议 Camera-Ready 格式检查报错
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# # ==========================================

# def plot_6x6_heatmaps(jsonl_file_path):
#     overall_list = []
#     turn_list = []
    
#     # 1. 读取并解析数据
#     print("正在读取数据并计算相关系数...")
#     with open(jsonl_file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if not line.strip():
#                 continue
            
#             data = json.loads(line)
#             binary_label = data.get('dialog_binary_label', None)
#             multi_label = data.get('dialog_multi_label', None)
            
#             # --- 提取 Global 数据 ---
#             overall = data.get('overall_analysis', {})
#             overall_list.append({
#                 'Obligation': overall.get('Obligation', {}).get('intensity_score', 0.0),
#                 'Constraint': overall.get('Constraint', {}).get('intensity_score', 0.0),
#                 'ValueJudgement': overall.get('ValueJudgement', {}).get('intensity_score', 0.0),
#                 'Toxicity': overall.get('Toxicity', {}).get('intensity_score', 0.0),
#                 'Binary_Label': binary_label,
#                 'Intensity': multi_label
#             })
            
#             # --- 提取 Turn 数据 ---
#             turns = data.get('turn_analysis', [])
#             for turn in turns:
#                 features = turn.get('features', {})
#                 if not features:
#                     continue
#                 turn_list.append({
#                     'Obligation': features.get('Obligation', {}).get('intensity_score', 0.0),
#                     'Constraint': features.get('Constraint', {}).get('intensity_score', 0.0),
#                     'ValueJudgement': features.get('ValueJudgement', {}).get('intensity_score', 0.0),
#                     'Toxicity': features.get('Toxicity', {}).get('intensity_score', 0.0),
#                     'Binary_Label': binary_label,  
#                     'Intensity': multi_label     
#                 })
                
#     # 2. 转换为 DataFrame 并清理缺失值
#     df_overall = pd.DataFrame(overall_list).dropna()
#     df_turn = pd.DataFrame(turn_list).dropna()
    
#     # 3. 计算 6x6 相关系数矩阵
#     corr_overall = df_overall.corr()
#     corr_turn = df_turn.corr()
    
#     # ================= 开始高级感绘图 =================
#     print("正在生成顶级期刊风格热力图...")
    
#     # 【审美升级 1】定制专属色系：灰蓝(-1) -> 纯白(0) -> 深铁锈红(1)
#     # 这让你的整篇论文色彩高度统一！
#     # colors = ["#5D6D7E", "#FFFFFF", "#C0392B"] 
#     colors = ["#3D83CD", "#FFFFFF", "#C0392B"] 
#     cmap_custom = LinearSegmentedColormap.from_list("morandi_corr", colors, N=256)
    
#     # 创建 1行2列 的画布，稍微加宽以容纳文字和色条
#     fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
#     # 统一的热力图参数配置
#     heatmap_kwargs = {
#         'annot': True, 
#         'cmap': cmap_custom, 
#         'fmt': ".2f", 
#         'vmin': 0,          # 【学术严谨】相关系数范围设为 -1 到 1
#         'vmax': 1, 
#         'square': True, 
#         'linewidths': 1.5,   # 【细节】加入白色网格线切割方块
#         'linecolor': 'white',
#         'annot_kws': {'size': 13, 'weight': 'medium'} # 放大方块内的数字
#     }
    
#     # 【审美升级 2】图1：Global 6x6 热力图 (去掉多余的色条 cbar=False)
#     sns.heatmap(corr_overall, ax=axes[0], cbar=False, **heatmap_kwargs)
#     axes[0].set_title('A: Dialogue-Level Correlation', fontsize=18, fontweight='bold', pad=20, loc='left')
    
#     # 【审美升级 3】图2：Turn 6x6 热力图 (保留色条，并进行精致化排版)
#     # cbar_kws shrink 缩短色条，aspect 让色条变得更修长高级
#     sns.heatmap(corr_turn, ax=axes[1], cbar=True, 
#                 cbar_kws={"shrink": 0.8, "aspect": 30}, **heatmap_kwargs)
#     axes[1].set_title('B: Turn-Level Correlation', fontsize=18, fontweight='bold', pad=20, loc='left')
    
#     # 【审美升级 4】坐标轴字体与对齐极致优化
#     for ax in axes:
#         # X 轴：倾斜40度，且锚定点在文字末端，治愈强迫症
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', rotation_mode='anchor', fontsize=14, color='#333333')
#         # Y 轴：水平放置，字体放大
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14, color='#333333')
        
#         # 隐藏周围讨厌的小刻度线
#         ax.tick_params(axis='both', which='both', length=0)
        
#         # 去掉默认的黑框，让图表更通透
#         for _, spine in ax.spines.items():
#             spine.set_visible(False)
            
#     # 调整布局
#     plt.tight_layout()
    
#     # --- 5. 保存 ---
#     os.makedirs('analysis', exist_ok=True)
#     plt.savefig('analysis/CORRELATION_HEATMAP/Correlation_Heatmap_deepseekv3.2_Elegant2.png', dpi=300, bbox_inches='tight')
#     plt.savefig('analysis/CORRELATION_HEATMAP/Correlation_Heatmap_deepseekv3.2_Elegant2.pdf', dpi=300, bbox_inches='tight', format='pdf')
#     print("高级感热力图已成功保存！")
    
#     # 显示图片
#     plt.show()

# # 调用函数（请将路径替换为实际路径）
# plot_6x6_heatmaps('inputters/data/test+feature_deepseekv3.2.jsonl')



import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import os


# ==========================================

def plot_6x6_heatmaps(jsonl_file_path):
    overall_list = []
    turn_list = []
    
    print("正在读取数据并计算相关系数...")
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            binary_label = data.get('dialog_binary_label', None)
            multi_label = data.get('dialog_multi_label', None)
            
            overall = data.get('overall_analysis', {})
            # 【优化 1】将原版的长标签在录入时就全部映射为顶会紧凑缩写 (BL 和 Int)
            overall_list.append({
                'OB': overall.get('Obligation', {}).get('intensity_score', 0.0),
                'CS': overall.get('Constraint', {}).get('intensity_score', 0.0),
                'VJ': overall.get('ValueJudgement', {}).get('intensity_score', 0.0),
                'TX': overall.get('Toxicity', {}).get('intensity_score', 0.0),
                'BL': binary_label,
                'Int': multi_label
            })
            
            turns = data.get('turn_analysis', [])
            for turn in turns:
                features = turn.get('features', {})
                if not features:
                    continue
                turn_list.append({
                    'OB': features.get('Obligation', {}).get('intensity_score', 0.0),
                    'CS': features.get('Constraint', {}).get('intensity_score', 0.0),
                    'VJ': features.get('ValueJudgement', {}).get('intensity_score', 0.0),
                    'TX': features.get('Toxicity', {}).get('intensity_score', 0.0),
                    'BL': binary_label,  
                    'Int': multi_label     
                })
                
    df_overall = pd.DataFrame(overall_list).dropna()
    df_turn = pd.DataFrame(turn_list).dropna()
    
    # 确保矩阵的排列顺序完全符合学术逻辑
    column_order = ['OB', 'CS', 'VJ', 'TX', 'BL', 'Int']
    corr_overall = df_overall.corr().reindex(index=column_order, columns=column_order)
    corr_turn = df_turn.corr().reindex(index=column_order, columns=column_order)
    
    print("正在生成完全去重叠、高度紧凑的顶会热力图...")
    
    colors = ["#3D83CD", "#FFFFFF", "#C0392B"] 
    cmap_custom = LinearSegmentedColormap.from_list("morandi_corr", colors, N=256)
    
    # 维持 15x7.5 的黄金优雅单双栏通用比例
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    
    heatmap_kwargs = {
        'annot': True, 
        'cmap': cmap_custom, 
        'fmt': ".2f", 
        'vmin': 0, 
        'vmax': 1, 
        'square': True, 
        'linewidths': 1.5, 
        'linecolor': 'white',
        'annot_kws': {'size': 14} # 维持高清晰度的粗体数字
    }
    
    # 渲染左图：Dialogue-Level
    sns.heatmap(corr_overall, ax=axes[0], cbar=False, **heatmap_kwargs)
    axes[0].set_title('(a) Dialogue-Level', fontsize=22, pad=15, loc='left')
    
    # 渲染右图：Turn-Level
    # 【优化 2】微调 pad=0.03，将色条位置优雅放开，防止右侧拥挤
    sns.heatmap(corr_turn, ax=axes[1], cbar=True, 
                cbar_kws={"shrink": 0.85, "aspect": 30, "pad": 0.03}, **heatmap_kwargs)
    axes[1].set_title('(b) Turn-Level', fontsize=22, pad=15, loc='left')
    
    cbar = axes[1].collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    
    for ax in axes:
        # 【优化 3】标签缩写后，横向投影长度大减，将倾斜度调至更工整的 30 度，右对齐完美避开撞车
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor', fontsize=18, color='#000000')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18, color='#000000')
        
        ax.tick_params(axis='both', which='both', length=0)
        for _, spine in ax.spines.items():
            spine.set_visible(False)
            
    # 【优化 4】将 wspace 调整为 0.15 黄金阈值，彻底拉开两个子图的物理安全防线
    plt.subplots_adjust(wspace=0.15, left=0.06, right=0.94, top=0.92, bottom=0.15)
    
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/CORRELATION_HEATMAP/Correlation_Heatmap_deepseekv3.2_Elegant2.png', dpi=300, bbox_inches='tight')
    plt.savefig('analysis/CORRELATION_HEATMAP/Correlation_Heatmap_deepseekv3.2_Elegant2.pdf', dpi=300, bbox_inches='tight', format='pdf')
    print("去重叠高清热力图已成功输出并保存！")
    
    plt.show()

# 调用函数
plot_6x6_heatmaps('inputters/data/test+feature_deepseekv3.2.jsonl')