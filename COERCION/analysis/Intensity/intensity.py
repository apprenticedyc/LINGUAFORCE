# import matplotlib.pyplot as plt
# import numpy as np

# # 提取 Table V 中所有类的 F1 Score 以及底部的 Macro F1
# classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Macro']
# x = np.arange(len(classes))  # 标签位置
# width = 0.35  # 柱子宽度

# # --- GLM-4-9B 数据 ---
# glm_f1_zero = [0.0233, 0.0000, 0.0302, 0.2483, 0.2843, 0.2559, 0.1403]
# glm_f1_ft   = [0.8739, 0.6306, 0.6087, 0.6923, 0.6882, 0.8671, 0.7268]

# # --- Qwen2.5-14B 数据 ---
# qwen_f1_zero = [0.4545, 0.0000, 0.0156, 0.0442, 0.2605, 0.5090, 0.2140]
# qwen_f1_ft   = [0.8837, 0.6897, 0.5590, 0.6418, 0.6776, 0.8788, 0.7218]

# # --- Llama2-13B 数据 ---
# llama_f1_zero = [0.4400, 0.0000, 0.0000, 0.0300, 0.0500, 0.2000, 0.1200]
# llama_f1_ft   = [0.8400, 0.5300, 0.5500, 0.6700, 0.6200, 0.8800, 0.6800]

# # 全局字体和格式设置
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams['axes.labelsize'] = 11
# plt.rcParams['xtick.labelsize'] = 9  # 类别多，字号稍微缩小防止重叠
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 10

# # 创建 1x3 横向画布（适合跨双栏）
# fig, axs = plt.subplots(1, 3, figsize=(14, 3.5))

# # 颜色配置：Zero-shot 浅灰色，Fine-tuned 深蓝色
# color_zero = '#B0BEC5' 
# color_ft = '#1565C0'

# def plot_class_bar(ax, data_zero, data_ft, title):
#     # 绘制分组柱状图
#     rects1 = ax.bar(x - width/2, data_zero, width, label='Zero-shot', color=color_zero, edgecolor='black', linewidth=0.5)
#     rects2 = ax.bar(x + width/2, data_ft, width, label='Fine-tuned', color=color_ft, edgecolor='black', linewidth=0.5)
    
#     ax.set_title(title, fontsize=12, fontweight='bold')
#     ax.set_ylabel('F1 Score')
#     ax.set_xticks(x)
#     ax.set_xticklabels(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'Macro']) # 简写节约空间
#     ax.set_ylim(0, 1.05)
#     ax.grid(axis='y', linestyle='--', alpha=0.6)
    
#     # 添加数值标签（仅保留两位小数，保持整洁）
#     for rect in rects1 + rects2:
#         height = rect.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 2),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=7, rotation=90) # 旋转90度防止拥挤

# # 绘制三个模型的子图
# plot_class_bar(axs[0], glm_f1_zero, glm_f1_ft, 'GLM-4-9B')
# plot_class_bar(axs[1], qwen_f1_zero, qwen_f1_ft, 'Qwen2.5-14B')
# plot_class_bar(axs[2], llama_f1_zero, llama_f1_ft, 'Llama2-13B')

# # 统一图例
# axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

# # 紧凑布局
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.2)
# plt.savefig('analysis/FEW-SHOT/full_class_f1_bars.pdf', format='pdf', bbox_inches='tight')
# print("全类别 F1 柱状图已生成！")


import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 数据准备 (已去掉 Macro 列) =================
classes = ['0', '1', '2', '3', '4', '5']
x = np.arange(len(classes))

# --- Zero-shot Data ---
rec_glm_zero   = [0.0118, 0.0000, 0.0385, 0.8438, 0.2617, 0.1677]
rec_qwen_zero  = [0.2941, 0.0000, 0.0128, 0.0625, 0.2897, 0.7019]
rec_llama_zero = [0.9600, 0.0000, 0.0000, 0.0200, 0.0300, 0.1200]

prec_glm_zero   = [1.0000, 0.0000, 0.0248, 0.1456, 0.3111, 0.5400]
prec_qwen_zero  = [1.0000, 0.0000, 0.0200, 0.0342, 0.2366, 0.3993]
prec_llama_zero = [0.2800, 0.0000, 0.0000, 0.1100, 0.1900, 0.6100]

f1_glm_zero   = [0.0233, 0.0000, 0.0302, 0.2483, 0.2843, 0.2559]
f1_qwen_zero  = [0.4545, 0.0000, 0.0156, 0.0442, 0.2605, 0.5090]
f1_llama_zero = [0.4400, 0.0000, 0.0000, 0.0300, 0.0500, 0.2000]

# --- Fine-tuned Data ---
rec_glm_ft   = [0.8765, 0.6481, 0.7179, 0.7031, 0.5981, 0.8509]
rec_qwen_ft  = [0.8941, 0.7407, 0.5769, 0.6719, 0.5794, 0.9006]
rec_llama_ft = [0.8400, 0.5000, 0.5900, 0.7200, 0.5300, 0.9200]

prec_glm_ft   = [0.8713, 0.6140, 0.5283, 0.6818, 0.8101, 0.8839]
prec_qwen_ft  = [0.8736, 0.6452, 0.5422, 0.6143, 0.8158, 0.8580]
prec_llama_ft = [0.8400, 0.5700, 0.5200, 0.6300, 0.7300, 0.8400]

f1_glm_ft   = [0.8739, 0.6306, 0.6087, 0.6923, 0.6882, 0.8671]
f1_qwen_ft  = [0.8837, 0.6897, 0.5590, 0.6418, 0.6776, 0.8788]
f1_llama_ft = [0.8400, 0.5300, 0.5500, 0.6700, 0.6200, 0.8800]


# ================= 2. 绘图配置 (极端防守：超大字号) =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# 为了抗击单栏缩放，使用震撼级的超大字号
plt.rcParams['axes.titlesize'] = 26   # 子图标题字号
plt.rcParams['axes.labelsize'] = 25   # X/Y轴标签字号
plt.rcParams['xtick.labelsize'] = 25  # X轴刻度字号
plt.rcParams['ytick.labelsize'] = 25  # Y轴刻度字号
plt.rcParams['legend.fontsize'] = 25  # 图例字号

# 创建 2行3列 的画布 (宽18, 高10，比例适合塞下所有粗线条)
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# 线条与标记点超级加粗放大
style_glm4  = {'color': '#b71c1c', 'marker': 'x', 'linestyle': '--', 'linewidth': 3.0, 'markersize': 12}
style_qwen  = {'color': '#00897b', 'marker': '*', 'linestyle': '--', 'linewidth': 3.0, 'markersize': 15}
style_llama = {'color': '#1e88e5', 'marker': 'o', 'linestyle': '--', 'linewidth': 3.0, 'markersize': 9}

# ================= 3. 循环绘图 =================
plot_config = [
    # 上排：Zero-shot
    (axs[0, 0], rec_glm_zero, rec_qwen_zero, rec_llama_zero, 'Recall', 'Zero-shot: Recall'),
    (axs[0, 1], prec_glm_zero, prec_qwen_zero, prec_llama_zero, 'Precision', 'Zero-shot: Precision'),
    (axs[0, 2], f1_glm_zero, f1_qwen_zero, f1_llama_zero, 'F1', 'Zero-shot: F1'),
    
    # 下排：Fine-tuned
    (axs[1, 0], rec_glm_ft, rec_qwen_ft, rec_llama_ft, 'Recall', 'Fine-tuned: Recall'),
    (axs[1, 1], prec_glm_ft, prec_qwen_ft, prec_llama_ft, 'Precision', 'Fine-tuned: Precision'),
    (axs[1, 2], f1_glm_ft, f1_qwen_ft, f1_llama_ft, 'F1', 'Fine-tuned: F1')
]

for ax, d_glm, d_qwen, d_llama, ylabel, title in plot_config:
    ax.plot(classes, d_glm, label='GLM-4-9B', **style_glm4)
    ax.plot(classes, d_qwen, label='Qwen2.5-14B', **style_qwen)
    ax.plot(classes, d_llama, label='Llama2-13B', **style_llama)
    
    ax.set_title(title, pad=12, fontweight='bold')
    # # 仅在底下一排显示 X 轴标签
    # if 'Fine-tuned' in title:
    #     ax.set_xlabel('Class')
        
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    
    # 统一 Y 轴范围 0-1
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5, color='#9E9E9E', linewidth=1.5)

# ================= 4. 统一图例与布局保存 =================
handles, labels = axs[0, 0].get_legend_handles_labels()
# 图例放在最底部居中
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)

plt.tight_layout()
# 调整间距：给底部图例留空间，增加子图之间的间距防止超大字体打架
plt.subplots_adjust(bottom=0.1, hspace=0.35, wspace=0.25) 
plt.savefig('analysis/Intensity/zero_vs_ft_2x3_large_font.pdf', format='pdf', bbox_inches='tight')
print("超大字号 2x3 折线图矩阵已生成！")