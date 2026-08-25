import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 数据准备 =================
k_shots = [0, 1, 2, 3, 4, 5, 6]

# GLM-4-9B 数据
glm_acc = [0.683, 0.722, 0.741, 0.746, 0.747, 0.742, 0.738]
glm_f1  = [0.759, 0.765, 0.772, 0.772, 0.773, 0.780, 0.774]

# Qwen2.5-14B 数据
qwen_acc = [0.724, 0.701, 0.710, 0.720, 0.717, 0.719, 0.729]
qwen_f1  = [0.764, 0.750, 0.752, 0.761, 0.763, 0.767, 0.765]

# Llama2-13B 数据
llama_acc = [0.505, 0.605, 0.601, 0.606, 0.548, 0.604, 0.579]
llama_f1  = [0.664, 0.703, 0.682, 0.681, 0.568, 0.597, 0.598]

# ================= 2. 全局学术字体与字号设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18

# 创建 1x2 横向画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# 【核心优化】：替换为高饱和深色，彻底移除 alpha 透明度，大幅加粗线条与标记
style_glm4  = {'color': '#b30000', 'marker': 'x', 'linestyle': '--', 'linewidth': 3.0, 'markersize': 10, 'markeredgewidth': 2.5}
style_qwen  = {'color': '#006d5b', 'marker': '*', 'linestyle': '--', 'linewidth': 3.0, 'markersize': 12, 'markeredgewidth': 1.5}
style_llama = {'color': '#0057b3', 'marker': 'o', 'linestyle': '--', 'linewidth': 3.0, 'markersize': 7}

def plot_lines(ax, data1, data2, data3, ylabel):
    ax.plot(k_shots, data1, label='GLM-4-9B', **style_glm4)
    ax.plot(k_shots, data2, label='Qwen2.5-14B', **style_qwen)
    ax.plot(k_shots, data3, label='Llama2-13B', **style_llama)
    
    ax.set_xlabel('Shot ($K$)', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xticks(k_shots)
    
    # 【核心优化】：加深网格线颜色（从浅灰改为中灰 #666666），提高对比度
    ax.grid(True, linestyle='--', alpha=0.8, color='#666666', linewidth=1.0)
    
    # 稍微加粗外框线
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# ================= 3. 执行绘图 =================
plot_lines(ax1, glm_acc, qwen_acc, llama_acc, 'Accuracy')
plot_lines(ax2, glm_f1, qwen_f1, llama_f1, 'F1 Score')

# ================= 4. 统一图例与加黑保存 =================
handles, labels = ax1.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False)

# 将图例文字也进行加粗
for text in leg.get_texts():
    text.set_weight('bold')

plt.tight_layout()
# 调整边距，给下方的加黑图例留出足够空间
plt.subplots_adjust(bottom=0.18) 
plt.savefig('analysis/FEWSHOT/few_shot_evaluation.pdf', format='pdf', bbox_inches='tight')
print("深色高对比度版 Few-shot 折线图已成功生成！")