# import matplotlib
# print(matplotlib.get_cachedir())

# import json
# import numpy as np
# from collections import Counter
# import matplotlib.pyplot as plt
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

# # 1. 配置文件路径
# input_file = "inputters/data/data_persuasion_techniques.jsonl"  

# # 【审美升级 1】采用高级学术配色
# COLOR_MC = '#C0392B'      # 深铁锈红 (Moral Coercion)
# COLOR_NORMAL = '#5D6D7E'  # 灰蓝色 (Not Moral Coercion)

# # 2. 初始化两个独立的计数器
# counter_1 = Counter() 
# counter_0 = Counter() 

# print("正在读取并分组统计预测结果...")
# try:
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             if not line.strip(): continue
#             data = json.loads(line)
            
#             techniques = data.get("predicted_techniques", [])
#             label = data.get("dialog_binary_label")
            
#             if label == 1:
#                 counter_1.update(techniques)
#             elif label == 0:
#                 counter_0.update(techniques)
# except FileNotFoundError:
#     print(f"找不到文件 {input_file}，请检查路径。")
#     exit()

# # 3. 获取所有出现过的技巧并排序
# all_techniques = set(counter_1.keys()).union(set(counter_0.keys()))
# techniques_list = sorted(list(all_techniques))
# techniques_list.reverse()

# counts_1 = [counter_1[t] for t in techniques_list]
# counts_0 = [counter_0[t] for t in techniques_list]

# # 4. 开始绘制分组条形图
# print("准备生成高级对比图表...")
# fig, ax = plt.subplots(figsize=(14, 12))

# y_positions = np.arange(len(techniques_list))
# # 【细节】柱子粗细设为0.4，组与组之间会有一点点呼吸空间，比0.5全部挤在一起更好看
# bar_height = 0.40 

# # 【审美升级 2】zorder=3 确保柱子图层在网格线之上，不被虚线切割
# bars_0 = ax.barh(y_positions - bar_height/2, counts_0, 
#                  height=bar_height, label='Not Moral Coercion', 
#                  color=COLOR_NORMAL, zorder=3)

# bars_1 = ax.barh(y_positions + bar_height/2, counts_1, 
#                  height=bar_height, label='Moral Coercion', 
#                  color=COLOR_MC, zorder=3)

# # 【审美升级 3】使用 bar_label 完美对齐数字，padding控制距离，告别手动计算
# ax.bar_label(bars_0, padding=5, fontsize=15, color='#333333')
# ax.bar_label(bars_1, padding=5, fontsize=15, color='#333333')

# # 5. 图表细节极致美化

# # 【审美升级 4】“呼吸感”边框：干掉顶部和右侧的黑线
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_color('#cccccc')
# ax.spines['bottom'].set_color('#cccccc')

# # 隐藏 Y 轴凸出的小短线刻度，更清爽
# ax.tick_params(axis='y', length=0)

# # 【审美升级 5】加入浅色虚线网格辅助阅读，层级设在底部 (zorder=0)
# ax.grid(axis='x', linestyle='--', alpha=0.4, color='#999999', zorder=0)

# # 动态扩展 X 轴的最大值 (增加 15% 的冗余空间)，防止最长的柱子末端的数字被截断
# max_val = max(max(counts_0), max(counts_1))
# ax.set_xlim(0, max_val * 1.15)

# # 设置字体大小和颜色
# ax.set_xlabel('Frequency', fontsize=18, color='#333333', weight='medium')
# ax.tick_params(axis='x', labelsize=16, colors='#333333')
# ax.set_yticks(y_positions)
# ax.set_yticklabels(techniques_list, fontsize=17, color='#333333')

# # 【审美升级 6】高级图例排版：移出主图表，顶部居中横向排列，无外框
# # 1. 提取当前图表的 handles 和 labels
# handles, labels = ax.get_legend_handles_labels()

# # 2. 颠倒列表顺序，实现左右位置交换 (把 Moral Coercion 放到左边)
# ax.legend(handles[::-1], labels[::-1], 
#           loc='lower right', bbox_to_anchor=(1.0, 0.9), 
#           ncol=2, fontsize=20, frameon=False)

# plt.tight_layout()

# # 6. 保存并显示
# os.makedirs('analysis/PERSUASION_TECHNIQUES', exist_ok=True)

# plt.savefig('analysis/PERSUASION_TECHNIQUES/technique_comparison.png', dpi=300, bbox_inches='tight')
# plt.savefig('analysis/PERSUASION_TECHNIQUES/technique_comparison.pdf', dpi=300, bbox_inches='tight', format='pdf')
# print("绝美图表已保存！")
# plt.show()



import matplotlib
print(matplotlib.get_cachedir())

import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ==========================================
# 【新增核心配置】全局修改 matplotlib 字体为 Linux Libertine
# ==========================================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Linux Libertine O', 'Linux Libertine', 'LinLibertine']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# ==========================================

input_file = "inputters/data/data_persuasion_techniques.jsonl"  

COLOR_MC = '#C0392B'      # 深铁锈红 (Moral Coercion)
COLOR_NORMAL = '#5D6D7E'  # 灰蓝色 (Not Moral Coercion)

counter_1 = Counter() 
counter_0 = Counter() 

print("正在读取并分组统计预测结果...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            techniques = data.get("predicted_techniques", [])
            label = data.get("dialog_binary_label")
            
            if label == 1:
                counter_1.update(techniques)
            elif label == 0:
                counter_0.update(techniques)
except FileNotFoundError:
    print(f"找不到文件 {input_file}，请检查路径。")
    exit()

all_techniques = set(counter_1.keys()).union(set(counter_0.keys()))
techniques_list = sorted(list(all_techniques))
techniques_list.reverse()

counts_1 = [counter_1[t] for t in techniques_list]
counts_0 = [counter_0[t] for t in techniques_list]

# ==========================================
# 4. 开始绘制分组条形图 (精简画布高度，消除上下空白)
# ==========================================
print("准备生成高级对比图表...")
# 【调整 1】将高度从 12 缩减到 10（或根据具体标签数量进一步缩减），迫使条形图更加紧凑
fig, ax = plt.subplots(figsize=(14, 9))

y_positions = np.arange(len(techniques_list))
bar_height = 0.40 

bars_0 = ax.barh(y_positions - bar_height/2, counts_0, 
                 height=bar_height, label='Not Moral Coercion', 
                 color=COLOR_NORMAL, zorder=3)

bars_1 = ax.barh(y_positions + bar_height/2, counts_1, 
                 height=bar_height, label='Moral Coercion', 
                 color=COLOR_MC, zorder=3)

# 【调整 2】调大柱状图末端的数字标签字体 (15 -> 18)
ax.bar_label(bars_0, padding=6, fontsize=18, color='#333333')
ax.bar_label(bars_1, padding=6, fontsize=18, color='#333333')

# 5. 图表细节极致美化
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

ax.tick_params(axis='y', length=0)
ax.grid(axis='x', linestyle='--', alpha=0.4, color='#999999', zorder=0)

max_val = max(max(counts_0), max(counts_1))
# 适当给右侧留出数字展示空间
ax.set_xlim(0, max_val * 1.15)

# 【调整 3】显著调大 X/Y 轴文本字体
ax.set_xlabel('Frequency', fontsize=22, color='#333333', weight='medium')
ax.tick_params(axis='x', labelsize=18, colors='#333333')
ax.set_yticks(y_positions)
ax.set_yticklabels(techniques_list, fontsize=20, color='#333333') # Y 轴标签放大到 20

# 【调整 4】精细化右上角图例排版，调大图例字体，并彻底解决上下空白问题
handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles[::-1], labels[::-1], 
    loc='lower right', 
    bbox_to_anchor=(1.0, 0.9), # 将图例基准线完美贴在主图顶端 (Y=1.01)，绝不往下压榨主图空间
    ncol=2, 
    fontsize=22,                # 图例字体调大到 22
    frameon=False
)

# 【关键】使用 bbox_inches='tight' 自动裁剪掉四周所有无用的绝对空白
plt.tight_layout()

# 6. 保存并显示
os.makedirs('analysis/PERSUASION_TECHNIQUES', exist_ok=True)

plt.savefig('analysis/PERSUASION_TECHNIQUES/technique_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('analysis/PERSUASION_TECHNIQUES/technique_comparison.pdf', dpi=300, bbox_inches='tight', format='pdf')
print("绝美紧凑版图表已保存！")
plt.show()