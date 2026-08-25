# import json
# import numpy as np
# from collections import Counter
# import matplotlib.pyplot as plt
# import matplotlib as mpl

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

# # ================= 配置区 =================
# input_file = "analysis/GOEMOTIONS/emotion_predictions.jsonl"

# # 【审美升级1】使用更高级的学术配色 (莫兰迪色系/高级灰)
# # 深铁锈红代表道德绑架，灰蓝色代表非道德绑架
# COLOR_MC = '#C0392B'  
# COLOR_NORMAL = '#5D6D7E' 

# # 准备计数器... (此处保留你的原有读取逻辑)
# utt_counter_1 = Counter()
# utt_counter_0 = Counter()
# dlg_counter_1 = Counter()
# dlg_counter_0 = Counter()

# print("正在读取并统计情感数据...")
# try:
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             if not line.strip(): continue
#             data = json.loads(line)
            
#             label = data.get("dialog_binary_label")
#             dlg_emotions = data.get("dialogue_emotions", [])
#             utt_analysis = data.get("utterance_emotions_analysis", [])
            
#             all_utt_emotions_in_dialogue = []
#             for item in utt_analysis:
#                 all_utt_emotions_in_dialogue.extend(item.get("emotions", []))
                
#             if label == 1:
#                 dlg_counter_1.update(dlg_emotions)
#                 utt_counter_1.update(all_utt_emotions_in_dialogue)
#             elif label == 0:
#                 dlg_counter_0.update(dlg_emotions)
#                 utt_counter_0.update(all_utt_emotions_in_dialogue)
# except FileNotFoundError:
#     print(f"找不到文件 {input_file}，请检查路径。")
#     exit()

# def get_sorted_data(c1, c0):
#     all_keys = set(c1.keys()).union(set(c0.keys()))
#     sorted_keys = sorted(list(all_keys), key=lambda x: c1[x] + c0[x], reverse=True)[:15]
#     sorted_keys.reverse()
    
#     counts_1 = [c1[k] for k in sorted_keys]
#     counts_0 = [c0[k] for k in sorted_keys]
#     return sorted_keys, counts_1, counts_0

# utt_labels, utt_counts_1, utt_counts_0 = get_sorted_data(utt_counter_1, utt_counter_0)
# dlg_labels, dlg_counts_1, dlg_counts_0 = get_sorted_data(dlg_counter_1, dlg_counter_0)

# # ================= 开始绘图 =================
# print("准备生成高级对比图表...")

# # 【审美升级2】调整为更修长的 16:8 比例，不再像正方形那么笨重
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8)) 

# bar_height = 0.38 # 柱子粗细适中

# # 辅助函数：统一给子图设置高级样式
# def format_axes(ax, labels, counts_0, counts_1, title):
#     y_pos = np.arange(len(labels))
    
#     # 画柱子 (设置 zorder=3 让柱子显示在网格线之上)
#     bars_0 = ax.barh(y_pos - bar_height/2, counts_0, height=bar_height, 
#                      label='Not Moral Coercion', color=COLOR_NORMAL, zorder=3)
#     bars_1 = ax.barh(y_pos + bar_height/2, counts_1, height=bar_height, 
#                      label='Moral Coercion', color=COLOR_MC, zorder=3)
    
#     # 【审美升级3】使用 bar_label 完美对齐数字，不再手动算位置！
#     ax.bar_label(bars_0, padding=4, fontsize=12, color='#333333')
#     ax.bar_label(bars_1, padding=4, fontsize=12, color='#333333')

#     # 【审美升级4】去掉顶部和右侧的黑色边框，左侧和底部边框变淡
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_color('#cccccc')
#     ax.spines['bottom'].set_color('#cccccc')
    
#     # 字体与标签设置
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(labels, fontsize=16)
#     ax.set_xlabel('Frequency', fontsize=16, color='#333333', weight='medium')
#     ax.tick_params(axis='x', labelsize=14, colors='#333333')
#     ax.tick_params(axis='y', length=0) # 去掉Y轴凸出的小刻度线，更干净
    
#     # 【审美升级5】添加浅灰色虚线网格，辅助阅读但不喧宾夺主
#     ax.grid(axis='x', linestyle='--', alpha=0.4, color='#999999', zorder=0)
    
#     # 添加子图标题
#     ax.set_title(title, fontsize=15, fontweight='bold', color='#222222', loc='left', pad=15)

# # 绘制左右两张图
# format_axes(ax1, utt_labels, utt_counts_0, utt_counts_1, 'A: Utterance-Level Emotions (Micro)')
# format_axes(ax2, dlg_labels, dlg_counts_0, dlg_counts_1, 'B: Dialogue-Level Emotions (Macro)')

# # 【审美升级6】全局共享一个图例，放在正上方中间，不占用画板内部空间
# handles, labels = ax1.get_legend_handles_labels()
# fig.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.1), 
#            ncol=2, fontsize=17, frameon=False)

# plt.tight_layout()

# # 保存并显示
# import os
# os.makedirs('analysis/GOEMOTIONS', exist_ok=True)
# plt.savefig('analysis/GOEMOTIONS/emotion_comparison_elegant2.png', dpi=300, bbox_inches='tight')
# plt.savefig('analysis/GOEMOTIONS/emotion_comparison_elegant2.pdf', dpi=300, bbox_inches='tight', format='pdf')
# print("绝美图表已保存！")
# plt.show()


# import json
# import numpy as np
# from collections import Counter
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import os

# # 全局字体配置
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['Linux Libertine O', 'Linux Libertine', 'LinLibertine']
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42

# input_file = "analysis/GOEMOTIONS/emotion_predictions.jsonl"

# COLOR_MC = '#C0392B'      # 深铁锈红
# COLOR_NORMAL = '#5D6D7E'  # 灰蓝色

# # [此处保持你的数据读取逻辑一致，提取计数器...]
# utt_counter_1, utt_counter_0 = Counter(), Counter()
# dlg_counter_1, dlg_counter_0 = Counter(), Counter()

# try:
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             if not line.strip(): continue
#             data = json.loads(line)
#             label = data.get("dialog_binary_label")
#             dlg_emotions = data.get("dialogue_emotions", [])
#             utt_analysis = data.get("utterance_emotions_analysis", [])
#             all_utt_emotions_in_dialogue = []
#             for item in utt_analysis:
#                 all_utt_emotions_in_dialogue.extend(item.get("emotions", []))
                
#             if label == 1:
#                 dlg_counter_1.update(dlg_emotions)
#                 utt_counter_1.update(all_utt_emotions_in_dialogue)
#             elif label == 0:
#                 dlg_counter_0.update(dlg_emotions)
#                 utt_counter_0.update(all_utt_emotions_in_dialogue)
# except FileNotFoundError:
#     print(f"找不到文件 {input_file}")
#     exit()

# # 获取交集并按总体频次排序，取前 15 个
# all_keys = set(utt_counter_1.keys()).union(set(dlg_counter_1.keys()))
# sorted_keys = sorted(list(all_keys), key=lambda x: utt_counter_1[x] + utt_counter_0[x] + dlg_counter_1[x] + dlg_counter_0[x], reverse=True)[:15]
# sorted_keys.reverse() # 让高频在上方

# # ==========================================================
# # 开始绘制镜像图 (Butterfly Chart) 极致节省空间
# # ==========================================================
# fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
# fig.subplots_adjust(wspace=0.25) # 中间留出放情感标签的空间

# y_pos = np.arange(len(sorted_keys))
# bar_height = 0.35

# # --------- 左图：Utterance-Level (让柱子向左伸展，数值取负数) ---------
# u_counts_0 = np.array([utt_counter_0[k] for k in sorted_keys])
# u_counts_1 = np.array([utt_counter_1[k] for k in sorted_keys])

# bars_u0 = ax_left.barh(y_pos - bar_height/2, -u_counts_0, height=bar_height, color=COLOR_NORMAL, zorder=3, label='Not Moral Coercion')
# bars_u1 = ax_left.barh(y_pos + bar_height/2, -u_counts_1, height=bar_height, color=COLOR_MC, zorder=3, label='Moral Coercion')

# # 左图数字标签 (使用绝对值格式化 '%g')
# ax_left.bar_label(bars_u0, labels=[f'{x}' for x in u_counts_0], padding=5, fontsize=14, color='#333333')
# ax_left.bar_label(bars_u1, labels=[f'{x}' for x in u_counts_1], padding=5, fontsize=14, color='#333333')

# # --------- 右图：Dialogue-Level (向右伸展，正常正数) ---------
# d_counts_0 = np.array([dlg_counter_0[k] for k in sorted_keys])
# d_counts_1 = np.array([dlg_counter_1[k] for k in sorted_keys])

# bars_d0 = ax_right.barh(y_pos - bar_height/2, d_counts_0, height=bar_height, color=COLOR_NORMAL, zorder=3)
# bars_d1 = ax_right.barh(y_pos + bar_height/2, d_counts_1, height=bar_height, color=COLOR_MC, zorder=3)

# ax_right.bar_label(bars_d0, padding=5, fontsize=14, color='#333333')
# ax_right.bar_label(bars_d1, padding=5, fontsize=14, color='#333333')

# # --------- 样式美化 ---------
# for ax, title in zip([ax_left, ax_right], ['A: Utterance-Level (Micro)', 'B: Dialogue-Level (Macro)']):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_color('#cccccc')
#     ax.grid(axis='x', linestyle='--', alpha=0.4, color='#999999', zorder=0)
#     ax.tick_params(axis='x', labelsize=14, colors='#333333')
#     ax.set_title(title, fontsize=18, fontweight='bold', color='#222222', pad=15)

# # 处理左图的负数 X 轴刻度，使其显示为正数
# ax_left.set_xticklabels([str(abs(int(x))) for x in ax_left.get_xticks()])

# # 把 Y 轴（情感标签）精准放置在两张图的中央
# ax_left.tick_params(axis='y', length=0)
# ax_left.set_yticks(y_pos)
# ax_left.set_yticklabels([]) # 先清空左侧默认y轴

# # 在中间空白带绘制标签
# for y, label in zip(y_pos, sorted_keys):
#     ax_left.text(0.5, y, label, ha='center', va='center', fontsize=18, color='#333333',
#                  transform=ax_left.get_yaxis_transform())

# # 右侧限制与图例
# ax_left.set_xlim(ax_left.get_xlim()[0] * 1.15, 0)
# ax_right.set_xlim(0, ax_right.get_xlim()[1] * 1.15)

# # 精准无框右上角图例
# ax_right.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=18, frameon=False)

# plt.tight_layout()
# os.makedirs('analysis/GOEMOTIONS', exist_ok=True)
# plt.savefig('analysis/GOEMOTIONS/emotion_butterfly_compact.pdf', dpi=300, bbox_inches='tight', format='pdf')
# plt.show()



import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ==========================================================
# 【核心配置区】全局修改 Matplotlib 字体为 Linux Libertine
# ==========================================================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Linux Libertine O', 'Linux Libertine', 'LinLibertine']

# 确保 PDF 导出时嵌入矢量字体 (Type 42)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 配置文件路径
input_file = "analysis/GOEMOTIONS/emotion_predictions.jsonl"

# 学术配色
COLOR_MC = '#C0392B'      
COLOR_NORMAL = '#5D6D7E'  

# ==========================================================
# 数据读取与计数器初始化
# ==========================================================
utt_counter_1 = Counter()
utt_counter_0 = Counter()
dlg_counter_1 = Counter()
dlg_counter_0 = Counter()

print("正在读取并统计情感数据...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            data = json.loads(line)
            
            label = data.get("dialog_binary_label")
            dlg_emotions = data.get("dialogue_emotions", [])
            utt_analysis = data.get("utterance_emotions_analysis", [])
            
            all_utt_emotions_in_dialogue = []
            for item in utt_analysis:
                all_utt_emotions_in_dialogue.extend(item.get("emotions", []))
                
            if label == 1:
                dlg_counter_1.update(dlg_emotions)
                utt_counter_1.update(all_utt_emotions_in_dialogue)
            elif label == 0:
                dlg_counter_0.update(dlg_emotions)
                utt_counter_0.update(all_utt_emotions_in_dialogue)
except FileNotFoundError:
    print(f"找不到文件 {input_file}")
    exit()

# ==========================================================
# 数据提取与独立清洗过滤（最高频前 15 项）
# ==========================================================
def get_top_15_data(c1, c0):
    all_keys = set(c1.keys()).union(set(c0.keys()))
    sorted_keys = sorted(list(all_keys), key=lambda x: c1[x] + c0[x], reverse=True)[:15]
    sorted_keys.reverse()  
    counts_1 = [c1[k] for k in sorted_keys]
    counts_0 = [c0[k] for k in sorted_keys]
    return sorted_keys, counts_1, counts_0

utt_labels, utt_counts_1, utt_counts_0 = get_top_15_data(utt_counter_1, utt_counter_0)
dlg_labels, dlg_counts_1, dlg_counts_0 = get_top_15_data(dlg_counter_1, dlg_counter_0)

# ==========================================================
# 绘图区：画布大小雷打不动保持 15 x 7.5
# ==========================================================
print("正在优化个体间距并放大 Y 轴标签...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

# 【核心修改 1】将柱子高度从 0.38 压缩至 0.28。
# 这样红蓝两根柱子整体收紧，个体与个体（不同情感行）之间的留白间距会瞬间拉远变宽！
bar_height = 0.40 
y_pos = np.arange(15)

# --------- 左子图：Utterance-Level ---------
bars_u0 = ax1.barh(y_pos - bar_height/2, utt_counts_0, height=bar_height, color=COLOR_NORMAL, zorder=3, label='Not Moral Coercion')
bars_u1 = ax1.barh(y_pos + bar_height/2, utt_counts_1, height=bar_height, color=COLOR_MC, zorder=3, label='Moral Coercion')

ax1.bar_label(bars_u0, padding=4, fontsize=13, color='#000000', fontweight='medium')
ax1.bar_label(bars_u1, padding=4, fontsize=13, color='#000000', fontweight='medium')

ax1.set_yticks(y_pos)
# 【核心修改 2】Y 轴字体由 18 放大到 21，并追加 bold 粗体，极大提升辨识度
ax1.set_yticklabels(utt_labels, fontsize=21, color='#000000')
ax1.set_title('(a) Utterance-Level (Micro)', fontsize=20, fontweight='bold', pad=15, loc='left')
ax1.set_xlim(0, max(max(utt_counts_0), max(utt_counts_1)) * 1.15)

# --------- 右子图：Dialogue-Level ---------
bars_d0 = ax2.barh(y_pos - bar_height/2, dlg_counts_0, height=bar_height, color=COLOR_NORMAL, zorder=3)
bars_d1 = ax2.barh(y_pos + bar_height/2, dlg_counts_1, height=bar_height, color=COLOR_MC, zorder=3)

ax2.bar_label(bars_d0, padding=4, fontsize=13, color='#000000', fontweight='medium')
ax2.bar_label(bars_d1, padding=4, fontsize=13, color='#000000', fontweight='medium')

ax2.set_yticks(y_pos)
# 【核心修改 3】右图 Y 轴字体同步放大到 21 并加粗
ax2.set_yticklabels(dlg_labels, fontsize=21, color='#000000')
ax2.set_title('(b) Dialogue-Level (Macro)', fontsize=20, fontweight='bold', pad=15, loc='left')
ax2.set_xlim(0, max(max(dlg_counts_0), max(dlg_counts_1)) * 1.15)

# --------- 全局学术样式美化 ---------
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    ax.tick_params(axis='x', labelsize=15, colors='#333333')
    ax.tick_params(axis='y', length=0, pad=8)  
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='#999999', zorder=0)

# ==========================================================
# 避撞布局与图例微调
# ==========================================================
fig.legend(handles=[bars_u1, bars_u0], labels=['Moral Coercion', 'Not Moral Coercion'], 
           loc='lower center', bbox_to_anchor=(0.5, 0.94), ncol=2, fontsize=17, frameon=False)

# 【核心修改 4】将 left 边距从 0.12 扩展到 0.14，给刚刚放大了的 Y 轴大字体腾出充足的呼吸空间，防止被切边
plt.subplots_adjust(wspace=0.35, left=0.14, right=0.96, top=0.83, bottom=0.12)

# ==========================================================
# 矢量化导出
# ==========================================================
os.makedirs('analysis/GOEMOTIONS', exist_ok=True)
plt.savefig('analysis/GOEMOTIONS/emotion_comparison_clean.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('analysis/GOEMOTIONS/emotion_comparison_clean.png', dpi=300, bbox_inches='tight')

print("图表美化完成！个体间距已完美拉开，Y 轴字体大幅清晰化。")
plt.show()