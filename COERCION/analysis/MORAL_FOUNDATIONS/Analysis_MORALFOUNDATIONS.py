import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl # <--- 新增导入
import seaborn as sns
import os


# ==============================================
# 1. 配置参数
# ==============================================
input_file = "analysis/MORAL_FOUNDATIONS/moral_foundations.jsonl"
output_image = "analysis/MORAL_FOUNDATIONS/moral_overall_count2.png"
output_pdf = "analysis/MORAL_FOUNDATIONS/moral_overall_count2.pdf"
THRESHOLD = 0.5  # 判定该道德维度被触发的概率阈值

# 【审美升级 1】与你的前一张图表保持学术级配色统一
COLOR_MC = '#C0392B'      # 深铁锈红 (Moral Coercion)
COLOR_NORMAL = '#5D6D7E'  # 灰蓝色 (Not Moral Coercion)

label_names = [
    "care_virtue", "care_vice", "fairness_virtue", "fairness_vice", 
    "loyalty_virtue", "loyalty_vice", "authority_virtue", "authority_vice", 
    "sanctity_virtue", "sanctity_vice"
]

# 初始化统计字典
counts = {
    1: {k: 0 for k in label_names},
    0: {k: 0 for k in label_names}
}

# ==============================================
# 2. 读取数据并统计数量 (仅整体对话)
# ==============================================
print("正在读取数据并统计频次...")
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
            
        b_label = data.get("dialog_binary_label")
        if b_label not in [0, 1]: continue
            
        overall_scores = data["analysis"].get("overall_dialogue_scores", {})
        for mf, score in overall_scores.items():
            if score >= THRESHOLD:
                counts[b_label][mf] += 1

# ==============================================
# 3. 转换为 Pandas DataFrame 格式 (长表)
# ==============================================
records = []
for b_label, mfs in counts.items():
    category_name = "Moral Coercion" if b_label == 1 else "Not Moral Coercion"
    for mf_name, count_val in mfs.items():
        records.append({
            "Dialogue Type": category_name,
            "Moral Foundation": mf_name,
            "Count": count_val
        })

df = pd.DataFrame(records)

# ==============================================
# 4. 绘制高级感分组柱状图
# ==============================================
print("正在生成高级图表...")

# 清除 seaborn 默认沉重的白底灰线网格，使用极简干净的底色
sns.set_style("white")

# 【关键动作】：在 seaborn 设置完之后，立刻把字体抢回来！
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Linux Libertine O', 'Linux Libertine', 'LinLibertine']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.figure(figsize=(16, 8)) # 稍微加长画布，让10个维度有呼吸空间

# 绘制柱状图 (zorder=3 确保柱子覆盖在网格线之上)
ax = sns.barplot(
    data=df, 
    x="Moral Foundation", 
    y="Count", 
    hue="Dialogue Type",
    palette={"Moral Coercion": COLOR_MC, "Not Moral Coercion": COLOR_NORMAL},
    zorder=3,
    width=0.6  # <--- 在这里添加 width 参数，数值越小柱子越细
)

# 遍历图表中的柱子，并添加数值标签 (字体颜色换成柔和的深灰 #333333)
for container in ax.containers:
    ax.bar_label(container, fmt='%g', padding=4, fontsize=15, color='#333333')

# ==============================================
# 5. 极致的图表细节美化
# ==============================================

# 【审美升级 2】“呼吸感”边框：干掉顶部、右侧、甚至左侧的实线边框
sns.despine(top=True, right=True, left=True)
ax.spines['bottom'].set_color('#cccccc') # 底部基准线变淡

# 【审美升级 3】加入浅色虚线网格辅助阅读，层级设在柱子下方 (zorder=0)
plt.grid(axis='y', linestyle='--', alpha=0.4, color='#999999', zorder=0)

plt.xlabel("") # 强制清空 X 轴默认标签
plt.ylabel("Number of Dialogues", fontsize=20, color='#333333', weight='medium')

# 增加 Y 轴上限，防止最高的数字和图例/顶部边缘打架
max_count = df["Count"].max()
plt.ylim(0, max_count * 1.15) 

# 隐藏 Y 轴凸出的小刻度线，更清爽
ax.tick_params(axis='y', length=0)
plt.yticks(fontsize=20, color='#333333')

# 【审美升级 4】X 轴倾斜文字锚定对齐，视觉上更稳
plt.xticks(rotation=40, ha="right", rotation_mode='anchor', fontsize=20, color='#333333')

# 【审美升级 5】高级图例排版：移出主图表，顶部居中横向排列，无外框
plt.legend(
    title=None, 
    fontsize=22, 
    loc='lower right',          # 以图例自身的右下角作为对齐锚点
    bbox_to_anchor=(1.0, 0.8),  # 锚点精准定位在主图区域的右上角 (X=1.0, Y=1.02) 稍微偏上的位置
    ncol=2,                     # 保持 2 列横排（如果想变竖排，可以删掉 ncol 或者设为 1）
    frameon=False               # 依旧保持无外框的清爽设计
)

# 自动调整布局并保存
plt.tight_layout()
os.makedirs(os.path.dirname(output_image), exist_ok=True)
plt.savefig(output_image, dpi=300)
plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
print(f"高级感统计图已保存至: {output_image} 和 {output_pdf}")

plt.show()