import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv('/home/fsy23/UniPoly/moleculenet/regression/Lipophilicity.csv')

# 假设目标列名为 'exp'
target_col = 'exp' 

# 提取数据并计算统计量
data = df[target_col].dropna() # 去除空值
mean_val = data.mean()
std_val = data.std()
min_val = data.min()
max_val = data.max()

# 2. 设置全局绘图风格 (论文风格)
plt.rcParams['font.family'] = 'serif'  # 衬线字体，类似 Times New Roman
plt.rcParams['axes.linewidth'] = 1.2   # 坐标轴线宽
plt.rcParams['xtick.direction'] = 'in' # 刻度向内
plt.rcParams['ytick.direction'] = 'in' # 刻度向内

# 3. 创建画布
fig, ax = plt.subplots(figsize=(6, 5), dpi=150) # 屏幕显示用 150 dpi

# 4. 绘制直方图和密度曲线 (KDE)
# color='#4c72b0' 是经典的学术深蓝色
# sns.histplot(data=data, stat='density', kde=True, 
#              color='#105CA4', edgecolor='black', alpha=0.7, ax=ax)
sns.histplot(data=data, stat='density', kde=True, 
             color='#08336E', edgecolor='black', alpha=0.7, ax=ax)

# 5. 添加统计信息框
stats_text = (f"mean = {mean_val:.2f}\n"
              f"std = {std_val:.2f}\n"
              f"min = {min_val:.2f}\n"
              f"max = {max_val:.2f}")

# 使用 bbox 参数创建圆角文本框
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left', bbox=props)

# 6. 设置标签和标题
ax.set_xlabel('Experimental logD', fontsize=12, labelpad=10)
ax.set_ylabel('Density', fontsize=12, labelpad=10)
ax.set_title('Lipophilicity Distribution', fontsize=14, pad=15)

# 7. 调整布局
plt.tight_layout()

# --- 保存图片的关键代码 ---
# dpi=300: 设置为300分辨率，满足绝大多数期刊要求
# bbox_inches='tight': 自动裁剪周围多余的白边
plt.savefig('/home/fsy23/UniPoly/makeGraph/dataveiw/Lipophilicity_Distribution.png', dpi=300, bbox_inches='tight')

# 如果你想保存为矢量图 (PDF)，可以使用下面这行：
# plt.savefig('Lipophilicity_Distribution.pdf', format='pdf', bbox_inches='tight')

# 显示图片
plt.show()