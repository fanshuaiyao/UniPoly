import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 读取数据 (请确保文件名正确)
df = pd.read_csv('/home/fsy23/UniPoly/moleculenet/regression/FreeSolv.csv') 

# --- 关键修改：目标列名为 'expt' ---
target_col = 'expt' 

# 提取数据并计算统计量
data = df[target_col].dropna() # 去除空值
mean_val = data.mean()
std_val = data.std()
min_val = data.min()
max_val = data.max()

# 2. 设置全局绘图风格
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['axes.linewidth'] = 1.2  
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 3. 创建画布
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

# 4. 绘制直方图和密度曲线
# 这里的颜色我换成了深绿色 '#55A868'，用来区分 Lipophilicity (蓝色)
sns.histplot(data=data, stat='density', kde=True, 
             color='#55A868', edgecolor='black', alpha=0.7, ax=ax)

# 5. 添加统计信息框 (保持在左上角)
stats_text = (f"mean = {mean_val:.2f}\n"
              f"std = {std_val:.2f}\n"
              f"min = {min_val:.2f}\n"
              f"max = {max_val:.2f}")

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

# 左上角对齐
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left', bbox=props)

# 6. 设置标签和标题 (适配 FreeSolv)
# 单位是 kcal/mol
ax.set_xlabel('Hydration Free Energy (kcal/mol)', fontsize=12, labelpad=10)
ax.set_ylabel('Density', fontsize=12, labelpad=10)
ax.set_title('FreeSolv Dataset Distribution', fontsize=14, pad=15)

# 7. 调整布局并保存
plt.tight_layout()
plt.savefig('/home/fsy23/UniPoly/makeGraph/dataveiw/FreeSolv_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()