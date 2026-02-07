import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 读取数据
# 注意：文件名可能是 'esol.csv' 或 'delaney-processed.csv'
df = pd.read_csv('/home/fsy23/UniPoly/moleculenet/regression/ESOL.csv') 

# --- 关键：目标列名 ---
# 如果报错 key error，请把这就改成 'measured' 或 'logSolubility'
# 标准列名通常很长：
target_col = 'measured log solubility in mols per litre' 

# 提取数据
data = df[target_col].dropna()
mean_val = data.mean()
std_val = data.std()
min_val = data.min()
max_val = data.max()

# 2. 设置风格
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['axes.linewidth'] = 1.2  
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 3. 创建画布
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

# 4. 绘制 (颜色：学术红 #C44E52)
sns.histplot(data=data, stat='density', kde=True, 
             color='#6D011F', edgecolor='black', alpha=0.7, ax=ax)

# 5. 添加统计信息框 (左上角)
stats_text = (f"mean = {mean_val:.2f}\n"
              f"std = {std_val:.2f}\n"
              f"min = {min_val:.2f}\n"
              f"max = {max_val:.2f}")

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left', bbox=props)

# 6. 设置标签 (适配 ESOL)
ax.set_xlabel('Value (log(mol/L))', fontsize=12, labelpad=10)
ax.set_ylabel('Density', fontsize=12, labelpad=10)
ax.set_title('ESOL Dataset Distribution', fontsize=14, pad=15)

# 7. 保存
plt.tight_layout()
plt.savefig('/home/fsy23/UniPoly/makeGraph/dataveiw/ESOL_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()