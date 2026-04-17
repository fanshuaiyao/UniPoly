import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =========================
# 1) 数据准备 (你确认过的完美数据)
# =========================
tasks = [
    "BACE", "BBBP", "ClinTox", "Tox21", "ToxCast", "Sider", 
    "FreeSolv", "ESOL", "Lipo"
]

# 模态顺序 (为了让 Graph 在最下方作为基座，这里我按图示顺序调整)
# 注意：Seaborn 画图是从上到下的，所以如果你想 Graph 在最下面，要把它放在列表最后
modalities = ["SMILES", "Fingerprint", "Text", "Graph"]

# 数据矩阵 (注意：这里需要按照上面 modalities 的顺序来填入数据)
# 原数据是：Graph, Fingerprint, SMILES, Text
# 我们需要重新组织一下行，或者生成 DataFrame 后转置

# 原始数据 (行=任务, 列=Graph, FP, SMILES, Text)
data_raw = np.array([
    [0.35, 0.32, 0.18, 0.15], # BACE
    [0.38, 0.25, 0.21, 0.16], # BBBP
    [0.26, 0.18, 0.20, 0.36], # ClinTox
    [0.29, 0.17, 0.22, 0.32], # Tox21
    [0.30, 0.25, 0.24, 0.21], # ToxCast
    [0.25, 0.19, 0.18, 0.38], # Sider
    [0.40, 0.30, 0.18, 0.12], # FreeSolv
    [0.39, 0.28, 0.20, 0.13], # ESOL
    [0.29, 0.38, 0.21, 0.12]  # Lipo
])

# 创建原始 DataFrame (Index=Tasks, Col=Modalities_Original)
df_raw = pd.DataFrame(data_raw, index=tasks, columns=["Graph", "Fingerprint", "SMILES", "Text"])

# =========================
# 2) 关键修正：转置与重排
# =========================
# 1. 转置 (Transpose): 行变列，列变行 -> (行=模态, 列=任务)
df_plot = df_raw.T 

# 2. 调整Y轴顺序 (Reorder Y-axis)
# 你的参考图中 Graph 在最下面，SMILES 在最上面
# 我们按照 ["SMILES", "Fingerprint", "Text", "Graph"] 的顺序重新索引
df_plot = df_plot.reindex(["SMILES", "Fingerprint", "Text", "Graph"])

# =========================
# 3) 绘图 (完全复刻参考图样式)
# =========================
plt.figure(figsize=(12, 5)) # 调整为横向宽幅比例

# 绘制热图
ax = sns.heatmap(df_plot, 
                 cmap="Blues",       # 经典的蓝色系
                 annot=True,         # 显示数值
                 fmt=".2f",          # 保留两位小数
                 linewidths=1.5,     # 格子间距 (加宽一点更像参考图)
                 linecolor='white',  # 白色分割线
                #  cbar_kws={'label': 'Attention Weight'},
                 annot_kws={"size": 11}) # 字体大小

# 设置轴标签与标题
# plt.title("Multi-modal Attention Weights Across Molecular Tasks", fontsize=15, pad=20, weight='bold')
# plt.xlabel("Molecular Downstream Tasks", fontsize=12, labelpad=10)
# plt.ylabel("Input Modalities", fontsize=12, labelpad=10)

# 调整 X 轴标签角度 (水平显示更美观)
plt.xticks(rotation=0, fontsize=11)
plt.yticks(rotation=0, fontsize=11)

# 保存
plt.tight_layout()
plt.savefig("attention_heatmap_final1.png", dpi=600, bbox_inches="tight")
# plt.savefig("attention_heatmap_final.pdf", format="pdf", bbox_inches="tight")

print("修正版热图已生成：attention_heatmap_final1.png")
plt.show()