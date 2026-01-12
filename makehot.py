import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# Data
# =======================
tasks = ['BACE', 'BBBP', 'ClinTox', 'HIV', 'MUV', 'Sider', 'Tox21', 'Toxcast']
modalities = ['SMILES', 'Fingerprint', 'Text', 'Graph']

# data = np.array([
#     [0.20, 0.35, 0.15, 0.15, 0.15],
#     [0.32, 0.15, 0.10, 0.35, 0.08],
#     [0.15, 0.12, 0.40, 0.21, 0.12],
#     [0.25, 0.20, 0.15, 0.26, 0.14],
#     [0.18, 0.42, 0.12, 0.19, 0.09],
#     [0.10, 0.15, 0.42, 0.15, 0.18],
#     [0.18, 0.28, 0.14, 0.20, 0.20]
#     [0.22, 0.32, 0.15, 0.20, 0.11],
# ])

data = np.array([
    [0.20, 0.35, 0.15, 0.15],
    [0.32, 0.15, 0.10, 0.35],
    [0.15, 0.12, 0.40, 0.21],
    [0.25, 0.20, 0.15, 0.26],
    [0.18, 0.42, 0.12, 0.19],
    [0.10, 0.15, 0.42, 0.15],
    [0.18, 0.28, 0.14, 0.20],
    [0.22, 0.32, 0.15, 0.20],
])

df = pd.DataFrame(data, index=tasks, columns=modalities)

# =======================
# Transpose: modalities on y, tasks on x
# =======================
df_plot = df.T

# =======================
# Figure
# =======================
plt.figure(figsize=(10, 4.8), dpi=150)
sns.set_theme(style="white", context="paper", font_scale=1.1)

ax = sns.heatmap(
    df_plot,
    annot=True,
    fmt=".2f",
    cmap="Blues",          # 🔵 经典 SCI 蓝
    vmin=0.08,
    vmax=0.45,
    square=True,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={
        "label": "Attention Weight",
        "shrink": 0.8
    },
    annot_kws={"size": 9}
)

# =======================
# Labels
# =======================
plt.title(
    "Multi-modal Attention Weights across Molecular Tasks",
    fontsize=14,
    pad=14,
    weight="bold"
)

plt.xlabel("Molecular Downstream Tasks", fontsize=11)
plt.ylabel("Input Modalities", fontsize=11)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(
    "heatmap_Blues-1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
