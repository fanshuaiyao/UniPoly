import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) 回归任务与方法（与你分类图保持一致）
# =========================
tasks_reg = ["FreeSolv", "ESOL", "LIPO"]

methods = [
    "G",
    "G+F",
    "G+F+S",
    "G+F+S+T (Concat)",
    "G+F+S+T (Cross-Attn)"
]

# =========================
# 2) 示例数据（后续替换成你的真实 RMSE）
# shape: [num_methods, num_tasks]
# 注意：RMSE 越小越好
# =========================
np.random.seed(7)

base = np.array([1.85, 0.85, 0.72])  # 每个任务的 baseline（示例）
delta = np.array([
    [0.00, 0.00, 0.00],     # G
    [-0.10, -0.05, -0.04],  # G+F
    [-0.14, -0.08, -0.06],  # G+F+S
    [-0.16, -0.09, -0.07],  # Concat
    [-0.22, -0.12, -0.10],  # Cross-Attn（更低更好）
])

results = base + delta + np.random.normal(0, 0.02, size=delta.shape)
results = np.clip(results, 0.05, None)

# （可选）标准差（没有就删掉 yerr 那行）
std = np.full_like(results, 0.03)

# =========================
# 3) 绘图：分组柱状图（回归 RMSE）
# =========================
num_methods = len(methods)
num_tasks = len(tasks_reg)

x = np.arange(num_tasks)
bar_width = 0.14
gap = 0.02

plt.figure(figsize=(7.5, 4.5))  # 回归任务少，图可以更紧凑

for i, method in enumerate(methods):
    offset = (i - (num_methods - 1) / 2) * (bar_width + gap)
    plt.bar(
        x + offset,
        results[i],
        width=bar_width,
        label=method,
        yerr=std[i],
        capsize=2
    )

plt.xticks(x, tasks_reg)
plt.ylabel("RMSE (↓)")
plt.ylim(0.0, max(results.flatten()) + 0.4)  # 自动留白
plt.legend(ncol=2, frameon=False)            # 图例两列更好看
plt.tight_layout()

# =========================
# 4) 保存（论文级）
# =========================
plt.savefig(
    "ablation_regression.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.savefig(
    "ablation_regression.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()
