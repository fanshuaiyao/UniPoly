import numpy as np
import matplotlib.pyplot as plt


tasks_cls = ["BACE", "BBBP", "ClinTox", "Tox21", "ToxCast", "Sider"]

methods = [
    "w/o ALL",
    "w/o STC",
    "w/o TC",
    "w/o C",
    "KGAMA"
]

# 低调学术配色：深蓝灰渐变系，方案A留给ROC折线图
colors = ['#b0c4de', '#7a9bbf', '#4a7aa0', '#2a5a80', '#0d3b5e']


results = np.array([
    [0.825, 0.898, 0.805, 0.771, 0.695, 0.638],  # w/o ALL
    [0.842, 0.912, 0.845, 0.795, 0.701, 0.645],  # w/o STC
    [0.855, 0.915, 0.885, 0.822, 0.706, 0.655],  # w/o TC
    [0.862, 0.922, 0.915, 0.835, 0.709, 0.661],  # w/o C
    [0.870, 0.930, 0.939, 0.841, 0.712, 0.669],  # KGAMA
])

std = np.array([
    [0.008, 0.010, 0.012, 0.009, 0.011, 0.010],
    [0.007, 0.009, 0.011, 0.008, 0.010, 0.009],
    [0.006, 0.008, 0.010, 0.007, 0.009, 0.008],
    [0.005, 0.007, 0.009, 0.006, 0.008, 0.007],
    [0.006, 0.002, 0.012, 0.005, 0.011, 0.004],
])

# =========================
# 3) 全局样式
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# =========================
# 4) 绘图
# =========================
num_methods = len(methods)
num_tasks = len(tasks_cls)

x = np.arange(num_tasks)
bar_width = 0.13
gap = 0.01

fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

for i, (method, color) in enumerate(zip(methods, colors)):
    offset = (i - (num_methods - 1) / 2) * (bar_width + gap)
    ax.bar(
        x + offset,
        results[i],
        width=bar_width,
        label=method,
        color=color,
        yerr=std[i],
        capsize=2,
        error_kw={'elinewidth': 0.6, 'capthick': 0.6},
        edgecolor='#555555',
        linewidth=0.5,
        alpha=0.88,
    )

# =========================
# 5) 坐标轴设置
# =========================
ax.set_xticks(x)
ax.set_xticklabels(tasks_cls, fontsize=11)
ax.set_xlabel("Dataset", fontsize=12, labelpad=8)
ax.set_ylabel("AUROC", fontsize=12, labelpad=8)
ax.set_ylim(0.62, 0.98)
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
ax.tick_params(axis='y', which='minor', direction='in', length=2)

# 网格线（仅横向，淡色）
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', zorder=0)
ax.set_axisbelow(True)

# 子图编号
# ax.text(-0.06, 1.02, '(a)', transform=ax.transAxes,
#         fontsize=13, fontweight='bold', va='bottom')

# =========================
# 6) 图例
# =========================
ax.legend(
    loc='upper right',
    bbox_to_anchor=(0.99, 0.99),
    frameon=True,
    fontsize=9.5,
    edgecolor='#888888',
    facecolor='white',
    framealpha=0.9,
    borderpad=0.7,
    handlelength=1.2,
    handleheight=0.9,
)

plt.tight_layout()

# =========================
# 7) 保存
# =========================
plt.savefig("ablation_bar.png", dpi=300, bbox_inches="tight")
print("已保存：ablation_bar_new1.png")
plt.show()
