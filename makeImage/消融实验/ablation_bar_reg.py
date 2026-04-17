import numpy as np
import matplotlib.pyplot as plt


tasks_reg = ["FreeSolv", "ESOL", "LIPO"]

methods = [
    "w/o ALL",
    "w/o STC",
    "w/o TC",
    "w/o C",
    "KGAMA"
]

# 低调学术配色：深蓝灰渐变系（与分类图一致）
colors = ['#b0c4de', '#7a9bbf', '#4a7aa0', '#2a5a80', '#0d3b5e']

results = np.array([
    [2.102, 1.521, 1.244],  # w/o ALL
    [1.855, 1.476, 0.966],  # w/o STC
    [1.801, 1.226, 0.853],  # w/o TC
    [1.725, 1.001, 0.789],  # w/o C
    [1.512, 0.831, 0.656],  # KGAMA
])

std = np.array([
    [0.042, 0.030, 0.025],
    [0.038, 0.028, 0.020],
    [0.033, 0.024, 0.017],
    [0.028, 0.020, 0.014],
    [0.025, 0.016, 0.012],
])


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


num_methods = len(methods)
num_tasks = len(tasks_reg)

x = np.arange(num_tasks)
bar_width = 0.13
gap = 0.01

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

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

ax.set_xticks(x)
ax.set_xticklabels(tasks_reg, fontsize=11)
ax.set_xlabel("Dataset", fontsize=12, labelpad=8)
ax.set_ylabel("RMSE", fontsize=12, labelpad=8)
ax.set_ylim(0.4, 2.4)
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.tick_params(axis='y', which='minor', direction='in', length=2)

# 网格线（仅横向，淡色）
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', zorder=0)
ax.set_axisbelow(True)

# 子图编号
# ax.text(-0.08, 1.02, '(b)', transform=ax.transAxes,
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
plt.savefig("ablation_bar_reg1.png", dpi=300, bbox_inches="tight")
print("已保存：ablation_bar_reg.png")
plt.show()
