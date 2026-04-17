import numpy as np
import matplotlib.pyplot as plt


tasks = ["Pgp", "BBB", "CYP2D6 Inhibition", "Ames"]

methods = [
    "w/o Text",
    "w/o CoT",
    "w/o LoRA",
    "w/o Cross-Attn",
    "CoT-CMP",
]

# 深蓝灰渐变配色，完整模型用最深色突出
# 暖灰棕渐变配色（学术风格）
colors = ['#d4c5b2', '#b8a08a', '#967a62', '#74553e', '#4a3122']


results = np.array([
    [0.835, 0.803, 0.597, 0.805],  # w/o Text
    [0.805, 0.791, 0.593, 0.795],  # w/o CoT
    [0.775, 0.740, 0.566, 0.748],  # w/o LoRA
    [0.872, 0.825, 0.638, 0.820],  # w/o Cross-Attn
    [0.912, 0.850, 0.650, 0.838],  # CoT-CMP
])

std = np.array([
    [0.018, 0.016, 0.037, 0.017],  # w/o Text
    [0.024, 0.032, 0.027, 0.022],  # w/o CoT
    [0.012, 0.010, 0.029, 0.015],  # w/o LoRA
    [0.015, 0.021, 0.023, 0.019],  # w/o Cross-Attn
    [0.020, 0.019, 0.010, 0.017],  # CoT-CMP
])

# =========================
# 3) 全局样式
# =========================
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# =========================
# 4) 绘图
# =========================
num_methods = len(methods)
num_tasks = len(tasks)

x = np.arange(num_tasks)
bar_width = 0.14
gap = 0.01

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

for i, (method, color) in enumerate(zip(methods, colors)):
    offset = (i - (num_methods - 1) / 2) * (bar_width + gap)
    bars = ax.bar(
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
ax.set_xticklabels(tasks, fontsize=11)
ax.set_xlabel("Dataset", fontsize=12, labelpad=8)
ax.set_ylabel("AUROC", fontsize=12, labelpad=8)
ax.set_ylim(0.50, 0.98)
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))
ax.tick_params(axis='y', which='minor', direction='in', length=2)

# 网格线（仅横向，淡色）
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', zorder=0)
ax.set_axisbelow(True)


ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=5,
    frameon=True,
    fontsize=9,
    edgecolor='#888888',
    facecolor='white',
    framealpha=0.9,
    borderpad=0.5,
    handlelength=1.2,
    handleheight=0.9,
    columnspacing=1.0,
)

plt.tight_layout()


plt.savefig("ablation_bar_admet_gronw.png", dpi=300, bbox_inches="tight")
print("已保存：ablation_bar_admet_gronw.png")
plt.show()
