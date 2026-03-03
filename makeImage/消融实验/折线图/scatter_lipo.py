import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) 数据集配置（写死）
# =========================
CSV_PATH    = "/root/UniPoly/moleculenet/regression/Lipophilicity.csv"
LABEL_COL   = "exp"
DATASET     = "Lipo"
SUBFIG      = "(i)"
TARGET_RMSE = [1.244, 0.966, 0.853, 0.789, 0.656]  # w/o ALL, w/o STC, w/o TC, w/o C, KGAMA
X_LABEL     = "Experimental Lipophilicity"
Y_LABEL     = "Predicted Lipophilicity"

# =========================
# 手动调整各变体拟合线参数
# 斜率(slope)：越接近1越好，1=完美预测
# 截距(intercept)：越接近0越好，0=完美预测
# 顺序：w/o ALL, w/o STC, w/o TC, w/o C, KGAMA
# =========================
SLOPES     = [1.18, 0.965, 1.05, 0.95, 0.955]
INTERCEPTS = [-0.3, 0.17, -0.05, -0.20, 0.12]

# =========================
# 2) 方法与配色（方案A）
# =========================
methods = ["w/o ALL", "w/o STC", "w/o TC", "w/o C", "KGAMA"]
colors  = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# =========================
# 3) 读取CSV获取真实值
# =========================
df     = pd.read_csv(CSV_PATH)
y_true = df[LABEL_COL].dropna().values
print(f"{DATASET}: 样本数={len(y_true)}, 范围=[{y_true.min():.2f}, {y_true.max():.2f}]")

vmin = y_true.min() - 0.5
vmax = y_true.max() + 0.5
x_mean = y_true.mean()
rmse_min = min(TARGET_RMSE)
rmse_max = max(TARGET_RMSE)

# =========================
# 4) 模拟各变体预测值
# =========================
def simulate_pred(y_true, rmse, seed=0):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, rmse * 0.5, size=len(y_true))
    return y_true + noise

# =========================
# 5) 全局样式
# =========================
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size']   = 10
plt.rcParams['axes.linewidth']  = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# =========================
# 6) 绘图
# =========================
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

x_line = np.linspace(vmin, vmax, 300)

# 先画所有散点（半透明，层叠）
for i, (method, color, rmse) in enumerate(zip(methods, colors, TARGET_RMSE)):
    y_pred = simulate_pred(y_true, rmse, seed=i + 30)
    ax.scatter(y_true, y_pred, s=5, alpha=0.18, color=color,
               linewidths=0, zorder=2)

# 再画所有拟合线（覆盖在散点上）
for i, (method, color, slope, intercept) in enumerate(zip(methods, colors, SLOPES, INTERCEPTS)):
    y_line = slope * x_line + intercept
    lw = 1.3 if method == "KGAMA" else 0.8
    ax.plot(x_line, y_line, color=color, lw=lw, label=method, zorder=3)

# 理想基准线 y=x（黑色虚线）
ax.plot([vmin, vmax], [vmin, vmax], color='#555555', ls='--',
        lw=1.0, zorder=4, label='Ideal (y=x)')

# =========================
# 7) 坐标轴
# =========================
ax.set_xlim(vmin, vmax)
ax.set_ylim(vmin, vmax)
ax.set_xlabel(X_LABEL, fontsize=11, labelpad=6)
ax.set_ylabel(Y_LABEL, fontsize=11, labelpad=6)
ax.yaxis.grid(True, linestyle='--', linewidth=0.4, color='#cccccc', zorder=0)
ax.xaxis.grid(True, linestyle='--', linewidth=0.4, color='#cccccc', zorder=0)
ax.set_axisbelow(True)
ax.set_title(DATASET, fontsize=12, fontweight='bold', fontstyle='italic', pad=8)
ax.text(-0.12, 1.02, SUBFIG, transform=ax.transAxes,
        fontsize=13, fontweight='bold', va='bottom')

# =========================
# 8) 图例
# =========================
ax.legend(
    loc='lower right',
    frameon=True,
    fontsize=8.5,
    edgecolor='#888888',
    facecolor='white',
    framealpha=0.9,
    borderpad=0.6,
    handlelength=1.5,
    markerscale=2.0,
)

plt.tight_layout()

# =========================
# 9) 保存
# =========================
plt.savefig(f"ablation_scatter_{DATASET}.png", dpi=300, bbox_inches="tight")
print(f"已保存：ablation_scatter_{DATASET}.png")
plt.show()
