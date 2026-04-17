import numpy as np
import matplotlib.pyplot as plt

tau_values = [0.05, 0.1, 0.2, 0.5, 1.0]

y_clintox = [0.895, 0.941, 0.932, 0.910, 0.880]

y_bbbp = [0.885, 0.930, 0.925, 0.905, 0.875]

plt.figure(figsize=(8, 6))

plt.plot(tau_values, y_clintox, marker='o', linestyle='-', linewidth=2, label='ClinTox', color='#1f77b4') # 蓝色
plt.plot(tau_values, y_bbbp, marker='s', linestyle='--', linewidth=2, label='BBBP', color='#ff7f0e')   # 橙色


plt.scatter([0.1], [0.941], s=100, c='red', zorder=5) # 高亮 ClinTox 的最高点
plt.text(0.1, 0.945, 'Optimal', fontsize=10, color='red', ha='center')

plt.xscale('log') # 因为 0.05 到 1.0 跨度较大，用对数轴更合理
plt.xticks(tau_values, labels=[str(t) for t in tau_values]) # 强制显示这几个刻度

plt.xlabel(r'温度系数（$\tau$）', fontsize=12, fontname='SimSun')
plt.ylabel('AUROC', fontsize=12)
plt.title(r'Sensitivity Analysis of Temperature $\tau$', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("sensitivity_tau.png", dpi=600, bbox_inches="tight")
plt.show()