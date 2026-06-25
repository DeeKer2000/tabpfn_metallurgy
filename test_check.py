import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

algorithms = ['RandomForest', 'XGBoost', 'GradientBoosting', 'MLP', 'TabPFN']
colors = ['#7884B4', '#B4C0E4', '#E4CCD8', '#F0C0CC', '#484878']
display = ['Random Forest', 'XGBoost', 'Gradient Boosting', 'MLP', 'TabPFN']
data = {
    'R2': [0.84, 0.68, 0.77, -185966765.12, 0.99],
    'MAE': [0.24, 0.28, 0.22, 0.16, 0.04],
    'RMSE': [0.38, 0.42, 0.34, 0.26, 0.11],
}
errs = {
    'R2': [0.15, 0.25, 0.20, 1700000000, 0.02],
    'MAE': [0.10, 0.12, 0.08, 0.05, 0.015],
    'RMSE': [0.12, 0.15, 0.10, 0.08, 0.02],
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['legend.frameon'] = False
plt.rcParams['font.size'] = 9
plt.rcParams['svg.fonttype'] = 'none'

metrics = ['R2', 'MAE', 'RMSE']
metric_labels = {'R2': r'$R^2$', 'MAE': 'MAE', 'RMSE': 'RMSE'}
n_algos = 5
n_metrics = 3

fig = plt.figure(figsize=(8.0, 3.2))
gs = fig.add_gridspec(1, n_metrics + 1, width_ratios=[1]*n_metrics + [0.5], wspace=0.35)
axes = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
ax_leg = fig.add_subplot(gs[0, n_metrics])

bar_width = 0.7
w = bar_width / n_algos
x = np.arange(1)

for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
    all_vals = data[metric]
    all_errs = errs[metric]

    if metric == 'R2':
        y_lo, y_hi = -0.05, 1.05
        ax.axhline(0, color='#A0A0A0', linestyle='--', linewidth=0.6, zorder=0)
    elif metric == 'MAE':
        y_lo = 0
        y_hi = max(v + e for v, e in zip(all_vals, all_errs)) * 1.15
    else:
        y_lo = 0
        y_hi = max(v + e for v, e in zip(all_vals, all_errs)) * 1.15

    ax.set_ylim(y_lo, y_hi)

    for i, (algo, color) in enumerate(zip(algorithms, colors)):
        offset = (i - (n_algos - 1) / 2) * w
        val = all_vals[i]
        err = all_errs[i]
        bar_height = min(val, y_hi) if val > 0 else max(val, y_lo)
        bar = ax.bar(
            x + offset, bar_height, width=w,
            color=color, edgecolor='black', linewidth=0.6,
            yerr=err if val >= 0 else None,
            error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'capsize': 3},
            label=display[i] if ax_idx == 0 else '_nolegend_',
            clip_on=False,
        )
        text_y = val + err + 0.01 * (y_hi - y_lo)
        if y_lo <= text_y <= y_hi * 1.05:
            ax.text(x[0] + offset, text_y, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=6.5, color='#272727')
        elif val < y_lo:
            ax.annotate(f'{val:.1f}',
                        xy=(x[0] + offset, y_lo + 0.02 * (y_hi - y_lo)),
                        xytext=(x[0] + offset, y_lo - 0.12 * (y_hi - y_lo)),
                        ha='center', va='top', fontsize=5.5, color='#B64342',
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='#B64342'))

    ax.set_xticks([])
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=7.5, length=3, width=0.8)

handles, labels = axes[0].get_legend_handles_labels()
ax_leg.legend(handles, labels, fontsize=7.5, loc='center', frameon=False, handlelength=1.5, handletextpad=0.5)
ax_leg.set_axis_off()
for i, ax in enumerate(axes):
    ax.text(-0.08, 1.05, chr(97 + i), transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='#272727', ha='left', va='bottom')

fig.savefig('test_reverted.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Done')
