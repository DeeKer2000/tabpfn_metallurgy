"""
对比算法 Baseline — 与 TabPFN 进行性能对比
用法: conda run -n tabpfn python scripts/analysis/compare_baselines.py
"""

# ============================================================
#                    参数配置区
# ============================================================

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 路径（相对于项目根目录） ---
DATA_PATH   = r'data\3000_fixed_temp\ml_dataset.csv'
RESULTS_DIR = r'experiments\3000_fixed_temp\baselines'        # 基线算法结果目录

# --- TabPFN 实验结果路径（用于加载评估指标对比） ---
TABPFN_RESULTS_DIR = r'experiments\3000_fixed_temp\tabpfn\20260622_2247'

# --- 训练参数 ---
TEST_SIZE  = 0.2
RANDOM_STATE = 42

# --- 目标变量选择 ---
# None = 全部目标变量，或指定列表如 ['matte_Cu_pct', 'slag_Cu_pct']
TARGETS = None

# --- 输入/输出列（与 train_tabpfn.py 一致） ---
INPUT_COLS = [
    'T(℃)', 'P(atm)', 'Cu(g)', 'S(g)', 'Fe(g)', 'SiO2(g)',
    'Pb(g)', 'Zn(g)', 'MgO(g)', 'Al2O3(g)', 'Ni(g)', 'Sb(g)',
    'Bi(g)', 'CaO(g)', 'As(g)', 'O2(g)'
]

OUTPUT_COLS = [
    # 冰铜 (wt.%)
    'matte_Cu_pct', 'matte_Fe_pct', 'matte_S_pct',
    'matte_Ni_pct', 'matte_Zn_pct', 'matte_Pb_pct', 'matte_As_pct',
    'matte_mass_g',
    # 渣 (wt.%)
    'slag_Cu_pct', 'slag_FeO_pct', 'slag_SiO2_pct', 'slag_FeO_SiO2_ratio',
    'slag_Al2O3_pct', 'slag_CaO_pct', 'slag_MgO_pct', 'slag_ZnO_pct',
    'slag_mass_g',
    # 气相 (mol分数)
    'gas_SO2_mol', 'gas_S2_mol', 'gas_SO_mol', 'gas_SSO_mol', 'gas_SO3_mol',
    'gas_Zn_mol', 'gas_PbS_mol', 'gas_CuS_mol', 'gas_Pb_mol', 'gas_Sb_mol',
    'gas_As_mol',
    'gas_mass_g',
]

# ============================================================
#                    对比算法代码
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Nature publication style（来自 nature-figure skill） ─────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入 XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: 未安装 XGBoost，将跳过 XGBoost 对比")


def train_baseline_models(X_train, Y_train, X_test, Y_test, target_idx, target_name):
    """训练所有 baseline 模型并返回结果。"""
    y_train = Y_train[:, target_idx]
    y_test = Y_test[:, target_idx]

    # 跳过常数列
    if np.std(y_train) < 1e-10:
        return None

    results = {}

    # 1. Random Forest
    print(f"    训练 Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['RandomForest'] = {
        'R2': r2_score(y_test, y_pred_rf),
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'predictions': y_pred_rf
    }

    # 2. Gradient Boosting
    print(f"    训练 Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results['GradientBoosting'] = {
        'R2': r2_score(y_test, y_pred_gb),
        'MAE': mean_absolute_error(y_test, y_pred_gb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'predictions': y_pred_gb
    }

    # 3. MLP (需要标准化)
    print(f"    训练 MLP...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    results['MLP'] = {
        'R2': r2_score(y_test, y_pred_mlp),
        'MAE': mean_absolute_error(y_test, y_pred_mlp),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_mlp)),
        'predictions': y_pred_mlp
    }

    # 4. XGBoost (如果可用)
    if HAS_XGBOOST:
        print(f"    训练 XGBoost...")
        xgb = XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        results['XGBoost'] = {
            'R2': r2_score(y_test, y_pred_xgb),
            'MAE': mean_absolute_error(y_test, y_pred_xgb),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            'predictions': y_pred_xgb
        }

    # 保存真实值
    results['y_test'] = y_test

    return results


def plot_comparison_bar(all_results, output_dir):
    """生成 Nature 风格算法对比柱状图（grouped bar，每个指标一个 panel）。"""
    # ── Nature publication style ──────────────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['font.size'] = 9

    # ── NMI pastel palette (unified cool family) ─────────────────────────────
    METHOD_COLORS = {
        'TabPFN':            '#484878',   # baseline_dark — hero method
        'RandomForest':      '#7884B4',   # baseline_mid
        'XGBoost':           '#B4C0E4',   # baseline_soft
        'GradientBoosting':  '#E4CCD8',   # ours_base — warm accent
        'MLP':               '#F0C0CC',   # ours_large — warm accent
    }
    DISPLAY_NAMES = {
        'TabPFN':           'TabPFN',
        'RandomForest':     'Random Forest',
        'XGBoost':          'XGBoost',
        'GradientBoosting': 'Gradient Boosting',
        'MLP':              'MLP',
    }

    # ── Metric display config ────────────────────────────────────────────────
    metrics = ['R2', 'MAE', 'RMSE']
    metric_labels = {
        'R2':   r'$R^2$',
        'MAE':  'MAE',
        'RMSE': 'RMSE',
    }

    algorithms = [a for a in next(iter(all_results.values())).keys() if a != 'y_test']
    n_algos = len(algorithms)
    n_metrics = len(metrics)

    # ── Collect data: mean ± std per algorithm per metric ─────────────────────
    means = {m: [] for m in metrics}
    stds  = {m: [] for m in metrics}
    for algo in algorithms:
        for m in metrics:
            vals = [all_results[t][algo][m] for t in all_results]
            means[m].append(np.mean(vals))
            stds[m].append(np.std(vals))

    # ── Figure layout: 1×3 data panels + 1 legend panel ──────────────────────
    fig = plt.figure(figsize=(8.0, 3.2))
    gs = fig.add_gridspec(1, n_metrics + 1, width_ratios=[1]*n_metrics + [0.5],
                          wspace=0.35)

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_metrics)]
    ax_leg = fig.add_subplot(gs[0, n_metrics])

    bar_width = 0.7
    w = bar_width / n_algos
    x = np.arange(1)  # single category per panel

    for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
        colors = [METHOD_COLORS.get(a, '#999999') for a in algorithms]

        # ── Compute y-axis limits first (handle extreme outliers) ─────────────
        all_vals = means[metric]
        all_errs = stds[metric]
        positive_vals = [v for v in all_vals if v >= 0]
        positive_errs = [e for v, e in zip(all_vals, all_errs) if v >= 0]

        if metric == 'R2':
            # R²: cap at [-0.05, 1.05], show reference line at 0
            y_lo, y_hi = -0.05, 1.05
            ax.axhline(0, color='#A0A0A0', linestyle='--', linewidth=0.6, zorder=0)
        elif positive_vals:
            y_lo = max(0, min(v - e for v, e in zip(positive_vals, positive_errs)) * 0.85)
            y_hi = max(v + e for v, e in zip(positive_vals, positive_errs)) * 1.15
        else:
            y_lo, y_hi = 0, 1

        ax.set_ylim(y_lo, y_hi)

        # ── Draw bars and annotations ────────────────────────────────────────
        for i, (algo, color) in enumerate(zip(algorithms, colors)):
            offset = (i - (n_algos - 1) / 2) * w
            val = means[metric][i]
            err = stds[metric][i]

            # Clip bar height to visible range for drawing
            bar_height = min(val, y_hi) if val > 0 else max(val, y_lo)
            bar = ax.bar(
                x + offset, bar_height, width=w,
                color=color, edgecolor='black', linewidth=0.6,
                yerr=err if val >= 0 else None,
                error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'capsize': 3},
                label=DISPLAY_NAMES.get(algo, algo) if ax_idx == 0 else '_nolegend_',
                clip_on=False,
            )

            # ── Direct value annotation (only if within visible range) ───────
            text_y = val + err + 0.01 * (y_hi - y_lo)
            if y_lo <= text_y <= y_hi * 1.05:
                ax.text(
                    x[0] + offset, text_y,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=6.5,
                    color='#272727',
                )
            elif val < y_lo:
                # Mark extreme negative values with arrow annotation
                ax.annotate(
                    f'{val:.1f}',
                    xy=(x[0] + offset, y_lo + 0.02 * (y_hi - y_lo)),
                    xytext=(x[0] + offset, y_lo - 0.12 * (y_hi - y_lo)),
                    ha='center', va='top', fontsize=5.5, color='#B64342',
                    arrowprops=dict(arrowstyle='->', lw=0.6, color='#B64342'),
                )

        # ── Axis styling ─────────────────────────────────────────────────────
        ax.set_xticks([])
        ax.set_ylabel(metric_labels[metric], fontsize=10, fontweight='bold')
        ax.set_title('')
        ax.tick_params(axis='y', labelsize=7.5, length=3, width=0.8)

    # ── Dedicated legend panel ────────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    ax_leg.legend(handles, labels, fontsize=7.5, loc='center',
                  frameon=False, handlelength=1.5, handletextpad=0.5)
    ax_leg.set_axis_off()

    # ── Panel labels (a, b, c) ───────────────────────────────────────────────
    for i, ax in enumerate(axes):
        ax.text(-0.08, 1.05, chr(97 + i), transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='#272727',
                ha='left', va='bottom')

    # ── Save PNG ─────────────────────────────────────────────────────────────
    fig.savefig(output_dir / 'algorithm_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_grid(all_results, output_dir, n_cols=5):
    """生成真实值 vs 预测值散点图网格。"""
    targets = list(all_results.keys())
    n_targets = len(targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        ax = axes[idx]
        y_test = all_results[target]['y_test']

        # 绘制各算法的散点图
        algorithms = [a for a in all_results[target].keys() if a != 'y_test']
        for algo in algorithms:
            y_pred = all_results[target][algo]['predictions']
            ax.scatter(y_test, y_pred, alpha=0.5, s=20, label=algo)

        # 对角线
        min_val = min(y_test.min(), min(all_results[target][a]['predictions'].min() for a in algorithms))
        max_val = max(y_test.max(), max(all_results[target][a]['predictions'].max() for a in algorithms))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='完美预测')

        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(target, fontsize=9)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('真实值 vs 预测值对比', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_true_vs_pred_grid.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_line_comparison(all_results, output_dir, n_cols=5):
    """生成真实值 vs 预测值折线对比图（按真实值排序，观察趋势拟合程度）。"""
    # PALETTE_NMI_PASTEL（与 algorithm_comparison_bar 统一）
    METHOD_COLORS = {
        'TabPFN':            '#484878',
        'RandomForest':      '#7884B4',
        'XGBoost':           '#B4C0E4',
        'GradientBoosting':  '#E4CCD8',
        'MLP':               '#F0C0CC',
    }
    DISPLAY_NAMES = {
        'TabPFN':           'TabPFN',
        'RandomForest':     'Random Forest',
        'XGBoost':          'XGBoost',
        'GradientBoosting': 'Gradient Boosting',
        'MLP':              'MLP',
    }

    targets = list(all_results.keys())
    n_targets = len(targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    legend_handles = {}

    for idx, target in enumerate(targets):
        ax = axes[idx]
        y_test = all_results[target]['y_test']

        # 按真实值排序，使折线呈现趋势
        sort_idx = np.argsort(y_test)
        y_sorted = y_test[sort_idx]
        x_idx = np.arange(len(y_sorted))

        ax.plot(x_idx, y_sorted, 'k-', linewidth=1.5, alpha=0.9, label='真实值', zorder=5)

        algorithms = [a for a in all_results[target].keys() if a != 'y_test']
        for algo in algorithms:
            y_pred = all_results[target][algo]['predictions']
            y_pred_sorted = y_pred[sort_idx]
            color = METHOD_COLORS.get(algo, '#999999')
            display = DISPLAY_NAMES.get(algo, algo)
            line = ax.plot(x_idx, y_pred_sorted, '-', color=color, linewidth=1.0,
                           alpha=0.8, label=display, zorder=4)
            if display not in legend_handles:
                legend_handles[display] = line[0]

        ax.set_xlabel('样本（按真实值排序）', fontsize=8)
        ax.set_ylabel('值', fontsize=8)
        ax.set_title(target, fontsize=9)
        ax.tick_params(axis='both', labelsize=7, length=3, width=0.8)
        ax.grid(axis='y', alpha=0.2)

    # 隐藏多余子图
    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)

    # 全局图例
    if legend_handles:
        fig.legend(legend_handles.values(), legend_handles.keys(),
                   loc='lower center', ncol=min(len(legend_handles), 5),
                   fontsize=8, frameon=False, handlelength=1.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    fig.savefig(output_dir / 'line_true_vs_pred.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def load_tabpfn_results(tabpfn_dir, X_test, Y_test):
    """加载 TabPFN 模型并计算预测结果。"""
    from tabpfn import load_fitted_tabpfn_model
    import os

    tabpfn_dir = Path(tabpfn_dir)
    model_dir = tabpfn_dir / 'saved_models'

    if not model_dir.exists():
        print(f"警告: TabPFN 模型目录不存在: {model_dir}")
        return None

    print("加载 TabPFN 模型并计算预测...")
    results = {}

    for i, target in enumerate(OUTPUT_COLS):
        model_path = model_dir / f'{target}.tabpfn_fit'
        if not model_path.exists():
            continue

        try:
            reg = load_fitted_tabpfn_model(model_path, device='cuda')
            y_pred = reg.predict(X_test)
            y_test = Y_test[:, i]

            results[target] = {
                'R2': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'predictions': y_pred
            }
        except Exception as e:
            print(f"  跳过 {target}: {e}")

    return results


def main():
    # 生成时间戳，创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    exp_dir = Path(RESULTS_DIR) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Baseline 算法对比实验")
    print("=" * 60)
    print(f"实验目录: {exp_dir}")

    # 1. 加载数据
    print("\n加载数据...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    print(f"  样本数: {len(df)}, 列数: {len(df.columns)}")

    X = df[INPUT_COLS].values

    # 确定目标变量
    if TARGETS is not None:
        output_cols = TARGETS
    else:
        output_cols = OUTPUT_COLS
    Y = df[output_cols].values

    # 2. 划分训练集 / 测试集（与 TabPFN 一致的随机种子）
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 2.5 加载 TabPFN 结果（如果存在）
    tabpfn_results = None
    if TABPFN_RESULTS_DIR:
        tabpfn_results = load_tabpfn_results(TABPFN_RESULTS_DIR, X_test, Y_test)

    # 3. 逐目标变量训练 baseline 模型
    print(f"\n待分析目标: {len(output_cols)} 个")
    print("\n" + "=" * 60)
    print("开始训练 Baseline 模型...")
    print("=" * 60)

    all_results = {}

    for i, target in enumerate(output_cols):
        print(f"\n[{i+1}/{len(output_cols)}] 目标: {target}")

        results = train_baseline_models(X_train, Y_train, X_test, Y_test, i, target)
        if results is None:
            print(f"  跳过（常数列）")
            continue

        # 添加 TabPFN 结果（如果存在）
        if tabpfn_results and target in tabpfn_results:
            results['TabPFN'] = tabpfn_results[target]

        all_results[target] = results

        # 打印该目标的结果
        for algo, metrics in results.items():
            if algo != 'y_test':
                print(f"    {algo}: R2={metrics['R2']:.4f}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}")

    # 4. 汇总结果
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    # 创建汇总表
    summary_rows = []
    algorithms = [a for a in next(iter(all_results.values())).keys() if a != 'y_test']

    for target in all_results:
        for algo in algorithms:
            summary_rows.append({
                'target': target,
                'algorithm': algo,
                'R2': all_results[target][algo]['R2'],
                'MAE': all_results[target][algo]['MAE'],
                'RMSE': all_results[target][algo]['RMSE']
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(exp_dir / 'baseline_comparison.csv', index=False, encoding='utf-8-sig')

    # 打印平均指标
    print(f"\n{'算法':20s} {'平均R2':>10s} {'平均MAE':>10s} {'平均RMSE':>10s}")
    print("-" * 55)
    for algo in algorithms:
        algo_data = df_summary[df_summary['algorithm'] == algo]
        print(f"{algo:20s} {algo_data['R2'].mean():10.4f} {algo_data['MAE'].mean():10.4f} {algo_data['RMSE'].mean():10.4f}")

    # 5. 生成可视化
    print("\n生成可视化图表...")
    plot_comparison_bar(all_results, exp_dir)
    plot_scatter_grid(all_results, exp_dir)
    plot_line_comparison(all_results, exp_dir)

    print(f"\n结果已保存至: {exp_dir}/")
    print("  - baseline_comparison.csv (详细指标)")
    print("  - algorithm_comparison_bar.png (算法对比柱状图，300dpi)")
    print("  - scatter_true_vs_pred_grid.png (真实值vs预测值散点图)")
    print("  - line_true_vs_pred.png (真实值vs预测值折线对比图，300dpi)")


if __name__ == '__main__':
    main()
