"""
对比算法 Baseline — 与 TabPFN 进行性能对比
用法: conda run -n tabpfn python compare_baselines.py
"""

# ============================================================
#                    参数配置区
# ============================================================

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 路径 ---
DATA_PATH   = r'data\3000固定温度\ml_dataset.csv'
RESULTS_DIR = r'results'
EXPERIMENT_NAME = os.path.basename(os.path.dirname(DATA_PATH))

# --- TabPFN 实验结果路径（如果有） ---
TABPFN_RESULTS_DIR = r'results\20260622_2247_3000固定温度'  # TabPFN 评估结果目录

# --- 训练参数 ---
TEST_SIZE  = 0.2
RANDOM_STATE = 42

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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
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
    """生成算法对比柱状图。"""
    metrics = ['R2', 'MAE', 'RMSE']
    algorithms = list(next(iter(all_results.values())).keys())
    algorithms = [a for a in algorithms if a != 'y_test']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metrics):
        data = []
        for algo in algorithms:
            values = [all_results[target][algo][metric] for target in all_results]
            data.append(values)

        x = np.arange(len(algorithms))
        ax.bar(x, [np.mean(d) for d in data], yerr=[np.std(d) for d in data],
               capsize=5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'平均 {metric} 对比')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close()


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
    exp_dir = Path(RESULTS_DIR) / f'{timestamp}_{EXPERIMENT_NAME}_baselines'
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
    Y = df[OUTPUT_COLS].values

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
    print("\n" + "=" * 60)
    print("开始训练 Baseline 模型...")
    print("=" * 60)

    all_results = {}

    for i, target in enumerate(OUTPUT_COLS):
        print(f"\n[{i+1}/{len(OUTPUT_COLS)}] 目标: {target}")

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

    print(f"\n结果已保存至: {exp_dir}/")
    print("  - baseline_comparison.csv (详细指标)")
    print("  - algorithm_comparison_bar.png (算法对比柱状图)")
    print("  - scatter_true_vs_pred_grid.png (真实值vs预测值散点图)")


if __name__ == '__main__':
    main()
