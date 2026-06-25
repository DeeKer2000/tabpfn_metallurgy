"""
TabPFN SHAP 特征重要性分析
用法: conda run -n tabpfn python scripts/analysis/shap_analysis.py
"""

# ============================================================
#                    参数配置区
# ============================================================

import os
os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_LelHYWBZTv7hyvkS0GfY_H8oC4BMwsQ-VTUcP01sAVM'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 路径（相对于项目根目录） ---
DATA_PATH    = r'data\3000_fixed_temp\ml_dataset.csv'
TABPFN_DIR   = r'experiments\3000_fixed_temp\tabpfn'         # TabPFN 训练结果目录（含模型）

# --- 设备 ---
DEVICE = 'cuda'  # 'cpu' 或 'cuda'

# --- 分析设置 ---
TARGETS = None            # None = 全部目标变量，或指定列表如 ['matte_Cu_pct', 'slag_Cu_pct']
N_EXPLAIN = 50            # 解释的样本数（越多越慢）
BUDGET = 256              # SHAP 计算预算（2^特征数，16个特征用 65536 精确，256 近似）

# --- 实验目录选择 ---
# None = 自动选 TABPFN_DIR 下最新的目录，或手动指定如 '20260622_2247'
EXPERIMENT_DIR = None

# --- 输入特征列 ---
INPUT_COLS = [
    'T(℃)', 'P(atm)', 'Cu(g)', 'S(g)', 'Fe(g)', 'SiO2(g)',
    'Pb(g)', 'Zn(g)', 'MgO(g)', 'Al2O3(g)', 'Ni(g)', 'Sb(g)',
    'Bi(g)', 'CaO(g)', 'As(g)', 'O2(g)'
]

# --- 输出目标列（与 train_tabpfn.py 一致） ---
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
#                    分析代码
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import shap
from tabpfn import load_fitted_tabpfn_model
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq, shapiq_to_shap_explanation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_experiment(base_dir):
    """找到目录下最新的子目录（按名称排序，通常为时间戳）。"""
    base = Path(base_dir)
    matches = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        reverse=True
    )
    if not matches:
        raise FileNotFoundError(f"未找到任何实验目录: {base_dir}")
    return matches[0]


def analyze_single_target(target, model_dir, output_dir, X, feature_names):
    """对单个目标变量进行 SHAP 分析。"""
    model_path = model_dir / f'{target}.tabpfn_fit'
    if not model_path.exists():
        print(f"  跳过 {target}: 模型文件不存在")
        return False

    # 加载模型
    reg = load_fitted_tabpfn_model(model_path, device=DEVICE)

    # 创建 SHAP 解释器
    explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
        model=reg, data=X[:200], index='SV', max_order=1,
    )

    # 计算 SHAP 值
    n = min(N_EXPLAIN, len(X))
    explanation = shapiq_to_shap_explanation(
        explainer, X[:n], budget=BUDGET, feature_names=feature_names
    )

    # 生成可视化
    prefix = output_dir / target

    plt.figure()
    shap.summary_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f'{prefix}_summary.png', dpi=150)
    plt.close()

    plt.figure()
    shap.plots.bar(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f'{prefix}_bar.png', dpi=150)
    plt.close()

    plt.figure()
    shap.plots.waterfall(explanation[0], show=False)
    plt.tight_layout()
    plt.savefig(f'{prefix}_waterfall.png', dpi=150)
    plt.close()

    # 输出特征重要性排名
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])
    print(f"\n  特征重要性排名 ({target}):")
    for name, val in importance:
        bar = '█' * int(val / max(mean_abs_shap) * 30)
        print(f"    {name:12s} {val:.4f}  {bar}")

    return mean_abs_shap


def plot_shap_heatmap(shap_matrix, feature_names, target_names, output_dir):
    """生成 SHAP 特征重要性热力图（特征 × 目标）。

    Parameters
    ----------
    shap_matrix : np.ndarray, shape (n_targets, n_features)
        每行是一个目标的 mean |SHAP| 值
    feature_names : list[str]
    target_names : list[str]
    output_dir : Path
    """
    n_targets = len(target_names)
    n_features = len(feature_names)

    # 归一化：每行（每个目标）除以该行最大值，便于跨目标比较
    row_max = shap_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1  # 避免除零
    norm_matrix = shap_matrix / row_max

    fig_h = max(4, n_features * 0.35 + 1.5)
    fig_w = max(6, n_targets * 0.45 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(norm_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # 坐标轴
    ax.set_xticks(range(n_targets))
    ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_names, fontsize=8)

    # 在格子中标注原始数值
    for i in range(n_targets):
        for j in range(n_features):
            val = shap_matrix[i, j]
            # 深色格子用白字，浅色格子用黑字
            text_color = 'white' if norm_matrix[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=6, color=text_color)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('归一化 mean |SHAP|', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title('SHAP 特征重要性热力图', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'shap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_heatmap_from_csv(csv_path, output_dir=None):
    """从已保存的 shap_importance.csv 生成热力图（无需重跑 SHAP 计算）。

    用法:
        plot_heatmap_from_csv('results/xxx/shap_results/shap_importance.csv')
    """
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent

    df = pd.read_csv(csv_path, index_col=0, encoding='utf-8-sig')
    shap_matrix = df.values
    target_names = list(df.index)
    feature_names = list(df.columns)

    plot_shap_heatmap(shap_matrix, feature_names, target_names, output_dir)
    print(f"热力图已保存: {output_dir / 'shap_heatmap.png'}")


def main():
    # 选择实验目录：手动指定 > 自动找最新
    if EXPERIMENT_DIR:
        exp_dir = Path(TABPFN_DIR) / EXPERIMENT_DIR
        if not exp_dir.exists():
            raise FileNotFoundError(f"指定的实验目录不存在: {exp_dir}")
    else:
        exp_dir = find_latest_experiment(TABPFN_DIR)
    model_dir = exp_dir / 'saved_models'
    output_dir = exp_dir / 'shap_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"实验目录: {exp_dir}")

    # 1. 加载数据
    print("加载数据...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    X = df[INPUT_COLS].values
    feature_names = INPUT_COLS

    # 2. 确定要分析的目标变量
    if TARGETS is None:
        targets = OUTPUT_COLS
    else:
        targets = TARGETS

    print(f"待分析目标: {len(targets)} 个")

    # 3. 逐个分析，收集 SHAP 重要性矩阵
    success, fail = 0, 0
    shap_values = []  # 收集每个目标的 mean |SHAP|
    analyzed_targets = []

    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] 分析: {target}")
        result = analyze_single_target(target, model_dir, output_dir, X, feature_names)
        if result is not False:
            shap_values.append(result)
            analyzed_targets.append(target)
            success += 1
        else:
            fail += 1

    # 4. 保存 SHAP 重要性矩阵 + 生成热力图
    if shap_values:
        shap_matrix = np.array(shap_values)  # shape: (n_targets, n_features)
        df_shap = pd.DataFrame(shap_matrix, index=analyzed_targets, columns=feature_names)
        df_shap.to_csv(output_dir / 'shap_importance.csv', encoding='utf-8-sig')
        print(f"\nSHAP 重要性矩阵已保存: {output_dir / 'shap_importance.csv'}")
        plot_shap_heatmap(shap_matrix, feature_names, analyzed_targets, output_dir)
        print(f"热力图已保存: {output_dir / 'shap_heatmap.png'}")

    print(f"\n{'='*40}")
    print(f"分析完成: 成功 {success}, 跳过 {fail}")
    print(f"结果已保存至: {output_dir}/")


if __name__ == '__main__':
    main()
