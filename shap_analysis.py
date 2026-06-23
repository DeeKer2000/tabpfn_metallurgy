"""
TabPFN SHAP 特征重要性分析
用法: conda activate tabpfn && python shap_analysis.py
"""

# ============================================================
#                    参数配置区
# ============================================================

import os
os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_LelHYWBZTv7hyvkS0GfY_H8oC4BMwsQ-VTUcP01sAVM'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 路径 ---
DATA_PATH    = r'data\3000固定温度\ml_dataset.csv'
RESULTS_DIR  = r'results'                    # 实验结果总目录
EXPERIMENT_NAME = os.path.basename(os.path.dirname(DATA_PATH))  # 与 train_tabpfn.py 一致

# --- 设备 ---
DEVICE = 'cuda'  # 'cpu' 或 'cuda'

# --- 分析设置 ---
TARGETS = None            # None = 全部目标变量，或指定列表如 ['matte_Cu_pct', 'slag_Cu_pct']
N_EXPLAIN = 50            # 解释的样本数（越多越慢）
BUDGET = 256              # SHAP 计算预算（2^特征数，16个特征用 65536 精确，256 近似）

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


def find_latest_experiment(results_dir, experiment_name):
    """找到 results 下匹配实验名的最新目录（格式: 时间戳_实验名）。"""
    base = Path(results_dir)
    matches = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.endswith(f'_{experiment_name}')],
        reverse=True
    )
    if not matches:
        raise FileNotFoundError(f"未找到匹配 '{experiment_name}' 的实验目录: {results_dir}")
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

    return True


def main():
    # 根据 DATA_PATH 推导实验名，找到最新的匹配实验
    exp_dir = find_latest_experiment(RESULTS_DIR, EXPERIMENT_NAME)
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

    # 3. 逐个分析
    success, fail = 0, 0
    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] 分析: {target}")
        if analyze_single_target(target, model_dir, output_dir, X, feature_names):
            success += 1
        else:
            fail += 1

    print(f"\n{'='*40}")
    print(f"分析完成: 成功 {success}, 跳过 {fail}")
    print(f"结果已保存至: {output_dir}/")


if __name__ == '__main__':
    main()
