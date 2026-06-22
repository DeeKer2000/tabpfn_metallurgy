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

# --- 分析设置 ---
TARGET = 'matte_Cu_pct'  # 要分析的目标变量（必须是已训练过的目标）
N_EXPLAIN = 50            # 解释的样本数（越多越慢）
BUDGET = 256              # SHAP 计算预算（2^特征数，16个特征用 65536 精确，256 近似）

# --- 输入特征列 ---
INPUT_COLS = [
    'T(℃)', 'P(atm)', 'Cu(g)', 'S(g)', 'Fe(g)', 'SiO2(g)',
    'Pb(g)', 'Zn(g)', 'MgO(g)', 'Al2O3(g)', 'Ni(g)', 'Sb(g)',
    'Bi(g)', 'CaO(g)', 'As(g)', 'O2(g)'
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

    # 2. 加载训练好的模型
    model_path = model_dir / f'{TARGET}.tabpfn_fit'
    print(f"加载模型: {model_path}")
    if not model_path.exists():
        print(f"错误: 模型文件不存在，请先运行 train_tabpfn.py 训练 {TARGET}")
        return
    reg = load_fitted_tabpfn_model(model_path)

    # 3. 创建 SHAP 解释器
    print("创建 SHAP 解释器...")
    explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
        model=reg,
        data=X[:200],  # 用部分数据作为背景数据
        index='SV',
        max_order=1,
    )

    # 4. 计算 SHAP 值
    n = min(N_EXPLAIN, len(X))
    X_explain = X[:n]
    print(f"计算 {n} 个样本的 SHAP 值 (budget={BUDGET})...")
    explanation = shapiq_to_shap_explanation(
        explainer, X_explain, budget=BUDGET, feature_names=feature_names
    )

    # 5. 生成可视化
    prefix = output_dir / TARGET

    print("生成 SHAP summary plot...")
    plt.figure()
    shap.summary_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f'{prefix}_summary.png', dpi=150)
    plt.close()

    print("生成 SHAP bar plot...")
    plt.figure()
    shap.plots.bar(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f'{prefix}_bar.png', dpi=150)
    plt.close()

    print("生成 SHAP waterfall (第1个样本)...")
    plt.figure()
    shap.plots.waterfall(explanation[0], show=False)
    plt.tight_layout()
    plt.savefig(f'{prefix}_waterfall.png', dpi=150)
    plt.close()

    # 6. 输出特征重要性排名
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])
    print(f"\n{'='*40}")
    print(f"特征重要性排名 ({TARGET}):")
    print(f"{'='*40}")
    for name, val in importance:
        bar = '█' * int(val / max(mean_abs_shap) * 30)
        print(f"  {name:12s} {val:.4f}  {bar}")

    print(f"\nSHAP 结果已保存至: {output_dir}/")


if __name__ == '__main__':
    main()
