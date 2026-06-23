"""
TabPFN 多输出回归预测 — FactSage 配矿数据
用法: conda run -n tabpfn python train_tabpfn.py
"""

# ============================================================
#                    参数配置区（按需修改）
# ============================================================

# --- 环境 ---
import os
os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_LelHYWBZTv7hyvkS0GfY_H8oC4BMwsQ-VTUcP01sAVM'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 路径 ---
DATA_PATH   = r'data\3000固定温度\ml_dataset.csv'           # 数据文件
MODEL_PATH  = r'TabPFN-main\models\tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt'  # 预训练权重
RESULTS_DIR = r'results'                                     # 实验结果总目录
# EXPERIMENT_NAME = '3000固定温度'                              # 实验名称（会出现在文件夹名中）
EXPERIMENT_NAME = os.path.basename(os.path.dirname(DATA_PATH))
# --- 设备 ---
DEVICE = 'cuda'  # 'cpu' 或 'cuda'

# --- 训练参数 ---
TEST_SIZE  = 0.2       # 测试集比例
RANDOM_STATE = 42      # 随机种子（保证可复现）
BATCH_SIZE = 1000      # 预测时分批大小（大数据集时防OOM）

# --- 输入特征列 ---
INPUT_COLS = [
    'T(℃)', 'P(atm)', 'Cu(g)', 'S(g)', 'Fe(g)', 'SiO2(g)',
    'Pb(g)', 'Zn(g)', 'MgO(g)', 'Al2O3(g)', 'Ni(g)', 'Sb(g)',
    'Bi(g)', 'CaO(g)', 'As(g)', 'O2(g)'
]

# --- 输出目标列 ---
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
#                    训练代码（一般无需修改）
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tabpfn import TabPFNRegressor
from tabpfn import save_fitted_tabpfn_model, load_fitted_tabpfn_model
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def predict_batched(reg, X, batch_size=1000):
    """分批预测，防止大数据集OOM。"""
    if len(X) <= batch_size:
        return reg.predict(X)
    preds = []
    for start in range(0, len(X), batch_size):
        preds.append(reg.predict(X[start:start + batch_size]))
    return np.concatenate(preds)


def main():
    # 0. 生成时间戳，创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    exp_dir = Path(RESULTS_DIR) / f'{timestamp}_{EXPERIMENT_NAME}'
    save_dir = exp_dir / 'saved_models'
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / 'evaluation_results.csv'

    # 1. 加载数据
    print("=" * 60)
    print("加载数据...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    print(f"  样本数: {len(df)}, 列数: {len(df.columns)}")

    X = df[INPUT_COLS].values
    Y = df[OUTPUT_COLS].values

    # 2. 划分训练集 / 测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 3. 逐目标变量训练 TabPFN 回归模型
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    results = []
    predictions = {}  # 保存预测值用于绘图

    for i, target in enumerate(OUTPUT_COLS):
        y_train = Y_train[:, i]
        y_test = Y_test[:, i]

        if np.std(y_train) < 1e-10:
            print(f"\n[{i+1}/{len(OUTPUT_COLS)}] {target}: 常数列，跳过")
            results.append({
                'target': target, 'R2': 1.0, 'MAE': 0.0,
                'RMSE': 0.0, 'mean_val': np.mean(y_train)
            })
            predictions[target] = {'y_test': y_test, 'y_pred': y_test}  # 常数列预测=真实值
            continue

        print(f"\n[{i+1}/{len(OUTPUT_COLS)}] 训练: {target}")

        reg = TabPFNRegressor(
            device=DEVICE,
            model_path=MODEL_PATH,
            ignore_pretraining_limits=True
        )
        reg.fit(X_train, y_train)
        y_pred = predict_batched(reg, X_test, BATCH_SIZE)

        # 保存训练好的模型
        save_path = save_dir / f'{target}.tabpfn_fit'
        save_fitted_tabpfn_model(reg, save_path)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            'target': target, 'R2': r2, 'MAE': mae,
            'RMSE': rmse, 'mean_val': np.mean(y_test)
        })

        # 保存预测值
        predictions[target] = {'y_test': y_test, 'y_pred': y_pred}

        print(f"  R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # 4. 汇总结果
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    df_res = pd.DataFrame(results)
    df_res['MAE_pct'] = df_res['MAE'] / df_res['mean_val'].abs().replace(0, np.nan) * 100

    print(f"\n{'目标变量':25s} {'R2':>8s} {'MAE':>10s} {'RMSE':>10s} {'MAE%':>8s}")
    print("-" * 65)
    for _, row in df_res.iterrows():
        print(f"{row['target']:25s} {row['R2']:8.4f} {row['MAE']:10.4f} {row['RMSE']:10.4f} {row['MAE_pct']:7.1f}%")

    print(f"\n平均 R2: {df_res['R2'].mean():.4f}")

    df_res.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n评估结果已保存至: {output_path}")
    print(f"训练好的模型已保存至: {save_dir}/")

    # 5. 生成真实值 vs 预测值散点图
    print("\n生成真实值 vs 预测值散点图...")
    n_cols = 5
    n_rows = (len(OUTPUT_COLS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    for idx, target in enumerate(OUTPUT_COLS):
        ax = axes[idx]
        y_test = predictions[target]['y_test']
        y_pred = predictions[target]['y_pred']

        ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue')

        # 对角线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='完美预测')

        # 添加 R2 信息
        r2 = df_res[df_res['target'] == target]['R2'].values[0]
        ax.text(0.05, 0.95, f'R²={r2:.4f}', transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(target, fontsize=9)
        ax.legend(fontsize=6, loc='lower right')
        ax.grid(alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(OUTPUT_COLS), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('TabPFN 真实值 vs 预测值', fontsize=14, y=1.02)
    plt.tight_layout()
    scatter_path = exp_dir / 'scatter_true_vs_pred.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"散点图已保存至: {scatter_path}")


if __name__ == '__main__':
    main()
