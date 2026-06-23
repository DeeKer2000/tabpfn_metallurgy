# 项目交接文档

> 动态闭环智能配矿 — TabPFN 回归预测与 SHAP 可解释性分析

## 1. 项目概述

本项目使用 TabPFN（Tabular Prior-Data Fitted Networks，Nature 2025）对铜熔炼 FactSage 热力学计算结果进行多输出回归预测。输入 16 个配料参数，预测 29 个输出变量（冰铜成分、渣成分、气相成分及各相质量）。同时使用 SHAP 进行特征重要性分析，揭示各输入变量对输出的影响。

**核心特点：** TabPFN 是预训练基础模型，fit() 不做梯度下降，仅将数据编码进 Transformer 上下文，因此训练极快（几秒到几分钟）。

## 2. 目录结构

```
动态闭环智能配矿/
├── train_tabpfn.py              # 训练脚本（主要入口）
├── shap_analysis.py             # SHAP 可解释性分析脚本
├── compare_baselines.py         # Baseline 算法对比脚本
├── test_tabpfn.py               # TabPFN 环境验证
├── test_extensions.py           # tabpfn-extensions 环境验证
├── README_TabPFN.md             # 使用文档
├── HANDOVER.md                  # 本文件（交接文档）
├── .gitignore
│
├── data/
│   └── 3000固定温度/
│       ├── ml_dataset.csv                # ML 训练数据（3000 样本）
│       ├── 合成炉料样本3000组_输入数据.txt  # FactSage 输入（GBK 编码）
│       ├── convert_results_to_csv.py     # FactSage 结果 → CSV 转换脚本
│       └── results/                      # FactSage 原始计算结果（result_*.txt）
│
├── results/                             # 实验结果（按时间戳组织，不提交 git）
│   ├── 20260622_2247_3000固定温度/
│   │   ├── evaluation_results.csv        # 评估指标（R2, MAE, RMSE）
│   │   ├── saved_models/                 # 29 个训练好的模型文件
│   │   └── shap_results/                 # SHAP 分析图表
│   └── 20260623_1007_3000固定温度_baselines/
│       ├── baseline_comparison.csv       # Baseline 对比详细指标
│       ├── algorithm_comparison_bar.png  # 算法对比柱状图
│       └── scatter_true_vs_pred_grid.png # 真实值vs预测值散点图
│
├── TabPFN-main/                         # TabPFN 源码（不提交 git）
│   └── models/                          # 预训练权重文件
│       ├── tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt
│       └── tabpfn-v3-regressor-v3_default.ckpt
│
└── tabpfn-extensions-main/              # tabpfn-extensions 源码（不提交 git）
    └── src/tabpfn_extensions/           # SHAP、PDP、特征选择等扩展功能
```

## 3. 环境配置

### 3.1 Conda 环境

环境名称：`tabpfn`，Python 3.12，使用 CUDA。

```bash
# 激活环境
conda activate tabpfn

# 验证环境
python test_tabpfn.py
python test_extensions.py
```

### 3.2 关键依赖版本

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.13 | 从 3.9 升级，需重装所有 C 扩展包 |
| PyTorch | 2.7.1+cu118 | CUDA 11.8 版本 |
| tabpfn | 8.0.8 | 支持 v3 架构 |
| tabpfn-extensions | 0.4.2 | 从本地目录安装（非 PyPI） |
| shap | 0.52.0 | SHAP 可视化 |
| shapiq | 1.5.2 | Shapley 值计算 |
| numpy | 2.4.6 | <2.5 以兼容 numba |
| pandas | 2.3.3 | <3 以兼容 tabpfn-common-utils |
| scikit-learn | 1.9.0 | |

### 3.3 环境安装注意事项

升级 Python 3.9 → 3.12 后，所有含 C 扩展的包都需要重装：

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --force-reinstall numpy "numpy<2.5" pandas "pandas<3" scikit-learn scipy matplotlib shap pydantic pydantic-core
```

`tabpfn-extensions` 必须从本地源码安装（PyPI 版本 0.3.0 锁死了 tabpfn<8）：

```bash
pip install -e "tabpfn-extensions-main[interpretability]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.4 HuggingFace 认证

TabPFN v3 是 HuggingFace 门控模型，需要：

1. 在 https://huggingface.co/PriorLabs/TabPFN-v3 接受 license
2. 获取 API key（脚本中已配置 `TABPFN_TOKEN`）
3. 国内需要设置 `HF_ENDPOINT=https://hf-mirror.com`（脚本中已配置）

## 4. 数据流水线

### 4.1 原始数据

- **输入数据：** `data/3000固定温度/合成炉料样本3000组_输入数据.txt`（GBK 编码）
- **计算结果：** `data/3000固定温度/results/result_*.txt`（FactSage 批量计算输出）

### 4.2 数据转换

```bash
cd data/3000固定温度
python convert_results_to_csv.py
```

生成 `ml_dataset.csv`，格式：

| 列 | 数量 | 说明 |
|---|---|---|
| 输入特征 | 16 | T(℃), P(atm), Cu(g), S(g), Fe(g), SiO2(g), Pb(g), Zn(g), MgO(g), Al2O3(g), Ni(g), Sb(g), Bi(g), CaO(g), As(g), O2(g) |
| 输出目标 | 29 | 冰铜 8 列 + 渣 9 列 + 气相 12 列 |

**注意：**
- 输入文件是 GBK 编码，转换脚本中已处理
- 渣中 Cu 由 Cu2O 转换为 Cu 元素 wt.%
- FeO/SiO2 是质量比，不是元素比
- 气相成分是摩尔分数

## 5. 训练流程

### 5.1 运行训练

```bash
conda run -n tabpfn python train_tabpfn.py
```

### 5.2 关键参数（train_tabpfn.py 顶部）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DATA_PATH` | `data\3000固定温度\ml_dataset.csv` | 数据文件路径 |
| `MODEL_PATH` | `TabPFN-main\models\tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt` | 预训练权重 |
| `RESULTS_DIR` | `results` | 实验结果总目录 |
| `EXPERIMENT_NAME` | 自动从 DATA_PATH 推导 | 实验名称（出现在文件夹名中） |
| `DEVICE` | `cuda` | 计算设备 |
| `TEST_SIZE` | `0.2` | 测试集比例 |
| `RANDOM_STATE` | `42` | 随机种子（保证可复现） |
| `BATCH_SIZE` | `1000` | 分批预测大小（防 OOM） |

### 5.3 训练原理

TabPFN 是预训练 Transformer 基础模型。`fit()` 不做梯度下降，仅将训练数据编码进模型上下文（in-context learning）。预测时通过注意力机制一次性输出结果，类似 GPT 根据 prompt 生成文本。因此训练极快，真正的耗时在模型加载和 GPU 推理。

### 5.4 输出

每次训练创建带时间戳的实验目录：

```
results/20260622_2247_3000固定温度/
├── evaluation_results.csv      # 29 个目标的 R2, MAE, RMSE
└── saved_models/               # 29 个 .tabpfn_fit 模型文件
```

### 5.5 换数据集

只需修改 `DATA_PATH`，`EXPERIMENT_NAME` 会自动从路径推导：

```python
DATA_PATH = r'data\1400固定温度\ml_dataset.csv'
# → 实验名自动变为 "1400固定温度"
```

## 6. SHAP 分析流程

### 6.1 运行分析

```bash
conda run -n tabpfn python shap_analysis.py
```

### 6.2 关键参数（shap_analysis.py 顶部）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `TARGETS` | `None` | None = 分析全部 29 个目标，或指定列表如 `['matte_Cu_pct', 'slag_Cu_pct']` |
| `N_EXPLAIN` | `50` | 解释的样本数（越多越慢） |
| `BUDGET` | `256` | SHAP 精度预算（16 个特征用 65536 精确，256 近似） |
| `DEVICE` | `cuda` | 计算设备 |

### 6.3 输出

结果保存在对应实验目录的 `shap_results/` 下：

```
results/20260622_2247_3000固定温度/shap_results/
├── matte_Cu_pct_summary.png      # 蜂群图：特征影响分布
├── matte_Cu_pct_bar.png          # 条形图：特征重要性排名
├── matte_Cu_pct_waterfall.png    # 瀑布图：单样本预测分解
├── matte_Fe_pct_summary.png
├── ...
```

### 6.4 图表含义

| 图表 | 含义 |
|---|---|
| `summary.png` | 蜂群图，每个点是一个样本的一个特征。点越分散说明该特征影响越大，颜色表示特征值高低 |
| `bar.png` | 特征重要性排名（mean\|SHAP\|），直观看哪些特征最重要 |
| `waterfall.png` | 单样本瀑布图，展示预测值如何从基准值被各特征推向最终预测 |

## 7. Baseline 算法对比

### 7.1 运行对比实验

```bash
conda run -n tabpfn python compare_baselines.py
```

### 7.2 对比算法

| 算法 | 说明 |
|---|---|
| Random Forest | 随机森林，100棵树 |
| Gradient Boosting | 梯度提升树，100棵树 |
| MLP | 多层感知机，(128, 64, 32) 结构 |
| XGBoost | 极端梯度提升（需安装：`pip install xgboost`） |

### 7.3 输出

每次运行创建带时间戳的实验目录：

```
results/20260623_1007_3000固定温度_baselines/
├── baseline_comparison.csv       # 29个目标×4个算法的详细指标
├── algorithm_comparison_bar.png  # 算法对比柱状图（R2, MAE, RMSE）
└── scatter_true_vs_pred_grid.png # 真实值vs预测值散点图（29个目标）
```

### 7.4 实验结果摘要

基于 3000 固定温度数据集（2400训练/600测试）：

| 算法 | 平均 R2 | 平均 MAE | 平均 RMSE |
|---|---|---|---|
| Random Forest | 0.8373 | 0.2429 | 0.3774 |
| XGBoost | 0.7663 | 0.2182 | 0.3399 |
| Gradient Boosting | 0.6792 | 0.2831 | 0.4164 |
| MLP | 不稳定 | 0.1639 | 0.2633 |

**注意：** MLP 在部分目标上出现负 R2（如气相微量成分），可能是因为：
- 数据标准化后，小数值目标的预测偏差被放大
- 某些目标变量值域极小（如 gas_Pb_mol ~1e-8），MLP 难以学习

**结论：** Random Forest 和 XGBoost 表现最稳定，可作为 TabPFN 的 baseline 对比。

## 8. 版本历史

| 日期 | 事件 |
|---|---|
| 2026-06-22 | 初始版本，10 样本 demo 数据，完成数据转换脚本 |
| 2026-06-22 | 完成 TabPFN 训练脚本、模型保存、SHAP 分析脚本 |
| 2026-06-22 | 升级 Python 3.9→3.12，TabPFN 7.1.1→8.0.8，支持 CUDA |

## 8. 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `KeyError: 'tabpfn_v3'` | TabPFN v7.1.1 不支持 v3 架构 | 升级到 v8.0+ |
| `tabpfn-extensions` 要求 `tabpfn<8` | PyPI 版本 0.3.0 依赖过旧 | 从本地源码安装 0.4.2 |
| Python 3.9 下无法安装 extensions | extensions 要求 Python>=3.10 | 升级 Python 到 3.12 |
| 升级 Python 后包报错 | C 扩展包需要重编译 | `pip install --force-reinstall` 所有含 C 扩展的包 |
| HuggingFace 下载失败 | 国内网络 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| `cannot import name 'save_fitted_tabpfn_model' from 'tabpfn.utils'` | v8.0 改了导入路径 | `from tabpfn import save_fitted_tabpfn_model` |

## 9. 后续开发建议

1. **✅ 对比其他算法：** 已完成！使用 `compare_baselines.py` 对比了 XGBoost、Random Forest、MLP、GradientBoosting
2. **超参数调优：** 调整 `N_EXPLAIN`、`BUDGET` 提高 SHAP 分析精度
3. **PDP 分析：** 使用 `tabpfn_extensions.interpretability.pdp` 生成部分依赖图
4. **特征选择：** 使用 `tabpfn_extensions.interpretability.feature_selection` 进行自动特征选择
5. **高阶交互：** 将 SHAP 的 `max_order` 改为 2，分析特征间交互效应
6. **新数据集：** 将新的 FactSage 结果放入 `data/` 目录，运行 `convert_results_to_csv.py` 生成 CSV，修改 `DATA_PATH` 即可训练
7. **模型部署：** 训练好的模型可以用 `load_fitted_tabpfn_model` 加载后直接调用 `predict()` 做推理
8. **补充验证图：** 根据 nature-figure skill 建议，可补充残差分析图、特征重要性热力图等

## 10. 联系与参考

- TabPFN 官方仓库：https://github.com/PriorLabs/TabPFN
- tabpfn-extensions 仓库：https://github.com/PriorLabs/tabpfn-extensions
- TabPFN 论文：Hollmann et al., "Accurate predictions on small data with a tabular foundation model", Nature 2025
