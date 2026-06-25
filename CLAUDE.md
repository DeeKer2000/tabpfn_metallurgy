# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

铜熔炼动态闭环智能配矿系统。使用 TabPFN（Nature 2025 预训练表格基础模型）对 FactSage 热力学计算结果进行多输出回归预测：输入 16 个配料参数，预测 29 个冶金输出变量（冰铜成分、渣成分、气相成分及各相质量）。同时使用 SHAP 进行特征重要性分析，使用 NSGA-II 进行配矿优化。

TabPFN 的 `fit()` 不做梯度下降，仅将训练数据编码进 Transformer 上下文（in-context learning），因此训练极快（秒级到分钟级）。

## Environment

Conda 环境名 `tabpfn`，Python 3.12 + CUDA 11.8。

```bash
conda activate tabpfn
```

关键依赖：tabpfn 8.0.8、tabpfn-extensions 0.4.2（从本地 `tabpfn-extensions-main/` 源码安装）、shap 0.52.0、PyTorch 2.7.1+cu118、pymoo。

TabPFN v3 是 HuggingFace 门控模型，脚本中已配置 `TABPFN_TOKEN` 和 `HF_ENDPOINT=https://hf-mirror.com`（国内镜像）。

## Directory Structure

```
├── scripts/                          # 所有可执行脚本
│   ├── data_processing/
│   │   ├── generate_samples.py       # 合成炉料样本生成
│   │   └── convert_factsage_to_csv.py # FactSage结果转CSV
│   ├── training/
│   │   └── train_tabpfn.py           # TabPFN模型训练
│   ├── analysis/
│   │   ├── shap_analysis.py          # SHAP特征重要性
│   │   └── compare_baselines.py      # 基线算法对比
│   └── optimization/
│       ├── nsga2_optimization.py     # NSGA-II配矿优化
│       └── plot_optimization.py      # 优化结果图表生成
├── data/                             # 源数据（不随实验变化）
│   ├── 矿源.xlsx                     # 11种矿料化学组成
│   └── 3000_fixed_temp/              # 数据集
│       ├── factsage_inputs.txt       # FactSage输入（GBK编码）
│       ├── factsage_results/         # FactSage原始计算结果
│       └── ml_dataset.csv            # ML训练数据集
├── experiments/                      # 所有实验结果（按数据集分组）
│   └── 3000_fixed_temp/
│       ├── tabpfn/                   # TabPFN训练结果
│       │   └── {timestamp}/
│       │       ├── evaluation_results.csv
│       │       ├── saved_models/     # 29个 .tabpfn_fit 模型
│       │       └── shap_results/     # SHAP分析产物
│       ├── baselines/                # 基线算法对比
│       │   └── {timestamp}/
│       └── optimization/             # 配矿优化结果
│           ├── optimization_results.csv
│           ├── optimization_pareto_all.csv
│           └── figures/
├── 文章/                             # 论文材料
│   ├── 02_论文正文.md
│   └── ...
├── TabPFN-main/                      # TabPFN源码+权重
└── tabpfn-extensions-main/           # 扩展库源码
```

## Common Commands

```bash
# 生成合成炉料样本
conda run -n tabpfn python scripts/data_processing/generate_samples.py

# FactSage 结果转 CSV（原始数据 → ml_dataset.csv）
conda run -n tabpfn python scripts/data_processing/convert_factsage_to_csv.py

# 训练 TabPFN 模型
conda run -n tabpfn python scripts/training/train_tabpfn.py

# SHAP 特征重要性分析
conda run -n tabpfn python scripts/analysis/shap_analysis.py

# Baseline 算法对比（RF / XGBoost / GB / MLP）
conda run -n tabpfn python scripts/analysis/compare_baselines.py

# 配矿优化（NSGA-II）
conda run -n tabpfn --env PYTHONIOENCODING=utf-8 python scripts/optimization/nsga2_optimization.py

# 从优化结果生成图表
conda run -n tabpfn python scripts/optimization/plot_optimization.py
```

所有命令从项目根目录运行。

## Data Pipeline

```
data/矿源.xlsx (11种矿料化学组成)
    ↓ scripts/data_processing/generate_samples.py
data/合成炉料样本.txt (N组配料方案, GBK编码)
    ↓ FactSage 8.3 批量计算
data/3000_fixed_temp/factsage_results/result_*.txt (热力学计算原始结果)
    ↓ scripts/data_processing/convert_factsage_to_csv.py
data/3000_fixed_temp/ml_dataset.csv (ML训练数据)
    ↓ scripts/training/train_tabpfn.py
experiments/3000_fixed_temp/tabpfn/{timestamp}/ (模型 + 评估指标)
    ↓ scripts/analysis/shap_analysis.py
experiments/3000_fixed_temp/tabpfn/{timestamp}/shap_results/ (SHAP图表)
    ↓ scripts/optimization/nsga2_optimization.py
experiments/3000_fixed_temp/optimization/ (优化结果 + 图表)
```

## Key Configuration

所有脚本的可调参数集中在文件顶部的**参数配置区**，无需修改下方的代码逻辑。

| 参数 | 位置 | 说明 |
|------|------|------|
| `DATA_PATH` | 各脚本顶部 | 数据文件路径，改此值即切换数据集 |
| `MODEL_PATH` | `train_tabpfn.py` | TabPFN 预训练权重路径 |
| `RESULTS_DIR` | 各脚本顶部 | 实验结果输出目录 |
| `DEVICE` | 各脚本顶部 | `'cuda'` 或 `'cpu'` |
| `TARGETS` | `shap_analysis.py` / `compare_baselines.py` | `None` = 全部 29 个目标，或指定列表 |
| `OUTPUT_COLS` | `train_tabpfn.py` | 29 个预测目标列名定义，所有脚本共享同一份 |

## Input/Output Schema

**16 个输入特征**：`T(℃)`, `P(atm)`, `Cu(g)`, `S(g)`, `Fe(g)`, `SiO2(g)`, `Pb(g)`, `Zn(g)`, `MgO(g)`, `Al2O3(g)`, `Ni(g)`, `Sb(g)`, `Bi(g)`, `CaO(g)`, `As(g)`, `O2(g)`

**29 个输出目标**：
- 冰铜 (wt.%): `matte_Cu_pct`, `matte_Fe_pct`, `matte_S_pct`, `matte_Ni_pct`, `matte_Zn_pct`, `matte_Pb_pct`, `matte_As_pct`, `matte_mass_g`
- 渣 (wt.%): `slag_Cu_pct`, `slag_FeO_pct`, `slag_SiO2_pct`, `slag_FeO_SiO2_ratio`, `slag_Al2O3_pct`, `slag_CaO_pct`, `slag_MgO_pct`, `slag_ZnO_pct`, `slag_mass_g`
- 气相 (mol分数): `gas_SO2_mol`, `gas_S2_mol`, `gas_SO_mol`, `gas_SSO_mol`, `gas_SO3_mol`, `gas_Zn_mol`, `gas_PbS_mol`, `gas_CuS_mol`, `gas_Pb_mol`, `gas_Sb_mol`, `gas_As_mol`, `gas_mass_g`

注意：渣中 Cu 由 Cu2O 转换为 Cu 元素 wt.%（×127.09/143.09）；FeO/SiO₂ 是氧化物质量比；气相成分是摩尔分数。

## Architecture Notes

- `scripts/training/train_tabpfn.py`：逐目标变量训练（29 个独立模型），每个目标单独 fit + predict + 保存。使用 `predict_batched()` 分批预测防 OOM。
- `scripts/analysis/shap_analysis.py`：使用 `tabpfn_extensions.interpretability.shapiq` 计算 Shapley 值，再转换为 SHAP explanation 对象。支持从已保存的 `shap_importance.csv` 直接生成热力图（`plot_heatmap_from_csv()`）。
- `scripts/analysis/compare_baselines.py`：对比 TabPFN 与 RF/GB/MLP/XGBoost，加载已训练的 TabPFN 模型与新训练的 baseline 模型。生成 Nature 风格柱状图（NMI pastel 配色）。
- `scripts/data_processing/generate_samples.py`：基于矿源数据生成合成炉料方案，支持冶金成分约束（Cu/S/Fe/SiO₂/Fe-SiO₂ 比范围）。
- `scripts/data_processing/convert_factsage_to_csv.py`：解析 FactSage 原始文本输出（正则匹配相名和组分），转换为结构化 CSV。
- `scripts/optimization/nsga2_optimization.py`：NSGA-II 多目标优化，12D 决策变量（11矿料配比 + SiO₂熔剂），目标 min 渣含Cu，约束 Fe/SiO₂ ∈ [1.15, 1.30]、冰铜品位 ∈ [58, 60]。

## Gotchas

- 输入数据文件是 **GBK 编码**（`factsage_inputs.txt`），CSV 输出是 **utf-8-sig**
- TabPFN 不需要特征缩放/标准化，直接使用原始数值
- `BATCH_SIZE` 控制预测分批大小，数据量大时需调小防 OOM
- `BUDGET` 参数控制 SHAP 精度：16 个特征用 65536 为精确计算，256 为近似
- tabpfn-extensions 必须从本地源码安装（PyPI 版本锁死了 tabpfn<8）
- 升级 Python 后所有含 C 扩展的包需要 `pip install --force-reinstall`
- `experiments/` 和 `data/3000_fixed_temp/factsage_results/` 在 .gitignore 中，不提交到仓库
- Windows 上 `conda run` 需要 `--env PYTHONIOENCODING=utf-8` 避免 GBK 编码错误

## Paper / 文章

`文章/` 目录存放论文相关文件，`02_论文正文.md` 是论文正文。不要修改 `文章/` 下的文件除非明确要求。
