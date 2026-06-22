# TabPFN 配矿数据回归预测

## 环境准备

```bash
# 激活已安装 TabPFN 的 conda 环境
conda activate tabpfn
```

依赖: `tabpfn`, `numpy`, `pandas`, `scikit-learn`

## 文件说明

| 文件 | 用途 |
|------|------|
| `test_tabpfn.py` | 验证 TabPFN 环境和模型加载是否正常 |
| `train_tabpfn.py` | 训练 + 评估多输出回归模型 |
| `data/3000固定温度/ml_dataset.csv` | 训练数据 (由 `convert_results_to_csv.py` 生成) |
| `TabPFN-main/models/*.ckpt` | 预训练模型权重 |

## 快速开始

### 1. 验证环境

```bash
conda run -n tabpfn python test_tabpfn.py
```

输出 `TabPFN 环境验证通过!` 即正常。

### 2. 训练与评估

```bash
conda run -n tabpfn python train_tabpfn.py
```

训练完成后输出各目标变量的 R2、MAE、RMSE 评估指标，结果保存至 `data/3000固定温度/evaluation_results.csv`。

## 参数配置

所有可调参数在 `train_tabpfn.py` 顶部的 **参数配置区**：

### 设备

```python
DEVICE = 'cpu'   # 'cpu' 或 'cuda'
```

- `cpu`: 通用，3000 样本约需几分钟
- `cuda`: 需要 PyTorch CUDA 版本，速度更快

### 数据路径

```python
DATA_PATH      = r'data\3000固定温度\ml_dataset.csv'
MODEL_PATH     = r'TabPFN-main\models\tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt'
RESULTS_DIR    = r'results'               # 实验结果总目录
EXPERIMENT_NAME = '3000固定温度'           # 实验名称
```

### 训练参数

```python
TEST_SIZE    = 0.2      # 测试集比例
RANDOM_STATE = 42       # 随机种子
BATCH_SIZE   = 1000     # 预测分批大小（数据量大时防OOM）
```

## 实验结果目录结构

每次训练会自动创建带时间戳的实验目录，所有产物保存在其中：

```
results/
└── 20260622_1430_3000固定温度/
    ├── evaluation_results.csv      # 评估指标
    ├── saved_models/               # 29个训练好的模型文件
    │   ├── matte_Cu_pct.tabpfn_fit
    │   ├── matte_Fe_pct.tabpfn_fit
    │   └── ...
    └── shap_results/               # SHAP 分析图表（运行 shap_analysis.py 后生成）
        ├── matte_Cu_pct_summary.png
        ├── matte_Cu_pct_bar.png
        └── matte_Cu_pct_waterfall.png
```

每次实验互不覆盖，方便对比不同数据集或参数的训练结果。

### 输入/输出列

- `INPUT_COLS`: 16 个输入特征 (T, P, Cu, S, Fe, SiO2, ...)
- `OUTPUT_COLS`: 29 个预测目标 (冰铜成分 + 渣成分 + 气相成分 + 各相质量)

可通过修改 `OUTPUT_COLS` 来选择只预测部分目标变量。

## 数据格式

`ml_dataset.csv` 由 `data/3000固定温度/convert_results_to_csv.py` 从 FactSage 批量结果转换而来。如需重新生成：

```bash
cd data/3000固定温度
python convert_results_to_csv.py
```

## 注意事项

- TabPFN **不需要**特征缩放或标准化，直接使用原始数值即可
- 预测大数据集时建议设置 `BATCH_SIZE` 分批预测
- 模型首次加载会验证 license，需要网络连接（国内建议设置 `HF_ENDPOINT` 镜像）

---

## SHAP 特征重要性分析

训练完成后，使用 `shap_analysis.py` 对训练好的模型进行 SHAP 可解释性分析。

### 分析流程

**第 1 步：确认训练已完成**

训练结果保存在 `results/` 下，包含 `saved_models/` 目录：

```
results/20260622_2247_3000固定温度/
├── evaluation_results.csv
└── saved_models/          ← SHAP 分析需要这些模型文件
    ├── matte_Cu_pct.tabpfn_fit
    ├── matte_Fe_pct.tabpfn_fit
    └── ...
```

**第 2 步：配置分析参数**

在 `shap_analysis.py` 顶部修改：

```python
# --- 分析设置 ---
TARGET = 'matte_Cu_pct'  # 要分析的目标变量（必须是已训练过的目标）
N_EXPLAIN = 50            # 解释的样本数（越多越慢，建议 30~100）
BUDGET = 256              # SHAP 精度预算（越大越精确越慢）
```

常用目标变量：
- 冰铜：`matte_Cu_pct`, `matte_Fe_pct`, `matte_S_pct`
- 渣：`slag_Cu_pct`, `slag_FeO_pct`, `slag_FeO_SiO2_ratio`
- 气相：`gas_SO2_mol`, `gas_S2_mol`

**第 3 步：运行分析**

```bash
conda run -n tabpfn python shap_analysis.py
```

程序会自动找到 `results/` 下最新的实验目录，加载对应模型，生成分析结果。

**第 4 步：查看结果**

输出位置在对应实验目录的 `shap_results/` 下：

```
results/20260622_2247_3000固定温度/shap_results/
├── matte_Cu_pct_summary.png      ← 蜂群图：所有样本的特征影响分布
├── matte_Cu_pct_bar.png          ← 条形图：特征重要性排名
└── matte_Cu_pct_waterfall.png    ← 瀑布图：单个样本的预测分解
```

### 输出图表说明

| 图表 | 含义 |
|------|------|
| `summary.png` | 蜂群图，展示每个特征对所有样本的 SHAP 值分布。点越分散说明该特征影响越大，颜色表示特征值高低 |
| `bar.png` | 特征重要性条形图，按 mean\|SHAP\| 排序，直观看哪些特征最重要 |
| `waterfall.png` | 单样本瀑布图，展示该样本预测值如何从基准值一步步被各特征推向最终预测 |

### 对多个目标重复分析

修改 `TARGET` 后重新运行即可，结果文件会以目标变量名区分：

```python
TARGET = 'slag_Cu_pct'   # 改为渣含铜
```

```bash
conda run -n tabpfn python shap_analysis.py
```
