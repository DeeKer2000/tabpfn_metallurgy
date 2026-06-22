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
DATA_PATH   = r'data\3000固定温度\ml_dataset.csv'
MODEL_PATH  = r'TabPFN-main\models\tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt'
OUTPUT_PATH = r'data\3000固定温度\evaluation_results.csv'
```

### 训练参数

```python
TEST_SIZE    = 0.2      # 测试集比例
RANDOM_STATE = 42       # 随机种子
BATCH_SIZE   = 1000     # 预测分批大小（数据量大时防OOM）
```

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

## tabpfn-extensions 可解释性分析

`tabpfn-extensions` 提供 SHAP、PDP、特征选择等分析工具，已安装在 `tabpfn` 环境中。

### 安装

```bash
pip install tabpfn-extensions -i https://pypi.tuna.tsinghua.edu.cn/simple
# SHAP 绑图需要额外安装
pip install shap -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1. SHAP 特征重要性分析

```python
from tabpfn_extensions import TabPFNRegressor
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq, shapiq_to_shap_explanation
import shap

# 训练模型（启用 KV 缓存加速 SHAP 计算）
reg = TabPFNRegressor(device='cpu', model_path=MODEL_PATH, fit_mode='fit_with_cache')
reg.fit(X_train, y_train)

# 创建 SHAP 解释器
explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
    model=reg, data=X_train, index='SV', max_order=1
)

# 计算 SHAP 值（budget=2^d，d 为特征数）
explanation = shapiq_to_shap_explanation(
    explainer, X_test[:30], budget=256, feature_names=feature_names
)

# 可视化
shap.summary_plot(explanation)        # 蜂群图
shap.plots.bar(explanation)           # 特征重要性条形图
shap.plots.waterfall(explanation[0])  # 单样本瀑布图
```

### 2. PDP 部分依赖图

```python
from tabpfn_extensions.interpretability.pdp import partial_dependence_plots

disp = partial_dependence_plots(
    estimator=reg, X=X_test,
    features=[0, 1, 2, (0, 3)],  # 单特征 + 交互特征对
    grid_resolution=30, kind='average'
)
disp.figure_.suptitle('Partial Dependence')
plt.savefig('pdp_plot.png')
```

### 3. 特征选择

```python
from tabpfn_extensions import interpretability

result = interpretability.feature_selection.feature_selection(
    estimator=reg, X=X, y=y,
    n_features_to_select=5,
    feature_names=feature_names
)
print(f'选择的特征: {result.selected_names}')
print(f'CV得分: {result.baseline_score_mean:.4f} -> {result.selected_score_mean:.4f}')
```

### 4. SHAPIQ 高阶交互分析

```python
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
    model=reg, data=X_train, index='SV', max_order=2  # 2阶交互
)
sv = explainer.explain(X_test[0], budget=256)
sv.plot_graph()  # 特征交互图
```

详细示例参见 `tabpfn-extensions-main/examples/interpretability/`
