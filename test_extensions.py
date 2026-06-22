import tabpfn_extensions
print(f'tabpfn-extensions: {tabpfn_extensions.__version__}')
from tabpfn_extensions.interpretability import shap, shapiq, pdp, feature_selection
print('SHAP, SHAPIQ, PDP, FeatureSelection 全部导入成功')
