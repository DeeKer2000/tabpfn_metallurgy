"""
TabPFN 环境验证脚本
用法: conda run -n tabpfn python test_tabpfn.py
"""

import os
os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_LelHYWBZTv7hyvkS0GfY_H8oC4BMwsQ-VTUcP01sAVM'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from tabpfn import TabPFNRegressor
import numpy as np

MODEL_PATH = r'TabPFN-main\models\tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt'

print(f"模型路径: {MODEL_PATH}")
print(f"文件存在: {os.path.exists(MODEL_PATH)}")

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1.0, 2.0, 3.0, 4.0])

reg = TabPFNRegressor(device='cuda', model_path=MODEL_PATH)
reg.fit(X, y)
pred = reg.predict(X)

print(f"真实值: {y}")
print(f"预测值: {pred}")
print(f"TabPFN 环境验证通过!")
