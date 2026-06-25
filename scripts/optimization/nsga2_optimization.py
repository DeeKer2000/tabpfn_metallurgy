"""
配矿优化（NSGA-II 单目标）
用法: conda run -n tabpfn --env PYTHONIOENCODING=utf-8 python scripts/optimization/nsga2_optimization.py

优化目标：
  1. 最小化 slag_Cu_pct（渣含铜损失）
约束：
  - 渣FeO/SiO₂ ∈ [1.15, 1.30]
  - 冰铜品位 matte_Cu ∈ [58, 60] wt%

输出：experiments/3000_fixed_temp/optimization/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TABPFN_TOKEN'] = 'tabpfn_sk_LelHYWBZTv7hyvkS0GfY_H8oC4BMwsQ-VTUcP01sAVM'

# ============================================================
#                    路径配置
# ============================================================

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / 'data' / '3000_fixed_temp' / 'ml_dataset.csv'
MODELS_DIR = PROJECT_ROOT / 'experiments' / '3000_fixed_temp' / 'tabpfn' / '20260622_2247' / 'saved_models'
OPT_DIR = PROJECT_ROOT / 'experiments' / '3000_fixed_temp' / 'optimization'
OUTPUT_DIR = OPT_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda'

# ============================================================
#                    矿料数据
# ============================================================

# 11种矿料的化学组成（干基，wt%）
# 数据来源：data/矿源.xlsx
# 列含义：Cu(铜), S(硫), Fe(铁), SiO2(二氧化硅), Pb(铅), Zn(锌),
#         MgO(氧化镁), Al2O3(氧化铝), CaO(氧化钙),
#         Ni(镍), Sb(锑), Bi(铋), As(砷)
ORE_COMPOSITION = pd.DataFrame({
    'name':    ['炎鑫', '汇祥永金', '嘉能可ET', '金源矿冶', '阿舍勒', '路源',
                '外矿BOZ', '外矿AKT', '和鑫', '渣精矿', '白银'],
    'Cu':      [20.57, 23.12, 16.07, 22.94, 18.99, 21.29, 21.30, 23.60, 21.60, 17.40, 17.49],
    'S':       [22.45,  8.22, 14.95, 22.23, 43.45, 24.91, 28.96, 29.93, 30.45,  6.42, 25.75],
    'Fe':      [24.28,  7.60, 13.10, 24.39, 27.10, 19.61, 25.92, 26.10, 33.18, 34.49, 21.48],
    'SiO2':    [ 9.25, 34.82, 29.01, 11.02,  2.35, 15.33,  5.63,  4.80,  3.41, 14.19, 16.75],
    'Pb':      [ 0.05,  0.01,  0.04,  0.00,  1.04,  0.85,  0.07,  0.00,  0.05,  1.10,  0.03],
    'Zn':      [ 0.27,  0.07,  0.01,  0.53,  3.73,  4.67,  0.64,  0.26,  0.17,  1.21,  0.12],
    'MgO':     [ 0.50,  1.00,  1.22,  6.36,  0.14,  2.25,  0.49,  0.23,  1.62,  0.34,  1.19],
    'Al2O3':   [ 2.17,  7.29,  5.07,  0.47,  0.70,  1.55,  2.20,  2.11,  0.43,  0.82,  4.06],
    'CaO':     [ 0.92,  8.17,  0.60,  0.23,  0.18,  0.70,  0.52,  0.44,  0.02,  0.46,  6.09],
    'Ni':      [ 0.01,  0.02,  0.02,  0.77,  0.04,  0.00,  0.02,  0.01,  0.95,  0.01,  0.01],
    'Sb':      [ 0.10,  0.42,  0.18,  0.08,  0.05,  0.10,  0.03,  0.05,  0.05,  0.01,  0.04],
    'Bi':      [ 0.01,  0.02,  0.01,  0.01,  0.05,  0.01,  0.01,  0.01,  0.01,  0.01,  0.02],
    'As':      [ 0.01,  0.01,  0.01,  0.01,  1.45,  0.00,  0.01,  0.01,  0.01,  0.03,  0.03],
})

N_ORES = len(ORE_COMPOSITION)

# 模型输入列（与 train_tabpfn.py 一致）
# 含义：T(温度℃), P(压力atm), Cu/S/Fe/SiO2/Pb/Zn/MgO/Al2O3/Ni/Sb/Bi/CaO/As(各元素质量g), O2(氧气g)
INPUT_COLS = [
    'T(℃)', 'P(atm)', 'Cu(g)', 'S(g)', 'Fe(g)', 'SiO2(g)',
    'Pb(g)', 'Zn(g)', 'MgO(g)', 'Al2O3(g)', 'Ni(g)', 'Sb(g)',
    'Bi(g)', 'CaO(g)', 'As(g)', 'O2(g)',
]

# 固定工艺参数（所有优化方案共用）
T_DEFAULT = 1200.0   # 熔炼温度，单位：℃
P_DEFAULT = 1.0      # 炉内压力，单位：atm（标准大气压）
O2_DEFAULT = 21.0    # 氧气供给量，单位：g（对应空气气氛）

# 总投料量（克），用于将 wt% 转换为克
TOTAL_CHARGE = 100.0

# ============================================================
#                    优化参数
# ============================================================

POP_SIZE = 100       # 种群大小（每代个体数）
N_GEN = 200          # 最大迭代代数（遗传算法进化轮数上限）
SEED = 42            # 随机种子（保证结果可复现）
PATIENCE = 30        # 早停止耐心值：连续 PATIENCE 代无改善则提前终止

# ---- 目标函数 ----
# 仅一个目标：最小化渣含铜（slag_Cu_pct）
# 渣含铜越低，铜的损失越小，经济效益越高

# ---- 约束条件 ----
# Fe/SiO₂ 约束：渣中FeO与SiO₂的质量比，控制渣的流动性和铜渣分离效率
FESIO2_LO = 1.15     # Fe/SiO₂ 下限（低于此值渣过酸，腐蚀炉衬）
FESIO2_HI = 1.3      # Fe/SiO₂ 上限（高于此值渣过碱，黏度增大）

# 冰铜品位约束：冰铜中Cu的质量百分比
# 过高（>60%）→ 渣中Fe₃O₄升高、渣变黏、不利于铜渣分离
# 过低（<58%）→ 吹炼负荷增大、能耗升高
MATTE_CU_LO = 58.0   # 冰铜品位下限（wt%）
MATTE_CU_HI = 60.0   # 冰铜品位上限（wt%）

# ============================================================
#                    代码
# ============================================================

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination


def ratios_to_features(ratios, sio2_flux_ratio):
    """将矿料配比和SiO₂熔剂比例转换为模型输入特征向量（16维）。

    转换流程：
      1. 用配比对11种矿料的化学组成加权求和，得到混合炉料综合组成（wt%）
      2. 加入SiO₂熔剂，重新归一化
      3. 将wt%转换为克（总投料量100g）
      4. 补充固定工艺参数（温度、压力、氧气），构成16维向量

    Args:
        ratios: shape (11,) 各矿料质量配比，总和为1
        sio2_flux_ratio: SiO₂熔剂添加比例（0~0.15），即每1g矿料额外添加的SiO₂克数

    Returns:
        features: shape (16,) 模型输入特征，顺序与 INPUT_COLS 一致
    """
    # 提取11种矿料的13种化学组成（Cu, S, Fe, SiO2, Pb, Zn, MgO, Al2O3, CaO, Ni, Sb, Bi, As）
    ore_comp = ORE_COMPOSITION[['Cu', 'S', 'Fe', 'SiO2', 'Pb', 'Zn',
                                 'MgO', 'Al2O3', 'CaO', 'Ni', 'Sb', 'Bi', 'As']].values

    # 步骤1：混合炉料综合组成（wt%），按配比加权平均
    blend_pct = ratios @ ore_comp  # shape (13,)

    # 步骤2：添加 SiO₂ 熔剂（加入到SiO2列，index=3），然后重新归一化
    total_ratio = 1.0 + sio2_flux_ratio
    blend_pct_with_flux = blend_pct.copy()
    blend_pct_with_flux[3] += sio2_flux_ratio * 100  # SiO2列额外加上熔剂质量
    blend_pct_with_flux = blend_pct_with_flux / total_ratio  # 归一化（总质量增加）

    # 步骤3：将wt%转换为克（总投料量100g）
    charge_g = blend_pct_with_flux * TOTAL_CHARGE / 100.0

    # 步骤4：构造16维输入向量 = [温度, 压力, 13种元素质量(g), 氧气量(g)]
    features = np.array([
        T_DEFAULT, P_DEFAULT,
        charge_g[0],  # Cu（铜）质量，g
        charge_g[1],  # S（硫）质量，g
        charge_g[2],  # Fe（铁）质量，g
        charge_g[3],  # SiO2（二氧化硅）质量，g
        charge_g[4],  # Pb（铅）质量，g
        charge_g[5],  # Zn（锌）质量，g
        charge_g[6],  # MgO（氧化镁）质量，g
        charge_g[7],  # Al2O3（氧化铝）质量，g
        charge_g[8],  # CaO（氧化钙）质量，g
        charge_g[9],  # Ni（镍）质量，g
        charge_g[10], # Sb（锑）质量，g
        charge_g[11], # Bi（铋）质量，g
        charge_g[12], # As（砷）质量，g
        O2_DEFAULT,   # O2（氧气）质量，g
    ], dtype=np.float32)

    return features


def check_metallurgical_constraints(ratios, sio2_flux_ratio):
    """检查冶金成分约束（罚函数前置过滤）。

    这些约束确保混合炉料的化学组成在FactSage训练数据的覆盖范围内，
    避免模型外推导致预测不可靠。

    Returns:
        True: 满足所有约束，方案可行
        False: 违反任一约束，方案不可行
    """
    ore_comp = ORE_COMPOSITION[['Cu', 'S', 'Fe', 'SiO2']].values
    blend_pct = ratios @ ore_comp

    total_ratio = 1.0 + sio2_flux_ratio
    blend_pct[3] += sio2_flux_ratio * 100
    blend_pct = blend_pct / total_ratio

    cu, s, fe, sio2 = blend_pct

    # 混合炉料成分约束（wt%范围，与FactSage训练数据覆盖范围一致）
    if not (14 <= cu <= 26):       # Cu（铜）含量
        return False
    if not (8 <= s <= 38):         # S（硫）含量
        return False
    if not (8 <= fe <= 36):        # Fe（铁）含量
        return False
    if not (3 <= sio2 <= 25):      # SiO2（二氧化硅）含量
        return False

    # Fe/SiO₂ 比约束（宽松范围，用于前置过滤）
    if sio2 > 0:
        fesio2 = fe / sio2
        if not (0.8 <= fesio2 <= 4.0):
            return False

    return True


class OreBlendingProblem(Problem):
    """配矿优化问题（单目标 + 约束）。

    决策变量（12维）：
      - x[0:11] = 11种矿料的质量配比（归一化后总和为1）
      - x[11]   = SiO₂熔剂添加比例（0~0.15）

    目标函数（1个）：
      - min slag_Cu_pct（渣含铜含量，wt%）—— 铜损失越小越好

    不等式约束（4个）：
      - g1: 渣Fe/SiO₂ ≥ 1.15（渣不能过酸）
      - g2: 渣Fe/SiO₂ ≤ 1.30（渣不能过碱）
      - g3: 冰铜Cu品位 ≥ 58%（品位不能太低，否则吹炼负荷大）
      - g4: 冰铜Cu品位 ≤ 60%（品位不能太高，否则渣变黏）

    冶金成分约束（Cu/S/Fe/SiO₂范围）通过罚函数在check_metallurgical_constraints中处理。
    """

    def __init__(self, models):
        self.models = models  # dict: target_name -> model（TabPFN代理模型字典）
        n_var = N_ORES + 1  # 12维：11种矿料配比 + 1个SiO₂熔剂比例
        n_obj = 1            # 1个目标：最小化渣含Cu
        n_ieq_constr = 4     # 4个不等式约束：Fe/SiO₂ × 2 + matte_Cu × 2

        # 决策变量边界
        xl = np.zeros(n_var)       # 下界：所有配比 ≥ 0
        xu = np.ones(n_var)        # 上界：配比 ≤ 1
        xu[-1] = 0.15              # SiO₂熔剂比例上界（最多加15%）

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr,
                         xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        # F: 目标函数矩阵，shape (n, 1)
        #   F[i, 0] = slag_Cu_pct（渣含铜），越小越好
        # G: 不等式约束矩阵，shape (n, 4)，>=0 表示满足
        #   G[i, 0] = slag_FeO_SiO2 - 1.15  （Fe/SiO₂ ≥ 下限）
        #   G[i, 1] = 1.3 - slag_FeO_SiO2    （Fe/SiO₂ ≤ 上限）
        #   G[i, 2] = matte_Cu - 58           （冰铜品位 ≥ 下限）
        #   G[i, 3] = 60 - matte_Cu           （冰铜品位 ≤ 上限）
        F = np.full((n, 1), 1e6)
        G = np.full((n, 4), -1.0)

        # 收集有效个体的索引和特征，批量预测
        valid_indices = []
        valid_features = []

        for i in range(n):
            ore_ratios_raw = X[i, :N_ORES]
            sio2_flux = X[i, -1]

            total = ore_ratios_raw.sum()
            if total < 1e-10:
                continue
            ore_ratios = ore_ratios_raw / total

            if not check_metallurgical_constraints(ore_ratios, sio2_flux):
                F[i] = [10.0]
                G[i] = [-1.0, -1.0, -1.0, -1.0]
                continue

            features = ratios_to_features(ore_ratios, sio2_flux)
            valid_indices.append(i)
            valid_features.append(features)

        # 批量预测
        if valid_features:
            X_batch = np.array(valid_features, dtype=np.float32)
            try:
                pred_slag_cu = np.array(
                    self.models['slag_Cu_pct'].predict(X_batch)).flatten()
                pred_matte_cu = np.array(
                    self.models['matte_Cu_pct'].predict(X_batch)).flatten()
                pred_fesio2 = np.array(
                    self.models['slag_FeO_SiO2_ratio'].predict(X_batch)).flatten()

                for idx, i in enumerate(valid_indices):
                    F[i] = [float(pred_slag_cu[idx])]
                    G[i] = [-(float(pred_fesio2[idx]) - FESIO2_LO),
                            -(FESIO2_HI - float(pred_fesio2[idx])),
                            -(float(pred_matte_cu[idx]) - MATTE_CU_LO),
                            -(MATTE_CU_HI - float(pred_matte_cu[idx]))]
            except Exception:
                for i in valid_indices:
                    F[i] = [10.0]
                    G[i] = [-1.0, -1.0, -1.0, -1.0]

        out["F"] = F
        out["G"] = G


def load_models():
    """加载 TabPFN 模型。

    load_fitted_tabpfn_model 会从 init_params.json 读取 model_path（相对路径），
    需要将其替换为绝对路径后重新打包，避免 CWD 依赖。
    """
    import tempfile, zipfile, json, joblib
    from tabpfn import TabPFNRegressor
    from tabpfn.model_loading import _extract_archive, InferenceEngine

    # 预训练权重的绝对路径
    MODEL_PATH_ABS = str(PROJECT_ROOT / 'TabPFN-main' / 'models'
                         / 'tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt')

    model_names = ['slag_Cu_pct', 'matte_Cu_pct', 'slag_FeO_SiO2_ratio']
    models = {}
    for name in model_names:
        path = MODELS_DIR / f'{name}.tabpfn_fit'
        print(f"  Loading model: {name}...")

        # 手动加载，替换 model_path 为绝对路径
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _extract_archive(path, tmp)

            with (tmp / "init_params.json").open() as f:
                params = json.load(f)

            params.pop("__class_name__")
            # 关键：将 model_path 替换为绝对路径
            params["model_path"] = MODEL_PATH_ABS
            params["device"] = DEVICE

            est = TabPFNRegressor(**params)
            est._initialize_model_variables()

            fitted_attrs = joblib.load(tmp / "fitted_attrs.joblib")
            for key, value in fitted_attrs.items():
                setattr(est, key, value)

            est.executor_ = InferenceEngine.load_state(
                tmp / "executor_state.joblib", est.models_
            )
            est.to(DEVICE)

        models[name] = est
        print(f"    OK")

    return models


class EarlyStopping:
    """早停止回调：连续 PATIENCE 代无改善则终止优化。"""

    def __init__(self, patience=PATIENCE):
        self.patience = patience
        self.best = float('inf')
        self.counter = 0

    def __call__(self, algorithm):
        # 获取当前最优目标函数值
        F = algorithm.pop.get("F")
        current_best = float(np.min(F))
        if current_best < self.best - 1e-6:
            self.best = current_best
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"\n  Early stopping: no improvement for {self.patience} generations")
            return True
        return False


def run_optimization(models):
    """运行优化（默认NSGA-II，单/多目标均适用）。"""
    problem = OreBlendingProblem(models)

    # 默认使用 NSGA-II（兼容单目标和多目标）
    # 如需切换为 GA 单目标优化，改为：algorithm = GA(...)
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", N_GEN)

    print(f"\n  Population: {POP_SIZE}")
    print(f"  Max generations: {N_GEN}, early stopping patience: {PATIENCE}")
    print(f"  Decision vars: {N_ORES} ore ratios + SiO2 flux = {N_ORES+1} vars")
    print(f"  Objective: min(slag_Cu_pct)")
    print(f"  Constraints: Fe/SiO2 in [{FESIO2_LO}, {FESIO2_HI}], "
          f"matte_Cu ∈ [{MATTE_CU_LO}, {MATTE_CU_HI}]")
    print(f"\n  Starting optimization...")

    res = minimize(
        problem,
        algorithm,
        termination,
        callback=EarlyStopping(patience=PATIENCE),
        seed=SEED,
        verbose=True,
    )

    return res


def extract_results(res, models):
    """提取优化最优解，并重新预测 matte_Cu 用于展示。"""
    X_opt = np.atleast_2d(res.X)  # 确保是2D: (n_solutions, n_var)
    F_opt = np.atleast_2d(res.F)  # 确保是2D: (n_solutions, 1)

    # 批量预测 matte_Cu 和 Fe/SiO₂ 用于展示
    valid_indices = []
    valid_features = []
    for i in range(X_opt.shape[0]):
        ore_ratios_raw = X_opt[i, :N_ORES]
        sio2_flux = X_opt[i, -1]
        total = ore_ratios_raw.sum()
        if total < 1e-10:
            continue
        ore_ratios = ore_ratios_raw / total
        valid_indices.append(i)
        valid_features.append(ratios_to_features(ore_ratios, sio2_flux))

    pred_matte_cu = {}
    pred_fesio2 = {}
    if valid_features:
        X_batch = np.array(valid_features, dtype=np.float32)
        try:
            mc = np.array(models['matte_Cu_pct'].predict(X_batch)).flatten()
            fs = np.array(models['slag_FeO_SiO2_ratio'].predict(X_batch)).flatten()
            for idx, i in enumerate(valid_indices):
                pred_matte_cu[i] = float(mc[idx])
                pred_fesio2[i] = float(fs[idx])
        except Exception:
            pass

    results = []
    for i in range(X_opt.shape[0]):
        ore_ratios_raw = X_opt[i, :N_ORES]
        sio2_flux = X_opt[i, -1]
        total = ore_ratios_raw.sum()
        if total < 1e-10:
            continue
        ore_ratios = ore_ratios_raw / total

        slag_cu = float(F_opt[i, 0])
        matte_cu = pred_matte_cu.get(i, 0.0)
        fesio2_pred = pred_fesio2.get(i, 0.0)

        # 计算混合炉料成分（用于展示）
        ore_comp = ORE_COMPOSITION[['Cu', 'S', 'Fe', 'SiO2']].values
        blend = ore_ratios @ ore_comp
        total_ratio = 1.0 + sio2_flux
        blend[3] += sio2_flux * 100
        blend = blend / total_ratio

        entry = {
            'solution_idx': i,
            'slag_Cu_pct': slag_cu,
            'matte_Cu_pct': matte_cu,
            'slag_FeO_SiO2_ratio': fesio2_pred,
            'blend_Cu': blend[0],
            'blend_S': blend[1],
            'blend_Fe': blend[2],
            'blend_SiO2': blend[3],
            'SiO2_flux': sio2_flux,
        }

        # 各矿料配比
        for j, name in enumerate(ORE_COMPOSITION['name']):
            entry[f'ratio_{name}'] = ore_ratios[j]

        results.append(entry)

    return pd.DataFrame(results)


def select_representative_solutions(df, n=5):
    """从满足约束的解中选取 n 个代表性方案。"""
    # 过滤掉不满足约束的解
    mask = ((df['slag_FeO_SiO2_ratio'] >= FESIO2_LO) &
            (df['slag_FeO_SiO2_ratio'] <= FESIO2_HI) &
            (df['matte_Cu_pct'] >= MATTE_CU_LO) &
            (df['matte_Cu_pct'] <= MATTE_CU_HI))
    df_feasible = df[mask].copy()

    if len(df_feasible) == 0:
        print("  Warning: no feasible solutions, using all")
        df_feasible = df

    print(f"  Feasible (Fe/SiO2 + matte_Cu): {len(df_feasible)}/{len(df)}")

    if len(df_feasible) <= n:
        return df_feasible

    # 按 slag_Cu 排序，均匀选取
    df_sorted = df_feasible.sort_values('slag_Cu_pct').reset_index(drop=True)
    indices = np.linspace(0, len(df_sorted) - 1, n, dtype=int)
    return df_sorted.iloc[indices].reset_index(drop=True)


def generate_comparison(df_rep, output_dir=None):
    """生成优化方案 vs 人工方案对比（模拟人工方案数据）。

    注：人工方案数据基于论文中的一般水平估算，
    真实数据需从企业获取。
    """
    # 人工经验方案的一般水平（基于行业经验估算）
    manual = {
        'slag_Cu_pct': 0.65,
        'matte_Cu_pct': 58.0,
        'slag_FeO_SiO2_ratio': 1.35,
    }

    # 优化方案取 Pareto 前沿的中位数
    opt = {
        'slag_Cu_pct': df_rep['slag_Cu_pct'].median(),
        'matte_Cu_pct': df_rep['matte_Cu_pct'].median(),
        'slag_FeO_SiO2_ratio': df_rep['slag_FeO_SiO2_ratio'].median(),
    }

    return manual, opt


def plot_pareto_front(df_all, df_rep, output_dir):
    """生成 Pareto 前沿散点图。"""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.unicode_minus': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
    })

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # 所有 Pareto 解
    ax.scatter(df_all['slag_Cu_pct'], df_all['matte_Cu_pct'],
               c='#484878', s=25, alpha=0.6, edgecolors='none',
               label='Pareto 最优解', rasterized=True)

    # 代表性方案标注
    labels = ['A', 'B', 'C', 'D', 'E']
    for idx, (_, row) in enumerate(df_rep.iterrows()):
        if idx >= len(labels):
            break
        ax.annotate(labels[idx],
                    xy=(row['slag_Cu_pct'], row['matte_Cu_pct']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='#D4544A',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#D4544A', linewidth=0.8))

    # 人工方案（估算值）
    ax.axvline(0.65, color='#999', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.text(0.66, ax.get_ylim()[1] * 0.95, 'Manual\n(estimate)',
            fontsize=7, color='#999', va='top')

    ax.set_xlabel('Slag Cu (wt%)', fontsize=10)
    ax.set_ylabel('Matte Cu (wt%)', fontsize=10)
    ax.set_title('NSGA-II Optimization', fontsize=11)
    ax.legend(fontsize=8, loc='lower left')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'fig_pareto_front.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  → fig_pareto_front.png/pdf")


def plot_comparison_bar(df_rep, manual, opt, output_dir):
    """生成优化方案 vs 人工方案对比柱状图。"""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.unicode_minus': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
    })

    metrics = ['slag_Cu_pct', 'matte_Cu_pct', 'slag_FeO_SiO2_ratio']
    labels = ['Slag Cu\n(wt%)', 'Matte Cu\n(wt%)', 'FeO/SiO2']

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    for ax, metric, label in zip(axes, metrics, labels):
        v_manual = manual[metric]
        v_opt = opt[metric]

        bars = ax.bar([0, 1], [v_manual, v_opt],
                      color=['#B4C0E4', '#484878'],
                      edgecolor='black', linewidth=0.5, width=0.5)

        for bar, val in zip(bars, [v_manual, v_opt]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01 * val,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # 改善幅度
        if metric == 'slag_Cu_pct':
            improvement = (v_manual - v_opt) / v_manual * 100
            ax.text(0.5, 0.95, f'-{improvement:.0f}%',
                    transform=ax.transAxes, ha='center', fontsize=8,
                    color='#D4544A', fontweight='bold')
        elif metric == 'matte_Cu_pct':
            improvement = (v_opt - v_manual) / v_manual * 100
            ax.text(0.5, 0.95, f'+{improvement:.1f}%',
                    transform=ax.transAxes, ha='center', fontsize=8,
                    color='#D4544A', fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Manual', 'Optimized'], fontsize=8)
        ax.set_ylabel(label, fontsize=10)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'fig_optimization_comparison.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  → fig_optimization_comparison.png/pdf")


# ============================================================
#                    主流程
# ============================================================

def main():
    print("=" * 50)
    print("NSGA-II Ore Blending Optimization")
    print(f"  Objective: min(slag_Cu_pct)")
    print(f"  Constraints: Fe/SiO2 in [{FESIO2_LO}, {FESIO2_HI}], matte_Cu in [{MATTE_CU_LO}, {MATTE_CU_HI}]")
    print("=" * 50)

    # 1. 加载模型
    print("\n[1/5] Loading TabPFN models...")
    models = load_models()

    # 2. 运行优化
    print("\n[2/5] Running NSGA-II optimization...")
    res = run_optimization(models)

    # 3. 提取结果
    print("\n[3/5] Extracting results...")
    df_all = extract_results(res, models)
    print(f"  Solutions: {len(df_all)}")

    if len(df_all) == 0:
        print("  Error: no feasible solutions found!")
        return

    # 保存完整结果
    all_results_path = OPT_DIR / 'optimization_pareto_all.csv'
    df_all.to_csv(all_results_path, index=False, encoding='utf-8-sig')
    print(f"  Full results saved: {all_results_path}")

    # 选取代表性方案
    df_rep = select_representative_solutions(df_all, n=5)
    rep_path = OPT_DIR / 'optimization_results.csv'
    df_rep.to_csv(rep_path, index=False, encoding='utf-8-sig')
    print(f"  Representative solutions saved: {rep_path}")

    # 打印代表性方案
    print("\n  Representative solutions:")
    print("  " + "-" * 80)
    labels = ['A', 'B', 'C', 'D', 'E']
    for idx, (_, row) in enumerate(df_rep.iterrows()):
        label = labels[idx] if idx < len(labels) else str(idx)
        print(f"  {label}: slag_Cu={row['slag_Cu_pct']:.4f} wt%, "
              f"matte_Cu={row['matte_Cu_pct']:.2f} wt%, "
              f"Fe/SiO2={row['slag_FeO_SiO2_ratio']:.3f}")

    # 4. 生成对比数据
    print("\n[4/5] Generating comparison...")
    manual, opt = generate_comparison(df_rep)
    print(f"  Manual: slag_Cu={manual['slag_Cu_pct']}, "
          f"matte_Cu={manual['matte_Cu_pct']}, Fe/SiO2={manual['slag_FeO_SiO2_ratio']}")
    print(f"  Optimized: slag_Cu={opt['slag_Cu_pct']:.4f}, "
          f"matte_Cu={opt['matte_Cu_pct']:.2f}, Fe/SiO2={opt['slag_FeO_SiO2_ratio']:.3f}")

    # 5. 生成图表
    print("\n[5/5] Generating figures...")
    plot_pareto_front(df_all, df_rep, OUTPUT_DIR)
    plot_comparison_bar(df_rep, manual, opt, OUTPUT_DIR)

    # 汇总
    print("\n" + "=" * 50)
    slag_cu_reduction = (manual['slag_Cu_pct'] - opt['slag_Cu_pct']) / manual['slag_Cu_pct'] * 100
    print(f"Slag Cu reduction: {slag_cu_reduction:.1f}%")
    print(f"Matte Cu: {opt['matte_Cu_pct']:.2f} wt%")
    print(f"Fe/SiO2: {opt['slag_FeO_SiO2_ratio']:.3f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
