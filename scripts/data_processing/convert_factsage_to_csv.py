"""
将 FactSage 批量计算结果转换为机器学习训练用 CSV 表格。

输入: results/result_XXXXX.txt + 合成炉料样本3000组_输入数据.txt
输出: ml_dataset.csv

输出变量:
  冰铜 (wt.%): Cu, Fe, S, Ni, Zn, Pb, As, 质量(g)
  渣   (wt.%): Cu(来自Cu2O), FeO, SiO2, FeO/SiO2比, Al2O3, CaO, MgO, ZnO, 质量(g)
  气相 (mol分数): SO2, S2, SO, SSO, SO3, Zn, PbS, CuS, Pb, Sb, As, 质量(g)
"""

import re
import os
import csv


def parse_input_file(filepath):
    """解析输入数据文件，返回每行的输入特征字典列表。"""
    samples = []
    with open(filepath, 'r', encoding='gbk') as f:
        header = f.readline().strip().split('\t')
        feature_cols = header  # T(°C), P(atm), Cu(g), S(g), Fe(g), ...
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            values = [float(x) for x in parts]
            samples.append(dict(zip(feature_cols, values)))
    return samples


def find_phase_line(lines, phase_name):
    """找到某个相的起始行号。匹配 '+ X gram  相名' 格式。"""
    pattern = re.compile(r'^\s+\+\s+([\d.E+\-]+)\s+gram\s+' + re.escape(phase_name) + r'\s*$')
    for i, line in enumerate(lines):
        if pattern.match(line):
            return i
    return -1


def parse_composition(lines, start_idx):
    """
    通用分量解析：从 start_idx 行之后开始，解析 "数值 + [wt.%|mol] + 名称" 格式。
    跳过温度、压力、a= 等行，直到遇到实际分量数据。
    返回 {名称: 数值} 字典。
    """
    comps = {}
    num_pattern = re.compile(r'([\d.E+\-]+)\s+(wt\.%|mol)\s+(\S+)')
    i = start_idx + 1
    found_any = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        m = num_pattern.search(line)
        if m:
            comps[m.group(3)] = float(m.group(1))
            found_any = True
            i += 1
            continue
        if found_any:
            break
        i += 1
    return comps


def parse_gas_composition(lines, start_idx):
    """
    气相分量解析：格式为 "数值 + 名称"（无 mol/wt.% 标记）。
    """
    comps = {}
    num_pattern = re.compile(r'([\d.E+\-]+)\s+([A-Za-z][A-Za-z0-9_]*)')
    i = start_idx + 1
    found_any = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if 'gram' in line and 'mol' in line:
            i += 1
            continue
        if line.startswith('(') and ('C,' in line or 'atm' in line):
            i += 1
            continue
        m = num_pattern.search(line)
        if m:
            name = m.group(2)
            value = float(m.group(1))
            comps[name] = value
            found_any = True
            i += 1
            continue
        if found_any:
            break
        i += 1
    return comps


def parse_result_file(filepath):
    """解析单个 FactSage 结果文件，提取所有输出变量。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = {}

    # --- 冰铜 (Matte) ---
    idx = find_phase_line(lines, 'Matte')
    if idx >= 0:
        m = re.search(r'([\d.E+\-]+)\s+gram', lines[idx])
        result['matte_mass_g'] = float(m.group(1)) if m else 0.0
        comps = parse_composition(lines, idx)
        result['matte_Cu_pct'] = comps.get('Cu', 0.0)
        result['matte_Fe_pct'] = comps.get('Fe', 0.0)
        result['matte_S_pct'] = comps.get('S', 0.0)
        result['matte_Ni_pct'] = comps.get('Ni', 0.0)
        result['matte_Zn_pct'] = comps.get('Zn', 0.0)
        result['matte_Pb_pct'] = comps.get('Pb', 0.0)
        result['matte_As_pct'] = comps.get('As', 0.0)
    else:
        for k in ['matte_mass_g', 'matte_Cu_pct', 'matte_Fe_pct', 'matte_S_pct',
                   'matte_Ni_pct', 'matte_Zn_pct', 'matte_Pb_pct', 'matte_As_pct']:
            result[k] = 0.0

    # --- 渣 (Slag-liq#1) ---
    idx = find_phase_line(lines, 'Slag-liq#1')
    if idx >= 0:
        m = re.search(r'([\d.E+\-]+)\s+gram', lines[idx])
        result['slag_mass_g'] = float(m.group(1)) if m else 0.0
        comps = parse_composition(lines, idx)
        cu2o = comps.get('Cu2O', 0.0)
        result['slag_Cu_pct'] = cu2o * 127.09 / 143.09  # Cu2O -> Cu
        result['slag_FeO_pct'] = comps.get('FeO', 0.0)
        result['slag_SiO2_pct'] = comps.get('SiO2', 0.0)
        result['slag_Al2O3_pct'] = comps.get('Al2O3', 0.0)
        result['slag_CaO_pct'] = comps.get('CaO', 0.0)
        result['slag_MgO_pct'] = comps.get('MgO', 0.0)
        result['slag_ZnO_pct'] = comps.get('ZnO', 0.0)
        sio2 = result['slag_SiO2_pct']
        feo = result['slag_FeO_pct']
        result['slag_FeO_SiO2_ratio'] = feo / sio2 if sio2 > 0 else 0.0
    else:
        for k in ['slag_mass_g', 'slag_Cu_pct', 'slag_FeO_pct', 'slag_SiO2_pct',
                   'slag_Al2O3_pct', 'slag_CaO_pct', 'slag_MgO_pct', 'slag_ZnO_pct',
                   'slag_FeO_SiO2_ratio']:
            result[k] = 0.0

    # --- 气相 (gas_real) ---
    gas_comps = {}
    gas_mass = 0.0
    for i, line in enumerate(lines):
        if 'gas_real' in line and 'mol' in line:
            m = re.search(r'([\d.E+\-]+)\s+gram', lines[i + 1])
            if m:
                gas_mass = float(m.group(1))
            gas_comps = parse_gas_composition(lines, i)
            break

    result['gas_mass_g'] = gas_mass
    # 气相主要成分 (摩尔分数)
    gas_species = ['SO2', 'S2', 'SO', 'SSO', 'SO3', 'Zn', 'PbS', 'CuS', 'Pb', 'Sb', 'As']
    for sp in gas_species:
        result[f'gas_{sp}_mol'] = gas_comps.get(sp, 0.0)

    return result


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(project_root, 'data', '3000_fixed_temp')
    input_file = os.path.join(dataset_dir, 'factsage_inputs.txt')
    results_dir = os.path.join(dataset_dir, 'factsage_results')
    output_file = os.path.join(dataset_dir, 'ml_dataset.csv')

    # 解析输入数据
    print("正在解析输入数据...")
    inputs = parse_input_file(input_file)
    print(f"  共 {len(inputs)} 组输入样本")

    # 获取所有结果文件并排序
    result_files = sorted([f for f in os.listdir(results_dir) if f.startswith('result_') and f.endswith('.txt')])
    print(f"  共 {len(result_files)} 个结果文件")

    if len(inputs) != len(result_files):
        print(f"  警告: 输入({len(inputs)})与结果({len(result_files)})数量不匹配，取较小值")

    n = min(len(inputs), len(result_files))

    # 输入特征列名
    input_cols = list(inputs[0].keys())
    # 输出列名
    output_cols = [
        # 冰铜 (wt.%)
        'matte_Cu_pct', 'matte_Fe_pct', 'matte_S_pct',
        'matte_Ni_pct', 'matte_Zn_pct', 'matte_Pb_pct', 'matte_As_pct',
        'matte_mass_g',
        # 渣 (wt.%)
        'slag_Cu_pct', 'slag_FeO_pct', 'slag_SiO2_pct', 'slag_FeO_SiO2_ratio',
        'slag_Al2O3_pct', 'slag_CaO_pct', 'slag_MgO_pct', 'slag_ZnO_pct',
        'slag_mass_g',
        # 气相 (mol分数 + 质量)
        'gas_SO2_mol', 'gas_S2_mol', 'gas_SO_mol', 'gas_SSO_mol', 'gas_SO3_mol',
        'gas_Zn_mol', 'gas_PbS_mol', 'gas_CuS_mol', 'gas_Pb_mol', 'gas_Sb_mol', 'gas_As_mol',
        'gas_mass_g'
    ]

    all_cols = input_cols + output_cols

    # 逐个解析并写入CSV
    print("正在解析结果文件并生成CSV...")
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_cols)
        writer.writeheader()
        for i in range(n):
            filepath = os.path.join(results_dir, result_files[i])
            result = parse_result_file(filepath)
            row = dict(inputs[i])
            row.update(result)
            writer.writerow(row)
            if (i + 1) % 500 == 0:
                print(f"  已处理 {i+1}/{n}")

    print(f"\n完成! 共 {n} 条数据，已保存至: {output_file}")
    print(f"输入特征: {len(input_cols)} 列 -> {input_cols}")
    print(f"输出变量: {len(output_cols)} 列 -> {output_cols}")


if __name__ == '__main__':
    main()
