"""从 NSGA-II 优化结果生成论文图表。
用法: conda run -n tabpfn python scripts/optimization/plot_optimization.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OPT_DIR = PROJECT_ROOT / 'experiments' / '3000_fixed_temp' / 'optimization'
OUTPUT_DIR = OPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fe/SiO2 constraint range（与 nsga2_optimization.py 一致）
FESIO2_LO = 1.15
FESIO2_HI = 1.30


def load_and_filter(csv_path):
    """Load Pareto results and filter by Fe/SiO2 constraint."""
    df = pd.read_csv(csv_path)
    print(f"  Total Pareto solutions: {len(df)}")

    mask = ((df['slag_FeO_SiO2_ratio'] >= FESIO2_LO) &
            (df['slag_FeO_SiO2_ratio'] <= FESIO2_HI))
    df_feasible = df[mask].copy()
    print(f"  Feasible (Fe/SiO2 in [{FESIO2_LO}, {FESIO2_HI}]): {len(df_feasible)}")
    return df_feasible


def select_representative(df, n=5):
    """Select n representative solutions evenly spaced by slag_Cu."""
    if len(df) <= n:
        return df
    df_sorted = df.sort_values('slag_Cu_pct').reset_index(drop=True)
    indices = np.linspace(0, len(df_sorted) - 1, n, dtype=int)
    return df_sorted.iloc[indices].reset_index(drop=True)


def plot_pareto_front(df_all, df_feasible, df_rep, output_dir):
    """Generate Pareto front scatter plot."""
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

    # All Pareto solutions (infeasible in gray)
    if len(df_all) > len(df_feasible):
        infeasible = df_all[~df_all.index.isin(df_feasible.index)]
        ax.scatter(infeasible['slag_Cu_pct'], infeasible['matte_Cu_pct'],
                   c='#CCCCCC', s=15, alpha=0.4, edgecolors='none',
                   label='Infeasible', rasterized=True)

    # Feasible Pareto solutions
    ax.scatter(df_feasible['slag_Cu_pct'], df_feasible['matte_Cu_pct'],
               c='#484878', s=25, alpha=0.7, edgecolors='none',
               label='Feasible Pareto', rasterized=True)

    # Representative solutions
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

    ax.set_xlabel('Slag Cu (wt%)', fontsize=10)
    ax.set_ylabel('Matte Cu (wt%)', fontsize=10)
    ax.set_title('NSGA-II Pareto Front', fontsize=11)
    ax.legend(fontsize=8, loc='lower left')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'fig_pareto_front.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> fig_pareto_front.png/pdf")


def plot_comparison_bar(df_rep, manual, opt, output_dir):
    """Generate optimization vs manual comparison bar chart."""
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
    print(f"  -> fig_optimization_comparison.png/pdf")


def main():
    print("=" * 50)
    print("NSGA-II Figure Generation")
    print("=" * 50)

    # Load data
    all_csv = OPT_DIR / 'optimization_pareto_all.csv'
    rep_csv = OPT_DIR / 'optimization_results.csv'

    df_all = pd.read_csv(all_csv)
    print(f"\n  Loaded {len(df_all)} Pareto solutions")

    # Filter feasible solutions
    df_feasible = load_and_filter(all_csv)
    df_rep = select_representative(df_feasible, n=5)

    # Print representative solutions
    labels = ['A', 'B', 'C', 'D', 'E']
    print("\n  Representative Pareto solutions (feasible):")
    print("  " + "-" * 80)
    for idx, (_, row) in enumerate(df_rep.iterrows()):
        label = labels[idx] if idx < len(labels) else str(idx)
        print(f"  {label}: slag_Cu={row['slag_Cu_pct']:.4f} wt%, "
              f"matte_Cu={row['matte_Cu_pct']:.2f} wt%, "
              f"Fe/SiO2={row['slag_FeO_SiO2_ratio']:.3f}")

    # Comparison data
    manual = {
        'slag_Cu_pct': 0.65,
        'matte_Cu_pct': 58.0,
        'slag_FeO_SiO2_ratio': 1.35,
    }
    opt = {
        'slag_Cu_pct': df_rep['slag_Cu_pct'].median(),
        'matte_Cu_pct': df_rep['matte_Cu_pct'].median(),
        'slag_FeO_SiO2_ratio': df_rep['slag_FeO_SiO2_ratio'].median(),
    }

    print(f"\n  Manual (est.): slag_Cu={manual['slag_Cu_pct']}, "
          f"matte_Cu={manual['matte_Cu_pct']}, Fe/SiO2={manual['slag_FeO_SiO2_ratio']}")
    print(f"  Optimized (median): slag_Cu={opt['slag_Cu_pct']:.4f}, "
          f"matte_Cu={opt['matte_Cu_pct']:.2f}, Fe/SiO2={opt['slag_FeO_SiO2_ratio']:.3f}")

    # Generate figures
    print("\n  Generating figures...")
    plot_pareto_front(df_all, df_feasible, df_rep, OUTPUT_DIR)
    plot_comparison_bar(df_rep, manual, opt, OUTPUT_DIR)

    # Summary
    slag_reduction = (manual['slag_Cu_pct'] - opt['slag_Cu_pct']) / manual['slag_Cu_pct'] * 100
    matte_improvement = (opt['matte_Cu_pct'] - manual['matte_Cu_pct']) / manual['matte_Cu_pct'] * 100
    print(f"\n  Summary:")
    print(f"  Slag Cu reduction: {slag_reduction:.1f}%")
    print(f"  Matte Cu improvement: {matte_improvement:.1f}%")
    print(f"  Fe/SiO2: {opt['slag_FeO_SiO2_ratio']:.3f}")
    print("=" * 50)

    # Save updated representative solutions
    df_rep.to_csv(rep_csv, index=False, encoding='utf-8-sig')
    print(f"  Updated representative solutions saved: {rep_csv}")


if __name__ == '__main__':
    main()
