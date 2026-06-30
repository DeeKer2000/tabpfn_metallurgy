"""
图1 技术路线图 — 热力学信息增强机器学习智能配矿框架
用法: conda run -n tabpfn python 文章/fig1_roadmap.py
输出: 文章/figures/fig1_roadmap.png + .pdf + .svg
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl
from pathlib import Path

# ============================================================
# 样式配置
# ============================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

# Nature 材料/机理页配色
C_STAGE1 = "#4DB8B0"   # 数据生成 — teal
C_STAGE2 = "#E8A040"   # 热力学计算 — amber
C_STAGE3 = "#0F4D92"   # ML训练 — deep blue (hero)
C_STAGE4 = "#3A7D44"   # 优化 — green
C_ARROW  = "#555555"   # 箭头
C_BG_LIGHT = ["#E8F6F5", "#FFF5EB", "#E8EEF6", "#ECF4ED"]  # 各阶段背景
C_TEXT    = "#333333"
C_WHITE   = "#FFFFFF"
C_GREY    = "#E0E0E0"

# 画布尺寸 (Nature 全宽 183mm)
FIG_W = 183 / 25.4  # inches
FIG_H = 9.0


def draw_rounded_box(ax, xy, width, height, color, text="", text_color=C_TEXT,
                     fontsize=7.5, fontweight="normal", edge_color=None, alpha=1.0,
                     text_align="center"):
    """绘制圆角矩形框并添加文字。"""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor=edge_color or color,
        linewidth=0.8, alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
                ha=text_align, va="center", fontsize=fontsize, fontweight=fontweight,
                color=text_color, zorder=4)


def draw_arrow(ax, start, end, color=C_ARROW, linewidth=0.8, style="simple"):
    """绘制箭头。"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="->" if style == "simple" else "-|>",
        color=color, linewidth=linewidth,
        mutation_scale=12, zorder=2
    )
    ax.add_patch(arrow)


def draw_stage_background(ax, y, height, color):
    """绘制阶段背景色条。"""
    rect = mpatches.Rectangle(
        (0.02, y), 0.96, height,
        facecolor=color, edgecolor="none", alpha=0.25, zorder=0,
        transform=ax.transAxes
    )
    ax.add_patch(rect)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # ============================================================
    # 标题
    # ============================================================
    ax.text(5, 9.65, "Thermodynamics-Informed Machine Learning Framework for Intelligent Ore Blending",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_TEXT)

    # ============================================================
    # 列坐标定义
    # ============================================================
    # 列: Stage | Input | Arrow | Process | Arrow | Output | Details
    X_STAGE   = 0.3   # 阶段标签
    X_INPUT   = 1.1   # 输入框
    X_ARROW1  = 2.45  # 箭头1
    X_PROCESS = 2.75  # 过程框
    X_ARROW2  = 5.55  # 箭头2
    X_OUTPUT  = 5.85  # 输出框
    X_DETAIL  = 7.5   # 细节文字

    W_BOX    = 1.2    # 小框宽度
    W_WIDE   = 2.5    # 宽框宽度
    H_BOX    = 0.55   # 框高度
    H_STAGE  = 0.65   # 阶段标签高度

    # 各阶段 Y 坐标
    Y_STAGES = [8.0, 5.8, 3.6, 1.4]
    Y_GAP = 0.85  # 行间距

    # ============================================================
    # 阶段定义
    # ============================================================
    stages = [
        {
            "label": "a",
            "title": "Synthetic Ore\nBlending Data\nGeneration",
            "input": "11 ore sources\n(13 components)",
            "process": "Dirichlet random sampling\n+\nMetallurgical constraints\n(Cu/S/Fe/SiO2/Fe-SiO2 ratio)",
            "output": "3,000 synthetic\nfurnace charge\ncompositions",
            "detail": "N_ore = 2–6 per blend\nSiO2 flux = 0–15%\nCu: 14–26 wt%\nS: 8–38 wt%\nFe: 8–36 wt%\nSiO2: 3–25 wt%",
            "color": C_STAGE1,
            "bg": C_BG_LIGHT[0],
        },
        {
            "label": "b",
            "title": "FactSage 8.3\nThermodynamic\nBatch Calculation",
            "input": "3,000 synthetic\nfurnace charge\ncompositions",
            "process": "Equilib module\nGibbs free energy\nminimization\n(FToxid/FTmisc/FTsalt/FTps)",
            "output": "29 metallurgical\nindicators\n(Matte / Slag / Gas)",
            "detail": "T = 1200 degC, P = 1 atm\nO2 = 21 g\n\nMatte: Cu/Fe/S/Ni/Zn/Pb/As\nSlag: Cu/FeO/SiO2/CaO/...\nGas: SO2/S2/SO/...",
            "color": C_STAGE2,
            "bg": C_BG_LIGHT[1],
        },
        {
            "label": "c",
            "title": "TabPFN Surrogate\nModel Training\n& Evaluation",
            "input": "16 input features\n(composition + T/P/O2)",
            "process": "TabPFN in-context learning\n(no gradient descent)\n+\nRF / XGBoost / GB / MLP\nbaseline comparison",
            "output": "29 independent\npredictive models\n(mean R² = 0.989)",
            "detail": "80/20 train-test split\nMetrics: R², MAE, RMSE\n\nMatte Cu R² = 0.9998\nSlag FeO/SiO2 R² = 0.9972\nSlag Cu R² = 0.890",
            "color": C_STAGE3,
            "bg": C_BG_LIGHT[2],
        },
        {
            "label": "d",
            "title": "NSGA-II\nOre Blending\nOptimization",
            "input": "Trained TabPFN\nsurrogate models\n(3 targets)",
            "process": "NSGA-II (Pop=100, Gen=200)\nSBX crossover + PM mutation\nEarly stopping (patience=30)\n+\nFeasibility filtering",
            "output": "Optimal ore\nblending recipe\n(slag Cu = 0.386%)",
            "detail": "12D decision variables\n(11 ore ratios + SiO2 flux)\n\nObjective: min slag_Cu_pct\nConstraints: FeO/SiO2: [1.15, 1.30]\nmatte Cu: [58, 60] wt%",
            "color": C_STAGE4,
            "bg": C_BG_LIGHT[3],
        },
    ]

    for i, st in enumerate(stages):
        y = Y_STAGES[i]

        # 阶段背景
        draw_stage_background(ax, (y - 0.5) / 10, 2.3 / 10, st["bg"])

        # 阶段标签 (a, b, c, d)
        draw_rounded_box(ax, (X_STAGE, y - H_STAGE / 2), 0.35, H_STAGE,
                         st["color"], st["label"],
                         text_color=C_WHITE, fontsize=13, fontweight="bold")

        # 阶段标题
        ax.text(X_STAGE + 0.55, y + 0.1, st["title"],
                ha="left", va="center", fontsize=7.5, fontweight="bold", color=st["color"])

        # 输入框
        draw_rounded_box(ax, (X_INPUT, y - H_BOX / 2), W_BOX, H_BOX,
                         C_WHITE, st["input"],
                         edge_color=st["color"], fontsize=6.2, alpha=0.9)

        # 箭头1: 输入 → 过程
        draw_arrow(ax, (X_INPUT + W_BOX, y), (X_PROCESS - 0.12, y), st["color"])

        # 过程框
        draw_rounded_box(ax, (X_PROCESS, y - H_BOX / 2), W_WIDE, H_BOX,
                         st["color"], st["process"],
                         text_color=C_WHITE, fontsize=6.5, fontweight="bold")

        # 箭头2: 过程 → 输出
        draw_arrow(ax, (X_PROCESS + W_WIDE, y), (X_OUTPUT - 0.12, y), st["color"])

        # 输出框
        draw_rounded_box(ax, (X_OUTPUT, y - H_BOX / 2), W_BOX, H_BOX,
                         C_WHITE, st["output"],
                         edge_color=st["color"], fontsize=6.2, alpha=0.9)

        # 细节文字
        ax.text(X_DETAIL, y, st["detail"],
                ha="left", va="center", fontsize=5.8, color=C_TEXT,
                linespacing=1.3)

        # 阶段间大箭头（除最后一行）
        if i < 3:
            arrow_y = y - 0.85
            draw_arrow(ax, (5, arrow_y + 0.55), (5, arrow_y - 0.05),
                       color=C_ARROW, linewidth=1.0, style="simple")

    # ============================================================
    # 图例：标注列含义
    # ============================================================
    ax.text(5, 0.15, "Input  →  Method  →  Output",
            ha="center", va="center", fontsize=7, fontweight="bold", color=C_GREY)

    # ============================================================
    # 保存
    # ============================================================
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 300), ("pdf", None), ("svg", None)]:
        path = output_dir / f"fig1_roadmap.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  -> {path}")

    plt.close(fig)
    print("Done: fig1_roadmap.png / .pdf / .svg")


if __name__ == "__main__":
    main()