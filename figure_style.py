"""Shared publication style: Arial (fallback Liberation Sans, then DejaVu Sans),
mathematics set in the same font, sizes given at final print size."""

import matplotlib as mpl
from matplotlib import font_manager

FONT_CANDIDATES = ("Arial", "Liberation Sans", "DejaVu Sans")
FS_TEXT = 9.0      # axis labels, symbols in schematics
FS_TICK = 8.0      # tick labels
FS_PANEL = 10.0    # panel letters (bold)
LW_AXIS = 0.7
LW_LINE = 1.0
LW_GUIDE = 0.6
MM = 1 / 25.4

C_BETA = "#C42238"     # closure defect in the main figures
C_REF = "#034A29"      # reference signal (as x(t) in the Lorenz figure)
C_SEL = "#2C7FB8"      # selected-scale markers in the main figures
C_SRAW = "#B8CFE3"     # raw S(T)
C_SSMOOTH = "#1f77b4"  # smoothed S(T)
C_A1, C_A2 = "#0072B2", "#CC3399"   # local rules in panel (a)


def pick_font():
    for name in FONT_CANDIDATES:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name),
                                  fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Sans"


def apply_style():
    fam = pick_font()
    if fam != "Arial":
        print(f"[figure_style] Arial not found; using {fam}")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [fam],
        "mathtext.fontset": "custom",
        "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold",
        "mathtext.sf": fam,
        "mathtext.cal": fam,
        "mathtext.tt": fam,
        "mathtext.fallback": "stixsans",
        "font.size": FS_TEXT,
        "axes.labelsize": FS_TEXT,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "axes.linewidth": LW_AXIS,
        "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.labelpad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": LW_LINE,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600,
    })
    return fam
