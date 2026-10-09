"""Graphical abstract for the scale-resolved local-closure manuscript.

Designed for the Nonlinear Dynamics 2:1 graphical-abstract format.
The figure keeps only the main logic:

    local closure  ->  scale scan  ->  hierarchy
"""

from pathlib import Path
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import closure_pipeline as cp
import figure_style as st
import fig2_lorenz as f2


# =============================================================================
# Canvas and style
# =============================================================================
ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"

W_MM, H_MM = 174.0, 87.0
MM = getattr(st, "MM", 1.0 / 25.4)

FS_HEAD = max(9.0, float(getattr(st, "FS_TEXT", 7.0)) + 1.8)
FS_TEXT = max(7.4, float(getattr(st, "FS_TEXT", 7.0)) + 0.4)
FS_SMALL = max(6.6, float(getattr(st, "FS_TICK", 6.0)) + 0.3)

LW_AXIS = float(getattr(st, "LW_AXIS", 0.65))

C_BETA = getattr(st, "C_BETA", "#B2182B")
C_T1 = getattr(st, "C_A1", "#2878B5")
C_T2 = getattr(st, "C_A2", "#C03A76")
C_S = "#3A3A3A"
C_REF = getattr(st, "C_REF", "0.60")

# Global geometry.  Outer margins stay tight, while both inter-panel gaps are
# widened to 9 mm so the flow arrows can sit in clean white space.
LEFT = (1.5, 4.5, 48.0, 75.0)       # ends at x = 49.5
# Map top at 72 mm so all three panels start ~7 mm below the titles.
MAP = (58.5, 14.0, 36.0, 58.0)
SPROF = (96.5, 14.0, 7.0, 58.0)     # middle group ends at x = 103.5
RIGHT = (112.5, 8.0, 59.5, 69.0)    # starts at x = 112.5


# =============================================================================
# Helpers
# =============================================================================
def axes_mm(fig, box, schematic=False):
    x, y, w, h = box
    ax = fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])
    if schematic:
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.axis("off")
    return ax


def arrow(ax, p0, p1, color="0.35", lw=0.8, ms=7.0, z=5):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def safe_savefig(fig, target, **kwargs):
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"
    with tempfile.TemporaryDirectory(prefix="graphical_abstract_") as tmp_dir:
        tmp = Path(tmp_dir) / f"render{suffix}"
        fig.savefig(tmp, **kwargs)
        try:
            shutil.copy2(tmp, target)
        except OSError:
            fig.savefig(target, **kwargs)
    return target


# =============================================================================
# Left: workflow-like local closure (closer to Fig. 1b)
# =============================================================================
def _stacked_rows(ax, x, y, w, nrows, row_h, f, color="0.35", fc="0.94", lw_tr=0.70):
    ax.add_patch(Rectangle((x, y), w, nrows * row_h, fc=fc, ec="0.25", lw=0.75, zorder=1))
    u = np.linspace(0, 1, 180)
    for i in range(nrows):
        yc = y + (nrows - i - 0.5) * row_h
        ax.plot(x + 0.8 + (w - 1.6) * u, yc + 0.28 * row_h * f(i, u),
                color=color, lw=lw_tr, solid_capstyle="round", zorder=2)
        if i > 0:
            ax.plot([x + 0.4, x + w - 0.4], [y + (nrows - i) * row_h] * 2,
                    color="white", lw=0.6, zorder=2)


def draw_local_closure(fig):
    ax = axes_mm(fig, LEFT, schematic=True)
    _, _, w, h = LEFT

    # ---- upper tier: time series -> reduced temporal representation
    yu = h - 15.0
    u = np.linspace(0, 1, 200)

    # Three coupled observables share one explicit time direction.
    x0, xw_trace = 1.4, 15.0
    for k, dy in enumerate((3.1, 0.0, -3.1)):
        tr = (
            np.sin(2 * np.pi * (2.0 * u + 0.18 * k))
            * (0.74 + 0.18 * np.cos(2 * np.pi * (0.75 * u + 0.10 * k)))
            + 0.10 * np.sin(2 * np.pi * (6.0 * u + 0.13 * k))
        )
        ax.plot(x0 + xw_trace * u, yu + dy + 1.0 * tr,
                color="0.25", lw=0.82, solid_capstyle="round")

    ax.text(x0, yu + 6.1, r"$\mathbf{x}(t)$",
            ha="left", va="center", fontsize=FS_TEXT)

    # A common time arrow makes the upper tier read as a trajectory record.
    time_y = yu - 5.3
    arrow(ax, (x0 + 0.4, time_y), (x0 + xw_trace - 0.2, time_y),
          color="0.48", lw=0.62, ms=5.0)
    ax.text(x0 + xw_trace + 0.4, time_y, r"$t$",
            ha="left", va="center", fontsize=FS_SMALL)

    # Clean transformation arrow between the record and V; it has its own channel.
    arrow(ax, (17.0, yu), (19.0, yu), color="0.45", lw=0.72, ms=5.8)

    xY, yY = 20.5, yu - 5.9
    row_h = 1.68
    nrows = 4
    freqs = (1.8, 2.7, 1.2, 3.1)

    def fY(i, v):
        return np.sin(2 * np.pi * (freqs[i] * v + 0.16 * i))

    _stacked_rows(ax, xY, yY, 20.0, nrows, row_h, fY,
                  color="0.30", fc="0.94", lw_tr=0.62)
    # Reduced-coordinate sequence Y; windows cut from it are V_t (main text).
    ax.text(xY + 10.0, yu + 6.1, r"$Y$",
            ha="center", va="center", fontsize=FS_TEXT)

    # One temporal window cut from Y.
    xw, ww = xY + 7.7, 4.8
    ax.add_patch(Rectangle((xw, yY), ww, nrows * row_h,
                           fc="none", ec="0.08", lw=1.05, zorder=4))
    ax.text(xw + ww / 2, yY - 2.0, r"$T$",
            ha="center", va="top", fontsize=FS_TEXT)

    # ---- lower tier: compact Fig.1b-like equation blocks
    yl = 30.0
    hz, wz, wa = 9.3, 6.2, 7.0
    y0 = yl - hz / 2
    xdv = 5.8
    xA = xdv + wz + 4.0
    xV = xA + wa + 1.2
    xR = xV + wz + 4.0

    u0 = (xw - xY) / 20.0
    u1 = (xw - xY + ww) / 20.0
    def win(i, v):
        return fY(i, u0 + (u1 - u0) * v)
    def dwin(i, v):
        return np.cos(2 * np.pi * (freqs[i] * (u0 + (u1 - u0) * v) + 0.16 * i))

    _stacked_rows(ax, xdv, y0, wz, 3, hz / 3.0, dwin, color="0.30", fc="0.94", lw_tr=0.70)
    ax.text(xdv + wz + 2.0, yl, "=", ha="center", va="center", fontsize=FS_HEAD)

    ax.add_patch(Rectangle((xA, y0), wa, hz, fc="white", ec="0.25", lw=0.75))
    for k in range(1, 3):
        ax.plot([xA + k * wa / 3] * 2, [y0, y0 + hz], color="0.80", lw=0.45)
        ax.plot([xA, xA + wa], [y0 + k * hz / 3] * 2, color="0.80", lw=0.45)
    A_MAG = np.array([[0.18, 0.80, 0.28], [0.78, 0.15, 0.55], [0.30, 0.52, 0.22]])
    cw, ch = wa / 3.0, hz / 3.0
    for i in range(3):
        for j in range(3):
            s = 0.74 * cw * np.sqrt(A_MAG[i, j])
            xc = xA + (j + 0.5) * cw
            yc = y0 + hz - (i + 0.5) * ch
            ax.add_patch(Rectangle((xc - s / 2, yc - s / 2), s, s, fc="0.40", ec="none"))

    _stacked_rows(ax, xV, y0, wz, 3, hz / 3.0, win, color="0.30", fc="0.94", lw_tr=0.70)
    ax.text(xV + wz + 2.0, yl, "+", ha="center", va="center", fontsize=FS_HEAD)

    def fR(i, v):
        env = -0.42 + 1.0 / (1.0 + np.exp(-(v - 0.55) / 0.07))
        return 1.45 * env * dwin(i, v)
    _stacked_rows(ax, xR, y0, wz, 3, hz / 3.0, fR, color=C_BETA, fc="#F7E7EA", lw_tr=0.78)

    for x, ww_, lab in ((xdv, wz, r"$\dot V_t$"), (xA, wa, r"$A_t$"), (xV, wz, r"$V_t$")):
        ax.text(x + ww_ / 2, y0 - 1.7, lab, ha="center", va="top", fontsize=FS_SMALL)
    ax.text(xR + wz / 2.0, y0 + hz + 1.5, r"$R_t$",
            ha="center", va="bottom", fontsize=FS_SMALL)

    # Magnification guides from upper window to V_t.
    ax.plot([xw, xV], [yY, y0 + hz], color="0.58", lw=0.55)
    ax.plot([xw + ww, xV + wz], [yY, y0 + hz], color="0.58", lw=0.55)

    # Bottom: beta(t) shown as a short time series rather than a single icon.
    xbeta = xR + wz / 2.0
    beta_y0 = 5.0
    arrow(ax, (xbeta, y0 - 0.3), (xbeta, beta_y0 + 2.8),
          color="0.35", lw=0.8, ms=5.8)

    # A small horizontal time axis makes the bottom schematic read explicitly as a trace.
    xbt = np.linspace(xbeta - 4.2, xbeta + 4.2, 140)
    tt = (xbt - (xbeta - 4.2)) / 8.4
    beta_trace = (
        0.22
        + 0.18 * np.sin(2 * np.pi * (1.05 * tt + 0.08))
        + 0.10 * np.sin(2 * np.pi * (2.10 * tt + 0.22))
        + 1.55 * np.exp(-((tt - 0.36) / 0.085) ** 2)
        + 0.55 * np.exp(-((tt - 0.76) / 0.10) ** 2)
    )
    ax.plot(xbt, beta_y0 + beta_trace, color=C_BETA, lw=1.35)
    arrow(ax, (xbeta - 4.2, beta_y0), (xbeta + 4.6, beta_y0),
          color="0.55", lw=0.65, ms=4.8)
    ax.text(xbeta + 5.1, beta_y0, r"$t$", color="0.45",
            ha="left", va="center", fontsize=FS_SMALL)
    ax.text(xbeta, 2.4, r"$\beta$", color=C_BETA,
            ha="center", va="top", fontsize=FS_HEAD)


# =============================================================================
# Middle: scale scan
# =============================================================================
def draw_scale_scan(fig, analysis):
    ax_map = axes_mm(fig, MAP)
    ax_s = axes_mm(fig, SPROF)

    selected = list(analysis["selected"][:2])
    if len(selected) < 2:
        raise RuntimeError(f"Graphical abstract needs two selected scales, got {selected}.")
    T1, T2 = sorted(selected[:2])

    display_T = np.asarray(analysis["display_T_values"], dtype=float)
    display_map = np.asarray(analysis["display_scale_map"], dtype=float)
    common_time = np.asarray(analysis["common_time"], dtype=float)

    ylo = float(display_T[0])
    yhi = min(float(display_T[-1]), max(3.0 * T2, 1.28 * T2))
    if yhi <= T2:
        yhi = float(display_T[-1])

    ax_map.pcolormesh(common_time, display_T, cp.globally_scaled_map(display_map),
                      shading="auto", rasterized=True)
    ax_map.set_yscale("log")
    ax_map.set_xlim(float(f2.TMIN), float(f2.TMAX))
    ax_map.set_ylim(ylo, yhi)
    ax_map.set_xticks([f2.TMIN, f2.TMAX])
    ax_map.set_xlabel(r"$t$", fontsize=FS_TEXT, labelpad=0.8)
    ax_map.set_ylabel("")
    ax_map.text(-0.035, 1.015, r"$T$", transform=ax_map.transAxes,
                ha="right", va="bottom", fontsize=FS_TEXT)
    ax_map.tick_params(axis="both", labelsize=FS_SMALL, length=2.0, width=LW_AXIS, pad=1.0)
    ax_map.spines["top"].set_visible(False)
    ax_map.spines["right"].set_visible(False)
    ax_map.spines["left"].set_linewidth(LW_AXIS)
    ax_map.spines["bottom"].set_linewidth(LW_AXIS)

    diag = analysis["diag"]
    T_smooth = np.exp(np.asarray(diag["log_grid"], dtype=float))
    S_smooth = np.asarray(diag["smooth"], dtype=float)
    keep = np.isfinite(T_smooth) & np.isfinite(S_smooth) & (T_smooth >= ylo) & (T_smooth <= yhi)
    T_smooth = T_smooth[keep]
    S_smooth = S_smooth[keep]

    ax_s.plot(S_smooth, T_smooth, color=C_S, lw=1.35)
    ax_s.set_yscale("log")
    ax_s.set_ylim(ylo, yhi)
    # which="both": the log axis otherwise leaves floating minor ticks
    # in the gap between the map and the profile.
    ax_s.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_s.tick_params(axis="x", labelsize=FS_SMALL, length=2.0, width=LW_AXIS, pad=0.9)
    ax_s.set_xlabel(r"$S(T)$", fontsize=FS_TEXT, labelpad=0.8)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.spines["left"].set_visible(False)
    ax_s.spines["bottom"].set_linewidth(LW_AXIS)

    if S_smooth.size:
        slo, shi = float(np.nanmin(S_smooth)), float(np.nanmax(S_smooth))
        span = max(shi - slo, 1e-4)
        ax_s.set_xlim(slo - 0.10 * span, shi + 0.04 * span)
        ax_s.set_xticks([round(shi, 2)])

    for j, (T, color) in enumerate(((T1, C_T1), (T2, C_T2)), start=1):
        ax_map.axhline(T, color=color, lw=1.2, ls=(0, (4, 2.2)), alpha=0.98)
        ax_s.axhline(T, color=color, lw=1.2, ls=(0, (4, 2.2)), alpha=0.98)
        sval = float(np.interp(np.log(T), np.asarray(diag["log_grid"], dtype=float),
                               np.asarray(diag["smooth"], dtype=float)))
        # The T1 point sits at the maximum of S, i.e. on the axis edge,
        # so the marker must not be clipped.
        ax_s.plot(sval, T, "o", ms=4.0, color=color, mec="white", mew=0.45,
                  zorder=5, clip_on=False)
        # Opaque white tag: a translucent box over viridis turns grey-violet.
        ax_map.text(0.985, T, rf"$T_{j}$", transform=ax_map.get_yaxis_transform(),
                    ha="right", va="bottom", color=color, fontsize=FS_TEXT,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=0.55))

    return T1, T2


# =============================================================================
# Right: hierarchy on the Lorenz attractor
# =============================================================================
def _find_segments(x, z, T1, T2):
    n = len(x)

    half_local = int(np.clip(0.16 * T1, 45, 150))
    candidates = []
    for i in range(half_local + 2, n - half_local - 2, 8):
        sl = slice(i - half_local, i + half_local + 1)
        if np.all(x[sl] > 1.5):
            score = float(x[i]) - 0.16 * abs(float(z[i]) - 29.0)
            candidates.append((score, i))
    i_local = max(candidates)[1] if candidates else int(np.argmax(x))
    local = slice(max(0, i_local - half_local), min(n, i_local + half_local + 1))

    half_switch = int(np.clip(0.18 * T2, 80, 260))
    changes = np.where(np.signbit(x[1:]) != np.signbit(x[:-1]))[0] + 1
    scored = []
    for i in changes:
        if i - half_switch < 0 or i + half_switch >= n:
            continue
        a = float(x[i - half_switch])
        b = float(x[i + half_switch])
        if a * b < 0:
            scored.append((abs(a) + abs(b), i))
    if scored:
        i_switch = max(scored)[1]
    elif len(changes):
        i_switch = int(changes[len(changes) // 2])
    else:
        i_switch = n // 2
    switch = slice(max(0, i_switch - half_switch), min(n, i_switch + half_switch + 1))
    return local, switch


def draw_hierarchy(fig, core, T1, T2):
    ax = axes_mm(fig, RIGHT)

    raw = core["data"]
    x = np.asarray(raw["x"][f2.START:f2.END], dtype=float)
    z = np.asarray(raw["z"][f2.START:f2.END], dtype=float)

    step = max(1, len(x) // 4200)
    ax.plot(x[::step], z[::step], color="0.77", lw=0.45, alpha=0.82, zorder=1)

    local, switch = _find_segments(x, z, T1, T2)
    ax.plot(x[local], z[local], color=C_T1, lw=1.8, zorder=4)
    ax.plot(x[switch], z[switch], color=C_T2, lw=1.8, zorder=5)

    for sl, color in ((local, C_T1), (switch, C_T2)):
        inds = np.arange(len(x))[sl]
        if len(inds) > 1:
            i0, i1 = inds[0], inds[-1]
            ax.plot(x[i0], z[i0], "o", ms=2.8, color=color, mec="white", mew=0.45, zorder=6)
            ax.plot(x[i1], z[i1], "o", ms=3.4, color=color, mec="white", mew=0.45, zorder=6)

    xp = np.nanpercentile(x, [0.5, 99.5])
    zp = np.nanpercentile(z, [0.5, 99.5])
    dx = max(float(xp[1] - xp[0]), 1e-9)
    dz = max(float(zp[1] - zp[0]), 1e-9)
    ax.set_xlim(float(xp[0] - 0.14 * dx), float(xp[1] + 0.14 * dx))
    # Small top pad: attractor top lands at ~72.5 mm, level with the map top.
    ax.set_ylim(float(zp[0] - 0.14 * dz), float(zp[1] + 0.08 * dz))
    ax.axis("off")

    FS_HIER = FS_HEAD + 0.7
    ax.text(0.02, 0.135, r"$T_1$  intra-lobe", transform=ax.transAxes,
            ha="left", va="center", color=C_T1, fontsize=FS_HIER, fontweight="semibold")
    ax.text(0.02, 0.025, r"$T_2$  switching", transform=ax.transAxes,
            ha="left", va="center", color=C_T2, fontsize=FS_HIER, fontweight="semibold")


def draw_titles(fig):
    y = 83.0 / H_MM
    left_c = LEFT[0] + 0.5 * LEFT[2]
    mid_c = 0.5 * (MAP[0] + SPROF[0] + SPROF[2])
    right_c = RIGHT[0] + 0.5 * RIGHT[2]
    title_fs = FS_HEAD + 1.0
    fig.text(left_c / W_MM, y, "Local closure", ha="center", va="top",
             fontsize=title_fs, fontweight="semibold")
    fig.text(mid_c / W_MM, y, "Scale scan", ha="center", va="top",
             fontsize=title_fs, fontweight="semibold")
    fig.text(right_c / W_MM, y, "Hierarchy", ha="center", va="top",
             fontsize=title_fs, fontweight="semibold")


# =============================================================================
# Assembly
# =============================================================================
def draw_flow_arrows(fig):
    ax = axes_mm(fig, (0.0, 0.0, W_MM, H_MM), schematic=True)
    ax.set_zorder(-10)

    # Both inter-panel arrows use exactly the same length and vertical position.
    # Their centres are computed from the two equal white gaps.
    flow_y = 43.5
    flow_len = 4.4

    # The two gaps are both 9 mm, but the arrows are intentionally NOT centered.
    # 1->2 is parked near the panel-1 side; 2->3 near the panel-3 side.
    # This is a large, deliberate displacement rather than a subtle nudge.
    left_gap_c = 52.2
    right_gap_c = 110.0

    for xc in (left_gap_c, right_gap_c):
        arrow(ax, (xc - 0.5 * flow_len, flow_y),
              (xc + 0.5 * flow_len, flow_y),
              color="0.55", lw=1.20, ms=7.2, z=1)


def main():
    fam = st.apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    core = f2.build_lorenz_core()
    analysis = f2.analyse_timescales(core)

    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM), facecolor="white")
    draw_titles(fig)
    draw_flow_arrows(fig)
    draw_local_closure(fig)
    T1, T2 = draw_scale_scan(fig, analysis)
    draw_hierarchy(fig, core, T1, T2)

    pdf = FIGURE_DIR / "graphical_abstract.pdf"
    png_hi = FIGURE_DIR / "graphical_abstract.png"
    png_preview = FIGURE_DIR / "graphical_abstract_400x200.png"

    safe_savefig(fig, pdf, format="pdf", facecolor="white")
    dpi_hi = 1200.0 / (W_MM * MM)
    safe_savefig(fig, png_hi, dpi=dpi_hi, facecolor="white")
    dpi_preview = 400.0 / (W_MM * MM)
    safe_savefig(fig, png_preview, dpi=dpi_preview, facecolor="white")

    plt.close(fig)

    print("Graphical abstract complete")
    print("  font:", fam)
    print("  selected scales:", analysis["selected"][:2])
    print("  PDF:", pdf)
    print("  1200x600 PNG:", png_hi)
    print("  400x200 preview:", png_preview)


if __name__ == "__main__":
    main()
