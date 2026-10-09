"""Generate manuscript Fig. 1: mechanism, workflow and synthetic demo.

Panel (a) illustrates why mixing two local rules produces a finite closure
defect.  Panel (b) summarizes the anchor-referenced local-linear workflow.
Panel (c) is a hierarchical two-rule synthetic demo analysed with the same
closure_pipeline.py used for the physical systems; characteristic scales are
selected by the shared pipeline and are not prescribed for the illustration.

Schematic panels are drawn in millimetres (1 data unit = 1 mm).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from pathlib import Path
import tempfile
import shutil
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import figure_style as st
import closure_pipeline as cp
import fig1_synthetic as fs

# Project layout:
#   project_root/
#       closure_pipeline.py
#       figure_style.py
#       fig1_synthetic.py
#       fig1.py
#       data/               # not used by Fig. 1
#       figures/            # created automatically
ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"

W, H = 174.0, 110.0
A_BOX = (0.0, 60.0, 54.0, 50.0)       # x, y, w, h (mm)
B_BOX = (59.0, 60.0, 115.0, 50.0)
C_BOX = (0.0, 0.0, 174.0, 55.0)

Q_ROWS, R_ROWS = 8, 3
ROW = 1.5          # mm per coordinate row
P_W = 34.0         # mm for the p time samples (H and Y)
T_W = P_W / 4      # mm for a window of T samples
ZOOM = 2.0

C_MAT = "#EFEFEF"
C_EDGE = "0.25"
C_TRACE = "0.35"
C_RFILL = "#F6DCE0"


# =============================================================================
# Drawing helpers
# =============================================================================
def axes_mm(fig, box, schematic=False):
    x, y, w, h = box
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    if schematic:
        ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")
    return ax


def txt(ax, x, y, s, **kw):
    kw.setdefault("fontsize", st.FS_TEXT); kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, **kw)


def arrow(ax, p0, p1, color="0.35", lw=0.8, ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0))


def rect(ax, x, y, w, h, fc=C_MAT, ec=C_EDGE, lw=0.7, z=1):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))


def matrix(ax, x, y, w, nrows, row_h, f, color=C_TRACE, fc=C_MAT, amp=0.30, lw_tr=0.55):
    """Matrix drawn as stacked row signals; row 0 at the top."""
    rect(ax, x, y, w, nrows * row_h, fc=fc, ec="none", z=1)
    u = np.linspace(0, 1, 220)
    for i in range(nrows):
        yc = y + (nrows - i - 0.5) * row_h
        ax.plot(x + 0.6 + (w - 1.2) * u, yc + amp * row_h * f(i, u),
                color=color, lw=lw_tr, zorder=2, solid_capstyle="round")
        if i > 0:
            ax.plot([x + 0.3, x + w - 0.3], [y + (nrows - i) * row_h] * 2,
                    color="white", lw=0.6, zorder=2)
    rect(ax, x, y, w, nrows * row_h, fc="none", ec=C_EDGE, lw=0.7, z=3)


def panel_letter(fig, x, y_top, s):
    fig.text(x / W, y_top / H, s, fontsize=st.FS_PANEL, fontweight="bold",
             ha="left", va="top")


# =============================================================================
# Panel (a): two-rule mixing mechanism
# =============================================================================
def panel_a(fig):
    ax = axes_mm(fig, A_BOX, schematic=True)
    xl, xr = 4.0, 47.0
    xc = 0.5 * (xl + xr)                       # rule switch on the time line
    tau = 12.0                                 # window width (mm)
    centres = (xl + 1.5 + tau / 2, xc, xr - 1.5 - tau / 2)

    # ---- state-space view of each window.
    # Every column shows the SAME orbit: the rules differ in the velocity they
    # assign to a state, not in where the state is.  A = omega J, so the local
    # velocity is tangent to the orbit; its length is the rule's speed.
    cy, R = 37.5, 6.0
    L1, L2 = 3.4, 7.0                          # A1 slow, A2 fast
    GAP, OFF = 0.55, 0.85                      # clearance from orbit; pair separation
    # Base states: top and bottom of the orbit; counter-clockwise tangents.
    bases = (((0.0, R), (-1.0, 0.0), (0.0, 1.0)),
             ((0.0, -R), (1.0, 0.0), (0.0, -1.0)))
    col_rules = (((st.C_A1, L1, GAP),),
                 ((st.C_A2, L2, GAP + OFF), (st.C_A1, L1, GAP)),
                 ((st.C_A2, L2, GAP),))
    for c, rules in zip(centres, col_rules):
        ax.add_patch(Circle((c, cy), R, fc="none", ec="0.62", lw=0.9, zorder=2))
        for (bx, by), (tx, ty), (nx, ny) in bases:
            for color, L, off in rules:            # magenta first, blue on top
                x0, y0 = c + bx + off * nx, cy + by + off * ny
                arrow(ax, (x0, y0), (x0 + L * tx, y0 + L * ty),
                      color=color, lw=1.4, ms=6.0)
            ax.plot(c + bx, cy + by, "o", ms=2.0, color="0.30", mew=0, zorder=5)

    # ---- time line with one switch at the centre
    ys, hs = 18.0, 3.0
    rect(ax, xl, ys, xc - xl, hs, fc=st.C_A1, ec="none")
    rect(ax, xc, ys, xr - xc, hs, fc=st.C_A2, ec="none")

    # three equal windows
    yb = ys + hs + 0.8
    for c in centres:
        ax.plot([c - tau / 2, c - tau / 2, c + tau / 2, c + tau / 2],
                [yb, yb + 1.7, yb + 1.7, yb], color="0.1", lw=0.9)
    for c, lab, col in zip(centres, (r"$A_1$", "mixed", r"$A_2$"),
                           (st.C_A1, "0.28", st.C_A2)):
        txt(ax, c, yb + 2.85, lab, fontsize=7.8, color=col)

    # ---- closure defect of each window, at the window centre.
    # beta is a root-SSE quantity, so for two-rule mixing beta ~ sqrt(p(1-p)):
    # a half-ellipse with vertical tangents where the window first touches
    # the switch.
    base, amp = 4.0, 8.0
    pfrac = lambda c: np.clip((xc - (c - tau / 2)) / tau, 0, 1)
    prof = lambda c: base + 0.35 + amp * 2.0 * np.sqrt(pfrac(c) * (1 - pfrac(c)))
    cc = np.r_[np.linspace(xl + tau / 2, xc - tau / 2, 60),
               xc - tau / 2 + 0.5 * tau * (1 - np.cos(np.linspace(0, np.pi, 400))),
               np.linspace(xc + tau / 2, xr - tau / 2, 60)]
    ax.plot([xl, xr], [base, base], color="0.55", lw=0.6)
    arrow(ax, (xr, base), (xr + 3.5, base), color="0.55", lw=0.6, ms=5)
    txt(ax, xr + 5.2, base, r"$t$")
    ax.plot(cc, prof(cc), color=st.C_BETA, lw=1.2, solid_joinstyle="round")
    for c in centres:
        yv = prof(c)
        ax.plot([c, c], [yv + 0.9, ys - 0.3], color="0.6", lw=0.5, ls=(0, (1, 1.2)))
        ax.plot(c, yv, "o", ms=3.4, color=st.C_BETA, mew=0)
    txt(ax, xl - 1.8, base + 5.0, r"$\beta$")


# =============================================================================
# Panel (b): analysis workflow
# =============================================================================
def panel_b(fig):
    ax = axes_mm(fig, B_BOX, schematic=True)

    # ---- upper tier: observations -> H -> SVD -> Y
    yu = 37.0
    u = np.linspace(0, 1, 200)
    rng_obs = np.random.default_rng(21)
    for k, dy in enumerate((3.2, 0.0, -3.2)):
        backbone = np.sin(2 * np.pi * (2.6 * u + 0.23 * k)) \
            * (0.72 + 0.24 * np.cos(2 * np.pi * (0.9 * u + 0.10 * k)))
        fine = 0.11 * np.sin(2 * np.pi * ((10.0 + 0.8 * k) * u + 0.17 * k))
        eta = rng_obs.standard_normal(len(u))
        eta = np.convolve(eta, np.ones(7) / 7.0, mode="same")
        eta /= max(np.std(eta), 1e-12)
        obs = backbone + fine + 0.10 * eta
        ax.plot(1.0 + 11.0 * u, yu + dy + 0.92 * obs,
                color="0.2", lw=0.72, solid_capstyle="round")
    txt(ax, 6.5, yu + 6.2, r"$\mathbf{x}(t)$", fontsize=st.FS_TEXT)
    arrow(ax, (13.0, yu), (16.2, yu))

    xH, hH = 20.2, Q_ROWS * ROW
    rng_h = np.random.default_rng(31)
    uh = np.linspace(0, 1, 220)
    hnoise = np.empty((Q_ROWS, len(uh)))
    for i in range(Q_ROWS):
        e = rng_h.standard_normal(len(uh))
        e = np.convolve(e, np.ones(9) / 9.0, mode="same")
        e /= max(np.std(e), 1e-12)
        hnoise[i] = e

    def fH(i, v):
        base = np.sin(2 * np.pi * (2.6 * v - 0.06 * i)) \
            * (0.75 + 0.25 * np.cos(2 * np.pi * (0.9 * v - 0.04 * i)))
        fine = 0.07 * np.sin(2 * np.pi * ((9.0 + 0.35 * i) * v + 0.11 * i))
        irr = 0.07 * np.interp(v, uh, hnoise[i])
        return base + fine + irr

    matrix(ax, xH, yu - hH / 2, P_W, Q_ROWS, ROW, fH)
    txt(ax, xH + P_W / 2, yu + hH / 2 + 2.3, r"$H$")

    xa0, xa1 = xH + P_W + 1.2, xH + P_W + 8.8
    arrow(ax, (xa0, yu), (xa1, yu))
    txt(ax, 0.5 * (xa0 + xa1), yu + 2.3, "SVD", fontsize=st.FS_TICK)

    xY, hY = xa1 + 4.0, R_ROWS * ROW
    freqs = (2.6, 1.3, 3.9)
    fY = lambda i, v: np.sin(2 * np.pi * (freqs[i] * v + 0.15 * i))
    matrix(ax, xY, yu - hY / 2, P_W, R_ROWS, ROW, fY)
    txt(ax, xY - 2.0, yu, r"$r$")
    txt(ax, xY + P_W / 2, yu + hY / 2 + 2.3, r"$Y$")
    xw = xY + 1.5                                   # window of T columns
    rect(ax, xw, yu - hY / 2, T_W, hY, fc="none", ec="0.05", lw=1.1, z=5)
    # The window slides along Y (single-headed: motion, not a dimension).
    arrow(ax, (xw + 1.0, yu + hY / 2 + 1.1), (xw + T_W + 3.5, yu + hY / 2 + 1.1),
          color="0.35", lw=0.7, ms=5)
    txt(ax, xw + T_W / 2, yu - hY / 2 - 2.2, r"$T$")

    # ---- lower tier: dV_t = A_t V_t + R_t  -> beta   (window magnified)
    # The window cut out of Y is V_t; its velocity dV_t is explained by one
    # linear rule A_t through the origin, and R_t is what that rule misses.
    yl = 12.0
    hz, wz, wa = hY * ZOOM, T_W * ZOOM, hY * ZOOM
    y0 = yl - hz / 2
    xs = 11.0
    xdv, xA = xs, xs + wz + 4.5
    xV = xA + wa + 1.0
    xR = xV + wz + 4.5
    u0, u1 = (xw - xY) / P_W, (xw - xY + T_W) / P_W
    win = lambda i, v: fY(i, u0 + (u1 - u0) * v)
    dwin = lambda i, v: np.cos(2 * np.pi * (freqs[i] * (u0 + (u1 - u0) * v) + 0.15 * i))
    matrix(ax, xdv, y0, wz, R_ROWS, ROW * ZOOM, dwin, amp=0.28)
    txt(ax, xdv + wz + 2.25, yl, "=")
    # A_t: fitted operator, r x r, no time axis.  White grid with Hinton
    # squares (side ~ |coefficient|) so it reads as a set of fitted numbers,
    # distinct from the grey data matrices.
    rect(ax, xA, y0, wa, hz, fc="white", ec="none")
    for k in range(1, R_ROWS):
        ax.plot([xA + k * wa / R_ROWS] * 2, [y0, y0 + hz], color="0.78", lw=0.45)
        ax.plot([xA, xA + wa], [y0 + k * hz / R_ROWS] * 2, color="0.78", lw=0.45)
    A_MAG = np.array([[0.15, 0.90, 0.35],         # rotation-like generator:
                      [0.90, 0.10, 0.60],         # strong off-diagonal pairs,
                      [0.35, 0.60, 0.20]])        # weak diagonal
    cw, ch = wa / R_ROWS, hz / R_ROWS
    for i in range(R_ROWS):
        for j in range(R_ROWS):
            s = 0.80 * cw * np.sqrt(A_MAG[i, j])  # area ~ |a_ij|
            xcen = xA + (j + 0.5) * cw
            ycen = y0 + hz - (i + 0.5) * ch
            ax.add_patch(Rectangle((xcen - s / 2, ycen - s / 2), s, s,
                                   fc="0.38", ec="none", zorder=2))
    rect(ax, xA, y0, wa, hz, fc="none", ec=C_EDGE, lw=0.7, z=3)
    matrix(ax, xV, y0, wz, R_ROWS, ROW * ZOOM, win, amp=0.28)
    txt(ax, xV + wz + 2.25, yl, "+")
    # R_t is the structured component left unexplained by the single fitted rule.
    V_SW, W_SW = 0.55, 0.045
    def fR(i, v):
        env = -0.45 + 1.0 / (1.0 + np.exp(-(v - V_SW) / W_SW))
        return 1.65 * env * dwin(i, v)
    matrix(ax, xR, y0, wz, R_ROWS, ROW * ZOOM, fR, color=st.C_BETA, fc=C_RFILL)
    for x, w_, lab in ((xdv, wz, r"$\dot V_t$"), (xA, wa, r"$A_t$"),
                       (xV, wz, r"$V_t$"), (xR, wz, r"$R_t$")):
        txt(ax, x + w_ / 2, y0 - 2.4, lab)

    # Magnification lines: the window in Y is V_t.
    for (xa, xb) in ((xw, xV), (xw + T_W, xV + wz)):
        ax.plot([xa, xb], [yu - hY / 2, y0 + hz], color="0.55", lw=0.5, zorder=0)

    # beta(t) is the row-averaged root-SSE of R_t, one value per window
    # position; sliding the window turns it into a time trace.
    xb0 = xR + wz + 1.2
    arrow(ax, (xb0, yl), (xb0 + 4.0, yl))
    xb = xb0 + 5.2
    tt = np.linspace(0, 1, 120)
    bb = 0.2 + 0.45 * np.exp(-((tt - 0.3) / 0.04) ** 2) + np.exp(-((tt - 0.7) / 0.08) ** 2)
    ax.plot(xb + 13.0 * tt, y0 + 0.3 + (hz - 0.6) * (bb - bb.min()) / (bb.max() - bb.min()),
            color=st.C_BETA, lw=1.1)
    # Time axis drawn as in panel (a), so the output reads as beta(t), a trace
    # over window positions, rather than as a spectrum.
    arrow(ax, (xb, y0), (xb + 15.5, y0), color="0.55", lw=0.6, ms=5)
    txt(ax, xb + 17.2, y0, r"$t$")
    txt(ax, xb + 6.5, y0 - 2.4, r"$\beta$")


# =============================================================================
# Panel (c): hierarchical synthetic demo
# =============================================================================
def panel_c(fig, sig, res):
    """Scale-resolved synthetic demo, using the same visual grammar as Lorenz.

    (c1) D(t,T) on the fine log-T grid: where in time each window size fails
         to close.
    (c2) S(T) on the baseline grid: how fast this temporal organization changes
         with T.  Dips are reorganizations; the guides mark the selected scales.
    (c3) beta(t) at T1 and T2 with the observed x1(t) for orientation.
    All quantities come from closure_pipeline.analyze_timescales.
    """
    _, y0, _, _ = C_BOX
    yb, ht = 9.0, 43.0
    Ts, d = res["Ts"], res["diag"]
    map_Ts = res.get("display_Ts", Ts)
    map_data = res.get("display_scale_map", res["scale_map"])
    sel = res["selected"]

    # (c1) Scale-time map.  One global display factor for the whole map, so
    # that levels at different T remain comparable by eye.
    ax_map = axes_mm(fig, (10.0, y0 + yb, 54.5, ht))
    mesh = ax_map.pcolormesh(
        res["common_time"], map_Ts, cp.globally_scaled_map(map_data),
        shading="auto", rasterized=True,
    )
    ax_map.set_title("Synthetic demo", loc="left", fontsize=st.FS_TEXT, pad=3.0)
    ax_map.set_yscale("log")
    ax_map.set_xlim(0, fs.T_TOTAL)
    ax_map.set_ylim(map_Ts[0], map_Ts[-1])
    ax_map.set_xticks([0, 10, 20, 30])
    for T in sel:
        ax_map.axhline(T, color="white", lw=st.LW_GUIDE, ls=(0, (3, 2)))
    ax_map.set_xlabel(r"$t$")
    ax_map.set_ylabel(r"Window size $T$")

    cax = axes_mm(fig, (66.6, y0 + yb, 1.6, ht))
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=1.8, width=0.5)
    cb.set_label(r"$\tilde{D}_{\mathrm{sample}}$", labelpad=1.5)

    # (c2) Neighbouring-scale similarity.  Raw values are deliberately faint;
    # the smoothed curve is the object used for selecting the scales.
    ax_s = axes_mm(fig, (88.8, y0 + yb, 28.7, ht))
    ax_s.plot(Ts, d["raw"], ls="none", marker="o", ms=1.6,
              markeredgewidth=0, color=st.C_SRAW, alpha=0.28)
    ax_s.plot(np.exp(d["log_grid"]), d["smooth"], color=st.C_SSMOOTH, lw=1.15)

    # Show enough range beyond T2 to make the later behaviour visible; this is
    # display-only and does not enter scale selection.
    xmax = min(float(Ts[-1]), 4.0 * float(max(sel)))
    ax_s.set_xscale("log")
    ax_s.set_xlim(float(Ts[0]), xmax)
    vis = Ts <= xmax
    sv = d["smooth_actual"][vis]
    sv = sv[np.isfinite(sv)]
    yr = max(float(sv.max() - sv.min()), 1e-4)
    ax_s.set_ylim(float(sv.min()) - 0.20 * yr, float(sv.max()) + 0.28 * yr)
    lo, hi = ax_s.get_ylim()
    yt = [v for v in MaxNLocator(nbins=4).tick_values(lo, hi)
          if lo <= v <= min(1.0, hi)]
    ax_s.set_yticks(yt)

    # Selected scales need not coincide with a displayed raw S(T) point, so
    # marker heights are always read from the smoothed log-T curve.
    for j, T in enumerate(sel, start=1):
        sval = float(np.interp(np.log(T), d["log_grid"], d["smooth"]))
        ax_s.axvline(T, ymin=0.0, ymax=0.90, ls="-.", lw=st.LW_GUIDE, color=st.C_SEL, alpha=0.85)
        ax_s.plot(T, sval, "o", ms=3.0, color=st.C_SEL)
        x_text, ha = (T * 0.985, "right") if j == 1 else (T * 1.015, "left")
        ax_s.text(x_text, 0.985, f"$T_{j}$", transform=ax_s.get_xaxis_transform(),
                  va="top", ha=ha, fontsize=st.FS_TEXT, color=st.C_SEL)

    ax_s.set_xlabel(r"Window size $T$")
    ax_s.set_ylabel(r"$S(T)$")

    # (c3) Closure signals at the selected scales.  Each beta trace uses its
    # own vertical range because beta amplitude depends on observation scale;
    # the observed x1(t) keeps a common right-axis range for orientation.
    x1 = sig["X"][0]

    for k, ((tb, beta), top) in enumerate(zip(res["traces"], (True, False))):
        axb = axes_mm(fig, (133.0, y0 + (32.4 if top else yb), 33.8, 19.4))
        axr = axb.twinx()

        # Reference observable: secondary information, lighter than beta.
        axr.plot(sig["t"], x1, color=st.C_REF, lw=0.48, alpha=0.38, zorder=1)
        axr.set_ylim(-1.08, 1.08)
        axr.set_yticks([-1, 0, 1])
        axr.tick_params(axis="y", colors=st.C_REF, labelsize=st.FS_TICK,
                        length=2.0, pad=1.5)
        axr.spines["right"].set_visible(True)
        axr.spines["right"].set_color(st.C_REF)
        axr.spines["right"].set_linewidth(st.LW_AXIS)
        axr.set_ylabel(r"$x_1$", color=st.C_REF, labelpad=0.4)
        axr.spines["top"].set_visible(False)
        axr.spines["left"].set_visible(False)

        # Closure defect: primary information.
        axb.plot(tb, beta, color=st.C_BETA, lw=1.05, alpha=0.95, zorder=3)
        axb.set_zorder(axr.get_zorder() + 1)
        axb.patch.set_visible(False)
        axb.set_xlim(0, fs.T_TOTAL)
        axb.set_xticks([0, 10, 20, 30])

        finite_beta = np.asarray(beta, dtype=float)
        finite_beta = finite_beta[np.isfinite(finite_beta)]
        if finite_beta.size:
            blo = float(np.min(finite_beta))
            bhi = float(np.max(finite_beta))
            bspan = max(bhi - blo, 1e-12)
            axb.set_ylim(blo - 0.10 * bspan, bhi + 0.10 * bspan)
        axb.yaxis.set_major_locator(MaxNLocator(nbins=3))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        axb.yaxis.set_major_formatter(formatter)
        axb.tick_params(axis="y", colors=st.C_BETA, labelsize=st.FS_TICK,
                        length=2.0, pad=1.5)
        axb.yaxis.get_offset_text().set_color(st.C_BETA)
        axb.yaxis.get_offset_text().set_fontsize(st.FS_TICK)
        axb.spines["left"].set_color(st.C_BETA)
        axb.set_ylabel(r"$\beta$", color=st.C_BETA, labelpad=1.0)
        axb.text(0.025, 0.955, f"$T_{k + 1}$", transform=axb.transAxes,
                 ha="left", va="top", fontsize=st.FS_TEXT,
                 color=st.C_SEL, zorder=6)

        if top:
            axb.set_xticklabels([])
        else:
            axb.set_xlabel(r"$t$")



def safe_savefig(fig, target, **kwargs):
    """Render to a temporary file and then copy the completed output."""
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"

    with tempfile.TemporaryDirectory(prefix="closure_fig1_") as tmp_dir:
        tmp = Path(tmp_dir) / f"render{suffix}"
        fig.savefig(tmp, **kwargs)
        try:
            shutil.copy2(tmp, target)
        except OSError as exc:
            print(f"[WARNING] Temporary render succeeded but copy failed: {exc}")
            print(f"          Trying direct save to: {target}")
            fig.savefig(target, **kwargs)
    return target

# =============================================================================
# Output and main figure
# =============================================================================

def save_summary(path, res):
    """Save the representation and selected synthetic characteristic scales."""
    selected = list(res["selected"])
    events = list(res["selected_events"])
    if len(selected) < 2 or len(events) < 2:
        raise RuntimeError(f"Fig. 1 requires two characteristic scales, got {selected}.")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "system,rank,rank_rule,t_scan_min,t_scan_max,"
            "T1,T1_depth,T1_prominence,T1_d_org,"
            "T2,T2_depth,T2_prominence,T2_d_org\n"
        )
        f.write(
            f"synthetic_demo,{res['rank']},cumulative_singular_values_0.98,"
            f"{res['t_min']},{res['t_max']},"
            f"{selected[0]},{events[0]['depth']},{events[0]['prominence']},{events[0]['d_org']},"
            f"{selected[1]},{events[1]['depth']},{events[1]['prominence']},{events[1]['d_org']}\n"
        )


def main():
    fam = st.apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    sig = fs.make_signal()
    res = fs.analyse(sig)
    if len(res["selected"]) < 2:
        raise RuntimeError(f"Fig. 1 requires two characteristic scales, got {res['selected']}.")
    save_summary(FIGURE_DIR / "fig1_summary.csv", res)

    fig = plt.figure(figsize=(W * st.MM, H * st.MM))
    panel_a(fig)
    panel_b(fig)
    panel_c(fig, sig, res)
    panel_letter(fig, A_BOX[0], A_BOX[1] + A_BOX[3], "(a)")
    panel_letter(fig, B_BOX[0], B_BOX[1] + B_BOX[3], "(b)")
    panel_letter(fig, C_BOX[0], C_BOX[1] + C_BOX[3], "(c)")

    pdf_path = FIGURE_DIR / "fig1.pdf"
    png_path = FIGURE_DIR / "fig1.png"
    safe_savefig(fig, pdf_path, format="pdf")
    safe_savefig(fig, png_path, dpi=300)
    plt.close(fig)

    print("Fig. 1 complete")
    print("  font:", fam)
    print("  retained rank:", res["rank"])
    print("  selected T:", res["selected"])
    print("  PDF:", pdf_path)
    print("  preview:", png_path)


if __name__ == "__main__":
    main()
