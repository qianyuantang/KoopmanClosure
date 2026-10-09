"""Supplementary Fig. S1: controlled two-regime validation of local closure response.

A planar rotation on the unit circle switches its angular velocity from w1 to
w2 at t = 0.  The state stays on the circle and its first and second moments
are the same on both sides; only the rule changes.  The closure field is
computed with the shared closure_pipeline exactly as in the main figures: one
linear rule through the anchor (here the rotation centre, i.e. the origin) is
fitted in every window of T samples, and the per-sample residual energy

    E = ||R||_F^2 / T = r D_sample,        r = 2,

is recorded.  The mixed-window relation predicts

    E(p) ~= p (1 - p) ||A1 - A2||_C^2 = p (1 - p) dw^2,      C = I/2,

where p is the fraction of the window lying in regime 1.  Three predictions
are tested separately:

  (a) mixing   : E / dw^2 collapses onto p (1 - p) for different contrasts;
  (b) contrast : max_p E = dw^2 / 4 over a wide range of dw;
  (c) timing   : the peak, indexed by window start, lies at -tau/2, so placing
                 each window at its centre (the pipeline convention
                 t_start + T dt / 2) recovers the switch at t = 0.

Run
---
    python figS1_two_regime.py

Outputs
-------
    figures/figS1_two_regime.pdf
    figures/figS1_two_regime.png
The numbers quoted in the SI text are printed to the console.
"""

from pathlib import Path
import shutil
import tempfile

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import closure_pipeline as cp
import figure_style as st

ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
STEM = "figS1_two_regime"

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
DT = 1e-3                                             # sampling interval
W1 = 2.0 * np.pi                                      # angular velocity for t < 0
TAU_MAIN = 4.0                                        # window duration T dt (T = 4000)
DW_COLLAPSE = np.pi * np.array([0.5, 1.0, 2.0, 4.0])  # panel (a)
DW_SCAN = np.pi * 2.0 ** (np.arange(-6, 7) / 2)       # panel (b): pi/8 ... 8 pi
DW_TIMING = np.pi                                     # panel (c)
TAU_SCAN = np.arange(1.0, 9.0, 1.0)                   # panel (c)
MARGIN = 0.5                                          # extra record on each side


# -----------------------------------------------------------------------------
# Model and closure
# -----------------------------------------------------------------------------
def rotation_trajectory(w1, w2, tmin, tmax, derivative="exact"):
    """Unit-circle rotation with angular velocity w1 (t < 0) and w2 (t >= 0).

    Time is built on an integer grid so that t = 0 is sampled exactly.  The
    velocity is either exact or the second-order central difference used for
    the numerically differentiated systems (Rössler, ADK).
    """
    t = np.arange(int(round(tmin / DT)), int(round(tmax / DT))) * DT
    theta = np.where(t < 0, w1 * t, w2 * t)
    V = np.vstack([np.cos(theta), np.sin(theta)])
    if derivative == "exact":
        Vdot = np.where(t < 0, w1, w2) * np.vstack([-np.sin(theta), np.cos(theta)])
    else:
        Vdot = np.gradient(V, DT, axis=1, edge_order=2)
    return t, V, Vdot


def sliding_energy(w1, w2, tau, derivative="exact"):
    """Window start times, regime-1 fractions p and energies E across the switch.

    The record spans the switch with tau + MARGIN on each side, so that every
    composition from p = 1 to p = 0 is visited.  Windows start at samples
    0, ..., n-T-1, as in the pipeline.
    """
    t, V, Vdot = rotation_trajectory(w1, w2, -tau - MARGIN, tau + MARGIN, derivative)
    T = int(round(tau / DT))
    _, D = cp.compute_metrics_for_T(*cp.prepare_window_sums(V, Vdot), T)
    starts = np.arange(len(D))
    # Number of regime-1 samples in [k, k + T): those before index n0 = #(t < 0).
    n0 = int(np.count_nonzero(t < 0))
    p = np.clip(n0 - starts, 0, T) / T
    return t[starts], p, V.shape[0] * D


# -----------------------------------------------------------------------------
# Computations
# -----------------------------------------------------------------------------
def compute(derivative="exact"):
    res = {}

    # (a) mixing: E / dw^2 against p for several contrasts
    res["a"] = []
    for dw in DW_COLLAPSE:
        _, p, E = sliding_energy(W1, W1 + dw, TAU_MAIN, derivative)
        res["a"].append((dw, p, E / dw ** 2))

    # (b) contrast: peak energy against dw at fixed window length
    res["b"] = (DW_SCAN, np.array([
        sliding_energy(W1, W1 + dw, TAU_MAIN, derivative)[2].max() for dw in DW_SCAN
    ]))

    # (c) timing: peak location, raw (window start) and at the window centre
    t_peak = []
    for tau in TAU_SCAN:
        t_start, _, E = sliding_energy(W1, W1 + DW_TIMING, tau, derivative)
        t_peak.append(t_start[np.argmax(E)])
    t_peak = np.array(t_peak)
    res["c"] = (TAU_SCAN, t_peak, t_peak + TAU_SCAN / 2)
    return res


def report(res, label):
    """The three numbers quoted in the SI text."""
    dev_a = max(np.max(np.abs(y - p * (1 - p))) for _, p, y in res["a"]) / 0.25
    bound = all(np.all(y <= p * (1 - p) + 1e-12) for _, p, y in res["a"])
    dw, Emax = res["b"]
    dev_b = np.max(np.abs(Emax / (dw ** 2 / 4) - 1))
    tau, raw, centre = res["c"]
    print(f"{label}")
    print(f"  (a) max |E/dw^2 - p(1-p)| / 0.25 = {dev_a:.4f};  all on or below bound: {bound}")
    print(f"  (b) max |E_max / (dw^2/4) - 1|   = {dev_b:.5f}")
    print(f"  (c) max |t_peak + tau/2|         = {np.max(np.abs(raw + tau / 2)):.4g};"
          f"  max |t_event|: {np.max(np.abs(centre)):.4g}")


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
W_MM, H_MM = 174.0, 56.0
PANEL_X = (14.0, 72.0, 130.0)
PANEL_Y, PANEL_W, PANEL_H = 12.0, 40.0, 37.0


def axes_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])


def safe_savefig(fig, target, **kwargs):
    """Render to a temporary path, then copy to the figure directory."""
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"
    with tempfile.TemporaryDirectory(prefix="closure_figS1_") as tmp_dir:
        tmp = Path(tmp_dir) / f"render{suffix}"
        fig.savefig(tmp, **kwargs)
        try:
            shutil.copy2(tmp, target)
        except OSError:
            fig.savefig(target, **kwargs)
    return target


def paper_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def add_panel_letter(fig, x_mm, y_mm, label):
    fig.text(
        x_mm / W_MM,
        y_mm / H_MM,
        label,
        fontsize=st.FS_PANEL,
        fontweight="bold",
        ha="left",
        va="top",
    )


def make_figure(res):
    st.apply_style()
    colors = mpl.colormaps["Blues"](np.linspace(0.45, 0.95, len(DW_COLLAPSE)))
    mk = dict(marker="o", ms=3.2, ls="none", mew=0, zorder=3)

    fig = plt.figure(figsize=(W_MM * st.MM, H_MM * st.MM), facecolor="white")
    axs = [axes_mm(fig, x, PANEL_Y, PANEL_W, PANEL_H) for x in PANEL_X]
    legend_style = dict(frameon=False, fontsize=st.FS_TICK, handlelength=1.4,
                        handletextpad=0.4, borderaxespad=0.3, labelspacing=0.25)
    dw_labels = (r"\pi/2", r"\pi", r"2\pi", r"4\pi")

    # (a) Mixing: rescaled energy against window composition.  Symbols are
    # thinned to about ten per contrast, offset slightly between contrasts.
    ax = axs[0]
    pp = np.linspace(0, 1, 400)
    ax.plot(pp, pp * (1 - pp), color="k", lw=1.0, zorder=2)
    grid = np.arange(0.0, 1.0001, 0.1)
    for k, (dw, p, y) in enumerate(res["a"]):
        idx = np.unique([np.argmin(np.abs(p - g)) for g in np.clip(grid + 0.025 * k, 0, 1)])
        ax.plot(p[idx], y[idx], color=colors[k], **mk)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.28)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.1, 0.2], ["0", "0.1", "0.2"])
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$E\,/\,\Delta\omega^{2}$")
    # Legend inside the arch, where the panel is empty.
    handles = [Line2D([0], [0], color=colors[k], **mk, label=rf"$\Delta\omega={lab}$")
               for k, lab in enumerate(dw_labels)]
    handles.append(Line2D([0], [0], color="k", lw=1.0, label=r"$p(1-p)$"))
    ax.legend(handles=handles, loc="lower center", **legend_style)

    # (b) Contrast: peak energy against dw; the contrasts of (a) in colour.
    ax = axs[1]
    dw, Emax = res["b"]
    xx = np.geomspace(dw.min() / 1.4, dw.max() * 1.4, 200)
    ax.plot(xx / np.pi, xx ** 2 / 4, color="k", lw=1.0, zorder=2)
    highlighted = [int(np.argmin(np.abs(dw - d))) for d in DW_COLLAPSE]
    others = np.setdiff1d(np.arange(len(dw)), highlighted)
    ax.plot(dw[others] / np.pi, Emax[others], color="0.55", **mk)
    for k, i in enumerate(highlighted):
        ax.plot(dw[i] / np.pi, Emax[i], color=colors[k], **mk)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1 / 8, 1 / 2, 2, 8])
    ax.set_xticklabels([r"$1/8$", r"$1/2$", r"$2$", r"$8$"])
    ax.set_xlabel(r"$\Delta\omega/\pi$")
    ax.set_ylabel(r"$E_{\max}$")
    handles = [
        Line2D([0], [0], color=colors[1], **mk, label="contrasts of (a)"),
        Line2D([0], [0], color="0.55", **mk, label="other contrasts"),
        Line2D([0], [0], color="k", lw=1.0, label=r"$\Delta\omega^{2}/4$"),
    ]
    ax.legend(handles=handles, loc="upper left", **legend_style)

    # (c) Timing: open symbols at the window start, filled at the window centre.
    ax = axs[2]
    tau, raw, centre = res["c"]
    tt = np.array([0.5, 8.5])
    ax.plot(tt, -tt / 2, color="k", ls=(0, (3, 2)), lw=0.8, zorder=2)
    ax.plot(tt, 0 * tt, color="k", lw=0.8, zorder=2)
    c = colors[1]                       # same colour as dw = pi in (a)
    ax.plot(tau, raw, marker="o", ms=3.6, ls="none", mfc="white", mec=c, mew=0.8, zorder=3)
    ax.plot(tau, centre, color=c, **{**mk, "ms": 3.6})
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(-5.2, 0.6)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_yticks([-4, -2, 0])
    ax.set_xlabel(r"$\tau_T$")
    ax.set_ylabel(r"$t_{\mathrm{peak}},\;t_{\mathrm{event}}$")
    handles = [
        Line2D([0], [0], marker="o", ms=3.6, ls="none", mfc="white", mec=c, mew=0.8,
               label=r"$t_{\mathrm{peak}}$ (window start)"),
        Line2D([0], [0], marker="o", ms=3.6, ls="none", mew=0, color=c,
               label=r"$t_{\mathrm{event}}$ (window centre)"),
        Line2D([0], [0], color="k", ls=(0, (3, 2)), lw=0.8, label=r"$-\tau_T/2$"),
    ]
    ax.legend(handles=handles, loc="lower left", **legend_style)

    for ax in axs:
        paper_axes(ax)
    for x, label in zip(PANEL_X, ("(a)", "(b)", "(c)")):
        add_panel_letter(fig, x - 12.0, H_MM - 1.0, label)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{STEM}.pdf"
    png_path = FIGURE_DIR / f"{STEM}.png"
    safe_savefig(fig, pdf_path, format="pdf", facecolor="white")
    safe_savefig(fig, png_path, dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    res = compute("exact")
    report(res, "exact derivatives")
    report(compute("central"), "central differences (np.gradient, as for Rössler and ADK)")
    make_figure(res)
