"""Generate manuscript Fig. 3: Kuramoto--Sivashinsky scale-resolved closure dynamics.

Expected project layout
-----------------------
project_root/
├── closure_pipeline.py
├── figure_style.py
├── fig1_synthetic.py
├── fig1.py
├── fig2_lorenz.py
├── fig3_ks.py
└── data/
    └── ks_raw_data.npz

Run
---
    python fig3_ks.py

Outputs
-------
    figures/fig3_ks.pdf
    figures/fig3_ks.png
    figures/fig3_ks_summary.csv

Method
------
This script only builds the KS observable representation and the state-space /
reference overlays; everything else comes from closure_pipeline.py.

Representation.  The observable is the vector of the 16 odd Fourier amplitudes
a_k of the truncated KS system.  Coordinates are measured from a = 0, the
trivial (flat) solution u = 0, which is a fixed point of the Galerkin system
and anchors the local linear closure; H is therefore not centred.  The rank
follows the shared rule on the singular values of H, and the exact velocity
from the Galerkin right-hand side is projected with the same U_r.

Closure field, S(T), reorganization events and characteristic scales come
from closure_pipeline.analyze_timescales.  The normal and sticky intervals are
fixed in advance for illustration and never enter the scale selection.
"""

from pathlib import Path
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator, FuncFormatter, NullFormatter

import closure_pipeline as cp
import figure_style as st


# =============================================================================
# Project layout
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
RAW_FILE = DATA_DIR / "ks_raw_data.npz"


# =============================================================================
# KS analysis settings
# =============================================================================
ANALYSIS_START = 1000
ANALYSIS_END = 20000
N_MODES = 16
RANK_FRACTION = cp.RANK_CUMULATIVE_SINGULAR_VALUE
NU_DEFAULT = 0.028509        # used only if the data file does not store it

# Observation-scale scan in samples, up to a quarter of the analysed record.
T_SCAN_MIN = 50
T_SCAN_MAX_FRACTION = 0.25
MAX_SELECTED_SCALES = 2

# Fixed representative intervals used only for state-space and trace annotations.
# They never enter the closure calculation or characteristic-scale selection.
NORMAL_INTERVAL = (38.5, 42.0)
STICKY_INTERVAL = (43.0, 49.0)

# System-specific display windows: the long-scale trace shows the coarse
# trajectory-scale organization, while the short-scale trace resolves the local
# interval containing the normal and sticky examples.
LONG_VIEW = (10.0, 200.0)
SHORT_VIEW = (18.0, 62.0)
LOCAL_INSPECTION_INTERVAL = SHORT_VIEW


# =============================================================================
# Figure geometry
# =============================================================================
W_MM, H_MM = 174.0, 148.0

C_GLOBAL = "#8F8F8F"
C_STICKY = "#E69F00"
C_NORMAL = "#4E79A7"
C_BG = "0.82"


# =============================================================================
# KS helpers
# =============================================================================
def require_raw_file():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Could not find data/ks_raw_data.npz.\n"
            "Expected project layout:\n"
            "  project_root/fig3_ks.py\n"
            "  project_root/data/ks_raw_data.npz"
        )
    return RAW_FILE


def _ks_index_tables(n_modes):
    """Index tables for the quadratic term of the odd KS Galerkin system.

    The amplitudes are extended to negative wavenumbers by a_{-m} = -a_m with
    a_0 = 0.  For mode k the quadratic term is sum_m a_m a_{k-m} over
    m = -N..N, where the partner a_{k-m} is kept only for 0 < |k-m| < N.
    """
    k = np.arange(1, n_modes + 1)
    m = np.arange(-n_modes, n_modes + 1)
    q = k[:, None] - m[None, :]
    partner_ok = (np.abs(q) < n_modes) & (q != 0)
    m_index = np.where(m == 0, 0, np.abs(m) - 1)
    m_sign = np.sign(m).astype(float)
    q_index = np.where(partner_ok, np.abs(q) - 1, 0)
    q_sign = np.where(partner_ok, np.sign(q), 0).astype(float)
    return k, m_index, m_sign, q_index, q_sign


def ks_rhs(a, nu):
    """Exact velocity of the truncated KS Fourier-mode system, column by column.

    da_k/dt = (k^2 - nu k^4) a_k - k sum_m a_m a_{k-m}.
    """
    a = np.asarray(a, dtype=float)
    k, m_index, m_sign, q_index, q_sign = _ks_index_tables(a.shape[0])
    a_m = m_sign[:, None] * a[m_index]                    # (2N+1, time)
    a_km = q_sign[:, :, None] * a[q_index]                # (N, 2N+1, time)
    nonlinear = np.einsum("mt,kmt->kt", a_m, a_km)
    return (k ** 2 - nu * k ** 4)[:, None] * a - k[:, None] * nonlinear


def build_ks_core(rank=None):
    """Load the KS Fourier data and build the fixed reduced representation.

    rank=None (Fig. 3) applies the shared cumulative-singular-value rule.  An
    explicit rank is used only by the representation-capacity scan of Fig. S3.
    """
    raw_file = require_raw_file()
    data = np.load(raw_file)

    if "t" not in data.files or "a" not in data.files:
        raise KeyError(
            f"{raw_file.name} must contain at least arrays 't' and 'a'."
        )

    t = np.asarray(data["t"], dtype=float)
    a = np.asarray(data["a"], dtype=float)
    if a.ndim != 2:
        raise ValueError(f"KS Fourier array 'a' must be 2D, got shape {a.shape}")
    if a.shape[0] != N_MODES and a.shape[1] == N_MODES:
        a = a.T
    if a.shape[0] != N_MODES:
        raise ValueError(
            f"Expected the first Fourier dimension to have {N_MODES} modes; got {a.shape}"
        )

    if ANALYSIS_END > a.shape[1]:
        raise ValueError(
            f"KS raw trajectory has only {a.shape[1]} frames, but Fig. 3 requires ANALYSIS_END={ANALYSIS_END}."
        )

    H = a[:, ANALYSIS_START:ANALYSIS_END]
    nu = float(data["mu"]) if "mu" in data.files else NU_DEFAULT
    Hdot = ks_rhs(H, nu)

    # Amplitudes are measured from the flat solution a = 0, which anchors the
    # local linear closure; H is therefore not centred.
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    if rank is None:
        rank = cp.retained_rank_from_singular_values(s, RANK_FRACTION)
    rank = int(rank)
    V = s[:rank, None] * Vh[:rank]       # U_r^T H
    Vdot = U[:, :rank].T @ Hdot          # exact velocity, same projection

    source_time = np.asarray(t[ANALYSIS_START:ANALYSIS_START + V.shape[1]], dtype=float)
    dt = float(t[1] - t[0])
    tmax = min(V.shape[1] - 1, int(np.floor(T_SCAN_MAX_FRACTION * V.shape[1])))

    return {
        "raw_file": raw_file,
        "data": data,
        "t": t,
        "a": a,
        "H": H,
        "singular_values": s,
        "rank": rank,
        "V": V,
        "Vdot": Vdot,
        "sums": cp.prepare_window_sums(V, Vdot),
        "source_time": source_time,
        "dt": dt,
        "nu": nu,
        "a1_trace": np.asarray(H[0], dtype=float),
        "a2_trace": np.asarray(H[1], dtype=float),
        "t_scan_max": tmax,
    }


def analyse_timescales(core):
    """Shared closure analysis over T up to a quarter of the record."""
    return cp.analyze_timescales(
        core["sums"],
        core["source_time"],
        core["dt"],
        t_min=T_SCAN_MIN,
        t_max=core["t_scan_max"],
        max_windows=MAX_SELECTED_SCALES,
        progress=True,
    )


def result_for_window(core, T_selected):
    T_selected = int(T_selected)
    beta, _ = cp.compute_metrics_for_T(*core["sums"], T_selected)
    beta_time = core["source_time"][:len(beta)] + 0.5 * T_selected * core["dt"]
    return {
        "T": T_selected,
        "beta": beta,
        "beta_time": beta_time,
        "a1_trace": core["a1_trace"],
    }


def time_interval_mask(time, interval):
    lo, hi = map(float, interval)
    time = np.asarray(time, dtype=float)
    return (time >= lo) & (time <= hi)


# =============================================================================
# Output helpers
# =============================================================================
def safe_savefig(fig, target, **kwargs):
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"
    with tempfile.TemporaryDirectory(prefix="closure_fig3_") as tmp_dir:
        tmp = Path(tmp_dir) / f"render{suffix}"
        fig.savefig(tmp, **kwargs)
        try:
            shutil.copy2(tmp, target)
        except OSError as exc:
            print(f"[WARNING] Temporary render succeeded but copy failed: {exc}")
            print(f"          Trying direct save to: {target}")
            fig.savefig(target, **kwargs)
    return target


def save_summary(path, core, analysis):
    """Save the fixed representation and the two selected characteristic scales."""
    (T1, T2), (e1, e2) = analysis["selected"][:2], analysis["selected_events"][:2]
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "system,rank,rank_rule,t_scan_min,t_scan_max,"
            "T1,T1_depth,T1_prominence,T1_d_org,"
            "T2,T2_depth,T2_prominence,T2_d_org,"
            "normal_t0,normal_t1,sticky_t0,sticky_t1\n"
        )
        f.write(
            f"KS,{core['rank']},cumulative_singular_values_0.98,"
            f"{analysis['t_min']},{analysis['t_max']},"
            f"{T1},{e1['depth']},{e1['prominence']},{e1['d_org']},"
            f"{T2},{e2['depth']},{e2['prominence']},{e2['d_org']},"
            f"{NORMAL_INTERVAL[0]},{NORMAL_INTERVAL[1]},"
            f"{STICKY_INTERVAL[0]},{STICKY_INTERVAL[1]}\n"
        )


# =============================================================================
# Figure helpers
# =============================================================================
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


def axes_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])


def paper_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out")


def format_scientific_y(ax, color):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_color(color)
    ax.yaxis.get_offset_text().set_fontsize(st.FS_TICK)


def beta_ylim_from(beta, beta_time, xlim):
    beta = np.asarray(beta, dtype=float)
    beta_time = np.asarray(beta_time, dtype=float)
    mask = (
        np.isfinite(beta)
        & np.isfinite(beta_time)
        & (beta_time >= float(xlim[0]))
        & (beta_time <= float(xlim[1]))
    )
    finite = beta[mask]
    if len(finite) == 0:
        finite = beta[np.isfinite(beta)]
    if len(finite) == 0:
        return (0.0, 1.0)
    bmin = float(np.min(finite))
    bmax = float(np.max(finite))
    if bmin >= 0.0:
        return (0.0, 1.22 * max(bmax, 1e-12))
    br = max(bmax - bmin, 1e-12)
    return (bmin - 0.08 * br, bmax + 0.16 * br)


def visible_signal_ylim(values, time, xlim, pad_fraction=0.12):
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    mask = (
        np.isfinite(values)
        & np.isfinite(time)
        & (time >= float(xlim[0]))
        & (time <= float(xlim[1]))
    )
    vals = values[mask]
    if len(vals) == 0:
        vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return (-1.0, 1.0)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, 1e-9)
    pad = float(pad_fraction) * span
    return (lo - pad, hi + pad)


# =============================================================================
# Drawing pieces
# =============================================================================
def draw_state_panel(ax, core, kind):
    x = core["a1_trace"]
    y = core["a2_trace"]
    tt = core["source_time"]

    ax.plot(x, y, color=C_BG, lw=0.7, alpha=0.85, zorder=1)

    if kind == "global":
        ax.plot(x, y, color=C_GLOBAL, lw=0.9, alpha=0.95, zorder=2)
        title = "global"
        color = C_GLOBAL
        interval = None
    elif kind == "sticky":
        mask = time_interval_mask(tt, STICKY_INTERVAL)
        ax.plot(x[mask], y[mask], color=C_STICKY, lw=1.25, zorder=3)
        title = "sticky"
        color = C_STICKY
        interval = STICKY_INTERVAL
    else:
        mask = time_interval_mask(tt, NORMAL_INTERVAL)
        ax.plot(x[mask], y[mask], color=C_NORMAL, lw=1.25, zorder=3)
        title = "normal"
        color = C_NORMAL
        interval = NORMAL_INTERVAL

    ax.text(
        0.03,
        0.97,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=st.FS_TEXT,
        color=color,
    )

    if interval is not None:
        ax.text(
            0.97,
            0.04,
            rf"$t\in[{interval[0]:g},{interval[1]:g}]$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=color,
        )

    paper_axes(ax)
    ax.set_aspect("equal", adjustable="box")


def draw_trace(
    ax,
    item,
    source_time,
    beta_ylim,
    a1_ylim,
    label,
    xlim,
    shading="none",
    show_xlabel=False,
    panel_tag=None,
):
    axr = ax.twinx()

    if shading == "coarse":
        ax.axvspan(
            LOCAL_INSPECTION_INTERVAL[0],
            LOCAL_INSPECTION_INTERVAL[1],
            color="0.55",
            alpha=0.13,
            lw=0,
            zorder=0,
        )
    elif shading == "local":
        ax.axvspan(
            NORMAL_INTERVAL[0],
            NORMAL_INTERVAL[1],
            color=C_NORMAL,
            alpha=0.14,
            lw=0,
            zorder=0,
        )
        ax.axvspan(
            STICKY_INTERVAL[0],
            STICKY_INTERVAL[1],
            color=C_STICKY,
            alpha=0.14,
            lw=0,
            zorder=0,
        )

    axr.plot(
        source_time,
        item["a1_trace"],
        color=st.C_REF,
        lw=0.82,
        alpha=0.62,
        zorder=1,
    )
    axr.set_xlim(*xlim)
    axr.set_ylim(*a1_ylim)
    axr.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axr.tick_params(axis="y", colors=st.C_REF, labelsize=st.FS_TICK, pad=1.5)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(st.C_REF)
    axr.spines["right"].set_linewidth(st.LW_AXIS)
    axr.spines["top"].set_visible(False)
    axr.spines["left"].set_visible(False)

    ax.plot(
        item["beta_time"],
        item["beta"],
        color=st.C_BETA,
        lw=1.08,
        alpha=0.97,
        zorder=3,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*beta_ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(axis="y", colors=st.C_BETA, labelsize=st.FS_TICK, pad=1.5)
    format_scientific_y(ax, st.C_BETA)
    ax.spines["left"].set_color(st.C_BETA)
    ax.spines["right"].set_visible(False)
    paper_axes(ax)

    # Keep only the scale label inside the data region.  Panel tags are placed
    # outside the axes at figure level, matching panels (a)--(c).
    ax.text(
        0.018,
        0.955,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=st.FS_TEXT,
        color=st.C_SEL,
    )

    if show_xlabel:
        ax.set_xlabel(r"$t$", labelpad=1.5)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    return axr


def main():
    fam = st.apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    core = build_ks_core()
    analysis = analyse_timescales(core)
    selected = list(analysis["selected"])
    if len(selected) < 2:
        raise RuntimeError(f"Fig. 3 needs two characteristic scales, got {selected}.")

    results = [result_for_window(core, T) for T in selected[:2]]
    save_summary(FIGURE_DIR / "fig3_ks_summary.csv", core, analysis)

    print("KS Fig. 3")
    print("  raw data:", core["raw_file"])
    print("  H shape:", core["H"].shape)
    print("  retained rank:", core["rank"])
    print("  selected T:", selected[:2])
    for j, ev in enumerate(analysis["selected_events"][:2], start=1):
        print(
            f"  event {j}: [{ev['T_left']:.1f}, {ev['T_min']:.1f}, {ev['T_right']:.1f}], "
            f"T_half={ev['T_half']:.1f}, T_pre={ev['T_pre_recovery']}, "
            f"depth={ev['depth']:.4f}, prominence={ev['prominence']:.4f}, "
            f"d_org={ev['d_org']:.3f}"
        )

    # The two event-level traces use system-specific time and amplitude ranges:
    # T2 shows the coarse trajectory-scale view and T1 the local interval.
    long_xlim = (
        max(float(core["source_time"][0]), LONG_VIEW[0]),
        min(float(core["source_time"][-1]), LONG_VIEW[1]),
    )
    short_xlim = (
        max(float(core["source_time"][0]), SHORT_VIEW[0]),
        min(float(core["source_time"][-1]), SHORT_VIEW[1]),
    )

    # State-space limits shared by the three top panels.
    x = core["a1_trace"]
    y = core["a2_trace"]
    xpad = 0.08 * max(float(np.nanmax(x) - np.nanmin(x)), 1e-9)
    ypad = 0.08 * max(float(np.nanmax(y) - np.nanmin(y)), 1e-9)
    xlim = (float(np.nanmin(x) - xpad), float(np.nanmax(x) + xpad))
    ylim = (float(np.nanmin(y) - ypad), float(np.nanmax(y) + ypad))

    fig = plt.figure(figsize=(W_MM * st.MM, H_MM * st.MM), facecolor="white")

    # ------------------------------------------------------------------ (a) three state-space panels
    top_y = 114.0
    top_h = 27.0
    gap = 4.0
    top_x0 = 13.0
    top_w = (W_MM - 2 * top_x0 - 2 * gap) / 3.0

    ax_a1 = axes_mm(fig, top_x0, top_y, top_w, top_h)
    ax_a2 = axes_mm(fig, top_x0 + top_w + gap, top_y, top_w, top_h)
    ax_a3 = axes_mm(fig, top_x0 + 2 * (top_w + gap), top_y, top_w, top_h)

    draw_state_panel(ax_a1, core, "global")
    draw_state_panel(ax_a2, core, "sticky")
    draw_state_panel(ax_a3, core, "normal")

    for ax in (ax_a1, ax_a2, ax_a3):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_a2.tick_params(labelleft=False)
    ax_a3.tick_params(labelleft=False)

    fig.text(7.2 / W_MM, (top_y + 0.5 * top_h) / H_MM, r"$a_2$", rotation=90,
             ha="center", va="center", fontsize=st.FS_TEXT)
    fig.text(0.5, (top_y - 4.8) / H_MM, r"$a_1$", ha="center", va="top", fontsize=st.FS_TEXT)

    # ------------------------------------------------------------------ (b) scale-time map
    # D(t,T) on the fine log-T grid; white guides mark the selected scales.
    ax_map = axes_mm(fig, 13.0, 64.0, 83.7, 39.5)
    mesh = ax_map.pcolormesh(
        analysis["common_time"],
        analysis["display_T_values"],
        cp.globally_scaled_map(analysis["display_scale_map"]),
        shading="auto",
        rasterized=True,
    )
    ax_map.set_yscale("log")
    ax_map.set_xlim(float(core["source_time"][0]), float(core["source_time"][-1]))
    ax_map.set_xlabel(r"$t$")
    ax_map.set_ylabel(r"Window size $T$")
    for T in selected[:2]:
        ax_map.axhline(T, color="white", lw=st.LW_GUIDE, ls=(0, (3, 2)), alpha=0.95)
    paper_axes(ax_map)

    cax = axes_mm(fig, 99.3, 64.0, 1.8, 39.5)
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=1.8, width=0.5, labelsize=st.FS_TICK)
    cb.set_label(r"$\widetilde{D}_{\mathrm{sample}}$", labelpad=0.5)

    # ------------------------------------------------------------------ (c) S(T)
    # Raw neighbouring-scale correlations and the smoothed S(T) used to locate
    # persistent reorganization events.
    ax_s = axes_mm(fig, 125.0, 64.0, 35.5, 39.5)
    Ts = analysis["T_values"]
    diag = analysis["diag"]

    ax_s.plot(Ts, diag["raw"], ls="none", marker="o", ms=1.8,
              markeredgewidth=0, color=st.C_SRAW, alpha=0.28)
    ax_s.plot(np.exp(diag["log_grid"]), diag["smooth"], color=st.C_SSMOOTH, lw=1.05)

    T2_display = float(selected[1]) if len(selected) >= 2 else float(Ts[-1])
    x_display_max = min(float(Ts[-1]), max(1.55 * T2_display, float(Ts[0]) * 1.2))
    # Keep logarithmic geometry while showing a compact set of major ticks.
    ax_s.set_xscale("log")
    ax_s.set_xlim(float(Ts[0]), x_display_max)

    c_tick_candidates = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
    c_ticks = c_tick_candidates[
        (c_tick_candidates >= float(Ts[0]))
        & (c_tick_candidates <= x_display_max)
    ]
    ax_s.xaxis.set_major_locator(FixedLocator(c_ticks))
    ax_s.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x / 100.0:g}")
    )
    ax_s.xaxis.set_minor_formatter(NullFormatter())
    ax_s.tick_params(axis="x", which="minor", labelbottom=False)

    visible = Ts <= x_display_max
    sv = np.asarray(diag["smooth_actual"])[visible]
    sv = sv[np.isfinite(sv)]
    if len(sv) > 0:
        yr = max(float(np.max(sv) - np.min(sv)), 1e-4)
        ylo = max(0.0, float(np.min(sv)) - 0.18 * yr)
        yhi = min(1.02, float(np.max(sv)) + 0.26 * yr)
        ax_s.set_ylim(ylo, yhi)

    for j, T in enumerate(selected[:2], start=1):
        sval = float(np.interp(np.log(T), diag["log_grid"], diag["smooth"]))
        ax_s.axvline(
            T,
            ymin=0.0,
            ymax=0.86,
            color=st.C_SEL,
            ls="-.",
            lw=st.LW_GUIDE,
            alpha=0.88,
        )
        ax_s.plot(T, sval, "o", ms=3.1, color=st.C_SEL)

        if j == 1:
            # T1 label: lower-right of the marker.
            offset = (6, -5)
            ha, va = "left", "top"
        else:
            # T2 label: upper-left of the marker.
            offset = (-6, 5)
            ha, va = "right", "bottom"

        ax_s.annotate(
            rf"$T_{j}={int(T)}$",
            xy=(T, sval),
            xycoords="data",
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=st.C_SEL,
            fontsize=st.FS_TICK,
            annotation_clip=False,
        )

    ax_s.set_xlabel(r"Window size $T$", labelpad=7.0)
    ax_s.set_ylabel(r"$S(T)$")
    ax_s.yaxis.set_major_locator(
        FixedLocator([0.90, 0.95, 1.00])
    )
    ax_s.text(
        0.985,
        -0.095,
        r"$\times 10^2$",
        transform=ax_s.transAxes,
        ha="right",
        va="top",
        fontsize=0.90 * st.FS_TICK,
    )
    paper_axes(ax_s)

    # ------------------------------------------------------------------ (d) traces
    # Event-level beta traces at T2 and T1 with a1(t) as an independent
    # reference for trajectory organization.
    ax_t2 = axes_mm(fig, 18.0, 33.0, 137.0, 18.5)
    ax_t1 = axes_mm(fig, 18.0, 7.0, 137.0, 18.5)

    beta_ylim_t2 = beta_ylim_from(
        results[1]["beta"], results[1]["beta_time"], long_xlim
    )
    beta_ylim_t1 = beta_ylim_from(
        results[0]["beta"], results[0]["beta_time"], short_xlim
    )
    a1_ylim_t2 = visible_signal_ylim(
        core["a1_trace"], core["source_time"], long_xlim, pad_fraction=0.22
    )
    a1_ylim_t1 = visible_signal_ylim(
        core["a1_trace"], core["source_time"], short_xlim, pad_fraction=0.22
    )

    draw_trace(
        ax_t2,
        results[1],
        core["source_time"],
        beta_ylim_t2,
        a1_ylim_t2,
        rf"$T_2={selected[1]}$",
        long_xlim,
        shading="coarse",
        show_xlabel=True,
        panel_tag=None,
    )
    draw_trace(
        ax_t1,
        results[0],
        core["source_time"],
        beta_ylim_t1,
        a1_ylim_t1,
        rf"$T_1={selected[0]}$",
        short_xlim,
        shading="local",
        show_xlabel=True,
        panel_tag=None,
    )

    fig.text(7.0 / W_MM, 42.0 / H_MM, r"$\beta$", color=st.C_BETA, fontsize=st.FS_TEXT,
             rotation=90, ha="center", va="center")
    fig.text(165.2 / W_MM, 42.0 / H_MM, r"$a_1(t)$", color=st.C_REF, fontsize=st.FS_TEXT,
             rotation=90, ha="center", va="center")
    fig.text(7.0 / W_MM, 16.0 / H_MM, r"$\beta$", color=st.C_BETA, fontsize=st.FS_TEXT,
             rotation=90, ha="center", va="center")
    fig.text(165.2 / W_MM, 16.0 / H_MM, r"$a_1(t)$", color=st.C_REF, fontsize=st.FS_TEXT,
             rotation=90, ha="center", va="center")

    add_panel_letter(fig, 1.0, 146.0, "(a)")
    add_panel_letter(fig, 1.0, 106.0, "(b)")
    add_panel_letter(fig, 109.0, 106.0, "(c)")
    add_panel_letter(fig, 1.0, 55.0, "(d1)")
    add_panel_letter(fig, 1.0, 29.0, "(d2)")

    pdf_path = FIGURE_DIR / "fig3_ks.pdf"
    png_path = FIGURE_DIR / "fig3_ks.png"

    safe_savefig(fig, pdf_path, format="pdf", facecolor="white")
    safe_savefig(fig, png_path, dpi=400, facecolor="white")
    plt.close(fig)

    print("  font:", fam)
    print("  PDF:", pdf_path)
    print("  preview:", png_path)


if __name__ == "__main__":
    main()
