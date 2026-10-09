"""Generate manuscript Fig. 2: Lorenz scale-resolved closure dynamics.

Expected project layout
-----------------------
project_root/
├── closure_pipeline.py
├── figure_style.py
├── fig1_synthetic.py
├── fig1.py
├── fig2_lorenz.py
└── data/
    └── lorenz_raw_data.npz

Run
---
    python fig2_lorenz.py

Outputs
-------
    figures/fig2_lorenz.pdf
    figures/fig2_lorenz.png
    figures/fig2_lorenz_summary.csv

Method
------
This script only builds the Lorenz observable representation and the
independent validation references; everything else comes from
closure_pipeline.py.

Representation.  The observed state (x, y, z) is embedded with five shifted
copies per coordinate, a 15-row delay matrix H.  Coordinates are measured from
the origin, which is the fixed point of the Lorenz flow left invariant by the
symmetry (x, y, z) -> (-x, -y, z), i.e. the point midway between the two lobes.
It is the anchor of the local linear closure, so H is not centred.  The rank
follows the shared cumulative-singular-value rule, and the exact Lorenz
velocity, embedded in the same way, is projected with the same U_r.

Closure field, S(T), reorganization events and characteristic scales come
from closure_pipeline.analyze_timescales.  The Poincare symbol changes and the
closure peaks are computed separately and only drawn on the T2 trace for
validation; neither enters the scale selection.
"""

from pathlib import Path
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator
from matplotlib.lines import Line2D

import closure_pipeline as cp
import figure_style as st


# =============================================================================
# Project layout
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

RAW_FILE = DATA_DIR / "lorenz_raw_data.npz"


# =============================================================================
# Lorenz analysis settings
# =============================================================================
# Fixed observable representation: a 20,000-sample segment and five one-sample
# shifts of each state coordinate, giving a 15-row delay representation.
START = 220000
END = 240000
EMBEDDING_LENGTH = 19996

RANK_FRACTION = cp.RANK_CUMULATIVE_SINGULAR_VALUE

# Observation-scale scan.  T is a window size in samples, not an intrinsic
# period.  The upper bound is limited only by the available reduced trajectory.
T_SCAN_MIN = 100
MAX_SELECTED_SCALES = 2


# =============================================================================
# Independent Lorenz validation references
# =============================================================================
# These parameters are used only after characteristic-scale selection.
# Closure events themselves are detected by the shared rule in closure_pipeline.
# The Lorenz-specific external reference is the z = 27 Poincare section.
POINCARE_Z_SECTION = 27.0
POINCARE_SHIFT_SAMPLES = 220


# =============================================================================
# Figure geometry
# =============================================================================
W_MM, H_MM = 174.0, 112.0
TMIN, TMAX = 220.0, 240.0
TIME_TICKS = np.arange(220, 241, 5)


# =============================================================================
# Data preparation
# =============================================================================
def require_raw_file():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Could not find data/lorenz_raw_data.npz.\n"
            "Expected project layout:\n"
            "  project_root/fig2_lorenz.py\n"
            "  project_root/data/lorenz_raw_data.npz"
        )
    return RAW_FILE


def build_lorenz_core(rank=None):
    """Load raw Lorenz data and build the fixed reduced representation.

    rank=None (Fig. 2) applies the shared cumulative-singular-value rule.  An
    explicit rank is used only by the representation-capacity scan of Fig. S3.
    """
    raw_file = require_raw_file()
    data = np.load(raw_file)

    required = ("t", "x", "y", "z", "dx_dt", "dy_dt", "dz_dt")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(
            f"{raw_file.name} is missing required arrays: {missing}"
        )

    n_raw = len(data["t"])
    if END > n_raw:
        raise ValueError(
            f"Lorenz raw trajectory has {n_raw} samples, "
            f"but Fig. 2 requires END={END}."
        )

    observed = np.vstack([
        data["x"][START:END],
        data["y"][START:END],
        data["z"][START:END],
    ])
    observed_velocity = np.vstack([
        data["dx_dt"][START:END],
        data["dy_dt"][START:END],
        data["dz_dt"][START:END],
    ])

    # Five shifted copies of each coordinate form a 15-row delay matrix; the
    # exact velocity is embedded with the same shifts.
    H = np.vstack([
        sliding_window_view(row, EMBEDDING_LENGTH)
        for row in observed
    ])
    Hdot = np.vstack([
        sliding_window_view(row, EMBEDDING_LENGTH)
        for row in observed_velocity
    ])

    # Coordinates are measured from the symmetric fixed point (the origin),
    # which anchors the local linear closure; H is therefore not centred.
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    if rank is None:
        rank = cp.retained_rank_from_singular_values(s, RANK_FRACTION)
    rank = int(rank)

    V = s[:rank, None] * Vh[:rank]       # U_r^T H
    Vdot = U[:, :rank].T @ Hdot          # exact velocity, same projection

    source_time = np.asarray(
        data["t"][START:START + V.shape[1]],
        dtype=float,
    )
    dt = float(data["t"][1] - data["t"][0])

    return {
        "raw_file": raw_file,
        "data": data,
        "H": H,
        "singular_values": s,
        "rank": rank,
        "V": V,
        "Vdot": Vdot,
        "sums": cp.prepare_window_sums(V, Vdot),
        "source_time": source_time,
        "dt": dt,
        "x_trace": np.asarray(data["x"][START:START + V.shape[1]], dtype=float),
    }


def analyse_timescales(core):
    """Shared closure analysis over T from T_SCAN_MIN to the full length."""
    return cp.analyze_timescales(
        core["sums"],
        core["source_time"],
        core["dt"],
        t_min=T_SCAN_MIN,
        t_max=core["V"].shape[1] - 1,
        max_windows=MAX_SELECTED_SCALES,
        progress=True,
    )


# =============================================================================
# Independent validation references
# =============================================================================
def closure_peak_reference(beta, core, T_selected):
    """Shared closure-event detector; no Lorenz-specific peak tuning."""
    beta = np.asarray(beta, dtype=float)
    peak_idx, peak_times, _ = cp.detect_closure_events(
        beta,
        core["source_time"][:len(beta)],
        int(T_selected),
        core["dt"],
    )
    if len(peak_times) == 0:
        return np.array([]), np.array([])

    t = np.asarray(core["data"]["t"], dtype=float)
    x = np.asarray(core["data"]["x"], dtype=float)
    peak_x = np.interp(peak_times, t, x)
    return peak_times, peak_x


def poincare_reference(data, segment_length):
    """Exact two-direction z=27 crossings, then symbolic changes and 220 shift."""
    t = np.asarray(data["t"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    z = np.asarray(data["z"], dtype=float)

    TT = t[START:START + segment_length]
    X = x[START:START + segment_length]
    Z = z[START:START + segment_length]

    values = Z - POINCARE_Z_SECTION
    crossing_idx = np.where(values[:-1] * values[1:] <= 0.0)[0]
    if len(crossing_idx) < 2:
        return np.array([]), np.array([])

    denom = Z[crossing_idx + 1] - Z[crossing_idx]
    alpha = np.zeros(len(crossing_idx), dtype=float)
    good = np.abs(denom) > 1e-14
    alpha[good] = (
        POINCARE_Z_SECTION - Z[crossing_idx[good]]
    ) / denom[good]

    crossing_time = (
        TT[crossing_idx]
        + alpha * (TT[crossing_idx + 1] - TT[crossing_idx])
    )
    crossing_x = (
        X[crossing_idx]
        + alpha * (X[crossing_idx + 1] - X[crossing_idx])
    )

    symbols = np.sign(crossing_x)
    valid = symbols != 0
    crossing_time = crossing_time[valid]
    symbols = symbols[valid]
    if len(symbols) < 2:
        return np.array([]), np.array([])

    change = np.where(symbols[1:] != symbols[:-1])[0] + 1
    if len(change) == 0:
        return np.array([]), np.array([])

    dt = float(t[1] - t[0])
    point_time = crossing_time[change] - POINCARE_SHIFT_SAMPLES * dt
    in_segment = (point_time >= TT[0]) & (point_time <= TT[-1])
    point_time = point_time[in_segment]
    point_x = np.interp(point_time, TT, X)
    return point_time, point_x


def result_for_window(core, T_selected):
    """Closure trace and validation references at one selected scale."""
    T_selected = int(T_selected)
    beta, _ = cp.compute_metrics_for_T(
        *core["sums"],
        T_selected,
    )
    beta_time = (
        core["source_time"][:len(beta)]
        + 0.5 * T_selected * core["dt"]
    )

    closure_t, closure_x = closure_peak_reference(
        beta,
        core,
        T_selected,
    )
    poincare_t, poincare_x = poincare_reference(
        core["data"],
        len(core["x_trace"]),
    )

    return {
        "T": T_selected,
        "beta": beta,
        "beta_time": beta_time,
        "x_trace": core["x_trace"],
        "closure_t": closure_t,
        "closure_x": closure_x,
        "poincare_t": poincare_t,
        "poincare_x": poincare_x,
    }


# =============================================================================
# Output helpers
# =============================================================================
def safe_savefig(fig, target, **kwargs):
    """Render to a unique temporary directory, then copy to the project."""
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"

    with tempfile.TemporaryDirectory(prefix="closure_fig2_") as tmp_dir:
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
            "T2,T2_depth,T2_prominence,T2_d_org\n"
        )
        f.write(
            f"Lorenz,{core['rank']},cumulative_singular_values_0.98,"
            f"{analysis['t_min']},{analysis['t_max']},"
            f"{T1},{e1['depth']},{e1['prominence']},{e1['d_org']},"
            f"{T2},{e2['depth']},{e2['prominence']},{e2['d_org']}\n"
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
    return fig.add_axes([
        x / W_MM,
        y / H_MM,
        w / W_MM,
        h / H_MM,
    ])


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


def draw_trace(
    ax,
    item,
    x_time,
    shared_beta_ylim,
    shared_x_ylim,
    label,
    show_xlabel=False,
    add_validation=False,
):
    """Draw one selected-scale beta trace and the independent x(t) reference."""
    axr = ax.twinx()

    # Secondary reference observable.
    axr.plot(
        x_time,
        item["x_trace"],
        color=st.C_REF,
        lw=0.78,
        alpha=0.58,
        zorder=1,
    )
    axr.set_ylim(*shared_x_ylim)
    axr.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axr.tick_params(
        axis="y",
        colors=st.C_REF,
        labelsize=st.FS_TICK,
        pad=1.5,
    )
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(st.C_REF)
    axr.spines["right"].set_linewidth(st.LW_AXIS)
    axr.spines["top"].set_visible(False)
    axr.spines["left"].set_visible(False)

    # Primary closure signal.
    ax.plot(
        item["beta_time"],
        item["beta"],
        color=st.C_BETA,
        lw=1.08,
        alpha=0.96,
        zorder=3,
    )
    ax.set_xlim(TMIN, TMAX)
    ax.set_ylim(*shared_beta_ylim)
    ax.xaxis.set_major_locator(FixedLocator(TIME_TICKS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(
        axis="y",
        colors=st.C_BETA,
        labelsize=st.FS_TICK,
        pad=1.5,
    )
    format_scientific_y(ax, st.C_BETA)
    ax.spines["left"].set_color(st.C_BETA)
    ax.spines["right"].set_visible(False)
    paper_axes(ax)

    ax.text(
        0.016,
        0.948,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=st.FS_TEXT,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="0.35",
            linewidth=0.55,
            alpha=0.98,
        ),
    )

    if add_validation:
        if len(item["poincare_t"]) > 0:
            axr.scatter(
                item["poincare_t"],
                item["poincare_x"],
                s=18,
                color="#6A5ACD",
                edgecolors="black",
                linewidth=0.35,
                zorder=6,
            )
        if len(item["closure_t"]) > 0:
            axr.scatter(
                item["closure_t"],
                item["closure_x"],
                s=18,
                color="#FF8C00",
                edgecolors="black",
                linewidth=0.35,
                zorder=6,
            )

        handles = [
            Line2D(
                [0], [0],
                marker="o",
                ls="none",
                ms=4.2,
                markerfacecolor="#6A5ACD",
                markeredgecolor="black",
                markeredgewidth=0.35,
                label="Poincaré",
            ),
            Line2D(
                [0], [0],
                marker="o",
                ls="none",
                ms=4.2,
                markerfacecolor="#FF8C00",
                markeredgecolor="black",
                markeredgewidth=0.35,
                label="Closure peaks",
            ),
        ]
        leg = ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.56, 1.085),
            ncol=2,
            frameon=True,
            fontsize=7.2,
            handletextpad=0.35,
            columnspacing=0.9,
            borderaxespad=0.0,
            fancybox=False,
            framealpha=0.96,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("0.35")
        leg.get_frame().set_linewidth(0.55)

    if show_xlabel:
        ax.set_xlabel(r"$t$")
    else:
        ax.tick_params(axis="x", labelbottom=False)

    return axr


# =============================================================================
# Main figure
# =============================================================================
def main():
    fam = st.apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    core = build_lorenz_core()
    analysis = analyse_timescales(core)
    selected = list(analysis["selected"])
    if len(selected) < 2:
        raise RuntimeError(f"Fig. 2 needs two characteristic scales, got {selected}.")

    results = [
        result_for_window(core, T)
        for T in selected[:2]
    ]

    print("Lorenz Fig. 2")
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

    save_summary(
        FIGURE_DIR / "fig2_lorenz_summary.csv",
        core,
        analysis,
    )

    # Shared panel-(c) ranges make the two selected scales directly comparable.
    beta_max = max(
        float(np.nanmax(item["beta"]))
        for item in results
    )
    beta_ylim = (0.0, 1.48 * beta_max)

    xmin = float(np.nanmin(core["x_trace"]))
    xmax = float(np.nanmax(core["x_trace"]))
    xpad = 0.34 * max(xmax - xmin, 1e-9)
    x_ylim = (xmin - xpad, xmax + xpad)

    fig = plt.figure(
        figsize=(W_MM * st.MM, H_MM * st.MM),
        facecolor="white",
    )

    # ------------------------------------------------------------------ (a)
    # D(t,T) on the fine log-T grid; white guides mark the selected scales.
    ax_map = axes_mm(fig, 13.0, 65.5, 73.0, 39.5)
    mesh = ax_map.pcolormesh(
        analysis["common_time"],
        analysis["display_T_values"],
        cp.globally_scaled_map(analysis["display_scale_map"]),
        shading="auto",
        rasterized=True,
    )
    ax_map.set_yscale("log")
    ax_map.set_xlim(TMIN, TMAX)
    ax_map.xaxis.set_major_locator(FixedLocator(TIME_TICKS))
    ax_map.set_xlabel(r"$t$")
    ax_map.set_ylabel(r"Window size $T$")

    for T in selected[:2]:
        ax_map.axhline(
            T,
            color="white",
            lw=st.LW_GUIDE,
            ls=(0, (3, 2)),
            alpha=0.95,
        )
    paper_axes(ax_map)

    cax = axes_mm(fig, 88.0, 65.5, 2.0, 39.5)
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(
        length=1.8,
        width=0.5,
        labelsize=st.FS_TICK,
    )
    cb.set_label(
        r"$\widetilde{D}_{\mathrm{sample}}$",
        labelpad=1.5,
    )

    # ------------------------------------------------------------------ (b)
    # Raw neighbouring-scale correlations and the smoothed S(T) used to locate
    # persistent reorganization events.
    ax_s = axes_mm(fig, 114.0, 65.5, 46.5, 39.5)
    Ts = analysis["T_values"]
    diag = analysis["diag"]

    ax_s.plot(
        Ts,
        diag["raw"],
        ls="none",
        marker="o",
        ms=1.8,
        markeredgewidth=0,
        color=st.C_SRAW,
        alpha=0.28,
    )
    ax_s.plot(
        np.exp(diag["log_grid"]),
        diag["smooth"],
        color=st.C_SSMOOTH,
        lw=1.05,
    )

    # Display range only; characteristic-scale selection uses the full scan.
    x_display_max = min(float(Ts[-1]), 2.1e3)
    ax_s.set_xscale("log")
    ax_s.set_xlim(float(Ts[0]), x_display_max)

    visible = Ts <= x_display_max
    sv = np.asarray(diag["smooth_actual"])[visible]
    sv = sv[np.isfinite(sv)]

    if len(sv) > 0:
        yr = max(float(np.max(sv) - np.min(sv)), 1e-4)
        ylo = max(0.80, float(np.min(sv)) - 0.18 * yr)
        yhi = float(np.max(sv)) + 0.26 * yr
        ax_s.set_ylim(ylo, yhi)

    for j, T in enumerate(selected[:2], start=1):
        sval = float(np.interp(
            np.log(T),
            diag["log_grid"],
            diag["smooth"],
        ))

        ax_s.axvline(
            T,
            ymin=0.0,
            ymax=0.88,
            color=st.C_SEL,
            ls="-.",
            lw=st.LW_GUIDE,
            alpha=0.88,
        )
        ax_s.plot(
            T,
            sval,
            "o",
            ms=3.1,
            color=st.C_SEL,
        )

        if j == 1:
            x_text, ha = T * 0.985, "right"
        else:
            x_text, ha = T * 1.015, "left"

        ax_s.text(
            x_text,
            0.965,
            rf"$T_{j}={T}$",
            transform=ax_s.get_xaxis_transform(),
            ha=ha,
            va="top",
            color=st.C_SEL,
            fontsize=st.FS_TEXT,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.90,
            ),
        )

    ax_s.set_xlabel(r"Window size $T$")
    ax_s.set_ylabel(r"$S(T)$")
    paper_axes(ax_s)

    # ------------------------------------------------------------------ (c)
    # Event-level beta traces at T1 and T2 with x(t) as an independent reference.
    # Validation markers are added only after scale selection.
    ax_t1 = axes_mm(fig, 18.0, 35.0, 137.0, 18.5)
    ax_t2 = axes_mm(fig, 18.0, 9.0, 137.0, 18.5)

    draw_trace(
        ax_t1,
        results[0],
        core["source_time"],
        beta_ylim,
        x_ylim,
        r"$T_1$",
        show_xlabel=False,
        add_validation=False,
    )
    draw_trace(
        ax_t2,
        results[1],
        core["source_time"],
        beta_ylim,
        x_ylim,
        r"$T_2$",
        show_xlabel=True,
        add_validation=True,
    )

    fig.text(
        7.0 / W_MM,
        31.0 / H_MM,
        r"$\beta$",
        color=st.C_BETA,
        fontsize=st.FS_TEXT,
        rotation=90,
        ha="center",
        va="center",
    )
    fig.text(
        169.0 / W_MM,
        31.0 / H_MM,
        r"$x(t)$",
        color=st.C_REF,
        fontsize=st.FS_TEXT,
        rotation=90,
        ha="center",
        va="center",
    )

    add_panel_letter(fig, 1.0, 110.0, "(a)")
    add_panel_letter(fig, 100.0, 110.0, "(b)")
    add_panel_letter(fig, 1.0, 57.5, "(c)")

    pdf_path = FIGURE_DIR / "fig2_lorenz.pdf"
    png_path = FIGURE_DIR / "fig2_lorenz.png"

    safe_savefig(
        fig,
        pdf_path,
        format="pdf",
        facecolor="white",
    )
    safe_savefig(
        fig,
        png_path,
        dpi=400,
        facecolor="white",
    )
    plt.close(fig)

    print("  font:", fam)
    print("  PDF:", pdf_path)
    print("  preview:", png_path)


if __name__ == "__main__":
    main()
