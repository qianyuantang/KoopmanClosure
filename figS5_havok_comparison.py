"""Supplementary Fig. S5: HAVOK comparison for Lorenz dynamics.

The comparison uses the production Lorenz representation, rank rule and
automatically selected T2 from the current Fig. 2 analysis. HAVOK forcing is
evaluated on the same Lorenz trajectory as an external diagnostic.

Expected project layout
-----------------------
project_root/
├── figS5_havok_comparison.py
├── closure_pipeline.py
├── figure_style.py
├── data/
│   └── lorenz_raw_data.npz
└── figures/

Outputs
-------
figures/figS5_havok_comparison.pdf
figures/figS5_havok_comparison.png
figures/figS5_havok_comparison_metadata.csv
figures/figS5_havok_comparison_summary.csv
figures/figS5_havok_comparison_episodes.csv

Panels
------
(a) HAVOK forcing coordinate v11 over t = 220--240, with |v11| > 0.008
    highlighted.
(b) Closure defect beta at the selected T2 together with Lorenz x(t) as an
    independent reference observable.
"""

from pathlib import Path
import csv
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator

import closure_pipeline as cp
import figure_style as st


# =============================================================================
# Project layout
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
RAW_FILE = DATA_DIR / "lorenz_raw_data.npz"
STEM = "figS5_havok_comparison"


# =============================================================================
# Production Lorenz preprocessing copied from the current Fig. 2 program
# =============================================================================
START = 220000
END = 240000
EMBEDDING_LENGTH = 19996
RANK_FRACTION = 0.98
T_SCAN_MIN = 100
MAX_SELECTED_SCALES = 2


# =============================================================================
# S5 comparison settings
# =============================================================================
VIEW_START = 220.0
VIEW_END = 240.0
TIME_TICKS = np.arange(220.0, 240.1, 5.0)

HAVOK_DELAY_ROWS = 100
HAVOK_COMPONENT = 11          # v_11 in one-based notation
HAVOK_THRESHOLD = 0.008
HAVOK_GAP_THRESHOLD = 330

# Figure geometry in mm.  The two traces have the same horizontal extent and
# use the visual grammar of the current main figures.
W_MM, H_MM = 174.0, 80.0
AX_X, AX_W = 18.0, 140.0
TOP_Y, TOP_H = 45.0, 27.0
BOT_Y, BOT_H = 10.0, 27.0
LETTER_X = 2.0

FS_PANEL = st.FS_PANEL
FS_TEXT = st.FS_TEXT


# =============================================================================
# Production Lorenz representation and shared scale selection
# =============================================================================
def require_raw_file() -> Path:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Could not find data/lorenz_raw_data.npz.\n"
            "Expected project layout:\n"
            "  project_root/figS5_havok_comparison.py\n"
            "  project_root/closure_pipeline.py\n"
            "  project_root/figure_style.py\n"
            "  project_root/data/lorenz_raw_data.npz"
        )
    return RAW_FILE


def build_lorenz_core():
    """Build exactly the Lorenz reduced representation used by current Fig. 2."""
    raw_file = require_raw_file()
    data = np.load(raw_file)

    required = ("t", "x", "y", "z", "dx_dt", "dy_dt", "dz_dt")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{raw_file.name} is missing required arrays: {missing}")

    n_raw = len(data["t"])
    if END > n_raw:
        raise ValueError(
            f"Lorenz raw trajectory has {n_raw} samples, but END={END} is required."
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

    # Five shifted copies of each coordinate -> 15-row multivariate delay
    # representation.  The origin is the Lorenz symmetry-fixed anchor, so H is
    # deliberately not centred.
    H = np.vstack([
        sliding_window_view(row, EMBEDDING_LENGTH)
        for row in observed
    ])
    Hdot = np.vstack([
        sliding_window_view(row, EMBEDDING_LENGTH)
        for row in observed_velocity
    ])

    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    rank = cp.retained_rank_from_singular_values(s, RANK_FRACTION)

    V = s[:rank, None] * Vh[:rank]   # U_r^T H
    Vdot = U[:, :rank].T @ Hdot      # exact velocity, same projection

    source_time = np.asarray(
        data["t"][START:START + V.shape[1]],
        dtype=float,
    )
    dt = float(data["t"][1] - data["t"][0])

    return {
        "raw_file": raw_file,
        "data": data,
        "rank": rank,
        "V": V,
        "Vdot": Vdot,
        "sums": cp.prepare_window_sums(V, Vdot),
        "source_time": source_time,
        "dt": dt,
        "x_trace": np.asarray(
            data["x"][START:START + V.shape[1]],
            dtype=float,
        ),
    }


def analyse_timescales(core):
    """Run the shared production pipeline and return the Lorenz hierarchy."""
    return cp.analyze_timescales(
        core["sums"],
        core["source_time"],
        core["dt"],
        t_min=T_SCAN_MIN,
        t_max=core["V"].shape[1] - 1,
        max_windows=MAX_SELECTED_SCALES,
        progress=True,
    )


def result_for_window(core, T_selected):
    """Production closure trace beta(t) at one selected window size."""
    T_selected = int(T_selected)
    beta, _ = cp.compute_metrics_for_T(*core["sums"], T_selected)
    beta_time = (
        core["source_time"][:len(beta)]
        + 0.5 * T_selected * core["dt"]
    )
    return beta_time, beta


# =============================================================================
# HAVOK diagnostic
# =============================================================================
def get_havok_v11(
    x,
    t,
    start_t=VIEW_START,
    end_t=VIEW_END,
    q=HAVOK_DELAY_ROWS,
    component=HAVOK_COMPONENT,
    threshold=HAVOK_THRESHOLD,
    gap_threshold=HAVOK_GAP_THRESHOLD,
):
    """Compute the HAVOK right-singular-vector coordinate shown in panel (a).

    The sign of a singular vector is arbitrary; the highlighted activity uses
    |v11| and is therefore sign-invariant. HAVOK is evaluated after the closure
    hierarchy is selected.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = (t >= float(start_t)) & (t <= float(end_t))
    xc = x[mask]
    tc = t[mask]

    q = int(q)
    if len(xc) < q + 2:
        raise ValueError("Comparison interval is too short for the HAVOK delay matrix.")

    p = len(xc) - q + 1
    H = np.empty((q, p), dtype=float)
    for i in range(q):
        H[i] = xc[i:i + p]

    _, _, VT = np.linalg.svd(H, full_matrices=False)
    row = min(max(int(component) - 1, 0), len(VT) - 1)
    v = VT[row]
    tv = tc[q - 1:q - 1 + len(v)]

    active = np.flatnonzero(np.abs(v) > float(threshold))
    segments = []
    if active.size:
        current = [int(active[0])]
        for idx in active[1:]:
            idx = int(idx)
            if idx - current[-1] <= int(gap_threshold):
                current.append(idx)
            else:
                segments.append(np.asarray(current, dtype=int))
                current = [idx]
        segments.append(np.asarray(current, dtype=int))

    return tv, v, segments


# =============================================================================
# Quantitative HAVOK--closure comparison
# =============================================================================
def quantify_havok_closure(tv, v11, active_segments, tb, beta):
    """Quantify shared episode structure in HAVOK forcing and closure failure.

    Metrics are evaluated on the time interval shown in Fig. S5.  Pointwise
    association, episode strength, and within-episode timing are reported
    separately, making clear both the agreement and the distinct temporal
    constructions of the two diagnostics.
    """
    tv = np.asarray(tv, dtype=float)
    v11 = np.asarray(v11, dtype=float)
    tb = np.asarray(tb, dtype=float)
    beta = np.asarray(beta, dtype=float)

    lo = max(VIEW_START, float(tv[0]), float(tb[0]))
    hi = min(VIEW_END, float(tv[-1]), float(tb[-1]))
    mask = np.isfinite(tb) & np.isfinite(beta) & (tb >= lo) & (tb <= hi)
    t_common = tb[mask]
    beta_common = beta[mask]

    if t_common.size < 10:
        raise RuntimeError("Insufficient common support for HAVOK--closure comparison.")

    abs_v_common = np.interp(t_common, tv, np.abs(v11))
    active = abs_v_common > float(HAVOK_THRESHOLD)

    active_fraction = float(np.mean(active))
    beta_mean = float(np.mean(beta_common))
    beta_active_mean = float(np.mean(beta_common[active])) if np.any(active) else np.nan
    beta_inactive_mean = float(np.mean(beta_common[~active])) if np.any(~active) else np.nan
    beta_active_to_inactive = (
        beta_active_mean / beta_inactive_mean
        if np.isfinite(beta_inactive_mean) and abs(beta_inactive_mean) > 1e-15
        else np.nan
    )

    pointwise_corr = float(np.corrcoef(abs_v_common, beta_common)[0, 1])

    beta_q90 = float(np.quantile(beta_common, 0.90))
    top_beta = beta_common >= beta_q90
    top_beta_active_fraction = (
        float(np.mean(active[top_beta])) if np.any(top_beta) else np.nan
    )
    top_beta_enrichment = (
        top_beta_active_fraction / active_fraction
        if active_fraction > 0.0
        else np.nan
    )

    finite_beta = beta_common[np.isfinite(beta_common)]
    episode_rows = []

    for episode, seg in enumerate(active_segments, start=1):
        seg = np.asarray(seg, dtype=int)
        if seg.size == 0:
            continue

        # Restrict each threshold-defined forcing episode to the displayed
        # common support.
        seg = seg[(tv[seg] >= lo) & (tv[seg] <= hi)]
        if seg.size == 0:
            continue

        t0 = float(tv[seg[0]])
        t1 = float(tv[seg[-1]])

        # HAVOK episode strength and its time are defined directly from the
        # strongest |v11| sample among the active points of that episode.
        abs_seg = np.abs(v11[seg])
        jh = int(np.argmax(abs_seg))
        havok_peak_index = int(seg[jh])
        havok_peak_time = float(tv[havok_peak_index])
        havok_peak = float(abs(v11[havok_peak_index]))

        # Closure strength is summarized over the same episode time span.
        emask = (t_common >= t0) & (t_common <= t1)
        if not np.any(emask):
            continue

        b = beta_common[emask]
        tt = t_common[emask]
        jb = int(np.argmax(b))
        beta_peak_time = float(tt[jb])
        beta_peak = float(b[jb])
        beta_peak_percentile = 100.0 * float(np.mean(finite_beta <= beta_peak))
        lag = beta_peak_time - havok_peak_time

        episode_rows.append({
            "episode": episode,
            "start_time": t0,
            "end_time": t1,
            "duration": t1 - t0,
            "havok_peak_time": havok_peak_time,
            "max_abs_v11": havok_peak,
            "beta_peak_time": beta_peak_time,
            "beta_peak": beta_peak,
            "beta_peak_percentile": beta_peak_percentile,
            "lag_beta_from_havok_peak": lag,
            "abs_peak_lag": abs(lag),
        })

    if len(episode_rows) >= 2:
        havok_strength = np.asarray(
            [r["max_abs_v11"] for r in episode_rows], dtype=float
        )
        closure_strength = np.asarray(
            [r["beta_peak"] for r in episode_rows], dtype=float
        )
        rho_episode, p_episode = spearmanr(havok_strength, closure_strength)
        rho_episode = float(rho_episode)
        p_episode = float(p_episode)
    else:
        rho_episode, p_episode = np.nan, np.nan

    if episode_rows:
        lags = np.asarray(
            [r["lag_beta_from_havok_peak"] for r in episode_rows], dtype=float
        )
        abs_lags = np.abs(lags)
        peak_percentiles = np.asarray(
            [r["beta_peak_percentile"] for r in episode_rows], dtype=float
        )
        lag_q25, lag_q75 = np.quantile(lags, [0.25, 0.75])
    else:
        lags = np.asarray([], dtype=float)
        abs_lags = np.asarray([], dtype=float)
        peak_percentiles = np.asarray([], dtype=float)
        lag_q25 = lag_q75 = np.nan

    summary = {
        "common_start": lo,
        "common_end": hi,
        "n_common_samples": int(t_common.size),
        "n_havok_episodes": int(len(episode_rows)),
        "havok_active_fraction": active_fraction,
        "corr_abs_v11_beta": pointwise_corr,
        "beta_mean": beta_mean,
        "beta_active_mean": beta_active_mean,
        "beta_inactive_mean": beta_inactive_mean,
        "beta_active_to_inactive": beta_active_to_inactive,
        "beta_q90": beta_q90,
        "top10_beta_active_fraction": top_beta_active_fraction,
        "top10_beta_enrichment": top_beta_enrichment,
        "median_episode_beta_peak_percentile": (
            float(np.median(peak_percentiles)) if peak_percentiles.size else np.nan
        ),
        "spearman_episode_strength": rho_episode,
        "spearman_episode_strength_p": p_episode,
        "median_peak_lag": float(np.median(lags)) if lags.size else np.nan,
        "lag_q25": float(lag_q25),
        "lag_q75": float(lag_q75),
        "median_abs_peak_lag": (
            float(np.median(abs_lags)) if abs_lags.size else np.nan
        ),
    }
    return summary, episode_rows


def save_comparison_summary(path, summary, core, selected, T2):
    """Save one-row quantitative comparison for the SI analysis."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lorenz_rank",
            "T1",
            "T2",
            "common_start",
            "common_end",
            "n_common_samples",
            "n_havok_episodes",
            "havok_active_fraction",
            "corr_abs_v11_beta",
            "beta_mean",
            "beta_active_mean",
            "beta_inactive_mean",
            "beta_active_to_inactive",
            "beta_q90",
            "top10_beta_active_fraction",
            "top10_beta_enrichment",
            "median_episode_beta_peak_percentile",
            "spearman_episode_strength",
            "spearman_episode_strength_p",
            "median_peak_lag",
            "lag_q25",
            "lag_q75",
            "median_abs_peak_lag",
        ])
        writer.writerow([
            int(core["rank"]),
            int(selected[0]),
            int(T2),
            summary["common_start"],
            summary["common_end"],
            summary["n_common_samples"],
            summary["n_havok_episodes"],
            summary["havok_active_fraction"],
            summary["corr_abs_v11_beta"],
            summary["beta_mean"],
            summary["beta_active_mean"],
            summary["beta_inactive_mean"],
            summary["beta_active_to_inactive"],
            summary["beta_q90"],
            summary["top10_beta_active_fraction"],
            summary["top10_beta_enrichment"],
            summary["median_episode_beta_peak_percentile"],
            summary["spearman_episode_strength"],
            summary["spearman_episode_strength_p"],
            summary["median_peak_lag"],
            summary["lag_q25"],
            summary["lag_q75"],
            summary["median_abs_peak_lag"],
        ])
    return path


def save_episode_table(path, rows):
    """Save episode-level HAVOK and closure strengths and peak times."""
    fields = [
        "episode",
        "start_time",
        "end_time",
        "duration",
        "havok_peak_time",
        "max_abs_v11",
        "beta_peak_time",
        "beta_peak",
        "beta_peak_percentile",
        "lag_beta_from_havok_peak",
        "abs_peak_lag",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# =============================================================================
# Figure helpers
# =============================================================================
def safe_savefig(fig, target, **kwargs):
    """Render to a temporary path and then copy to figures/."""
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"

    with tempfile.TemporaryDirectory(prefix="closure_figS5_") as tmp_dir:
        tmp = Path(tmp_dir) / f"render{suffix}"
        fig.savefig(tmp, **kwargs)
        try:
            shutil.copy2(tmp, target)
        except OSError as exc:
            print(f"[WARNING] Temporary render succeeded but copy failed: {exc}")
            print(f"          Trying direct save to: {target}")
            fig.savefig(target, **kwargs)


def axes_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])


def add_panel_letter(fig, y_mm, label):
    fig.text(
        LETTER_X / W_MM,
        (y_mm + 0.6) / H_MM,
        label,
        fontsize=FS_PANEL,
        fontweight="bold",
        ha="left",
        va="top",
    )


def paper_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def format_scientific_y(ax, color):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_color(color)
    ax.yaxis.get_offset_text().set_fontsize(st.FS_TICK)


def values_in_view(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (
        np.isfinite(t)
        & np.isfinite(y)
        & (t >= VIEW_START)
        & (t <= VIEW_END)
    )
    return y[mask]


def padded_limits(values, pad=0.12, symmetric=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)

    if symmetric:
        vmax = max(float(np.max(np.abs(values))), 1e-12) * (1.0 + pad)
        return -vmax, vmax

    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 1e-12)
    return lo - pad * span, hi + pad * span


# =============================================================================
# Main
# =============================================================================
def main():
    st.apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Current Fig. 2 production representation and shared scale selector.
    core = build_lorenz_core()
    analysis = analyse_timescales(core)
    selected = list(analysis.get("selected", []))
    if len(selected) < 2:
        raise RuntimeError(
            f"The shared Lorenz pipeline did not return T2; selected={selected}."
        )
    T2 = int(selected[1])

    # HAVOK uses the same raw Lorenz trajectory only as an external comparison.
    data = core["data"]
    tv, v11, active_segments = get_havok_v11(data["x"], data["t"])

    # Production beta trace at the automatically selected T2.
    tb, beta = result_for_window(core, T2)
    comparison, episode_rows = quantify_havok_closure(
        tv, v11, active_segments, tb, beta
    )
    tx = np.asarray(core["source_time"], dtype=float)
    xref = np.asarray(core["x_trace"], dtype=float)

    fig = plt.figure(figsize=(W_MM * st.MM, H_MM * st.MM), facecolor="white")

    # ------------------------------------------------------------------ (a)
    ax1 = axes_mm(fig, AX_X, TOP_Y, AX_W, TOP_H)
    ax1.plot(tv, v11, color="0.18", lw=0.72, zorder=1)
    for seg in active_segments:
        if len(seg):
            ax1.plot(tv[seg], v11[seg], color=st.C_BETA, lw=1.02, zorder=2)

    ax1.axhline(+HAVOK_THRESHOLD, color="0.58", lw=0.55,
                ls=(0, (3, 2)), zorder=0)
    ax1.axhline(-HAVOK_THRESHOLD, color="0.58", lw=0.55,
                ls=(0, (3, 2)), zorder=0)
    ax1.set_xlim(VIEW_START, VIEW_END)
    ax1.set_ylim(*padded_limits(values_in_view(tv, v11), pad=0.12, symmetric=True))
    ax1.xaxis.set_major_locator(FixedLocator(TIME_TICKS))
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.set_ylabel(r"$v_{11}$")
    paper_axes(ax1)

    ax1.text(
        0.012, 0.94, "HAVOK",
        transform=ax1.transAxes,
        ha="left", va="top",
        fontsize=FS_TEXT,
    )
    ax1.text(
        0.988, 0.94, rf"$|v_{{11}}|>{HAVOK_THRESHOLD:g}$",
        transform=ax1.transAxes,
        ha="right", va="top",
        fontsize=st.FS_TICK,
        color=st.C_BETA,
    )

    # ------------------------------------------------------------------ (b)
    ax2 = axes_mm(fig, AX_X, BOT_Y, AX_W, BOT_H)
    ax2r = ax2.twinx()
    ax2.set_zorder(ax2r.get_zorder() + 1)
    ax2.patch.set_visible(False)

    ax2.plot(tb, beta, color=st.C_BETA, lw=1.00, alpha=0.98)
    ax2r.plot(tx, xref, color=st.C_REF, lw=0.76, alpha=0.68)

    ax2.set_xlim(VIEW_START, VIEW_END)
    ax2.xaxis.set_major_locator(FixedLocator(TIME_TICKS))

    beta_view = values_in_view(tb, beta)
    if beta_view.size:
        ax2.set_ylim(0.0, 1.20 * float(np.max(beta_view)))
    ax2r.set_ylim(*padded_limits(values_in_view(tx, xref), pad=0.12))

    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\beta$", color=st.C_BETA)
    ax2r.set_ylabel(r"$x(t)$", color=st.C_REF, labelpad=2.0)
    ax2.tick_params(axis="y", colors=st.C_BETA)
    ax2r.tick_params(axis="y", colors=st.C_REF)
    ax2.spines["left"].set_color(st.C_BETA)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2r.yaxis.set_major_locator(MaxNLocator(nbins=3))
    format_scientific_y(ax2, st.C_BETA)
    paper_axes(ax2)

    ax2r.spines["right"].set_visible(True)
    ax2r.spines["right"].set_color(st.C_REF)
    ax2r.spines["right"].set_linewidth(st.LW_AXIS)
    ax2r.spines["top"].set_visible(False)
    ax2r.spines["left"].set_visible(False)
    ax2r.spines["bottom"].set_visible(False)

    ax2.text(
        0.012, 0.94, rf"Closure-based, $T_2={T2}$",
        transform=ax2.transAxes,
        ha="left", va="top",
        fontsize=FS_TEXT,
        color=st.C_SEL,
    )

    add_panel_letter(fig, TOP_Y + TOP_H, "(a)")
    add_panel_letter(fig, BOT_Y + BOT_H, "(b)")

    # Reproducibility metadata.
    metadata = FIGURE_DIR / f"{STEM}_metadata.csv"
    with open(metadata, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lorenz_rank", "T1", "T2", "rank_rule",
            "view_start", "view_end", "havok_delay_rows",
            "havok_component", "havok_threshold",
        ])
        writer.writerow([
            int(core["rank"]), int(selected[0]), T2,
            "cumulative_singular_values_0.98",
            VIEW_START, VIEW_END, HAVOK_DELAY_ROWS,
            HAVOK_COMPONENT, HAVOK_THRESHOLD,
        ])

    summary_path = save_comparison_summary(
        FIGURE_DIR / f"{STEM}_summary.csv",
        comparison,
        core,
        selected,
        T2,
    )
    episodes_path = save_episode_table(
        FIGURE_DIR / f"{STEM}_episodes.csv",
        episode_rows,
    )

    pdf_path = FIGURE_DIR / f"{STEM}.pdf"
    png_path = FIGURE_DIR / f"{STEM}.png"
    safe_savefig(fig, pdf_path, format="pdf", facecolor="white")
    safe_savefig(fig, png_path, dpi=600, facecolor="white")
    plt.close(fig)

    print("Supplementary Fig. S5")
    print("  raw data:", core["raw_file"])
    print("  retained Lorenz rank:", core["rank"])
    print("  selected scales:", selected[:2])
    print("  HAVOK threshold:", HAVOK_THRESHOLD)
    print(
        f"  common support: {comparison['common_start']:.3f}--"
        f"{comparison['common_end']:.3f}; "
        f"HAVOK-active fraction={comparison['havok_active_fraction']:.3f}"
    )
    print(
        f"  corr(|v11|, beta)={comparison['corr_abs_v11_beta']:.3f}; "
        f"mean beta active/inactive={comparison['beta_active_to_inactive']:.3f}x"
    )
    print(
        f"  top-10% beta during HAVOK activity="
        f"{comparison['top10_beta_active_fraction']:.3f}; "
        f"enrichment={comparison['top10_beta_enrichment']:.3f}x"
    )
    print(
        f"  HAVOK episodes={comparison['n_havok_episodes']}; "
        f"median episode beta-peak percentile="
        f"{comparison['median_episode_beta_peak_percentile']:.1f}%"
    )
    print(
        f"  episode-strength association: "
        f"Spearman rho={comparison['spearman_episode_strength']:.3f}, "
        f"p={comparison['spearman_episode_strength_p']:.3g}"
    )
    print(
        f"  within-episode peak lag (beta - HAVOK): "
        f"median={comparison['median_peak_lag']:.3g}, "
        f"IQR=[{comparison['lag_q25']:.3g}, {comparison['lag_q75']:.3g}], "
        f"median |lag|={comparison['median_abs_peak_lag']:.3g}"
    )
    print("  PDF:", pdf_path)
    print("  PNG:", png_path)
    print("  metadata:", metadata)
    print("  comparison summary:", summary_path)
    print("  episode table:", episodes_path)


if __name__ == "__main__":
    main()
