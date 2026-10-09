"""Supplementary Fig. S2: Rössler partial-observation validation.

Expected project layout
-----------------------
project_root/
├── closure_pipeline.py
├── figure_style.py
├── figS2_rossler.py
└── data/
    └── rossler_raw_data.npz

Run
---
    python figS2_rossler.py

Outputs
-------
    figures/figS2_rossler.pdf
    figures/figS2_rossler.png
    figures/figS2_rossler_summary.csv
    figures/figS2_rossler_hidden_event_alignment.csv

Partial observation
-------------------
Only x and y are observed; z is withheld from the closure calculation and is
used afterwards as an independent reference, because the large out-of-plane
excursions of the Rössler flow live in z.

Representation
--------------
* samples 2000:6000;
* coordinates measured from the inner equilibrium (x*, y*), the fixed point at
  the centre of the spiral, which anchors the local linear closure;
* 15 delays per observed coordinate, a 30-row delay matrix H (not centred);
* rank from the shared rule on the singular values of H;
* the reduced coordinates are differentiated numerically.

Closure field, S(T), reorganization events and the characteristic scale come
from closure_pipeline.analyze_timescales.  The figure follows the Lorenz
layout: scale-time map and S(T) on the top row, beta with the hidden z below.
"""

from pathlib import Path
import csv
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import spearmanr
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FixedLocator, FuncFormatter
from matplotlib.lines import Line2D

import closure_pipeline as cp
import figure_style as st


# =============================================================================
# Project layout
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
RAW_FILE = DATA_DIR / "rossler_raw_data.npz"
STEM = "figS2_rossler"


# =============================================================================
# Rössler analysis settings
# =============================================================================
SEGMENT_START = 2000
SEGMENT_END = 6000
DELAY_LENGTH = 15
RANK_FRACTION = cp.RANK_CUMULATIVE_SINGULAR_VALUE     # shared rule, 0.98

# Rössler parameters of the simulation behind rossler_raw_data.npz; they fix
# the anchor.  Values stored in the data file take precedence.
ROSSLER_A, ROSSLER_B, ROSSLER_C = 0.2, 0.2, 5.7

T_SCAN_MIN = 50
T_SCAN_MAX_FRACTION = 0.25
MAX_SELECTED_SCALES = 1


# =============================================================================
# Independent hidden-z reference
# =============================================================================
# z is withheld from scale selection and closure calculation. These settings
# identify its large out-of-plane excursions for independent comparison.
Z_PEAK_HEIGHT_QUANTILE = 0.90
Z_PEAK_PROMINENCE_FRACTION = 0.25
Z_PEAK_MIN_GAP_TIME = 2.0


# =============================================================================
# Figure geometry -- same grammar as Lorenz, shortened because there is one
# event-level trace rather than two.
# =============================================================================
W_MM, H_MM = 174.0, 92.0
MAP_TIME_MIN, MAP_TIME_MAX = 20.0, 60.0
MAP_TIME_TICKS = np.arange(20.0, 61.0, 10.0)


# =============================================================================
# Data preparation
# =============================================================================
def require_raw_file():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Could not find data/rossler_raw_data.npz.\n"
            "Expected project layout:\n"
            "  project_root/figS2_rossler.py\n"
            "  project_root/data/rossler_raw_data.npz"
        )
    return RAW_FILE


def inner_equilibrium(a, b, c):
    """(x*, y*) of the Rössler fixed point at the centre of the spiral.

    Fixed points satisfy y = -x/a, z = x/a and x^2 - c x + a b = 0; the inner
    one is the smaller root.
    """
    x_star = 0.5 * (c - np.sqrt(c * c - 4.0 * a * b))
    return x_star, -x_star / a


def build_rossler_core():
    """Partial-observation representation anchored at the inner equilibrium."""
    raw_file = require_raw_file()
    data = np.load(raw_file)

    required = ("t", "x", "y", "z")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(
            f"{raw_file.name} is missing required arrays: {missing}"
        )

    if SEGMENT_END > len(data["t"]):
        raise ValueError(
            f"Rössler trajectory has {len(data['t'])} samples, "
            f"but SEGMENT_END={SEGMENT_END}."
        )

    segment = slice(SEGMENT_START, SEGMENT_END)
    t_segment = np.asarray(data["t"][segment], dtype=float)
    x_segment = np.asarray(data["x"][segment], dtype=float)
    y_segment = np.asarray(data["y"][segment], dtype=float)
    z_segment = np.asarray(data["z"][segment], dtype=float)

    a, b, c = (
        float(data[key]) if key in data.files else default
        for key, default in (("a", ROSSLER_A), ("b", ROSSLER_B), ("c", ROSSLER_C))
    )
    x_star, y_star = inner_equilibrium(a, b, c)
    observed = np.vstack((x_segment - x_star, y_segment - y_star))

    blocks = []
    for row in observed:
        p = len(row) - DELAY_LENGTH + 1
        blocks.append(
            np.vstack([
                row[k:k + p]
                for k in range(DELAY_LENGTH)
            ])
        )
    H = np.vstack(blocks)

    # H is already measured from the anchor, so it is not centred.
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    rank = cp.retained_rank_from_singular_values(s, RANK_FRACTION)
    V = s[:rank, None] * Vh[:rank]       # U_r^T H

    # Delay-coordinate time is assigned to the center of each delay block.
    source_time = 0.5 * (
        t_segment[:H.shape[1]]
        + t_segment[
            DELAY_LENGTH - 1:
            DELAY_LENGTH - 1 + H.shape[1]
        ]
    )

    # Only positions are observed, so the velocity is a numerical derivative.
    Vdot = np.gradient(
        V,
        source_time,
        axis=1,
        edge_order=2,
    )
    dt = float(np.median(np.diff(source_time)))

    return {
        "raw_file": raw_file,
        "data": data,
        "t_segment": t_segment,
        "x_segment": x_segment,
        "y_segment": y_segment,
        "z_segment": z_segment,
        "H": H,
        "anchor": (x_star, y_star),
        "singular_values": s,
        "rank": rank,
        "V": V,
        "Vdot": Vdot,
        "source_time": source_time,
        "dt": dt,
        "sums": cp.prepare_window_sums(V, Vdot),
    }


def analyse_timescales(core):
    """Shared closure analysis over T up to a quarter of the record."""
    n = core["V"].shape[1]
    return cp.analyze_timescales(
        core["sums"],
        core["source_time"],
        core["dt"],
        t_min=T_SCAN_MIN,
        t_max=min(n - 1, int(np.floor(T_SCAN_MAX_FRACTION * n))),
        max_windows=MAX_SELECTED_SCALES,
        progress=True,
    )


# =============================================================================
# Independent hidden-coordinate reference
# =============================================================================
def hidden_z_excursions(core):
    """Detect prominent z excursions without using z in closure selection."""
    z = core["z_segment"]
    t = core["t_segment"]

    med = float(np.median(z))

    # The withheld reference is highly skewed: most of the time z stays close
    # to its low baseline and only occasionally makes a large excursion.
    # A high quantile therefore separates the visually meaningful excursions
    # more cleanly than a MAD threshold around the near-zero baseline.
    height = float(np.quantile(z, Z_PEAK_HEIGHT_QUANTILE))
    prominence = max(
        Z_PEAK_PROMINENCE_FRACTION * (height - med),
        1e-12,
    )

    dt_raw = float(np.median(np.diff(t)))
    min_distance = max(
        1,
        int(round(Z_PEAK_MIN_GAP_TIME / dt_raw)),
    )

    idx, props = find_peaks(
        z,
        height=height,
        prominence=prominence,
        distance=min_distance,
    )

    return {
        "indices": idx,
        "times": t[idx],
        "values": z[idx],
        "height_threshold": height,
        "prominence_threshold": prominence,
        "properties": props,
    }


def result_for_window(core, T_selected):
    T_selected = int(T_selected)
    beta, _ = cp.compute_metrics_for_T(
        *core["sums"],
        T_selected,
    )
    beta_time = (
        core["source_time"][:len(beta)]
        + 0.5 * T_selected * core["dt"]
    )

    closure_idx, closure_time, _ = cp.detect_closure_events(
        beta,
        core["source_time"][:len(beta)],
        T_selected,
        core["dt"],
    )
    closure_z = np.interp(
        closure_time,
        core["t_segment"],
        core["z_segment"],
    )

    return {
        "T": T_selected,
        "beta": beta,
        "beta_time": beta_time,
        "closure_idx": closure_idx,
        "closure_time": closure_time,
        "closure_z": closure_z,
        "z_reference": hidden_z_excursions(core),
    }


# =============================================================================
# Quantitative hidden-event comparison
# =============================================================================
def quantify_hidden_event_alignment(core, result):
    """Quantify closure organization around withheld-z excursions.

    Each z excursion is characterized by its peak and by the left
    half-prominence crossing, which provides a parameter-free marker of the
    rapid rising phase.  The closure response is then evaluated at and around
    these independently defined times.
    """
    beta_time = np.asarray(result["beta_time"], dtype=float)
    beta = np.asarray(result["beta"], dtype=float)
    zref = result["z_reference"]

    T1 = int(result["T"])
    dt = float(core["dt"])
    half_window = 0.5 * T1 * dt

    finite_beta = beta[np.isfinite(beta)]
    if finite_beta.size == 0:
        raise RuntimeError("No finite beta values available for hidden-event validation.")

    q90 = float(np.quantile(finite_beta, 0.90))
    q95 = float(np.quantile(finite_beta, 0.95))

    # The left half-prominence crossing follows directly from each detected
    # excursion geometry and adds no new event-selection parameter.
    z = np.asarray(core["z_segment"], dtype=float)
    tz_all = np.asarray(core["t_segment"], dtype=float)
    peak_idx = np.asarray(zref["indices"], dtype=int)

    if peak_idx.size:
        widths = peak_widths(z, peak_idx, rel_height=0.5)
        left_ips = np.asarray(widths[2], dtype=float)
        sample_index = np.arange(len(tz_all), dtype=float)
        rise_times = np.interp(left_ips, sample_index, tz_all)
    else:
        rise_times = np.asarray([], dtype=float)

    rows = []
    for event_id, (idx, tz_peak, zval, tz_rise) in enumerate(
        zip(peak_idx, zref["times"], zref["values"], rise_times),
        start=1,
    ):
        tz_peak = float(tz_peak)
        tz_rise = float(tz_rise)
        in_support = bool(
            tz_rise >= beta_time[0]
            and tz_rise <= beta_time[-1]
            and tz_peak >= beta_time[0]
            and tz_peak <= beta_time[-1]
        )

        row = {
            "event": event_id,
            "z_peak_index": int(idx),
            "z_rise_time": tz_rise,
            "z_peak_time": tz_peak,
            "z_peak_value": float(zval),
            "rise_to_peak_time": tz_peak - tz_rise,
            "half_window": half_window,
            "in_beta_support": in_support,
            "beta_at_z_rise": np.nan,
            "beta_at_z_rise_percentile": np.nan,
            "beta_at_z_peak": np.nan,
            "beta_at_z_peak_percentile": np.nan,
            "local_beta_peak_time": np.nan,
            "local_beta_peak": np.nan,
            "lag_beta_peak_from_z_rise": np.nan,
            "abs_lag_from_z_rise": np.nan,
            "lag_over_Tdt": np.nan,
            "local_peak_percentile": np.nan,
            "local_peak_ge_q90": False,
            "local_peak_ge_q95": False,
        }

        if in_support:
            beta_at_rise = float(np.interp(tz_rise, beta_time, beta))
            beta_at_peak = float(np.interp(tz_peak, beta_time, beta))

            rise_percentile = 100.0 * float(np.mean(finite_beta <= beta_at_rise))
            peak_percentile = 100.0 * float(np.mean(finite_beta <= beta_at_peak))

            # Use the same T1 temporal support around the independently defined
            # rising phase to summarize the associated closure response.
            mask = (
                (beta_time >= tz_rise - half_window)
                & (beta_time <= tz_rise + half_window)
                & np.isfinite(beta)
            )
            if np.any(mask):
                local_t = beta_time[mask]
                local_beta = beta[mask]
                j = int(np.argmax(local_beta))

                local_peak_time = float(local_t[j])
                local_peak = float(local_beta[j])
                lag = local_peak_time - tz_rise
                percentile = 100.0 * float(np.mean(finite_beta <= local_peak))

                row.update({
                    "beta_at_z_rise": beta_at_rise,
                    "beta_at_z_rise_percentile": rise_percentile,
                    "beta_at_z_peak": beta_at_peak,
                    "beta_at_z_peak_percentile": peak_percentile,
                    "local_beta_peak_time": local_peak_time,
                    "local_beta_peak": local_peak,
                    "lag_beta_peak_from_z_rise": lag,
                    "abs_lag_from_z_rise": abs(lag),
                    "lag_over_Tdt": lag / (T1 * dt),
                    "local_peak_percentile": percentile,
                    "local_peak_ge_q90": bool(local_peak >= q90),
                    "local_peak_ge_q95": bool(local_peak >= q95),
                })

        rows.append(row)

    valid = [r for r in rows if r["in_beta_support"]]
    if valid:
        signed_lags = np.asarray(
            [r["lag_beta_peak_from_z_rise"] for r in valid], dtype=float
        )
        abs_lags = np.abs(signed_lags)
        peak_percentiles = np.asarray(
            [r["local_peak_percentile"] for r in valid], dtype=float
        )
        rise_percentiles = np.asarray(
            [r["beta_at_z_rise_percentile"] for r in valid], dtype=float
        )
        z_amplitudes = np.asarray([r["z_peak_value"] for r in valid], dtype=float)
        local_peaks = np.asarray([r["local_beta_peak"] for r in valid], dtype=float)

        # With only a few excursions, rank association is the most transparent
        # summary of whether stronger hidden excursions produce stronger closure
        # responses.
        if len(valid) >= 2:
            rho_amp, p_amp = spearmanr(z_amplitudes, local_peaks)
            rho_amp = float(rho_amp)
            p_amp = float(p_amp)
        else:
            rho_amp, p_amp = np.nan, np.nan

        summary = {
            "n_hidden_events_total": len(rows),
            "n_hidden_events_in_beta_support": len(valid),
            "half_window": half_window,
            "median_signed_lag_from_z_rise": float(np.median(signed_lags)),
            "median_abs_lag_from_z_rise": float(np.median(abs_lags)),
            "max_abs_lag_from_z_rise": float(np.max(abs_lags)),
            "median_beta_at_z_rise_percentile": float(np.median(rise_percentiles)),
            "median_local_peak_percentile": float(np.median(peak_percentiles)),
            "n_local_peak_ge_q90": int(sum(r["local_peak_ge_q90"] for r in valid)),
            "n_local_peak_ge_q95": int(sum(r["local_peak_ge_q95"] for r in valid)),
            "spearman_z_peak_vs_local_beta_peak": rho_amp,
            "spearman_z_peak_vs_local_beta_peak_p": p_amp,
        }
    else:
        summary = {
            "n_hidden_events_total": len(rows),
            "n_hidden_events_in_beta_support": 0,
            "half_window": half_window,
            "median_signed_lag_from_z_rise": np.nan,
            "median_abs_lag_from_z_rise": np.nan,
            "max_abs_lag_from_z_rise": np.nan,
            "median_beta_at_z_rise_percentile": np.nan,
            "median_local_peak_percentile": np.nan,
            "n_local_peak_ge_q90": 0,
            "n_local_peak_ge_q95": 0,
            "spearman_z_peak_vs_local_beta_peak": np.nan,
            "spearman_z_peak_vs_local_beta_peak_p": np.nan,
        }

    return summary, rows


def save_hidden_event_alignment(path, rows):
    """Save one row for every independently detected withheld-z excursion."""

    fields = [
        "event",
        "z_peak_index",
        "z_rise_time",
        "z_peak_time",
        "z_peak_value",
        "rise_to_peak_time",
        "half_window",
        "in_beta_support",
        "beta_at_z_rise",
        "beta_at_z_rise_percentile",
        "beta_at_z_peak",
        "beta_at_z_peak_percentile",
        "local_beta_peak_time",
        "local_beta_peak",
        "lag_beta_peak_from_z_rise",
        "abs_lag_from_z_rise",
        "lag_over_Tdt",
        "local_peak_percentile",
        "local_peak_ge_q90",
        "local_peak_ge_q95",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# =============================================================================
# Output helpers
# =============================================================================
def safe_savefig(fig, target, **kwargs):
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"

    with tempfile.TemporaryDirectory(
        prefix="closure_figS2_"
    ) as tmp_dir:
        tmp = Path(tmp_dir) / f"render{suffix}"
        fig.savefig(tmp, **kwargs)
        try:
            shutil.copy2(tmp, target)
        except OSError as exc:
            print(
                f"[WARNING] Temporary render succeeded "
                f"but copy failed: {exc}"
            )
            print(f"          Trying direct save to: {target}")
            fig.savefig(target, **kwargs)

    return target


def save_summary(path, core, analysis, result, validation):
    """One-line record of the representation, selected scale and hidden-event test."""
    T1, ev = analysis["selected"][0], analysis["selected_events"][0]
    x_star, y_star = core["anchor"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "system,x_star,y_star,rank,rank_rule,"
            "T1,T1_Tmin,T1_Thalf,T1_Tpre,T1_depth,T1_prominence,T1_d_org,"
            "hidden_z_excursions_total,hidden_z_excursions_in_beta_support,"
            "validation_half_window,"
            "median_signed_lag_from_z_rise,median_abs_lag_from_z_rise,"
            "max_abs_lag_from_z_rise,median_beta_at_z_rise_percentile,"
            "median_local_peak_percentile,n_local_peak_ge_q90,n_local_peak_ge_q95,"
            "spearman_z_peak_vs_local_beta_peak,"
            "spearman_z_peak_vs_local_beta_peak_p\n"
        )
        f.write(
            f"Rossler,{x_star},{y_star},{core['rank']},cumulative_singular_values_0.98,"
            f"{T1},{ev['T_min']},{ev['T_half']},{ev['T_pre_recovery']},"
            f"{ev['depth']},{ev['prominence']},{ev['d_org']},"
            f"{validation['n_hidden_events_total']},"
            f"{validation['n_hidden_events_in_beta_support']},"
            f"{validation['half_window']},"
            f"{validation['median_signed_lag_from_z_rise']},"
            f"{validation['median_abs_lag_from_z_rise']},"
            f"{validation['max_abs_lag_from_z_rise']},"
            f"{validation['median_beta_at_z_rise_percentile']},"
            f"{validation['median_local_peak_percentile']},"
            f"{validation['n_local_peak_ge_q90']},"
            f"{validation['n_local_peak_ge_q95']},"
            f"{validation['spearman_z_peak_vs_local_beta_peak']},"
            f"{validation['spearman_z_peak_vs_local_beta_peak_p']}\n"
        )


# =============================================================================
# Figure helpers
# =============================================================================
def axes_mm(fig, x, y, w, h):
    return fig.add_axes([
        x / W_MM,
        y / H_MM,
        w / W_MM,
        h / H_MM,
    ])


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


def paper_axes(ax):
    """Shared single-axis grammar; twin axes restore the active right spine."""
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


def visible_ylim(values, times, xlim, pad=0.12, zero_floor=False):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    mask = (
        np.isfinite(values)
        & np.isfinite(times)
        & (times >= float(xlim[0]))
        & (times <= float(xlim[1]))
    )
    vals = values[mask]
    if len(vals) == 0:
        vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return (0.0, 1.0)

    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = max(hi - lo, 1e-12)

    if zero_floor and lo >= 0.0:
        return (0.0, hi + pad * max(hi, span))

    return (
        lo - pad * span,
        hi + pad * span,
    )


# =============================================================================
# Main figure
# =============================================================================
def main():
    fam = st.apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    core = build_rossler_core()
    analysis = analyse_timescales(core)
    selected = list(analysis["selected"])

    result = result_for_window(core, selected[0])
    validation, validation_rows = quantify_hidden_event_alignment(core, result)
    summary_path = FIGURE_DIR / f"{STEM}_summary.csv"
    alignment_path = FIGURE_DIR / f"{STEM}_hidden_event_alignment.csv"
    save_summary(summary_path, core, analysis, result, validation)
    save_hidden_event_alignment(alignment_path, validation_rows)

    print("Rössler partial-observation figure")
    print("  raw data:", core["raw_file"])
    print("  anchor (x*, y*):", np.round(core["anchor"], 5))
    print("  H shape:", core["H"].shape)
    print("  retained rank:", core["rank"])
    print("  selected T:", selected)
    # Only the first band is used for the figure; all retained bands are
    # listed so that the full hierarchy can be reported.
    print("  all retained bands (T*):", [int(T) for T in analysis["hierarchy"]])
    for j, e in enumerate(analysis["persistent_events"], start=1):
        print(f"    band {j}: [{e['T_left']:.1f}, {e['T_min']:.1f}, {e['T_right']:.1f}], "
              f"prominence={e['prominence']:.4f}")

    ev = analysis["selected_events"][0]
    print(
        f"  event: [{ev['T_left']:.1f}, {ev['T_min']:.1f}, {ev['T_right']:.1f}], "
        f"T_half={ev['T_half']:.1f}, T_pre={ev['T_pre_recovery']}, "
        f"depth={ev['depth']:.4f}, prominence={ev['prominence']:.4f}, "
        f"d_org={ev['d_org']:.3f}"
    )
    print(
        "  hidden-z excursion times:",
        np.round(result["z_reference"]["times"], 3),
    )
    print(
        "  shared beta-event times:",
        np.round(result["closure_time"], 3),
    )
    print(
        f"  withheld-z excursions: "
        f"{validation['n_hidden_events_in_beta_support']}/"
        f"{validation['n_hidden_events_total']} within centered beta support"
    )
    print(
        f"  rise-phase alignment: median lag="
        f"{validation['median_signed_lag_from_z_rise']:.3g}, "
        f"median |lag|={validation['median_abs_lag_from_z_rise']:.3g}"
    )
    print(
        f"  closure strength near excursions: median peak percentile="
        f"{validation['median_local_peak_percentile']:.1f}%; "
        f">=90th={validation['n_local_peak_ge_q90']}/"
        f"{validation['n_hidden_events_in_beta_support']}, "
        f">=95th={validation['n_local_peak_ge_q95']}/"
        f"{validation['n_hidden_events_in_beta_support']}"
    )
    print(
        f"  excursion-strength association: "
        f"Spearman rho={validation['spearman_z_peak_vs_local_beta_peak']:.3f}, "
        f"p={validation['spearman_z_peak_vs_local_beta_peak_p']:.3g}"
    )

    fig = plt.figure(
        figsize=(W_MM * st.MM, H_MM * st.MM),
        facecolor="white",
    )

    # ------------------------------------------------------------------ (a)
    # D(t,T) on the fine log-T grid; the white guide marks the selected scale.
    ax_map = axes_mm(fig, 13.0, 51.0, 73.0, 34.0)
    mesh = ax_map.pcolormesh(
        analysis["common_time"],
        analysis["display_T_values"],
        cp.globally_scaled_map(
            analysis["display_scale_map"]
        ),
        shading="auto",
        rasterized=True,
    )
    ax_map.set_yscale("log")
    ax_map.set_xlim(MAP_TIME_MIN, MAP_TIME_MAX)
    ax_map.xaxis.set_major_locator(
        FixedLocator(MAP_TIME_TICKS)
    )
    ax_map.set_xlabel(r"$t$")
    ax_map.set_ylabel(r"Window size $T$")
    ax_map.axhline(
        selected[0],
        color="white",
        lw=st.LW_GUIDE,
        ls=(0, (3, 2)),
        alpha=0.95,
    )
    paper_axes(ax_map)

    cax = axes_mm(fig, 88.0, 51.0, 2.0, 34.0)
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
    # S(T) on the baseline grid: raw neighbouring-scale correlations (faint)
    # and the smoothed S on which the reorganization event is found.
    ax_s = axes_mm(fig, 114.0, 51.0, 46.5, 34.0)
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
    # Keep the true logarithmic geometry, but use only a few readable major
    # labels.  The displayed numbers are scaled by 10^2 to avoid the crowded
    # default log labels.
    ax_s.set_xscale("log")
    ax_s.set_xlim(float(Ts[0]), float(Ts[-1]))
    ax_s.xaxis.set_major_locator(
        FixedLocator([100.0, 200.0, 500.0])
    )
    ax_s.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x/100:g}")
    )
    ax_s.tick_params(axis="x", which="minor", labelbottom=False)

    sv = np.asarray(
        diag["smooth_actual"],
        dtype=float,
    )
    sv = sv[np.isfinite(sv)]
    if len(sv) > 0:
        yr = max(
            float(np.max(sv) - np.min(sv)),
            1e-4,
        )
        ylo = max(
            0.0,
            float(np.min(sv)) - 0.15 * yr,
        )
        yhi = min(
            1.015,
            float(np.max(sv)) + 0.18 * yr,
        )
        ax_s.set_ylim(ylo, yhi)

    T1 = selected[0]
    sval = float(np.interp(
        np.log(T1),
        diag["log_grid"],
        diag["smooth"],
    ))
    ax_s.axvline(
        T1,
        ymin=0.0,
        ymax=0.84,
        color=st.C_SEL,
        ls="-.",
        lw=st.LW_GUIDE,
        alpha=0.88,
    )
    ax_s.plot(
        T1,
        sval,
        "o",
        ms=3.1,
        color=st.C_SEL,
    )
    ax_s.annotate(
        rf"$T_1={T1}$",
        xy=(T1, sval),
        xytext=(9, 12),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=st.C_SEL,
        fontsize=st.FS_TEXT,
        arrowprops=dict(
            arrowstyle="-",
            color=st.C_SEL,
            lw=0.6,
            shrinkA=2,
            shrinkB=2,
        ),
        annotation_clip=False,
    )
    ax_s.set_xlabel(r"Window size $T$")
    ax_s.set_ylabel(r"$S(T)$")
    ax_s.text(0.985, -0.16, r"$\times 10^2$", transform=ax_s.transAxes, ha="right", va="top", fontsize=st.FS_TICK)
    paper_axes(ax_s)

    # ------------------------------------------------------------------ (c)
    # beta(t) at T1 from (x, y) alone, against the withheld z and its large
    # excursions.
    ax_beta = axes_mm(fig, 18.0, 8.0, 137.0, 27.0)
    ax_z = ax_beta.twinx()

    beta_time = result["beta_time"]
    beta = result["beta"]

    # For Rössler, the selected window is several physical time units long.
    # Restrict the displayed trace to the common valid support of centered beta
    # rather than showing blank margins at the raw segment edges.
    trace_xlim = (
        float(beta_time[0]),
        float(beta_time[-1]),
    )

    ax_z.plot(
        core["t_segment"],
        core["z_segment"],
        color=st.C_REF,
        lw=0.84,
        alpha=0.65,
        zorder=1,
    )

    closure_time = np.asarray(result["closure_time"], dtype=float)
    closure_z = np.asarray(result["closure_z"], dtype=float)
    visible_events = (
        (closure_time >= trace_xlim[0])
        & (closure_time <= trace_xlim[1])
    )
    ax_z.scatter(
        closure_time[visible_events],
        closure_z[visible_events],
        s=18,
        color="#FF8C00",
        edgecolors="black",
        linewidth=0.35,
        zorder=5,
    )

    ax_beta.plot(
        beta_time,
        beta,
        color=st.C_BETA,
        lw=1.08,
        alpha=0.96,
        zorder=3,
    )

    ax_beta.set_xlim(*trace_xlim)
    ax_z.set_xlim(*trace_xlim)

    # Integer-ish 5-unit ticks inside the valid range.
    tick0 = 5.0 * np.ceil(trace_xlim[0] / 5.0)
    tick1 = 5.0 * np.floor(trace_xlim[1] / 5.0)
    if tick1 >= tick0:
        ax_beta.xaxis.set_major_locator(
            FixedLocator(
                np.arange(tick0, tick1 + 0.1, 5.0)
            )
        )

    ax_beta.set_ylim(
        *visible_ylim(
            beta,
            beta_time,
            trace_xlim,
            pad=0.12,
            zero_floor=True,
        )
    )
    ax_z.set_ylim(
        *visible_ylim(
            core["z_segment"],
            core["t_segment"],
            trace_xlim,
            pad=0.10,
            zero_floor=True,
        )
    )

    ax_beta.set_ylim(bottom=-0.03 * ax_beta.get_ylim()[1])
    ax_z.set_ylim(bottom=-0.03 * ax_z.get_ylim()[1])

    ax_beta.yaxis.set_major_locator(
        MaxNLocator(nbins=4)
    )
    ax_z.yaxis.set_major_locator(
        MaxNLocator(nbins=4)
    )
    ax_beta.tick_params(
        axis="y",
        colors=st.C_BETA,
        labelsize=st.FS_TICK,
        pad=1.5,
    )
    ax_z.tick_params(
        axis="y",
        colors=st.C_REF,
        labelsize=st.FS_TICK,
        pad=1.5,
    )
    format_scientific_y(ax_beta, st.C_BETA)

    ax_beta.spines["left"].set_color(st.C_BETA)
    ax_beta.spines["right"].set_visible(False)
    ax_beta.spines["top"].set_visible(False)

    ax_z.spines["right"].set_visible(True)
    ax_z.spines["right"].set_color(st.C_REF)
    ax_z.spines["right"].set_linewidth(st.LW_AXIS)
    ax_z.spines["left"].set_visible(False)
    ax_z.spines["top"].set_visible(False)

    ax_beta.set_xlabel(r"$t$")

    ax_beta.text(
        0.016,
        0.95,
        rf"$T_1={T1}$",
        transform=ax_beta.transAxes,
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

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            ls="none",
            ms=4.2,
            markerfacecolor="#FF8C00",
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=r"shared $\beta$ events",
        )
    ]
    leg = ax_beta.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.56, 1.04),
        frameon=True,
        fontsize=7.2,
        handletextpad=0.35,
        borderaxespad=0.0,
        fancybox=False,
        framealpha=0.96,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.35")
    leg.get_frame().set_linewidth(0.55)

    paper_axes(ax_beta)

    fig.text(
        7.0 / W_MM,
        21.5 / H_MM,
        r"$\beta$",
        color=st.C_BETA,
        fontsize=st.FS_TEXT,
        rotation=90,
        ha="center",
        va="center",
    )
    fig.text(
        169.0 / W_MM,
        21.5 / H_MM,
        r"$z(t)$",
        color=st.C_REF,
        fontsize=st.FS_TEXT,
        rotation=90,
        ha="center",
        va="center",
    )

    add_panel_letter(fig, 1.0, 90.0, "(a)")
    add_panel_letter(fig, 100.0, 90.0, "(b)")
    add_panel_letter(fig, 1.0, 40.0, "(c)")

    pdf_path = FIGURE_DIR / f"{STEM}.pdf"
    png_path = FIGURE_DIR / f"{STEM}.png"

    safe_savefig(
        fig,
        pdf_path,
        format="pdf",
        facecolor="white",
    )
    safe_savefig(
        fig,
        png_path,
        dpi=600,
        facecolor="white",
    )
    plt.close(fig)

    print("  font:", fam)
    print("  PDF:", pdf_path)
    print("  preview:", png_path)
    print("  summary:", summary_path)
    print("  hidden-event alignment:", alignment_path)


if __name__ == "__main__":
    main()
