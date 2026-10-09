from __future__ import annotations

"""Supplementary Fig. S4: multilevel AdK coordinate-noise robustness.

A post-selection coordinate-noise robustness test complementary to Fig. S3.

The production AdK scales T1 and T2 are kept fixed. Gaussian coordinate noise
is added only to the CORE observation after alignment to closedCA. The clean
CORE basis (rank 3) is fitted once and then held fixed for all noise levels.

Three vertically stacked panels:
    (a) first CORE reduced coordinate over a short representative interval;
    (b) standardized closure response z(beta) at T1;
    (c) standardized closure response z(beta) at T2.

The clean trace is a black dashed reference. Increasing noise levels are shown
by a perceptually ordered color gradient.
"""

from pathlib import Path
import csv
import gc
import shutil
import tempfile
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

import closure_pipeline as cp
import figure_style as st
import fig4_adk as adk

warnings.filterwarnings("ignore")


# =============================================================================
# Paths and test settings
# =============================================================================
ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
STEM = "figS4_adk_noise"

# MDTraj coordinates are in nm.
#
# Noise levels span moderate to strong coordinate perturbations while retaining
# a compact progression that is easy to compare in one figure.
NOISE_LEVELS_ANGSTROM = (0.2, 0.5, 1.0, 2.0)
PLOT_SEED = 7
SUMMARY_SEEDS = (3, 7, 11, 19, 29)

# Within each seed, one Gaussian realization is scaled across all sigma values.
# Independent seeds provide the realization-to-realization summary.
CORE_RANK = 3
DEFAULT_T1 = 100
DEFAULT_T2 = 501

# Panel (a) uses a shorter interval so the perturbation is actually visible.
TOP_WINDOW_FRAMES = 1500

TIME_TICK_STEP = 5000
TIME_EXPONENT = 4

# Figure geometry: one compact input panel, followed by two response panels.
W_MM, H_MM = 174.0, 132.0
LEFT_MM, RIGHT_MM = 16.0, 6.0
AX_W_MM = W_MM - LEFT_MM - RIGHT_MM

A_Y, A_H = 96.0, 21.0
B_Y, B_H = 57.0, 27.0
C_Y, C_H = 18.0, 27.0

LETTER_X = 2.0

CLEAN_COLOR = "0.10"
CLEAN_LS = (0, (4.0, 2.2))
CLEAN_LW = 1.15

# Perceptually ordered gradient: increasing sigma moves along viridis.
_cmap = plt.get_cmap("viridis")
NOISE_COLORS = [
    _cmap(x) for x in np.linspace(0.28, 0.86, len(NOISE_LEVELS_ANGSTROM))
]


# =============================================================================
# General helpers
# =============================================================================
def safe_savefig(fig, target, **kwargs):
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"
    with tempfile.TemporaryDirectory(prefix="closure_figS4_") as tmp_dir:
        tmp = Path(tmp_dir) / f"render{suffix}"
        fig.savefig(tmp, **kwargs)
        try:
            shutil.copy2(tmp, target)
        except OSError:
            fig.savefig(target, **kwargs)
    return target


def axes_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])


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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def add_axis_multiplier(ax, exponent, x=1.0):
    ax.annotate(
        rf"$\times 10^{{{int(exponent)}}}$",
        xy=(x, 0.0),
        xycoords="axes fraction",
        xytext=(0.0, -15.0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=0.90 * st.FS_TICK,
        annotation_clip=False,
    )


def format_time_axis(ax, t0, t1):
    """ADK time-axis grammar used consistently in all three panels."""
    t0, t1 = float(t0), float(t1)

    start = int(np.ceil(t0 / TIME_TICK_STEP) * TIME_TICK_STEP)
    end = int(np.floor(t1 / TIME_TICK_STEP) * TIME_TICK_STEP)
    ticks = np.arange(start, end + 1, TIME_TICK_STEP, dtype=float)

    # The short top panel may span less than one 5000-frame interval.
    if ticks.size < 2:
        ticks = np.linspace(t0, t1, 3)

    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x / 10.0**TIME_EXPONENT:.2g}")
    )
    add_axis_multiplier(ax, TIME_EXPONENT)
    ax.set_xlabel(r"$t$", labelpad=1.6)


def zscore(y):
    y = np.asarray(y, dtype=float)
    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(y)
    return (y - mu) / sd


def standardize_against_clean(y, clean_mu, clean_sd):
    y = np.asarray(y, dtype=float)
    if not np.isfinite(clean_sd) or clean_sd < 1e-12:
        return y - clean_mu
    return (y - clean_mu) / clean_sd


def headroom_ylim(values, qlo=0.01, qhi=0.99, pad_lo=0.10, pad_hi=0.18):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (-1.0, 1.0)

    lo = float(np.quantile(x, qlo))
    hi = float(np.quantile(x, qhi))
    span = max(hi - lo, 1e-8)
    return (lo - pad_lo * span, hi + pad_hi * span)


def read_selected_scales():
    path = FIGURE_DIR / "fig4_adk_selected_windows.csv"
    if not path.exists():
        return DEFAULT_T1, DEFAULT_T2

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return DEFAULT_T1, DEFAULT_T2

    row = rows[0]
    try:
        T1 = int(round(float(row.get("T1", DEFAULT_T1))))
        T2 = int(round(float(row.get("T2", DEFAULT_T2))))
        return T1, T2
    except Exception:
        return DEFAULT_T1, DEFAULT_T2


# =============================================================================
# Compatibility with current fig4_adk.py
# =============================================================================
def load_adk_segment():
    """Require the current four-return-value main-figure interface."""
    loaded = adk.load_aligned_analysis_segment(chunk_size=2000)

    if not isinstance(loaded, (tuple, list)) or len(loaded) != 4:
        n = len(loaded) if isinstance(loaded, (tuple, list)) else "non-tuple"
        raise RuntimeError(
            "figS4_adk_noise.py expects the current fig4_adk.py interface:\n"
            "  xyz, source_time, reference_xyz, residue_ids = "
            "load_aligned_analysis_segment(...)\n"
            f"but received {n} return values."
        )

    xyz, source_time, reference_xyz, residue_ids = loaded

    return (
        np.asarray(xyz, dtype=np.float32),
        np.asarray(source_time, dtype=float),
        np.asarray(reference_xyz, dtype=np.float32),
        np.asarray(residue_ids, dtype=int),
    )


def core_coordinate_rows(residue_ids):
    """Use the same PDB-residue definition of CORE as the main figure."""
    if hasattr(adk, "residue_groups_to_coordinate_rows"):
        rows = adk.residue_groups_to_coordinate_rows(residue_ids, ("CORE",))
        return np.asarray(rows, dtype=int)

    atom_idx = np.flatnonzero((residue_ids >= 1) & (residue_ids <= 20))
    if atom_idx.size != 20:
        raise RuntimeError("Could not map CORE residues 1--20.")

    rows = np.concatenate([3 * atom_idx + k for k in range(3)])
    return np.sort(rows)


# =============================================================================
# Fixed clean CORE representation
# =============================================================================
def fit_clean_core_basis(H_core, rank=CORE_RANK):
    H_core = np.asarray(H_core, dtype=np.float64)
    gram = np.asarray(H_core @ H_core.T, dtype=np.float64)

    evals, U = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    U = U[:, order]

    r = min(int(rank), U.shape[1])
    Ur = np.asarray(U[:, :r], dtype=np.float64)

    cumulative_energy = np.cumsum(evals) / np.sum(evals)
    retained_energy = float(cumulative_energy[r - 1])

    return Ur, r, retained_energy


def project_with_fixed_basis(H_core, Ur):
    return np.asarray(
        Ur.T @ np.asarray(H_core, dtype=np.float64),
        dtype=np.float64,
    )


def beta_from_reduced(V, source_time, T):
    source_time = np.asarray(source_time, dtype=float)
    dt = float(np.median(np.diff(source_time)))

    Vdot = np.gradient(V, dt, axis=1, edge_order=2)
    sums = cp.prepare_window_sums(V, Vdot)

    beta, _ = cp.compute_metrics_for_T(*sums, int(T))
    beta_time = source_time[:len(beta)] + 0.5 * int(T) * dt

    del Vdot, sums
    gc.collect()

    return np.asarray(beta_time, dtype=float), np.asarray(beta, dtype=float)


def clean_noisy_corr(t_clean, y_clean, t_noisy, y_noisy):
    lo = max(float(t_clean[0]), float(t_noisy[0]))
    hi = min(float(t_clean[-1]), float(t_noisy[-1]))

    grid = np.linspace(lo, hi, 4000)
    a = np.interp(grid, t_clean, y_clean)
    b = np.interp(grid, t_noisy, y_noisy)

    return float(np.corrcoef(a, b)[0, 1])


# =============================================================================
# Build multilevel noise test
# =============================================================================
def build_noise_test():
    T1, T2 = read_selected_scales()

    xyz, source_time, reference_xyz, residue_ids = load_adk_segment()

    # Reproduce the main-figure anchor convention q = x - x_closedCA.
    flat_abs = adk.flatten_coordinates(xyz).astype(np.float32, copy=False)
    reference_flat = reference_xyz.reshape(-1, 1).astype(np.float32, copy=False)
    H_clean = np.asarray(flat_abs - reference_flat, dtype=np.float32)

    rows = core_coordinate_rows(residue_ids)
    H_core_clean = np.asarray(H_clean[rows], dtype=np.float32)

    # Fit one clean CORE basis and hold it fixed for every seed and noise level.
    Ur, rank, retained_energy = fit_clean_core_basis(H_core_clean, CORE_RANK)
    V_clean = project_with_fixed_basis(H_core_clean, Ur)

    # Clean reference traces are computed once.
    t1_clean, beta1_clean = beta_from_reduced(V_clean, source_time, T1)
    t2_clean, beta2_clean = beta_from_reduced(V_clean, source_time, T2)
    zb1_clean = zscore(beta1_clean)
    zb2_clean = zscore(beta2_clean)

    # Five-seed summary. Within each seed, one Gaussian realization is scaled
    # across sigma; across seeds, independent realizations quantify repeatability.
    seedwise = []
    plot_V_by_sigma = None
    plot_trace_by_sigma = None

    for seed in SUMMARY_SEEDS:
        rng = np.random.default_rng(int(seed))
        epsilon = rng.standard_normal(H_core_clean.shape).astype(np.float32)

        V_by_sigma = {}
        trace_by_sigma = {}

        for sigma_A in NOISE_LEVELS_ANGSTROM:
            sigma_A = float(sigma_A)
            sigma_nm = 0.1 * sigma_A

            H_noisy = np.asarray(
                H_core_clean + sigma_nm * epsilon,
                dtype=np.float32,
            )
            V_noisy = project_with_fixed_basis(H_noisy, Ur)
            V_by_sigma[sigma_A] = V_noisy

            t1_n, b1_n = beta_from_reduced(V_noisy, source_time, T1)
            t2_n, b2_n = beta_from_reduced(V_noisy, source_time, T2)

            zb1_n = zscore(b1_n)
            zb2_n = zscore(b2_n)

            corr1 = clean_noisy_corr(t1_clean, zb1_clean, t1_n, zb1_n)
            corr2 = clean_noisy_corr(t2_clean, zb2_clean, t2_n, zb2_n)

            trace_by_sigma[sigma_A] = {
                "t1": t1_n,
                "zb1": zb1_n,
                "t2": t2_n,
                "zb2": zb2_n,
                "corr1": corr1,
                "corr2": corr2,
            }

            seedwise.append({
                "seed": int(seed),
                "sigma_A": sigma_A,
                "sigma_nm": sigma_nm,
                "corr1": corr1,
                "corr2": corr2,
            })

        if int(seed) == int(PLOT_SEED):
            plot_V_by_sigma = V_by_sigma
            plot_trace_by_sigma = trace_by_sigma

    if plot_V_by_sigma is None or plot_trace_by_sigma is None:
        raise RuntimeError(
            f"PLOT_SEED={PLOT_SEED} must be included in SUMMARY_SEEDS={SUMMARY_SEEDS}."
        )

    # Panel (a): same displayed realization as panels (b,c).
    n = V_clean.shape[1]
    start = max((n - TOP_WINDOW_FRAMES) // 2, 0)
    end = min(start + TOP_WINDOW_FRAMES, n)

    time_a = source_time[start:end]
    v1_clean = V_clean[0, start:end]

    # Use CLEAN statistics for every displayed v1 trace so noise amplitude is
    # not normalized away.
    mu_v1 = float(np.mean(v1_clean))
    sd_v1 = float(np.std(v1_clean))

    v1_clean_z = standardize_against_clean(v1_clean, mu_v1, sd_v1)
    v1_by_sigma = {
        float(sigma_A): standardize_against_clean(
            plot_V_by_sigma[float(sigma_A)][0, start:end],
            mu_v1,
            sd_v1,
        )
        for sigma_A in NOISE_LEVELS_ANGSTROM
    }

    # Aggregate seedwise correlations by noise level.
    aggregate = {}
    for sigma_A in NOISE_LEVELS_ANGSTROM:
        sigma_A = float(sigma_A)
        rows_sigma = [r for r in seedwise if r["sigma_A"] == sigma_A]

        c1 = np.asarray([r["corr1"] for r in rows_sigma], dtype=float)
        c2 = np.asarray([r["corr2"] for r in rows_sigma], dtype=float)

        aggregate[sigma_A] = {
            "corr1_median": float(np.median(c1)),
            "corr1_q25": float(np.quantile(c1, 0.25)),
            "corr1_q75": float(np.quantile(c1, 0.75)),
            "corr2_median": float(np.median(c2)),
            "corr2_q25": float(np.quantile(c2, 0.25)),
            "corr2_q75": float(np.quantile(c2, 0.75)),
        }

    print("AdK Fig. S4 multilevel coordinate-noise robustness")
    print(f"  CORE residues: 1--20 ({rows.size} Cartesian rows)")
    print(f"  fixed CORE rank: {rank}")
    print(f"  clean-basis retained energy at r={rank}: {retained_energy:.6f}")
    print(f"  scales: T1={T1}, T2={T2}")
    print(f"  displayed seed: {PLOT_SEED}")
    print(f"  summary seeds: {SUMMARY_SEEDS}")

    for sigma_A in NOISE_LEVELS_ANGSTROM:
        a = aggregate[float(sigma_A)]
        print(
            f"  sigma={sigma_A:g} A: "
            f"T1 rho={a['corr1_median']:.4f} "
            f"[{a['corr1_q25']:.4f}, {a['corr1_q75']:.4f}], "
            f"T2 rho={a['corr2_median']:.4f} "
            f"[{a['corr2_q25']:.4f}, {a['corr2_q75']:.4f}]"
        )

    del xyz, flat_abs, H_clean, H_core_clean, Ur
    gc.collect()

    return {
        "T1": int(T1),
        "T2": int(T2),
        "rank": int(rank),
        "retained_energy": retained_energy,
        "rows": rows,
        "time_a": time_a,
        "v1_clean": np.asarray(v1_clean_z, dtype=float),
        "v1_by_sigma": v1_by_sigma,
        "t1_clean": t1_clean,
        "zb1_clean": zb1_clean,
        "t2_clean": t2_clean,
        "zb2_clean": zb2_clean,
        "trace_by_sigma": plot_trace_by_sigma,
        "seedwise": seedwise,
        "aggregate": aggregate,
    }


# =============================================================================
# Save tabular outputs
# =============================================================================
def save_summary(res):
    """Per-sigma five-seed median and interquartile range."""
    path = FIGURE_DIR / f"{STEM}_summary.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sigma_A",
            "sigma_nm",
            "n_seeds",
            "plot_seed",
            "fixed_core_rank",
            "clean_basis_retained_energy",
            "T1",
            "T2",
            "corr_T1_median",
            "corr_T1_q25",
            "corr_T1_q75",
            "corr_T2_median",
            "corr_T2_q25",
            "corr_T2_q75",
        ])

        for sigma_A in NOISE_LEVELS_ANGSTROM:
            sigma_A = float(sigma_A)
            a = res["aggregate"][sigma_A]

            w.writerow([
                sigma_A,
                0.1 * sigma_A,
                len(SUMMARY_SEEDS),
                PLOT_SEED,
                res["rank"],
                res["retained_energy"],
                res["T1"],
                res["T2"],
                a["corr1_median"],
                a["corr1_q25"],
                a["corr1_q75"],
                a["corr2_median"],
                a["corr2_q25"],
                a["corr2_q75"],
            ])

    return path


def save_seedwise(res):
    """Underlying seed-by-noise correlations used for the summary."""
    path = FIGURE_DIR / f"{STEM}_seedwise.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "seed",
            "sigma_A",
            "sigma_nm",
            "corr_T1",
            "corr_T2",
        ])

        for row in res["seedwise"]:
            w.writerow([
                row["seed"],
                row["sigma_A"],
                row["sigma_nm"],
                row["corr1"],
                row["corr2"],
            ])

    return path


def save_traces(res):
    """Only the fixed-seed traces actually plotted in Fig. S4."""
    path = FIGURE_DIR / f"{STEM}_traces.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "seed", "sigma_A", "time", "value"])

        for t, y in zip(res["time_a"], res["v1_clean"]):
            w.writerow(["a_v1", PLOT_SEED, 0.0, float(t), float(y)])

        for sigma_A in NOISE_LEVELS_ANGSTROM:
            for t, y in zip(
                res["time_a"],
                res["v1_by_sigma"][float(sigma_A)],
            ):
                w.writerow([
                    "a_v1",
                    PLOT_SEED,
                    sigma_A,
                    float(t),
                    float(y),
                ])

        for t, y in zip(res["t1_clean"], res["zb1_clean"]):
            w.writerow(["b_T1", PLOT_SEED, 0.0, float(t), float(y)])

        for sigma_A in NOISE_LEVELS_ANGSTROM:
            item = res["trace_by_sigma"][float(sigma_A)]
            for t, y in zip(item["t1"], item["zb1"]):
                w.writerow([
                    "b_T1",
                    PLOT_SEED,
                    sigma_A,
                    float(t),
                    float(y),
                ])

        for t, y in zip(res["t2_clean"], res["zb2_clean"]):
            w.writerow(["c_T2", PLOT_SEED, 0.0, float(t), float(y)])

        for sigma_A in NOISE_LEVELS_ANGSTROM:
            item = res["trace_by_sigma"][float(sigma_A)]
            for t, y in zip(item["t2"], item["zb2"]):
                w.writerow([
                    "c_T2",
                    PLOT_SEED,
                    sigma_A,
                    float(t),
                    float(y),
                ])

    return path


# =============================================================================
# Plotting
# =============================================================================
def plot_reference(ax, x, y, label="clean"):
    ax.plot(
        x,
        y,
        color=CLEAN_COLOR,
        lw=CLEAN_LW,
        ls=CLEAN_LS,
        label=label,
        zorder=5,
    )


def plot_noise_family(ax, x_by_sigma, y_by_sigma):
    for color, sigma_A in zip(NOISE_COLORS, NOISE_LEVELS_ANGSTROM):
        x = x_by_sigma[float(sigma_A)]
        y = y_by_sigma[float(sigma_A)]
        ax.plot(
            x,
            y,
            color=color,
            lw=0.92,
            alpha=0.88,
            label=rf"$\sigma={sigma_A:g}\,\AA$",
            zorder=2,
        )


def make_figure(res):
    st.apply_style()

    fig = plt.figure(
        figsize=(W_MM * st.MM, H_MM * st.MM),
        facecolor="white",
    )

    axa = axes_mm(fig, LEFT_MM, A_Y, AX_W_MM, A_H)
    axb = axes_mm(fig, LEFT_MM, B_Y, AX_W_MM, B_H)
    axc = axes_mm(fig, LEFT_MM, C_Y, AX_W_MM, C_H)

    # ------------------------------------------------------------------ (a)
    plot_reference(axa, res["time_a"], res["v1_clean"])

    x_by_sigma_a = {
        float(s): res["time_a"] for s in NOISE_LEVELS_ANGSTROM
    }
    plot_noise_family(
        axa,
        x_by_sigma_a,
        res["v1_by_sigma"],
    )

    vals_a = [res["v1_clean"]]
    vals_a.extend(res["v1_by_sigma"][float(s)] for s in NOISE_LEVELS_ANGSTROM)
    axa.set_ylim(
        headroom_ylim(
            np.concatenate(vals_a),
            qlo=0.01,
            qhi=0.99,
            pad_lo=0.10,
            pad_hi=0.16,
        )
    )
    axa.set_ylabel(r"$z(v_1)$", labelpad=2.0)
    axa.set_title(
        "CORE reduced coordinate",
        fontsize=st.FS_TEXT,
        loc="left",
        pad=3.0,
    )
    format_time_axis(
        axa,
        float(res["time_a"][0]),
        float(res["time_a"][-1]),
    )
    paper_axes(axa)

    # ------------------------------------------------------------------ (b)
    plot_reference(axb, res["t1_clean"], res["zb1_clean"])

    x_by_sigma_b = {
        float(s): res["trace_by_sigma"][float(s)]["t1"]
        for s in NOISE_LEVELS_ANGSTROM
    }
    y_by_sigma_b = {
        float(s): res["trace_by_sigma"][float(s)]["zb1"]
        for s in NOISE_LEVELS_ANGSTROM
    }
    plot_noise_family(axb, x_by_sigma_b, y_by_sigma_b)

    vals_b = [res["zb1_clean"]]
    vals_b.extend(y_by_sigma_b[float(s)] for s in NOISE_LEVELS_ANGSTROM)
    axb.set_ylim(
        headroom_ylim(
            np.concatenate(vals_b),
            qlo=0.01,
            qhi=0.99,
            pad_lo=0.10,
            pad_hi=0.16,
        )
    )
    axb.set_ylabel(r"$z(\beta)$", labelpad=2.0)
    axb.set_title(
        rf"Short-scale response, $T_1={res['T1']}$",
        fontsize=st.FS_TEXT,
        loc="left",
        pad=3.0,
    )
    format_time_axis(
        axb,
        float(res["t1_clean"][0]),
        float(res["t1_clean"][-1]),
    )
    axb.yaxis.set_major_locator(MaxNLocator(nbins=4))
    paper_axes(axb)

    # ------------------------------------------------------------------ (c)
    plot_reference(axc, res["t2_clean"], res["zb2_clean"])

    x_by_sigma_c = {
        float(s): res["trace_by_sigma"][float(s)]["t2"]
        for s in NOISE_LEVELS_ANGSTROM
    }
    y_by_sigma_c = {
        float(s): res["trace_by_sigma"][float(s)]["zb2"]
        for s in NOISE_LEVELS_ANGSTROM
    }
    plot_noise_family(axc, x_by_sigma_c, y_by_sigma_c)

    vals_c = [res["zb2_clean"]]
    vals_c.extend(y_by_sigma_c[float(s)] for s in NOISE_LEVELS_ANGSTROM)
    axc.set_ylim(
        headroom_ylim(
            np.concatenate(vals_c),
            qlo=0.01,
            qhi=0.99,
            pad_lo=0.10,
            pad_hi=0.16,
        )
    )
    axc.set_ylabel(r"$z(\beta)$", labelpad=2.0)
    axc.set_title(
        rf"Long-scale response, $T_2={res['T2']}$",
        fontsize=st.FS_TEXT,
        loc="left",
        pad=3.0,
    )
    format_time_axis(
        axc,
        float(res["t2_clean"][0]),
        float(res["t2_clean"][-1]),
    )
    axc.yaxis.set_major_locator(MaxNLocator(nbins=4))
    paper_axes(axc)

    # ---------------------------------------------------------------- legend
    # Put one shared legend completely outside the data regions.
    handles, labels = axa.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.995),
        ncol=5,
        frameon=False,
        fontsize=0.88 * st.FS_TICK,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.35,
    )

    add_panel_letter(fig, LETTER_X, 121.5, "(a)")
    add_panel_letter(fig, LETTER_X, 87.5, "(b)")
    add_panel_letter(fig, LETTER_X, 48.5, "(c)")

    pdf_path = FIGURE_DIR / f"{STEM}.pdf"
    png_path = FIGURE_DIR / f"{STEM}.png"

    safe_savefig(fig, pdf_path, format="pdf", facecolor="white")
    safe_savefig(fig, png_path, dpi=600, facecolor="white")
    plt.close(fig)

    return pdf_path, png_path


# =============================================================================
# Main
# =============================================================================
def main():
    res = build_noise_test()

    pdf_path, png_path = make_figure(res)
    summary_path = save_summary(res)
    seedwise_path = save_seedwise(res)
    traces_path = save_traces(res)

    print("Saved:")
    print(" ", pdf_path)
    print(" ", png_path)
    print(" ", summary_path)
    print(" ", seedwise_path)
    print(" ", traces_path)


if __name__ == "__main__":
    main()
