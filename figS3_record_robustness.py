"""Supplementary Fig. S3: finite-record robustness across systems.

For Lorenz, KS and ADK, the production reduced representation is held fixed
while the analysed record is shortened by removing 5% of samples from each
temporal edge. The shared three-grid pipeline is rerun on the retained central
90%, and the resulting S(T) organization and independently selected T1/T2 are
compared with the reference record.

The displayed robustness metric eta_rec is the maximum separation between the
two smoothed S(T) curves over the same T range shown in each panel. The
full-common-domain sup norm eta_full is retained in the summary CSV as an
additional diagnostic of the far large-T tail only.  That tail is limited by
record length: windows there approach the analysed record, so removing 10% of
the record removes a large share of the windows at those T.  eta_full is
therefore not used to assess the robustness of the characteristic scales and
is not reported in the paper.

Second row: representation capacity.  The full record is rescanned with the
retained rank r varied around the production rank r0 (r0-2 ... r0+2) and at
the ranks given by both 98% rank criteria (cumulative singular values, the
shared rule; cumulative energy, the ADK rule).  All ranks of one system come
from the same SVD basis: V_r is the first r rows of one reduced representation,
and the velocity is projected (Lorenz, KS) or differentiated (ADK) row by row,
so truncation does not change the retained rows.  The production rank uses the
production representation itself.  Ranks below 2 or not below the smallest
scanned window are excluded and reported.  eta_r is the largest separation of
any scanned-rank S(T) from the production S(T) over the displayed T range.

Outputs added by the second row
    figures/figS3_record_robustness_rank_summary.csv
    figures/figS3_record_robustness_rank_curves.csv

Expected project layout
-----------------------
project_root/
├── closure_pipeline.py
├── figure_style.py
├── fig2_lorenz.py
├── fig3_ks.py
├── fig4_adk.py
├── figS3_record_robustness.py
├── data/
│   ├── lorenz_raw_data.npz
│   ├── ks_raw_data.npz
│   ├── closedCA.pdb
│   └── C1.0_1.dcd
└── figures/
    ├── fig2_lorenz_summary.csv
    ├── fig3_ks_summary.csv
    └── fig4_adk_selected_windows.csv
"""

from pathlib import Path
import csv
import gc
import shutil
import tempfile

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter, NullLocator

import closure_pipeline as cp
import figure_style as st
import fig2_lorenz as lor
import fig3_ks as ks
import fig4_adk as adk


# =============================================================================
# Configuration
# =============================================================================
ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
STEM = "figS3_record_robustness"

TRIM_FRACTION_EACH_SIDE = 0.05
MAX_SELECTED_SCALES = 2

# Representation-capacity scan (second row).
RANK_OFFSETS = (-2, -1, 0, 1, 2)
RANK_CRITERION_FRACTION = 0.98
MIN_SCAN_RANK = 2              # r = 1 reduces the local rule to a scalar

SUMMARY_FILES = {
    "Lorenz": FIGURE_DIR / "fig2_lorenz_summary.csv",
    "KS": FIGURE_DIR / "fig3_ks_summary.csv",
    "ADK": FIGURE_DIR / "fig4_adk_selected_windows.csv",
}

# Two-row supplementary layout.  The first row (record length) keeps the
# geometry of the former one-row figure and is shifted up by ROW_SHIFT_MM; the
# second row (representation capacity) uses the same columns.
ROW_SHIFT_MM = 64.0
W_MM, H_MM = 174.0, 72.0 + ROW_SHIFT_MM
LEFT_MM = 12.0
RIGHT_MM = 4.5
GAP_MM = 9.5
AX_Y_MM = 13.0
AX_H_MM = 40.0
AX_W_MM = (W_MM - LEFT_MM - RIGHT_MM - 2 * GAP_MM) / 3.0

SYSTEM_TITLES = {
    "Lorenz": "Lorenz",
    "KS": "KS",
    "ADK": "AdK",
}


# =============================================================================
# General helpers
# =============================================================================
def safe_savefig(fig, target, **kwargs):
    """Render to a temporary path first, then copy to the project."""
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".tmp"
    with tempfile.TemporaryDirectory(prefix="closure_figS3_") as tmp_dir:
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




def add_axis_multiplier(ax, text):
    ax.annotate(
        text,
        xy=(1.0, 0.0), xycoords="axes fraction",
        xytext=(0.0, -17.0), textcoords="offset points",
        ha="right", va="top", fontsize=0.92 * st.FS_TICK,
        annotation_clip=False,
    )


def configure_xaxis(ax, system, t_min, x_display_max):
    ax.set_xscale("log")
    ax.set_xlim(t_min, x_display_max)

    if system == "Lorenz":
        ticks = [100, 1000]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: rf"$10^{{{int(np.log10(x))}}}$" if x in ticks else ""
        ))
        return

    # For KS and AdK, pull out a common ×10^2 multiplier so the tick labels
    # stay compact and the two right panels do not collide visually.
    ticks = [60, 100, 200, 300, 600]
    ticks = [t for t in ticks if t_min <= t <= x_display_max]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    labels = {60: "0.6", 100: "1", 200: "2", 300: "3", 600: "6"}
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: labels.get(int(round(x)), "") if any(abs(x-t) < 1e-6 for t in ticks) else ""
    ))
    ax.xaxis.set_minor_locator(NullLocator())
    add_axis_multiplier(ax, r"$\times 10^{2}$")

def close_npz_if_present(obj):
    data = obj.get("data") if isinstance(obj, dict) else None
    if data is not None and hasattr(data, "close"):
        try:
            data.close()
        except Exception:
            pass


def selected_from_summary(system):
    """Read the main-figure T1/T2 if its summary CSV is already available."""
    path = SUMMARY_FILES[system]
    if not path.exists():
        return None, None

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None

    row = rows[0]

    def get_number(keys, as_int=True):
        for key in keys:
            if key in row and str(row[key]).strip() != "":
                value = float(row[key])
                return int(round(value)) if as_int else value
        return None

    T1 = get_number(("T1", "T_1", "selected_T1"))
    T2 = get_number(("T2", "T_2", "selected_T2"))
    rank = get_number(("rank", "production_rank", "retained_rank"))

    if T1 is None or T2 is None:
        # ADK event table is a robust fallback if an older selected-windows CSV
        # did not yet store T1/T2 explicitly.
        if system == "ADK":
            event_path = FIGURE_DIR / "fig4_adk_reorganization_events.csv"
            if event_path.exists():
                with open(event_path, newline="", encoding="utf-8") as f:
                    events = sorted(
                        list(csv.DictReader(f)),
                        key=lambda r: int(float(r["event"])),
                    )
                if len(events) >= 2:
                    T1 = int(round(float(events[0]["T_star"])))
                    T2 = int(round(float(events[1]["T_star"])))

    if T1 is None or T2 is None:
        return None, rank
    return [T1, T2], rank


def trim_reduced_record(V, Vdot, source_time, fraction=TRIM_FRACTION_EACH_SIDE):
    """Retain the central 1-2*fraction part of one fixed reduced representation."""
    V = np.asarray(V)
    Vdot = np.asarray(Vdot)
    source_time = np.asarray(source_time, dtype=float)

    n = V.shape[1]
    cut = max(1, int(round(float(fraction) * n)))
    lo, hi = cut, n - cut
    if hi - lo < 100:
        raise ValueError(
            f"Trim leaves only {hi - lo} samples from n={n}; reduce trim fraction."
        )

    return (
        np.asarray(V[:, lo:hi], dtype=np.float64),
        np.asarray(Vdot[:, lo:hi], dtype=np.float64),
        source_time[lo:hi],
        lo,
        hi,
    )


def baseline_similarity(sums, source_time, dt, t_min, t_max, progress=True):
    """Compute only the baseline-grid S(T), avoiding a full three-grid rerun."""
    baseline_Ts = cp.make_nested_log_grids(int(t_min), int(t_max))["baseline"]
    _, scale_map = cp.make_scale_map(
        sums,
        baseline_Ts,
        source_time,
        dt,
        progress=progress,
    )
    diag = cp.compute_similarity(baseline_Ts, scale_map)
    del scale_map
    gc.collect()
    return np.asarray(baseline_Ts, dtype=int), diag


def full_analysis(sums, source_time, dt, t_min, t_max):
    return cp.analyze_timescales(
        sums,
        source_time,
        dt,
        t_min=int(t_min),
        t_max=int(t_max),
        max_windows=MAX_SELECTED_SCALES,
        progress=True,
    )


def curve_eta(ref_diag, pert_diag, T_bounds=None):
    """Sup norm between smoothed S(T) curves on a common T domain.

    With T_bounds=None, the complete common scan domain is used. Passing
    (T_lo, T_hi) evaluates the same quantity on a stated displayed interval.
    """
    x0 = np.asarray(ref_diag["log_grid"], dtype=float)
    y0 = np.asarray(ref_diag["smooth"], dtype=float)
    x1 = np.asarray(pert_diag["log_grid"], dtype=float)
    y1 = np.asarray(pert_diag["smooth"], dtype=float)

    lo = max(float(x0[0]), float(x1[0]))
    hi = min(float(x0[-1]), float(x1[-1]))
    if T_bounds is not None:
        T_lo, T_hi = map(float, T_bounds)
        lo = max(lo, np.log(T_lo))
        hi = min(hi, np.log(T_hi))

    if hi <= lo:
        return np.nan

    mask = (x0 >= lo) & (x0 <= hi) & np.isfinite(y0)
    xx = x0[mask]
    yy0 = y0[mask]
    yy1 = np.interp(xx, x1, y1)
    finite = np.isfinite(yy0) & np.isfinite(yy1)
    if not np.any(finite):
        return np.nan
    return float(np.max(np.abs(yy1[finite] - yy0[finite])))


def display_T_range(ref_diag, pert_diag, ref_selected, pert_selected):
    """T range used both for plotting and for the displayed eta_rec."""
    x_ref = np.exp(np.asarray(ref_diag["log_grid"], dtype=float))
    x_pert = np.exp(np.asarray(pert_diag["log_grid"], dtype=float))

    T_min = max(float(x_ref[0]), float(x_pert[0]))
    T_max_common = min(float(x_ref[-1]), float(x_pert[-1]))

    selected = [
        float(T)
        for T in list(ref_selected) + list(pert_selected)
        if np.isfinite(T)
    ]
    max_selected = max(selected) if selected else 5.0 * T_min
    T_max = min(T_max_common, max(1.65 * max_selected, 4.0 * T_min))
    return T_min, T_max


def S_at_T(diag, T):
    return float(np.interp(np.log(float(T)), diag["log_grid"], diag["smooth"]))


# =============================================================================
# Representation-capacity helpers
# =============================================================================
def criterion_ranks(s, fraction=RANK_CRITERION_FRACTION):
    """Ranks given by the two 98% criteria from one singular-value spectrum."""
    s = np.asarray(s, dtype=float)
    rank_sigma = cp.retained_rank_from_singular_values(s, fraction)
    rank_energy = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), fraction) + 1)
    return rank_sigma, rank_energy


def capacity_ranks(r0, s, n_rows, t_min):
    """Ranks scanned for one system and the ranks excluded, with the reason.

    The closure fit in a window of T samples has r coefficients per row, so a
    rank must stay below the smallest scanned window (the pipeline requires
    t_min > r) for the closure test to be defined at every scale.
    """
    rank_sigma, rank_energy = criterion_ranks(s)
    candidates = {int(r0) + k for k in RANK_OFFSETS} | {rank_sigma, rank_energy}
    ranks, excluded = [], []
    for r in sorted(candidates):
        if r < MIN_SCAN_RANK or r > int(n_rows):
            excluded.append((r, f"outside [{MIN_SCAN_RANK}, {int(n_rows)}]"))
        elif r >= int(t_min):
            excluded.append((r, f"not below the smallest window T_min={int(t_min)}"))
        else:
            ranks.append(r)
    return dict(ranks=ranks, excluded=excluded,
                rank_sigma=rank_sigma, rank_energy=rank_energy)


# =============================================================================
# Production representations
# =============================================================================
def build_lorenz_representation():
    """Exactly the reduced representation constructed by fig2_lorenz.py."""
    core = lor.build_lorenz_core()
    result = {
        "V": np.asarray(core["V"], dtype=np.float64),
        "Vdot": np.asarray(core["Vdot"], dtype=np.float64),
        "source_time": np.asarray(core["source_time"], dtype=float),
        "dt": float(core["dt"]),
        "rank": int(core["rank"]),
        "t_min": int(lor.T_SCAN_MIN),
        "t_max": int(core["V"].shape[1] - 1),
    }
    scan = capacity_ranks(result["rank"], core["singular_values"],
                          core["H"].shape[0], result["t_min"])
    close_npz_if_present(core)
    del core
    gc.collect()

    # One representation at the largest scanned rank; smaller ranks are its
    # leading rows.
    wide = lor.build_lorenz_core(rank=max(scan["ranks"]))
    scan["V"] = np.asarray(wide["V"], dtype=np.float64)
    scan["Vdot"] = np.asarray(wide["Vdot"], dtype=np.float64)
    close_npz_if_present(wide)
    del wide
    gc.collect()
    result["scan"] = scan
    return result


def build_ks_representation():
    """Exactly the reduced representation constructed by fig3_ks.py."""
    core = ks.build_ks_core()
    result = {
        "V": np.asarray(core["V"], dtype=np.float64),
        "Vdot": np.asarray(core["Vdot"], dtype=np.float64),
        "source_time": np.asarray(core["source_time"], dtype=float),
        "dt": float(core["dt"]),
        "rank": int(core["rank"]),
        "t_min": int(ks.T_SCAN_MIN),
        "t_max": int(core["t_scan_max"]),
    }
    scan = capacity_ranks(result["rank"], core["singular_values"],
                          core["H"].shape[0], result["t_min"])
    close_npz_if_present(core)
    del core
    gc.collect()

    wide = ks.build_ks_core(rank=max(scan["ranks"]))
    scan["V"] = np.asarray(wide["V"], dtype=np.float64)
    scan["Vdot"] = np.asarray(wide["Vdot"], dtype=np.float64)
    close_npz_if_present(wide)
    del wide
    gc.collect()
    result["scan"] = scan
    return result


def build_adk_representation():
    """Exactly the full-coordinate reduced representation used by fig4_adk.py."""
    xyz, source_time, reference_xyz, _ = adk.load_aligned_analysis_segment(
        chunk_size=2000
    )
    source_time = np.asarray(source_time, dtype=float)
    dt = float(np.median(np.diff(source_time)))

    flat = adk.flatten_coordinates(xyz).astype(np.float32, copy=False)
    ref_flat = np.asarray(reference_xyz, dtype=np.float32).reshape(-1, 1)
    H = np.asarray(flat - ref_flat, dtype=np.float32)

    del xyz, flat, ref_flat
    gc.collect()

    V, Vdot, rank, s, _ = adk.compute_reduced_coordinates_adk(H, dt)
    n = V.shape[1]

    # Same Gram-matrix basis at the largest scanned rank; the velocity is the
    # row-wise numerical derivative, so smaller ranks are its leading rows.
    scan = capacity_ranks(rank, s, H.shape[0], adk.T_MIN)
    V_wide, Vdot_wide, _, _, _ = adk.compute_reduced_coordinates_adk(
        H, dt, rank=max(scan["ranks"]))
    scan["V"] = np.asarray(V_wide, dtype=np.float64)
    scan["Vdot"] = np.asarray(Vdot_wide, dtype=np.float64)
    del H, V_wide, Vdot_wide
    gc.collect()

    return {
        "V": np.asarray(V, dtype=np.float64),
        "Vdot": np.asarray(Vdot, dtype=np.float64),
        "source_time": source_time,
        "dt": dt,
        "rank": int(rank),
        "t_min": int(adk.T_MIN),
        "t_max": int(np.floor(0.25 * n)),
        "scan": scan,
    }


BUILDERS = {
    "Lorenz": build_lorenz_representation,
    "KS": build_ks_representation,
    "ADK": build_adk_representation,
}


def trimmed_tmax(system, n_trim):
    """Preserve each main figure's production rule for the upper scale bound."""
    if system == "Lorenz":
        return int(n_trim - 1)
    # KS and ADK both scan only to one quarter of the analysed record.
    return int(np.floor(0.25 * n_trim))


# =============================================================================
# Representation-capacity run
# =============================================================================
def run_rank_scan(system, rep, main_selected, T_bounds):
    """Full three-grid analysis of the full record at every scanned rank."""
    scan = rep["scan"]
    r0 = rep["rank"]
    print(f"{system}: rank criteria (98%): cumulative singular values -> "
          f"{scan['rank_sigma']}, cumulative energy -> {scan['rank_energy']}; "
          f"production rank {r0}")
    for r, why in scan["excluded"]:
        print(f"{system}: rank {r} excluded ({why})")

    entries = []
    for r in scan["ranks"]:
        if r == r0:
            V_r, Vdot_r = rep["V"], rep["Vdot"]
        else:
            V_r, Vdot_r = scan["V"][:r], scan["Vdot"][:r]
        print(f"{system}: capacity scan, r = {r}")
        sums = cp.prepare_window_sums(V_r, Vdot_r)
        entry = {"rank": r, "production": r == r0,
                 "criteria": [name for name, rr in (("sigma", scan["rank_sigma"]),
                                                    ("energy", scan["rank_energy"]))
                              if rr == r],
                 "hierarchy": [], "events": [], "diag": None, "Ts": None, "error": ""}
        try:
            res = full_analysis(sums, rep["source_time"], rep["dt"],
                                rep["t_min"], rep["t_max"])
            entry.update(hierarchy=[int(T) for T in res["hierarchy"]],
                         events=res["persistent_events"], diag=res["diag"],
                         Ts=np.asarray(res["T_values"], dtype=int))
            del res
        except RuntimeError as exc:
            # No retained event is a possible outcome at some rank; the S(T)
            # curve is still drawn.  Any other failure is reported and skipped.
            if "persistent" in str(exc):
                entry["Ts"], entry["diag"] = baseline_similarity(
                    sums, rep["source_time"], rep["dt"], rep["t_min"], rep["t_max"],
                    progress=True)
            else:
                entry["error"] = str(exc)
            print(f"{system}: r = {r}: {exc}")
        del sums
        gc.collect()
        entries.append(entry)

    production = next(e for e in entries if e["production"])
    print(f"{system}: production rank reproduces main-figure scales: "
          f"{production['hierarchy'][:2] == list(main_selected)} "
          f"({production['hierarchy'][:2]} vs {list(main_selected)})")
    for e in entries:
        e["eta"] = (np.nan if e["diag"] is None or e["production"]
                    else curve_eta(production["diag"], e["diag"], T_bounds=T_bounds))
        prom = [round(float(ev["prominence"]), 4) for ev in e["events"]]
        print(f"{system}: r = {e['rank']}: scales {e['hierarchy']}, "
              f"prominence {prom}, eta_r = {e['eta']:.4g}")
    eta_r = np.nanmax([e["eta"] for e in entries]) if len(entries) > 1 else np.nan
    print(f"{system}: eta_r (max over scanned ranks) = {eta_r:.4g}")
    return {"entries": entries, "eta_r": eta_r, "rank_sigma": scan["rank_sigma"],
            "rank_energy": scan["rank_energy"], "excluded": scan["excluded"]}


# =============================================================================
# One-system robustness run
# =============================================================================
def run_system(system):
    print("\n" + "=" * 72)
    print(f"Fig. S3 robustness: {system}")
    print("=" * 72)

    rep = BUILDERS[system]()
    V = rep["V"]
    Vdot = rep["Vdot"]
    source_time = rep["source_time"]
    dt = rep["dt"]
    rank = rep["rank"]
    t_min = rep["t_min"]
    t_max = rep["t_max"]

    main_selected, main_rank = selected_from_summary(system)
    if main_rank is not None and int(main_rank) != rank:
        raise RuntimeError(
            f"{system}: representation rank {rank} differs from main-figure "
            f"summary rank {main_rank}. Rerun the main figure with the current code."
        )

    # Reference S(T): baseline grid only.
    sums_ref = cp.prepare_window_sums(V, Vdot)
    if main_selected is None:
        print(
            f"{system}: main summary not found; running the full baseline pipeline "
            "once to recover reference T1/T2."
        )
        base_full = full_analysis(
            sums_ref, source_time, dt, t_min=t_min, t_max=t_max
        )
        main_selected = [int(T) for T in base_full["selected"][:2]]
        ref_diag = base_full["diag"]
        ref_Ts = np.asarray(base_full["T_values"], dtype=int)
        del base_full
    else:
        ref_Ts, ref_diag = baseline_similarity(
            sums_ref,
            source_time,
            dt,
            t_min=t_min,
            t_max=t_max,
            progress=True,
        )

    if len(main_selected) < 2:
        raise RuntimeError(
            f"{system}: baseline has fewer than two characteristic scales: "
            f"{main_selected}"
        )

    del sums_ref
    gc.collect()

    # Finite-record perturbation: same fixed representation, central 90%.
    V_trim, Vdot_trim, time_trim, lo, hi = trim_reduced_record(
        V, Vdot, source_time
    )
    n_ref = V.shape[1]
    n_trim = V_trim.shape[1]
    del V, Vdot
    gc.collect()

    sums_trim = cp.prepare_window_sums(V_trim, Vdot_trim)
    t_max_trim = trimmed_tmax(system, n_trim)
    pert = full_analysis(
        sums_trim,
        time_trim,
        dt,
        t_min=t_min,
        t_max=t_max_trim,
    )
    pert_selected = [int(T) for T in pert["selected"][:2]]
    pert_diag = pert["diag"]
    pert_Ts = np.asarray(pert["T_values"], dtype=int)

    pert_selected_padded = [
        pert_selected[0] if len(pert_selected) >= 1 else np.nan,
        pert_selected[1] if len(pert_selected) >= 2 else np.nan,
    ]
    display_t_min, display_t_max = display_T_range(
        ref_diag,
        pert_diag,
        main_selected[:2],
        pert_selected_padded,
    )
    eta_rec = curve_eta(
        ref_diag,
        pert_diag,
        T_bounds=(display_t_min, display_t_max),
    )
    eta_full = curve_eta(ref_diag, pert_diag)

    print(f"{system}: fixed representation rank = {rank}")
    print(
        f"{system}: retained central record = {n_trim}/{n_ref} "
        f"({100*n_trim/n_ref:.1f}%), indices [{lo}, {hi})"
    )
    print(f"{system}: reference scales = {main_selected}")
    print(f"{system}: trimmed-record scales = {pert_selected}")
    print(
        f"{system}: eta_rec = {eta_rec:.6g} "
        f"on T=[{display_t_min:.3g}, {display_t_max:.3g}]"
    )
    print(f"{system}: eta_full = {eta_full:.6g}")

    result = {
        "system": system,
        "rank": rank,
        "n_ref": n_ref,
        "n_trim": n_trim,
        "trim_fraction_each_side": TRIM_FRACTION_EACH_SIDE,
        "trim_lo": lo,
        "trim_hi": hi,
        "T1_ref": main_selected[0],
        "T2_ref": main_selected[1],
        "T1_trim": pert_selected_padded[0],
        "T2_trim": pert_selected_padded[1],
        "eta_rec": eta_rec,
        "eta_full": eta_full,
        "eta_rec_Tmin": display_t_min,
        "eta_rec_Tmax": display_t_max,
        "ref_Ts": ref_Ts,
        "ref_diag": ref_diag,
        "pert_Ts": pert_Ts,
        "pert_diag": pert_diag,
    }

    del V_trim, Vdot_trim, sums_trim, pert
    gc.collect()

    result["rank_scan"] = run_rank_scan(
        system, rep, main_selected[:2], (display_t_min, display_t_max))
    del rep
    gc.collect()
    return result


# =============================================================================
# Output tables
# =============================================================================
def save_summary(results):
    path = FIGURE_DIR / f"{STEM}_summary.csv"
    fields = [
        "system",
        "rank",
        "n_ref",
        "n_trim",
        "trim_fraction_each_side",
        "T1_ref",
        "T2_ref",
        "T1_trim",
        "T2_trim",
        "eta_rec",
        "eta_full",
        "eta_rec_Tmin",
        "eta_rec_Tmax",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    return path


def save_curves(results):
    path = FIGURE_DIR / f"{STEM}_curves.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "record", "T", "S_smooth"])
        for r in results:
            for record, Ts, diag in (
                ("reference", r["ref_Ts"], r["ref_diag"]),
                ("central90", r["pert_Ts"], r["pert_diag"]),
            ):
                smooth_actual = np.interp(
                    np.log(np.asarray(Ts, dtype=float)),
                    diag["log_grid"],
                    diag["smooth"],
                )
                for T, S in zip(Ts, smooth_actual):
                    w.writerow([r["system"], record, int(T), float(S)])
    return path


def save_rank_tables(results):
    summary = FIGURE_DIR / f"{STEM}_rank_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "rank", "production", "criteria", "n_levels",
                    "T1", "T2", "hierarchy", "prominence", "eta_r", "note"])
        for res in results:
            rs = res["rank_scan"]
            for e in rs["entries"]:
                h = e["hierarchy"]
                w.writerow([
                    res["system"], e["rank"], int(e["production"]),
                    ";".join(e["criteria"]), len(h),
                    h[0] if len(h) > 0 else "", h[1] if len(h) > 1 else "",
                    ";".join(str(T) for T in h),
                    ";".join(f"{float(ev['prominence']):.4g}" for ev in e["events"]),
                    e["eta"], e["error"],
                ])
            for r, why in rs["excluded"]:
                w.writerow([res["system"], r, 0, "", "", "", "", "", "", "", f"excluded: {why}"])
    curves = FIGURE_DIR / f"{STEM}_rank_curves.csv"
    with open(curves, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "rank", "T", "S_smooth"])
        for res in results:
            for e in res["rank_scan"]["entries"]:
                if e["diag"] is None:
                    continue
                Ts = np.asarray(e["Ts"], dtype=float)
                S = np.interp(np.log(Ts), e["diag"]["log_grid"], e["diag"]["smooth"])
                for T, val in zip(Ts, S):
                    w.writerow([res["system"], e["rank"], int(T), float(val)])
    return summary, curves


# =============================================================================
# Figure
# =============================================================================
def rank_colors(entries, r0):
    """Production rank in the main S(T) colour; lower ranks orange, higher blue,
    lighter with distance from r0."""
    lower = sorted([e["rank"] for e in entries if e["rank"] < r0], reverse=True)
    higher = sorted([e["rank"] for e in entries if e["rank"] > r0])
    colors = {r0: st.C_SSMOOTH}
    for ranks, cmap in ((lower, "Oranges"), (higher, "Blues")):
        if ranks:
            shades = mpl.colormaps[cmap](np.linspace(0.75, 0.42, len(ranks)))
            colors.update(dict(zip(ranks, shades)))
    return colors


def draw_rank_panel(ax, result):
    """S(T) of the full record at every scanned rank, on the first-row T range."""
    entries = [e for e in result["rank_scan"]["entries"] if e["diag"] is not None]
    r0 = result["rank"]
    colors = rank_colors(entries, r0)
    t_min = float(result["eta_rec_Tmin"])
    x_display_max = float(result["eta_rec_Tmax"])

    vals = []
    for e in sorted(entries, key=lambda e: e["production"]):   # production on top
        x = np.exp(e["diag"]["log_grid"])
        y = np.asarray(e["diag"]["smooth"], dtype=float)
        label = rf"$r={e['rank']}$" + (" (main)" if e["production"] else "")
        ax.plot(x, y, color=colors[e["rank"]],
                lw=1.15 if e["production"] else 0.85,
                zorder=3 if e["production"] else 2, label=label)
        m = np.isfinite(x) & np.isfinite(y) & (x >= t_min) & (x <= x_display_max)
        vals.extend(y[m].tolist())

        def s_at(T):
            return float(np.interp(np.log(T), e["diag"]["log_grid"], e["diag"]["smooth"]))

        for T in e["hierarchy"][:2]:
            if e["production"]:
                ax.axvline(T, color=st.C_SEL, ls="-.", lw=st.LW_GUIDE, alpha=0.52)
                ax.plot(T, s_at(T), marker="o", ms=3.0, color=st.C_SEL, ls="none", zorder=4)
            else:
                ax.plot(T, s_at(T), marker="o", ms=3.4, mfc="white",
                        mec=colors[e["rank"]], mew=0.85, ls="none", zorder=5)

    configure_xaxis(ax, result["system"], t_min, x_display_max)
    if vals:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        span = max(hi - lo, 1e-4)
        ax.set_ylim(max(-1.0, lo - 0.14 * span), min(1.02, hi + 0.20 * span))
    ax.set_xlabel(r"Window size $T$")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    paper_axes(ax)

    # Rank key above the panel, outside the data region, ordered by rank.
    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort([int(l.split("=")[1].split("$")[0]) for l in labels])
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, fontsize=0.80 * st.FS_TICK, handlelength=1.4,
              columnspacing=0.8, handletextpad=0.35, borderaxespad=0.2,
              title=rf"$\eta_r={result['rank_scan']['eta_r']:.3g}$",
              title_fontsize=0.84 * st.FS_TICK)


def draw_panel(ax, result):
    ref_diag = result["ref_diag"]
    pert_diag = result["pert_diag"]

    x_ref = np.exp(ref_diag["log_grid"])
    y_ref = np.asarray(ref_diag["smooth"], dtype=float)
    x_pert = np.exp(pert_diag["log_grid"])
    y_pert = np.asarray(pert_diag["smooth"], dtype=float)

    # Same visual language as the main S(T) panels: one shared curve color,
    # distinguished here by line style rather than adding a new palette.
    ax.plot(
        x_ref,
        y_ref,
        color=st.C_SSMOOTH,
        lw=1.15,
        label="reference record",
    )
    ax.plot(
        x_pert,
        y_pert,
        color=st.C_SSMOOTH,
        lw=1.00,
        ls=(0, (3, 2)),
        alpha=0.58,
        label="central 90%",
    )

    ref_sel = [result["T1_ref"], result["T2_ref"]]
    trim_sel = [
        result["T1_trim"],
        result["T2_trim"],
    ]

    # Reference scales: same guide grammar as the main figures.
    for T in ref_sel:
        if np.isfinite(T):
            sval = S_at_T(ref_diag, T)
            ax.axvline(
                T,
                color=st.C_SEL,
                ls="-.",
                lw=st.LW_GUIDE,
                alpha=0.52,
            )
            ax.plot(
                T,
                sval,
                marker="o",
                ms=3.0,
                color=st.C_SEL,
                ls="none",
                zorder=4,
            )

    # Perturbed scales use open markers; reference scales keep the main-figure guides.
    for T in trim_sel:
        if np.isfinite(T):
            sval = S_at_T(pert_diag, T)
            ax.plot(
                T,
                sval,
                marker="o",
                ms=3.7,
                mfc="white",
                mec=st.C_SEL,
                mew=0.85,
                ls="none",
                zorder=5,
            )

    # The plotted T interval is also the domain of the displayed eta_rec.
    t_min = float(result["eta_rec_Tmin"])
    x_display_max = float(result["eta_rec_Tmax"])
    configure_xaxis(ax, result["system"], t_min, x_display_max)

    # Compact y range based only on the displayed portion.
    vals = []
    for xx, yy in ((x_ref, y_ref), (x_pert, y_pert)):
        m = (
            np.isfinite(xx)
            & np.isfinite(yy)
            & (xx >= t_min)
            & (xx <= x_display_max)
        )
        vals.extend(yy[m].tolist())
    vals = np.asarray(vals, dtype=float)
    if vals.size:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        span = max(hi - lo, 1e-4)
        ax.set_ylim(max(0.0, lo - 0.14 * span), min(1.02, hi + 0.20 * span))

    ax.set_xlabel(r"Window size $T$")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # The numerical robustness summary is placed in the top margin, leaving the
    # S(T) curves unobstructed.

    paper_axes(ax)


def make_figure(results):
    st.apply_style()
    fig = plt.figure(
        figsize=(W_MM * st.MM, H_MM * st.MM),
        facecolor="white",
    )

    axes = []
    panel_centers = []
    for i, result in enumerate(results):
        x = LEFT_MM + i * (AX_W_MM + GAP_MM)
        ax = axes_mm(fig, x, AX_Y_MM + ROW_SHIFT_MM, AX_W_MM, AX_H_MM)
        draw_panel(ax, result)
        axes.append(ax)
        panel_centers.append((x + 0.5 * AX_W_MM) / W_MM)

        # Keep all prose outside the data region: panel letter, system title,
        # and the numerical robustness summary occupy the top margin.
        add_panel_letter(fig, x - 6.0, 69.5 + ROW_SHIFT_MM, f"({chr(ord('a') + i)})")
        fig.text(
            panel_centers[-1], (62.0 + ROW_SHIFT_MM) / H_MM,
            SYSTEM_TITLES[result["system"]],
            ha="center", va="center", fontsize=st.FS_TEXT,
        )

        T1t = f"{int(result['T1_trim'])}" if np.isfinite(result["T1_trim"]) else "--"
        T2t = f"{int(result['T2_trim'])}" if np.isfinite(result["T2_trim"]) else "--"
        summary = (
            rf"$T_1$: {int(result['T1_ref'])}$\rightarrow${T1t}"
            "\n"
            rf"$T_2$: {int(result['T2_ref'])}$\rightarrow${T2t}; "
            rf"$\eta_{{\rm rec}}={result['eta_rec']:.3g}$"
        )
        fig.text(
            panel_centers[-1], (58.2 + ROW_SHIFT_MM) / H_MM, summary,
            ha="center", va="top", fontsize=0.84 * st.FS_TICK,
            linespacing=1.12,
        )

    axes[0].set_ylabel(r"$S(T)$")

    # Second row: representation capacity, same columns and T ranges.
    rank_axes = []
    for i, result in enumerate(results):
        x = LEFT_MM + i * (AX_W_MM + GAP_MM)
        ax = axes_mm(fig, x, AX_Y_MM, AX_W_MM, AX_H_MM)
        draw_rank_panel(ax, result)
        rank_axes.append(ax)
        add_panel_letter(fig, x - 6.0, AX_Y_MM + AX_H_MM + 13.0,
                         f"({chr(ord('a') + len(results) + i)})")
    rank_axes[0].set_ylabel(r"$S(T)$")

    # Shared line-style key is also outside the plotting region.  It is placed
    # above panel (a), where it does not compete with the system titles.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2], labels[:2],
        loc="upper center",
        bbox_to_anchor=(panel_centers[0], 0.995),
        ncol=2,
        frameon=False,
        fontsize=0.82 * st.FS_TICK,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

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
    st.apply_style()

    results = []
    for system in ("Lorenz", "KS", "ADK"):
        results.append(run_system(system))

    summary_path = save_summary(results)
    curves_path = save_curves(results)
    rank_summary_path, rank_curves_path = save_rank_tables(results)
    pdf_path, png_path = make_figure(results)

    print("\nSaved:")
    print(" ", pdf_path)
    print(" ", png_path)
    print(" ", summary_path)
    print(" ", curves_path)
    print(" ", rank_summary_path)
    print(" ", rank_curves_path)


if __name__ == "__main__":
    main()
