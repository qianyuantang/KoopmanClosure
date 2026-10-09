from __future__ import annotations

"""Generate manuscript Fig. 4 analytical panels for adenylate kinase (ADK).

Expected project layout
-----------------------
root/
  closure_pipeline.py
  figure_style.py
  fig4_adk.py
  data/
    closedCA.pdb
    C1.0_1.dcd

This script draws panels (b)--(e); structural panel (a) is prepared separately:
  (b) scale-time map
  (c) neighboring-scale similarity S(T)
  (d) beta traces at the selected short and long scales with reference angles
  (e) standardized partial-coordinate responses (functional subsets, low-RMSF and random controls), compared against RMSD

The closure analysis is complete within this script; panel (a) contains only the
separate protein-structure illustration.
"""

from pathlib import Path
import os
import sys
import shutil
import warnings
import gc

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter, NullFormatter
from matplotlib.lines import Line2D

from figure_style import (
    apply_style, MM, FS_PANEL, FS_TEXT, FS_TICK,
    C_BETA, C_SEL, C_SRAW, C_SSMOOTH, LW_GUIDE, LW_AXIS,
)
from closure_pipeline import (
    prepare_window_sums,
    compute_metrics_for_T,
    globally_scaled_map,
    analyze_timescales,
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

PDB_CANDIDATES = [
    DATA / "closedCA.pdb",
    ROOT / "closedCA.pdb",
]
DCD_CANDIDATES = [
    DATA / "C1.0_1.dcd",
    ROOT / "C1.0_1.dcd",
]

# -----------------------------------------------------------------------------
# ADK configuration
# -----------------------------------------------------------------------------
ANALYSIS_START = 50000
ANALYSIS_END = 70000
# ADK uses a system-specific fixed representation: retain the smallest rank whose
# cumulative squared singular-value energy reaches 0.98.  Once this rank is
# fixed, the full T scan uses the shared closure pipeline without further
# scale-dependent reduction.  The cumulative-singular-value rank is reported
# only as a cross-system diagnostic and never enters scale selection.
SVD_ENERGY_FRACTION = 0.98
SIGMA_SUM_DIAGNOSTIC_FRACTION = 0.98

T_MIN = 50
T_SCAN_MAX_FRACTION = 0.25

# closedCA.pdb contains one CA atom for each ADK residue, numbered 1--214.
# Structural subsets are defined only by PDB residue numbers and converted to
# Cartesian rows after topology validation.  Each restricted subset uses the
# first 20 residues of the corresponding domain definition.
EXPECTED_RESIDUE_IDS = np.arange(1, 215, dtype=int)
PARTIAL_RESIDUE_GROUPS = {
    "CORE": (1, 20),
    "NMP": (30, 49),
    "LID": (122, 141),
}
PARTIAL_CASES = [
    {"name": "CORE+NMP+LID", "groups": ("CORE", "NMP", "LID"), "color": "#E69F00"},
    {"name": "CORE",         "groups": ("CORE",),               "color": "#56B4E9"},
]
PARTIAL_RANK = 3
LOW_RMSF_COUNT = 10
RANDOM_RESIDUE_COUNT = 10
RANDOM_RESIDUE_SEED = 7
LOW_RMSF_COLOR = "#3AA657"
RANDOM_RESIDUE_COLOR = "#8C6BB1"
RMSD_COLOR = "0.35"

ANGLE_RESIDUE_GROUPS_NMP = {
    "arm1": (115, 125),
    "vertex": (90, 100),
    "arm2": (35, 55),
}
ANGLE_RESIDUE_GROUPS_LID = {
    "arm1": (179, 185),
    "vertex": (115, 125),
    "arm2": (125, 153),
}


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError("Missing required file. Tried:\n  " + "\n  ".join(str(p) for p in paths))


def _is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _choose_ascii_stage_dir() -> Path:
    candidates = []
    try:
        candidates.append(Path(sys.executable).resolve().parent.parent / "_mdtraj_ascii_stage")
    except Exception:
        pass
    tmp = os.environ.get("TEMP") or os.environ.get("TMP")
    if tmp:
        candidates.append(Path(tmp) / "_mdtraj_ascii_stage")
    anchor = Path.cwd().anchor
    if anchor:
        candidates.append(Path(anchor) / "_mdtraj_ascii_stage")
    for candidate in candidates:
        if not _is_ascii_path(candidate):
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test = candidate / "_write_test.tmp"
            test.write_bytes(b"ok")
            test.unlink()
            return candidate
        except Exception:
            continue
    raise RuntimeError("Could not create an ASCII-only staging directory for MDTraj.")


def _stage_for_mdtraj(source_path: Path, stage_dir: Path, target_name: str) -> Path:
    source = Path(source_path).resolve()
    target = stage_dir / target_name
    needs_copy = True
    if target.exists():
        try:
            s1 = source.stat(); s2 = target.stat()
            needs_copy = (s1.st_size != s2.st_size) or (s1.st_mtime_ns != s2.st_mtime_ns)
        except OSError:
            needs_copy = True
    if needs_copy:
        shutil.copy2(source, target)
    return target


def load_aligned_analysis_segment(chunk_size=2000):
    """Load the analysed CA frames, align them to closedCA, and return the anchor.

    closedCA.pdb is used in two distinct but consistent ways:
      1) as the rigid-body alignment reference for every trajectory chunk;
      2) as the physical anchor x_ref in q = x - x_ref.

    The topology is validated to contain exactly one CA atom for each residue
    1--214, so all later residue selections can be specified by PDB residue ID.
    """
    pdb_path = first_existing(PDB_CANDIDATES)
    dcd_path = first_existing(DCD_CANDIDATES)

    if _is_ascii_path(pdb_path.resolve()) and _is_ascii_path(dcd_path.resolve()):
        pdb_for_mdtraj = pdb_path.resolve()
        dcd_for_mdtraj = dcd_path.resolve()
    else:
        stage = _choose_ascii_stage_dir()
        pdb_for_mdtraj = _stage_for_mdtraj(pdb_path, stage, "closedCA.pdb")
        dcd_for_mdtraj = _stage_for_mdtraj(dcd_path, stage, "C1.0_1.dcd")

    reference = md.load(str(pdb_for_mdtraj), top=str(pdb_for_mdtraj))
    if reference.n_frames != 1:
        raise ValueError("closedCA.pdb must contain exactly one reference structure.")

    atoms = list(reference.topology.atoms)
    residues = list(reference.topology.residues)
    if len(atoms) != len(residues) or any(atom.name != "CA" for atom in atoms):
        raise ValueError("closedCA.pdb must contain exactly one CA atom per residue.")

    residue_ids = np.asarray([res.resSeq for res in residues], dtype=int)
    if not np.array_equal(residue_ids, EXPECTED_RESIDUE_IDS):
        raise ValueError(
            "Unexpected closedCA residue numbering. Expected continuous PDB residue "
            "IDs 1--214; got "
            f"{residue_ids[0] if residue_ids.size else 'empty'}--"
            f"{residue_ids[-1] if residue_ids.size else 'empty'} with "
            f"{residue_ids.size} residues."
        )

    reference_xyz = np.asarray(reference.xyz[0], dtype=np.float32).copy()
    atom_indices = np.arange(reference.n_atoms, dtype=int)

    n_target = ANALYSIS_END - ANALYSIS_START
    xyz = np.empty((n_target, reference.n_atoms, 3), dtype=np.float32)
    times = np.empty(n_target, dtype=np.float64)

    filled = 0
    iterator = md.iterload(
        str(dcd_for_mdtraj),
        chunk=int(chunk_size),
        top=str(pdb_for_mdtraj),
        atom_indices=atom_indices,
        skip=ANALYSIS_START,
    )

    for chunk in iterator:
        if filled >= n_target:
            break

        # Remove rigid-body motion with respect to the same closed structure that
        # defines the origin of the closure coordinates.
        chunk.superpose(reference, frame=0)

        take = min(chunk.n_frames, n_target - filled)
        xyz[filled:filled + take] = chunk.xyz[:take]
        times[filled:filled + take] = chunk.time[:take]
        filled += take

        print(
            f"ADK chunk load: {filled}/{n_target} analysed frames "
            f"({reference.n_atoms} C-alpha atoms)"
        )
        del chunk

    if filled != n_target:
        raise ValueError(
            f"Could load only {filled} analysed frames; expected {n_target} "
            f"for [{ANALYSIS_START}, {ANALYSIS_END})."
        )

    print(
        f"ADK retained segment only: xyz shape={xyz.shape}, "
        f"memory={xyz.nbytes / 1024**2:.1f} MiB"
    )
    print(
        "ADK residue mapping validated: 214 CA atoms, PDB residue IDs 1--214; "
        "closedCA is both alignment reference and closure anchor."
    )

    return xyz, times, reference_xyz, residue_ids


def residue_range_atom_indices(residue_ids, start, end):
    """Atom indices for an inclusive PDB-residue interval [start, end]."""
    residue_ids = np.asarray(residue_ids, dtype=int)
    start, end = int(start), int(end)
    idx = np.flatnonzero((residue_ids >= start) & (residue_ids <= end))
    expected = end - start + 1
    if idx.size != expected:
        present = residue_ids[idx].tolist()
        raise ValueError(
            f"Residues {start}--{end} are not fully represented in closedCA.pdb; "
            f"found {present}."
        )
    return idx


def residue_groups_to_coordinate_rows(residue_ids, groups):
    """Cartesian row indices for one or more named partial-observation groups."""
    atom_idx = []
    for name in groups:
        if name not in PARTIAL_RESIDUE_GROUPS:
            raise KeyError(f"Unknown ADK residue group: {name}")
        start, end = PARTIAL_RESIDUE_GROUPS[name]
        atom_idx.extend(residue_range_atom_indices(residue_ids, start, end).tolist())

    atom_idx = np.asarray(sorted(set(atom_idx)), dtype=int)
    rows = np.concatenate([3 * atom_idx + k for k in range(3)])
    # Concatenating by Cartesian component above changes order; sort to recover
    # x,y,z rows residue by residue, matching flatten_coordinates().
    return np.sort(rows)


def flatten_coordinates(xyz):
    """frames x atoms x xyz -> (atoms*3) x frames."""
    xyz = np.asarray(xyz)
    return np.transpose(xyz, (1, 2, 0)).reshape(-1, xyz.shape[0])


def calculate_angle_from_xyz(xyz, residue_ids, residue_ranges):
    """Reference angle from inclusive PDB residue ranges on the aligned CA trace."""
    centers = {}
    for name, (start, end) in residue_ranges.items():
        idx = residue_range_atom_indices(residue_ids, start, end)
        centers[name] = xyz[:, idx, :].mean(axis=1)

    v1 = centers["arm1"] - centers["vertex"]
    v2 = centers["arm2"] - centers["vertex"]
    cosine = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def compute_rmsd_from_aligned_xyz(xyz, reference_xyz):
    """CA RMSD to the closedCA anchor after rigid-body alignment."""
    delta = xyz - np.asarray(reference_xyz, dtype=float)[None, :, :]
    return np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1))


def compute_local_residue_rmsf(anchored_coordinates, window_T, stride=None):
    """Median within-window residue RMSF at the selected short scale T1.

    RMSF is computed only after T1 has been selected by the shared closure
    pipeline.  It is therefore a post-selection observability control, not an
    input to the scale-selection procedure.
    """
    H = np.asarray(anchored_coordinates, dtype=float)
    if H.ndim != 2 or H.shape[0] % 3 != 0:
        raise ValueError("Anchored Cartesian matrix must have 3 rows per residue.")

    n_res = H.shape[0] // 3
    n_frames = H.shape[1]
    T = int(window_T)
    if T < 3 or T > n_frames:
        raise ValueError(f"Invalid RMSF window T={T} for n={n_frames}.")
    if stride is None:
        stride = max(T // 2, 1)
    stride = int(stride)

    xyz_view = H.reshape(n_res, 3, n_frames)
    values = []
    for start in range(0, n_frames - T + 1, stride):
        block = xyz_view[:, :, start:start + T]
        fluct = block - np.mean(block, axis=2, keepdims=True)
        values.append(np.sqrt(np.mean(np.sum(fluct * fluct, axis=1), axis=1)))
    if not values:
        raise RuntimeError("No windows were available for local RMSF.")
    return np.median(np.asarray(values, dtype=float), axis=0)


def atom_indices_to_coordinate_rows(atom_indices):
    """Cartesian rows x,y,z for zero-based CA atom/residue indices."""
    idx = np.sort(np.asarray(atom_indices, dtype=int))
    if idx.size == 0:
        return np.empty(0, dtype=int)
    return np.sort(np.concatenate([3 * idx + k for k in range(3)]))


def smooth_series(y, width=9):
    y = np.asarray(y, dtype=float)
    width = max(int(width), 1)
    if width <= 1 or y.size < 3:
        return y.copy()
    if width % 2 == 0:
        width += 1
    width = min(width, y.size if y.size % 2 == 1 else y.size - 1)
    if width < 3:
        return y.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    pad = width // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def zscore_series(y):
    y = np.asarray(y, dtype=float)
    mu = float(np.nanmean(y))
    sd = float(np.nanstd(y))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(y)
    return (y - mu) / sd


def estimate_prefix_memory_bytes(rank, n_samples, bytes_per_value=8):
    """Persistent memory of cVV, cDV and cD2."""
    rank = int(rank)
    n_samples = int(n_samples)
    b = int(bytes_per_value)
    return (
        2 * (n_samples + 1) * rank * rank * b
        + rank * (n_samples + 1) * b
    )


def report_rank_memory(label, rank, n_samples):
    gib = estimate_prefix_memory_bytes(rank, n_samples) / 1024**3
    print(f"{label}: rank={rank}, estimated prefix memory={gib:.3f} GiB")
    return gib


def compute_reduced_coordinates_adk(H, dt, rank=None):
    """Construct the fixed ADK reduced representation from H H^T.

    Diagonalizing the 642 x 642 Gram matrix gives the left singular vectors and
    singular values without allocating the full right-singular-vector matrix.
    The ADK analysis rank is the smallest rank reaching 98% cumulative squared
    singular-value energy.  A 98% cumulative-singular-value rank is returned as
    a diagnostic only and does not enter the closure analysis.

    rank=None (Fig. 4) keeps the energy rank.  An explicit rank is used only by
    the representation-capacity scan of Fig. S3; the returned rank is the one
    actually used for V.
    """
    H = np.asarray(H)
    if H.ndim != 2:
        raise ValueError("H must be a 2D coordinate matrix")

    # The Gram matrix is small enough to diagonalize in float64.
    gram = H @ H.T
    gram = np.asarray(gram, dtype=np.float64)

    evals, U = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    U = U[:, order]
    s = np.sqrt(evals)

    # ADK analysis rank: cumulative squared-singular-value energy.
    cumulative_energy = np.cumsum(evals) / np.sum(evals)
    rank_energy = int(
        np.searchsorted(cumulative_energy, SVD_ENERGY_FRACTION) + 1
    )

    # Cross-system cumulative-singular-value diagnostic; not used downstream.
    cumulative_sigma = np.cumsum(s) / np.sum(s)
    rank_sigma = int(
        np.searchsorted(cumulative_sigma, SIGMA_SUM_DIAGNOSTIC_FRACTION) + 1
    )

    print(
        f"ADK analysis rank = {rank_energy} "
        f"({100*SVD_ENERGY_FRACTION:.1f}% cumulative SVD energy)"
    )
    print(
        f"ADK cumulative-sigma diagnostic rank = {rank_sigma} "
        f"({100*SIGMA_SUM_DIAGNOSTIC_FRACTION:.1f}% cumulative singular values)"
    )
    report_rank_memory("  energy-rank analysis", rank_energy, H.shape[1])
    report_rank_memory("  sigma-sum diagnostic", rank_sigma, H.shape[1])

    # Reduced coordinates V = U_r^T H.
    rank_used = rank_energy if rank is None else int(rank)
    Ur = U[:, :rank_used]
    V = Ur.T @ H
    V = np.asarray(V, dtype=np.float64)
    Vdot = np.gradient(V, float(dt), axis=1, edge_order=2)

    del gram, evals, U, Ur
    gc.collect()

    return V, Vdot, rank_used, s, rank_sigma


def compute_partial_coordinate_beta(
    anchored_coordinate_matrix,
    coordinate_rows,
    start,
    end,
    time_grid,
    window_T,
    retained_rank=PARTIAL_RANK,
):
    """Partial-coordinate ADK beta from anchor-referenced Cartesian rows.

    Partial-observation controls use a fixed local rank r=3.  Their raw beta
    amplitudes are not compared across subsets; panel (e) standardizes each trace
    independently and compares temporal organization.
    """
    rows = np.asarray(coordinate_rows, dtype=int)
    H_local = np.asarray(anchored_coordinate_matrix[rows, start:end], dtype=np.float32)

    gram = np.asarray(H_local @ H_local.T, dtype=np.float64)
    evals, U = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    U = U[:, order]

    rank_local = min(int(retained_rank), U.shape[1])
    V_local = U[:, :rank_local].T @ H_local
    V_local = np.asarray(V_local, dtype=np.float64)

    local_time = np.asarray(time_grid[start:end], dtype=float)
    dt_local = float(np.median(np.diff(local_time)))
    Vdot_local = np.gradient(V_local, dt_local, axis=1, edge_order=2)

    sums_local = prepare_window_sums(V_local, Vdot_local)
    beta_local, _ = compute_metrics_for_T(*sums_local, int(window_T))
    beta_time_local = (
        local_time[:len(beta_local)]
        + 0.5 * int(window_T) * dt_local
    )

    energy_fraction = np.cumsum(evals) / np.sum(evals)
    print(
        "Partial coordinates:",
        f"n_rows={rows.size}",
        "| H.shape =",
        H_local.shape,
        "| retained rank =",
        rank_local,
        "| cumulative energy at retained rank =",
        float(energy_fraction[rank_local - 1]),
    )

    del H_local, gram, evals, U, V_local, Vdot_local, sums_local
    gc.collect()

    return beta_time_local, beta_local, rank_local


# -----------------------------------------------------------------------------
# Figure geometry (mm).  The top row reuses the Fig. 3 placement of the
# scale-time map, its colorbar and S(T); the trace rows reuse the horizontal
# extent of the Fig. 3 closure traces.
# -----------------------------------------------------------------------------
W_MM, H_MM = 174.0, 158.0

TOP_Y, TOP_H = 115.0, 39.5
B_X, B_W = 13.0, 83.7
CB_X, CB_W = 99.3, 1.8
C_X, C_W = 125.0, 35.5

TR_X, TR_W = 18.0, 137.0
D_H = 18.5
D_UPPER_Y = 78.5          # T2 trace
D_LOWER_Y = 56.0          # T1 trace, directly above panel (e) which tests T1
E_Y, E_H = 11.0, 30.0

YLABEL_LEFT_X = 7.0
YLABEL_RIGHT_X = 165.2
LETTER_X_LEFT = 1.0
LETTER_X_C = 109.0

# Physical offset of an axis multiplier below its axis.  Equal to the
# Fig. 3(c) placement (0.095 of a 39.5 mm axis), so every multiplier in both
# figures sits at the same distance from its axis.
MULT_OFFSET_PT = 0.095 * 39.5 / 25.4 * 72.0

TIME_TICK_STEP = 5000
TIME_EXPONENT = 4
S_TICKS = (0.90, 0.95, 1.00)

C_ANGLE_NMP = "#2B8C66"
C_ANGLE_LID = "#3B5AA9"
C_ANGLE_AXIS = "0.20"
ANGLE_LW = 0.70
ANGLE_ALPHA = 0.46
BETA_LW = 1.08


def axes_mm(fig, x, y, w, h):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])


def add_panel_letter(fig, x_mm, y_mm, label):
    fig.text(x_mm / W_MM, y_mm / H_MM, label, fontsize=FS_PANEL,
             fontweight="bold", ha="left", va="top")


def add_axis_multiplier(ax, exponent, x=1.0):
    ax.annotate(
        rf"$\times 10^{{{int(exponent)}}}$",
        xy=(x, 0.0), xycoords="axes fraction",
        xytext=(0.0, -MULT_OFFSET_PT), textcoords="offset points",
        ha="right", va="top", fontsize=0.90 * FS_TICK,
        annotation_clip=False,
    )


def time_ticks(t0, t1, step=TIME_TICK_STEP):
    start = int(np.ceil(t0 / step) * step)
    end = int(np.floor(t1 / step) * step)
    return np.arange(start, end + 1, step, dtype=float)


def format_time_axis(ax, t0, t1, show_labels=True):
    """Shared time axis for (b), (d) and (e): ticks every 5000 frames,
    labelled in units of 10^4 with one multiplier at the right end."""
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(FixedLocator(time_ticks(t0, t1)))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x / 10.0**TIME_EXPONENT:.1f}")
    )
    if show_labels:
        add_axis_multiplier(ax, TIME_EXPONENT)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def headroom_ylim(values, qlo, qhi, pad_lo=0.08, pad_hi=0.30):
    """Compact limits with extra room at the top for the in-axes scale label."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)
    lo = float(np.quantile(x, qlo))
    hi = float(np.quantile(x, qhi))
    span = max(hi - lo, 1e-8)
    return (lo - pad_lo * span, hi + pad_hi * span)


def compare_scale_traces(time_a, beta_a, time_b, beta_b, n_grid=3000):
    """Correlation and normalized RMS separation on their common support."""
    lo = max(float(time_a[0]), float(time_b[0]))
    hi = min(float(time_a[-1]), float(time_b[-1]))
    if hi <= lo:
        return np.nan, np.nan

    grid = np.linspace(lo, hi, int(n_grid))
    a = np.interp(grid, time_a, beta_a)
    b = np.interp(grid, time_b, beta_b)

    corr = float(np.corrcoef(a, b)[0, 1])

    za = (a - np.mean(a)) / (np.std(a) + 1e-12)
    zb = (b - np.mean(b)) / (np.std(b) + 1e-12)
    zrms = float(np.sqrt(np.mean((za - zb) ** 2)))
    return corr, zrms




def paper_axes(ax):
    """Spines only; tick geometry comes from figure_style, as in Fig. 3."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def save_figure(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", dpi=600, facecolor="white")
    print(f"Saved: {OUT / (stem + '.png')}")
    print(f"Saved: {OUT / (stem + '.pdf')}")


# -----------------------------------------------------------------------------
# Main figure construction
# -----------------------------------------------------------------------------

def main():
    apply_style()

    # Stream only the analysed interval.  Every frame is aligned to closedCA,
    # which is also the physical anchor x_ref used to form q = x - x_ref.
    xyz, source_time, reference_xyz, residue_ids = load_aligned_analysis_segment(
        chunk_size=2000
    )
    source_time = source_time.astype(float)
    dt = float(np.median(np.diff(source_time)))

    # Structural observables are reference labels only: they are evaluated after
    # alignment and never enter scale selection.  All ranges below are PDB
    # residue numbers, not zero-based array slices.
    angle_nmp = calculate_angle_from_xyz(
        xyz, residue_ids, ANGLE_RESIDUE_GROUPS_NMP
    )
    angle_lid = calculate_angle_from_xyz(
        xyz, residue_ids, ANGLE_RESIDUE_GROUPS_LID
    )
    rmsd_full = compute_rmsd_from_aligned_xyz(xyz, reference_xyz)

    # Anchor-referenced Cartesian representation.  flatten_coordinates orders
    # rows as x,y,z for residue 1, then x,y,z for residue 2, etc.
    flattened_abs = flatten_coordinates(xyz).astype(np.float32, copy=False)
    reference_flat = np.asarray(reference_xyz, dtype=np.float32).reshape(-1, 1)
    H = np.asarray(flattened_abs - reference_flat, dtype=np.float32)
    del xyz, flattened_abs, reference_flat
    gc.collect()

    # System-specific preprocessing fixes one reduced representation.  The rank
    # is then held fixed throughout the entire shared T scan.
    V, Vdot, rank, singular_values, sigma_rank_diag = (
        compute_reduced_coordinates_adk(H, dt)
    )
    sums = prepare_window_sums(V, Vdot)

    n_samples = V.shape[1]
    t_max = int(T_SCAN_MAX_FRACTION * n_samples)

    # Shared closure pipeline: same nested log-T grids, normalization, S(T),
    # cross-grid event retention, and recovery-side localization as Figs. 1--3.
    analysis = analyze_timescales(
        sums,
        source_time,
        dt,
        T_MIN,
        t_max,
        max_windows=2,
        progress=True,
    )
    if len(analysis["selected"]) < 2:
        raise RuntimeError(
            "ADK analysis retained fewer than two reorganization events; "
            "Fig. 4 requires both T1 and T2."
        )

    T1, T2 = map(int, analysis["selected"][:2])
    common_time = np.asarray(analysis["common_time"], dtype=float)
    display_Ts = np.asarray(analysis["display_T_values"], dtype=int)
    display_scale_map = np.asarray(analysis["display_scale_map"], dtype=float)
    baseline_Ts = np.asarray(analysis["T_values"], dtype=int)
    diag = analysis["diag"]
    selected_events = analysis["selected_events"]

    print(
        f"ADK shared T scan: fine={len(analysis['grids']['fine']['T_values'])}, "
        f"baseline={len(baseline_Ts)}, "
        f"coarse={len(analysis['grids']['coarse']['T_values'])}; "
        f"S smoothing grid={len(diag['log_grid'])}"
    )
    print(
        f"ADK selected scales: T1={T1}, T2={T2}; "
        f"T2/T1={T2 / T1:.2f}"
    )
    for j, ev in enumerate(selected_events[:2], start=1):
        print(
            f"  event {j}: band=[{ev['T_left']:.1f}, {ev['T_right']:.1f}], "
            f"T_min={ev['T_min']:.1f}, depth={ev['depth']:.5f}, "
            f"prominence={ev['prominence']:.5f}, d_org={ev['d_org']:.5f}"
        )

    # Closure traces are post-selection diagnostics at the two retained scales.
    beta1, _ = compute_metrics_for_T(*sums, T1)
    beta2, _ = compute_metrics_for_T(*sums, T2)
    beta1_time = source_time[:len(beta1)] + 0.5 * T1 * dt
    beta2_time = source_time[:len(beta2)] + 0.5 * T2 * dt

    corr12, zrms12 = compare_scale_traces(
        beta1_time, beta1, beta2_time, beta2
    )
    print(
        f"Scale-trace comparison T1={T1} vs T2={T2}: "
        f"corr={corr12:.4f}, normalized-RMS-separation={zrms12:.4f}"
    )

    # Post-selection partial-observation test at T1.  Functional subsets are
    # compared with low-RMSF and size-matched random controls.
    T_partial = T1
    partial_results = []
    for case in PARTIAL_CASES:
        rows = residue_groups_to_coordinate_rows(residue_ids, case["groups"])
        residue_text = "+".join(
            f"{name}:{PARTIAL_RESIDUE_GROUPS[name][0]}-{PARTIAL_RESIDUE_GROUPS[name][1]}"
            for name in case["groups"]
        )
        print(
            f"Partial case {case['name']}: {residue_text}; "
            f"{rows.size // 3} residues, {rows.size} Cartesian rows"
        )
        t_local, beta_local, local_rank = compute_partial_coordinate_beta(
            H, rows, 0, n_samples, source_time, T_partial,
            retained_rank=PARTIAL_RANK,
        )
        partial_results.append({
            "name": case["name"],
            "groups": case["groups"],
            "color": case["color"],
            "time": t_local,
            "beta": beta_local,
            "rank": local_rank,
        })

    residue_rmsf = compute_local_residue_rmsf(
        H, window_T=T1, stride=max(T1 // 2, 1)
    )
    low_rmsf_indices = np.argsort(residue_rmsf)[:LOW_RMSF_COUNT]
    low_rmsf_rows = atom_indices_to_coordinate_rows(low_rmsf_indices)
    t_low, beta_low, rank_low = compute_partial_coordinate_beta(
        H, low_rmsf_rows, 0, n_samples, source_time, T_partial,
        retained_rank=PARTIAL_RANK,
    )
    partial_results.append({
        "name": f"low-RMSF {LOW_RMSF_COUNT}",
        "color": LOW_RMSF_COLOR,
        "time": t_low,
        "beta": beta_low,
        "rank": rank_low,
        "residue_indices": low_rmsf_indices,
    })

    rng = np.random.default_rng(RANDOM_RESIDUE_SEED)
    random_indices = np.sort(
        rng.choice(np.arange(len(residue_ids), dtype=int),
                   size=RANDOM_RESIDUE_COUNT, replace=False)
    )
    random_low_overlap = np.intersect1d(random_indices, low_rmsf_indices)
    print(
        f"Random-{RANDOM_RESIDUE_COUNT} / low-RMSF overlap: "
        f"{len(random_low_overlap)} residue(s); "
        f"PDB IDs={residue_ids[random_low_overlap].tolist()}"
    )
    random_rows = atom_indices_to_coordinate_rows(random_indices)
    t_rand, beta_rand, rank_rand = compute_partial_coordinate_beta(
        H, random_rows, 0, n_samples, source_time, T_partial,
        retained_rank=PARTIAL_RANK,
    )
    partial_results.append({
        "name": f"random {RANDOM_RESIDUE_COUNT}",
        "color": RANDOM_RESIDUE_COLOR,
        "time": t_rand,
        "beta": beta_rand,
        "rank": rank_rand,
        "residue_indices": random_indices,
    })
    print(
        f"Low-RMSF PDB residues: {residue_ids[low_rmsf_indices].tolist()}"
    )
    print(
        f"Random PDB residues (seed={RANDOM_RESIDUE_SEED}): "
        f"{residue_ids[random_indices].tolist()}"
    )

    t0, t1 = float(source_time[0]), float(source_time[-1])

    # ----- figure canvas -----
    # Panels (b)--(e); structural panel (a) is assembled separately.
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM), facecolor="white")

    # ------------------------------------------------------------------
    # (b) scale-time map
    # ------------------------------------------------------------------
    axb = axes_mm(fig, B_X, TOP_Y, B_W, TOP_H)
    mesh = axb.pcolormesh(
        common_time,
        display_Ts,
        globally_scaled_map(display_scale_map),
        shading="auto",
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=4.0),
        rasterized=True,
    )
    axb.set_yscale("log")
    axb.set_ylim(display_Ts[0], display_Ts[-1])
    format_time_axis(axb, t0, t1, show_labels=True)
    axb.set_xlabel(r"$t$")
    axb.set_ylabel(r"Window size $T$")
    for T in (T1, T2):
        axb.axhline(T, color="white", lw=LW_GUIDE, ls=(0, (3, 2)), alpha=0.95)
    paper_axes(axb)

    cax = axes_mm(fig, CB_X, TOP_Y, CB_W, TOP_H)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=1.8, width=0.5, labelsize=FS_TICK)
    cbar.set_label(r"$\widetilde{D}_{\mathrm{sample}}$", labelpad=0.5)

    # ------------------------------------------------------------------
    # (c) S(T), drawn with the Fig. 3(c) conventions
    # ------------------------------------------------------------------
    axc = axes_mm(fig, C_X, TOP_Y, C_W, TOP_H)
    T_base = baseline_Ts.astype(float)

    axc.plot(T_base, diag["raw"], ls="none", marker="o", ms=1.8,
             markeredgewidth=0, color=C_SRAW, alpha=0.28)
    axc.plot(np.exp(diag["log_grid"]), diag["smooth"], color=C_SSMOOTH, lw=1.05)

    # Panel (c) resolves the first two retained reorganization bands; panel (b)
    # retains the complete scan.
    second_right = float(selected_events[1]["T_right"])
    x_display_max = min(
        float(T_base[-1]),
        max(1.20 * second_right, 1.15 * float(T2)),
    )
    axc.set_xscale("log")
    axc.set_xlim(float(T_base[0]), x_display_max)

    c_tick_candidates = np.array([50.0, 100.0, 200.0, 500.0, 1000.0])
    c_ticks = c_tick_candidates[
        (c_tick_candidates >= float(T_base[0]))
        & (c_tick_candidates <= x_display_max)
    ]
    axc.xaxis.set_major_locator(FixedLocator(c_ticks))
    axc.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 100.0:g}"))
    axc.xaxis.set_minor_formatter(NullFormatter())
    axc.tick_params(axis="x", which="minor", labelbottom=False)

    # Event-band shading separates each reorganization interval from its
    # representative recovery-side scale.
    for ev in selected_events[:2]:
        axc.axvspan(
            ev["T_left"], ev["T_right"],
            color=C_SEL, alpha=0.055, lw=0.0, zorder=0,
        )

    log_display = np.asarray(diag["log_grid"], dtype=float)
    smooth_display = np.asarray(diag["smooth"], dtype=float)
    in_view = np.exp(log_display) <= x_display_max
    sv = smooth_display[in_view]
    sv = sv[np.isfinite(sv)]
    if sv.size:
        smin, smax = float(np.min(sv)), float(np.max(sv))
        yr = max(smax - smin, 2e-4)
        ylo = max(0.0, smin - 0.12 * yr)
        yhi = min(1.002, smax + 0.14 * yr)
        if yhi <= ylo:
            yhi = ylo + 5e-4
        axc.set_ylim(ylo, yhi)
    # Use a sparse set of clean two-decimal ticks.  For the present AdK
    # range this gives 0.96 and 0.97 without repeated rounded labels.
    ylo, yhi = axc.get_ylim()
    tick_lo = np.ceil((ylo - 1e-12) * 100.0) / 100.0
    tick_hi = np.floor((yhi + 1e-12) * 100.0) / 100.0
    y_ticks = np.arange(tick_lo, tick_hi + 0.005, 0.01)
    if y_ticks.size > 3:
        keep = np.linspace(0, y_ticks.size - 1, 3).round().astype(int)
        y_ticks = y_ticks[np.unique(keep)]
    if y_ticks.size:
        axc.yaxis.set_major_locator(FixedLocator(y_ticks))
    axc.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))

    for j, T in enumerate((T1, T2), start=1):
        sval = float(np.interp(np.log(T), diag["log_grid"], diag["smooth"]))
        axc.axvline(T, ymin=0.0, ymax=0.86, color=C_SEL, ls="-.",
                    lw=LW_GUIDE, alpha=0.88)
        axc.plot(T, sval, "o", ms=3.1, color=C_SEL)
        if j == 1:
            offset, ha, va = (6, -5), "left", "top"
        else:
            offset, ha, va = (-6, 5), "right", "bottom"
        axc.annotate(
            rf"$T_{j}={int(T)}$",
            xy=(T, sval), xycoords="data",
            xytext=offset, textcoords="offset points",
            ha=ha, va=va, color=C_SEL, fontsize=FS_TICK,
            annotation_clip=False,
        )

    axc.set_xlabel(r"Window size $T$", labelpad=7.0)
    axc.set_ylabel(r"$S(T)$")
    add_axis_multiplier(axc, 2, x=0.985)
    paper_axes(axc)

    # ------------------------------------------------------------------
    # (d) full-coordinate traces: T2 above, T1 below (Fig. 3 order; T1 then
    # sits directly above panel e, which tests the T1 response)
    # ------------------------------------------------------------------
    d_items = [
        {"label": rf"$T_2={T2}$", "time": beta2_time, "beta": beta2,
         "y": D_UPPER_Y, "show_x": False},
        {"label": rf"$T_1={T1}$", "time": beta1_time, "beta": beta1,
         "y": D_LOWER_Y, "show_x": True},
    ]

    finite_angles = np.concatenate([
        angle_nmp[np.isfinite(angle_nmp)],
        angle_lid[np.isfinite(angle_lid)],
    ])
    angle_ylim = headroom_ylim(finite_angles, qlo=0.002, qhi=0.998,
                               pad_lo=0.05, pad_hi=0.30)

    for item in d_items:
        ax = axes_mm(fig, TR_X, item["y"], TR_W, D_H)
        axr = ax.twinx()
        # Draw beta above the reference angles.
        ax.set_zorder(axr.get_zorder() + 1)
        ax.patch.set_visible(False)

        axr.plot(source_time, angle_nmp, color=C_ANGLE_NMP,
                 lw=ANGLE_LW, alpha=ANGLE_ALPHA)
        axr.plot(source_time, angle_lid, color=C_ANGLE_LID,
                 lw=ANGLE_LW, alpha=ANGLE_ALPHA)
        axr.set_ylim(*angle_ylim)
        axr.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axr.tick_params(axis="y", colors=C_ANGLE_AXIS, labelsize=FS_TICK, pad=1.5)
        axr.spines["right"].set_visible(True)
        axr.spines["right"].set_color(C_ANGLE_AXIS)
        axr.spines["right"].set_linewidth(LW_AXIS)
        for side in ("top", "left", "bottom"):
            axr.spines[side].set_visible(False)

        ax.plot(item["time"], item["beta"], color=C_BETA, lw=BETA_LW, alpha=0.97)
        format_time_axis(ax, t0, t1, show_labels=item["show_x"])
        ax.set_ylim(*headroom_ylim(item["beta"], qlo=0.004, qhi=0.996,
                                   pad_lo=0.10, pad_hi=0.30))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis="y", colors=C_BETA, labelsize=FS_TICK, pad=1.5)
        ax.spines["left"].set_color(C_BETA)
        paper_axes(ax)

        ax.text(0.018, 0.955, item["label"], transform=ax.transAxes,
                ha="left", va="top", fontsize=FS_TEXT, color=C_SEL)
        if item["show_x"]:
            ax.set_xlabel(r"$t$", labelpad=1.5)

        yc = (item["y"] + 0.5 * D_H) / H_MM
        fig.text(YLABEL_LEFT_X / W_MM, yc, r"$\beta$", color=C_BETA,
                 fontsize=FS_TEXT, rotation=90, ha="center", va="center")
        fig.text(YLABEL_RIGHT_X / W_MM, yc, "angle (rad)", color=C_ANGLE_AXIS,
                 fontsize=FS_TEXT, rotation=90, ha="center", va="center")

    d_top = D_UPPER_Y + D_H
    legend_handles = [
        Line2D([0], [0], color=C_BETA, lw=BETA_LW, label="closure defect"),
        Line2D([0], [0], color=C_ANGLE_NMP, lw=ANGLE_LW, alpha=ANGLE_ALPHA, label="NMP-CORE"),
        Line2D([0], [0], color=C_ANGLE_LID, lw=ANGLE_LW, alpha=ANGLE_ALPHA, label="LID-CORE"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=((TR_X + 0.5 * TR_W) / W_MM, (d_top + 2.2) / H_MM),
        ncol=3, frameon=False, fontsize=FS_TICK,
        handlelength=2.0, columnspacing=1.15, handletextpad=0.5,
        borderpad=0.0, borderaxespad=0.0,
    )

    # ------------------------------------------------------------------
    # (e) standardized partial-coordinate responses + RMSD
    # ------------------------------------------------------------------
    axe = axes_mm(fig, TR_X, E_Y, TR_W, E_H)

    # Compare timing/shape only.  Every curve is independently z-scored.
    t_e = partial_results[0]["time"]
    rmsd_interp = np.interp(t_e, source_time, smooth_series(rmsd_full, width=21))

    e_series = [{
        "name": "RMSD", "color": RMSD_COLOR, "data": rmsd_interp,
        "lw": 1.10, "ls": (0, (4, 2)),
    }]
    for item in partial_results:
        e_series.append({
            "name": item["name"],
            "color": item["color"],
            "data": np.interp(t_e, item["time"], smooth_series(item["beta"], width=9)),
            "lw": 0.95,
            "ls": "-",
        })

    zcurves = []
    for item in e_series:
        z = zscore_series(item["data"])
        zcurves.append(z)
        axe.plot(t_e, z, color=item["color"], lw=item["lw"], ls=item["ls"],
                 alpha=0.95, label=item["name"])

    z_all = np.concatenate([z[np.isfinite(z)] for z in zcurves])
    if z_all.size:
        lo = float(np.quantile(z_all, 0.001))
        hi = float(np.quantile(z_all, 0.999))
        span = max(hi - lo, 1e-6)
        axe.set_ylim(lo - 0.04 * span, hi + 0.04 * span)

    format_time_axis(axe, t0, t1, show_labels=True)
    axe.set_xlabel(r"$t$", labelpad=1.5)
    axe.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    paper_axes(axe)
    fig.text(YLABEL_LEFT_X / W_MM, (E_Y + 0.5 * E_H) / H_MM, r"$z(\beta)$",
             fontsize=FS_TEXT, rotation=90, ha="center", va="center")

    e_top = E_Y + E_H
    fig.legend(
        handles=axe.get_legend_handles_labels()[0],
        labels=axe.get_legend_handles_labels()[1],
        loc="center",
        bbox_to_anchor=((TR_X + 0.5 * TR_W) / W_MM, (e_top + 2.2) / H_MM),
        ncol=5, frameon=False, fontsize=FS_TICK,
        handlelength=1.8, columnspacing=0.9, handletextpad=0.45,
        borderpad=0.0, borderaxespad=0.0,
    )

    # ------------------------------------------------------------------
    # Panel letters (same convention as Fig. 3)
    # ------------------------------------------------------------------
    top_letter_y = TOP_Y + TOP_H + 2.5
    add_panel_letter(fig, LETTER_X_LEFT, top_letter_y, "(b)")
    add_panel_letter(fig, LETTER_X_C, top_letter_y, "(c)")
    add_panel_letter(fig, LETTER_X_LEFT, d_top + 4.0, "(d)")
    add_panel_letter(fig, LETTER_X_LEFT, e_top + 4.0, "(e)")

    # Save residue definitions in PDB numbering together with Cartesian rows.
    residue_csv = OUT / "fig4_adk_residue_definitions.csv"
    with open(residue_csv, "w", encoding="utf-8") as f:
        f.write(
            "kind,label,residue_start,residue_end,n_residues,"
            "first_atom_index_0based,last_atom_index_0based,"
            "first_cartesian_row_0based,row_stop_exclusive\n"
        )
        for label, (r0, r1) in PARTIAL_RESIDUE_GROUPS.items():
            idx = residue_range_atom_indices(residue_ids, r0, r1)
            f.write(
                f"partial,{label},{r0},{r1},{idx.size},"
                f"{idx[0]},{idx[-1]},{3*idx[0]},{3*(idx[-1]+1)}\n"
            )
        for prefix, groups in (("NMP_angle", ANGLE_RESIDUE_GROUPS_NMP),
                               ("LID_angle", ANGLE_RESIDUE_GROUPS_LID)):
            for label, (r0, r1) in groups.items():
                idx = residue_range_atom_indices(residue_ids, r0, r1)
                f.write(
                    f"angle,{prefix}:{label},{r0},{r1},{idx.size},"
                    f"{idx[0]},{idx[-1]},{3*idx[0]},{3*(idx[-1]+1)}\n"
                )
    print(f"Saved: {residue_csv}")

    # Save post-selection control residue sets in PDB numbering.
    low_rmsf_csv = OUT / "fig4_adk_low_rmsf_residues.csv"
    with open(low_rmsf_csv, "w", encoding="utf-8") as f:
        f.write("rank,pdb_residue_id,atom_index_0based,rmsf\n")
        for j, idx in enumerate(low_rmsf_indices, start=1):
            f.write(
                f"{j},{int(residue_ids[idx])},{int(idx)},"
                f"{float(residue_rmsf[idx]):.10e}\n"
            )
    print(f"Saved: {low_rmsf_csv}")

    random_csv = OUT / "fig4_adk_random_residues.csv"
    with open(random_csv, "w", encoding="utf-8") as f:
        f.write("rank,pdb_residue_id,atom_index_0based,seed\n")
        for j, idx in enumerate(random_indices, start=1):
            f.write(
                f"{j},{int(residue_ids[idx])},{int(idx)},"
                f"{RANDOM_RESIDUE_SEED}\n"
            )
    print(f"Saved: {random_csv}")

    # Save every cross-grid persistent event and its representative scale.
    events_csv = OUT / "fig4_adk_reorganization_events.csv"
    with open(events_csv, "w", encoding="utf-8") as f:
        f.write(
            "event,T_star,T_left,T_min,T_right,depth,prominence,d_org,"
            "fine_T_min,coarse_T_min\n"
        )
        for j, (Tstar, ev) in enumerate(
            zip(analysis["hierarchy"], analysis["persistent_events"]), start=1
        ):
            f.write(
                f"{j},{int(Tstar)},{ev['T_left']:.8f},{ev['T_min']:.8f},"
                f"{ev['T_right']:.8f},{ev['depth']:.10f},"
                f"{ev['prominence']:.10f},{ev['d_org']:.10f},"
                f"{ev['fine_support']['T_min']:.8f},"
                f"{ev['coarse_support']['T_min']:.8f}\n"
            )
    print(f"Saved: {events_csv}")

    # Save the rank spectrum and both cumulative rank diagnostics.
    rank_csv = OUT / "fig4_adk_rank_spectrum.csv"
    sv = np.asarray(singular_values, dtype=float)
    cum_sigma = np.cumsum(sv) / np.sum(sv)
    cum_energy = np.cumsum(sv * sv) / np.sum(sv * sv)
    with open(rank_csv, "w", encoding="utf-8") as f:
        f.write("index,sigma,cumulative_sigma,cumulative_energy\n")
        for i, (sig, cs, ce) in enumerate(zip(sv, cum_sigma, cum_energy), start=1):
            f.write(f"{i},{sig:.12e},{cs:.12e},{ce:.12e}\n")
    print(f"Saved: {rank_csv}")

    # Quantitative contrast between the two selected event-level traces.
    contrast_csv = OUT / "fig4_adk_scale_contrast.csv"
    with open(contrast_csv, "w", encoding="utf-8") as f:
        f.write("scale_a,scale_b,correlation,zrms_separation\n")
        f.write(f"{T1},{T2},{corr12:.8f},{zrms12:.8f}\n")
    print(f"Saved: {contrast_csv}")

    # Save the fixed representation and selected characteristic scales.
    summary = OUT / "fig4_adk_summary.csv"
    e1, e2 = selected_events[:2]
    with open(summary, "w", encoding="utf-8") as f:
        f.write(
            "system,rank,rank_rule,sigma_sum_rank_diagnostic,"
            "t_scan_min,t_scan_max,T1,T1_depth,T1_prominence,T1_d_org,"
            "T2,T2_depth,T2_prominence,T2_d_org,T_partial,"
            "n_frames,start,end,anchor,selector\n"
        )
        f.write(
            f"ADK,{rank},cumulative_squared_singular_values_0.98,{sigma_rank_diag},"
            f"{analysis['t_min']},{analysis['t_max']},"
            f"{T1},{e1['depth']},{e1['prominence']},{e1['d_org']},"
            f"{T2},{e2['depth']},{e2['prominence']},{e2['d_org']},"
            f"{T_partial},{n_samples},{ANALYSIS_START},{ANALYSIS_END},"
            "closedCA.pdb,shared_three_grid_reorganization\n"
        )
    print(f"Saved: {summary}")

    save_figure(fig, "fig4_adk_no_structure")


if __name__ == "__main__":
    main()
