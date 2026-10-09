"""Scale-resolved local closure and characteristic observation scales.

This module is the shared core of Figs. 1-4.  It starts from a fixed reduced
representation V(t) of the observed dynamics and returns the characteristic
observation scales at which the temporal organization of closure failure is
reorganized.  The algorithm is one chain:

    V(t), Vdot(t)  ->  D(t,T)  ->  z_T(t)  ->  S(T)  ->  reorganization events  ->  T*

1. Closure field D(t,T).  Inside every window of T samples a single linear rule
   Vdot = A_{t,T} V is fitted through the origin.  The origin is the reference
   state (anchor) of the representation, fixed by each system adapter before
   the analysis; D is the residual energy per component and per sample.

2. Temporal organization z_T(t).  Dividing by r T already removes the
   sample-count scaling of the residual energy, but the absolute level of
   D(.,T) still depends on the scale and on the representation.  What carries
   dynamical information is *when* closure fails unusually strongly, so D(.,T)
   is robustly standardized scale by scale and only its excess over the
   typical level is kept.  Before this, a scale whose temporal closure
   variation (the range of D over time) lies below a fixed numerical-resolution
   floor relative to the velocity energy is marked unresolved: standardization
   would otherwise turn floating-point noise into an O(1) profile (an exact
   single-rule signal has D at rounding level, yet a nonzero MAD).  Resolved profiles are exactly
   invariant under D -> a D + b (a > 0) applied scale by scale.

3. Reorganization rate S(T).  Neighbouring scales are compared by the Pearson
   correlation of their profiles.  For centred unit vectors 1 - corr equals half
   the squared distance, so 1 - S(T) is a local proxy for how fast the closure
   organization moves along ln T: plateaus of S are ranges of T over which the
   same events dominate closure failure, dips are bands in which that
   organization is replaced.

4. Reorganization events.  A dip of the smoothed S(T) (a complete
   decrease-recovery) is retained only if it is resolved on three nested log-T
   grids, so that it is a property of the closure field rather than of the
   grid on which S is sampled.  No phenomenological amplitude threshold is
   used for retaining reorganization events; the only amplitude criterion is
   the numerical-resolution guard of step 2.  Depth, the enclosure prominence
   and the shoulder contrast d_org are reported.

5. Characteristic scales.  Retained events ordered by T form the hierarchy.
   Each is represented by one scale on its recovery branch, where the new
   organization is established.  The first event is referenced to the
   fine-scale plateau that precedes it (recovery to that level); later events
   are localized from their own geometry (half recovery from the minimum to
   the right shoulder), so that no level depends on how fully the previous one
   recovered.  Localization decides where a retained band is represented, never
   whether it is retained.

T is an observation-window size in samples, not an intrinsic period.  The rank
r is chosen by the adapter and held fixed during the scale scan.  No reference
labels, Poincare sections, structural angles or event times enter the
selection.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# -----------------------------------------------------------------------------
# Shared pipeline constants
# -----------------------------------------------------------------------------
# Default rank convention for adapters that use the shared cumulative-singular-value rule.
# A system adapter may define another fixed representation before entering this pipeline.
RANK_CUMULATIVE_SINGULAR_VALUE = 0.98

# Shared post-selection closure-event detector.  These constants are fixed
# across systems; adapters must not retune them system by system.
CLOSURE_EVENT_PROMINENCE = 0.13
CLOSURE_EVENT_MIN_DISTANCE_FRACTION = 1.0 / 20.0
CLOSURE_EVENT_MERGE_FRACTION = 0.30


# Closure field is sampled on a common time grid of window centres.
COMMON_TIME_POINTS = 1800

# Temporal organization: robust z-score (Gaussian-consistent MAD), positive
# part, capped so that a single extreme window cannot dominate a profile.
MAD_TO_SIGMA = 1.4826
PROFILE_CAP = 20.0

# Numerical resolvability of a closure profile.  A scale is resolved when the
# temporal range of D(.,T) (max - min over the window centres) exceeds this
# fraction of the velocity energy per component and sample (same units as D,
# so the ratio is dimensionless and unit-free).  The range is used, not the
# MAD: when closure failure is sparse (most windows exactly closed, a few
# windows straddling a rule change), the MAD is at rounding level although D
# varies by a large fraction of the velocity energy.  This is a floating-point
# guard, not a dynamical or event threshold.  The floor sits between the
# rounding level of an exactly closed signal (an exact single-rule signal has a
# range below ~1e-12 of the velocity energy) and the smallest closure variation
# of the analysed representations, which is lower for ill-conditioned ranks.
# make_scale_map(progress=True) logs both sides of this gap for every run.
CLOSURE_RESOLUTION_REL = 1e-10

# Reorganization rate: median correlation with +/-1, +/-2 neighbouring scales,
# a minimum common support, and Gaussian smoothing in ln T.
SIMILARITY_NEIGHBOURS = 2
MIN_COMMON_SUPPORT = 20
LOG_GRID_SIZE = 900
LOG_T_SMOOTH_WIDTH = 0.03

# Nested log-T grids: baseline and coarse are exact subsets of the fine grid.
LOG_POINTS_PER_DECADE_FINE = 60
BASELINE_STRIDE = 2        # ~30 points per decade
COARSE_STRIDE = 4          # ~15 points per decade

# Localization of later scales: fraction of recovery from the dip minimum
# towards the right shoulder.
RECOVERY_FRACTION = 0.5

# Shoulder organizations for d_org are taken over +/- three smoothing widths.
SHOULDER_HALF_WIDTH = 3.0 * LOG_T_SMOOTH_WIDTH
SHOULDER_SAMPLES = 7


# =============================================================================
# Representation
# =============================================================================
def retained_rank_from_singular_values(s, fraction=RANK_CUMULATIVE_SINGULAR_VALUE):
    """Smallest r with sum_{i<=r} sigma_i / sum_i sigma_i >= fraction.

    This is the shared rank convention used by the synthetic, Lorenz, KS and
    Rossler adapters.  The input singular values come from an anchor-referenced
    observable matrix; no centring is introduced inside the closure pipeline.
    """
    s = np.asarray(s, dtype=float)
    fraction = float(fraction)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("s must be a non-empty one-dimensional array of singular values.")
    if not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("singular values must be finite and non-negative.")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must lie in (0, 1].")
    total = float(np.sum(s))
    if total <= 0.0:
        raise ValueError("at least one singular value must be positive.")
    return int(np.searchsorted(np.cumsum(s) / total, fraction) + 1)


# =============================================================================
# Closure field D(t,T)
# =============================================================================
def prepare_window_sums(V, Vdot):
    """Prefix sums of V V^T, Vdot V^T and Vdot^2 for sliding-window fits."""
    V = np.asarray(V, dtype=float)
    Vdot = np.asarray(Vdot, dtype=float)
    if V.ndim != 2 or Vdot.ndim != 2 or V.shape != Vdot.shape:
        raise ValueError("V and Vdot must be two-dimensional arrays with identical shape.")
    if V.shape[1] < 2:
        raise ValueError("V and Vdot must contain at least two time samples.")
    r = V.shape[0]
    VV = np.einsum("ik,jk->kij", V, V)
    DV = np.einsum("ik,jk->kij", Vdot, V)
    cVV = np.concatenate([np.zeros((1, r, r)), np.cumsum(VV, axis=0)], axis=0)
    cDV = np.concatenate([np.zeros((1, r, r)), np.cumsum(DV, axis=0)], axis=0)
    cD2 = np.concatenate([np.zeros((r, 1)), np.cumsum(Vdot ** 2, axis=1)], axis=1)
    return cVV, cDV, cD2


def compute_metrics_for_T(cVV, cDV, cD2, T):
    """Closure defect of every window of T samples.

    In each window the through-origin rule A = C G^+ is fitted, with
    G = sum V V^T and C = sum Vdot V^T over the window.  The residual energy of
    row i follows from the window sums without materializing the window:

        SSE_i = E_i - 2 <A_i, C_i> + A_i G A_i^T.

    Returns
    -------
    beta : mean over rows of sqrt(SSE_i)       (root-SSE, used for traces)
    D    : sum_i SSE_i / (r T)                 (per component and sample, used for maps)

    Windows start at 0, ..., n-T-1 (n - T windows, the manuscript convention).
    """
    T = int(T)
    r = cD2.shape[0]
    n = cD2.shape[1] - 1
    n_windows = n - T

    beta = np.empty(n_windows)
    D = np.empty(n_windows)
    for first in range(0, n_windows, 4096):
        starts = np.arange(first, min(first + 4096, n_windows))
        ends = starts + T
        G = cVV[ends] - cVV[starts]
        C = cDV[ends] - cDV[starts]
        E = (cD2[:, ends] - cD2[:, starts]).T
        A = C @ np.linalg.pinv(G)
        row_sse = (
            E
            - 2.0 * np.einsum("bij,bij->bi", A, C)
            + np.einsum("bij,bjk,bik->bi", A, G, A)
        )
        row_sse = np.maximum(row_sse, 0.0)
        beta[starts] = np.mean(np.sqrt(row_sse), axis=1)
        D[starts] = np.sum(row_sse, axis=1) / (r * T)
    return beta, D


def interpolate_centered(values, T, source_time, common_time, dt):
    """Place window values at window centres and interpolate onto common_time.

    The window starting at sample k is placed at source_time[k] + T dt / 2
    (manuscript convention; a common dt/2 offset relative to the geometric
    centre of T samples).  Outside the span of available window centres the
    result is NaN, so that each scale keeps its own temporal support.
    """
    centered_time = source_time[:len(values)] + 0.5 * int(T) * float(dt)
    out = np.full(np.asarray(common_time).shape, np.nan)
    valid = (common_time >= centered_time[0]) & (common_time <= centered_time[-1])
    out[valid] = np.interp(common_time[valid], centered_time, values)
    return out


def velocity_energy_per_component(sums):
    """Mean squared velocity per retained component and per sample."""
    cD2 = np.asarray(sums[2], dtype=float)
    r, n = cD2.shape[0], cD2.shape[1] - 1
    return float(np.sum(cD2[:, -1]) / (r * n))


def temporal_range(row):
    """max - min of a closure row on its finite support (NaN if empty)."""
    row = np.asarray(row, dtype=float)
    valid = row[np.isfinite(row)]
    if valid.size == 0:
        return np.nan
    return float(np.max(valid) - np.min(valid))


def make_scale_map(sums, Ts, source_time, dt,
                   common_time_points=COMMON_TIME_POINTS, progress=False):
    """D(t,T) for a set of window sizes, on one common grid of window centres.

    A row whose temporal range is positive but not above
    CLOSURE_RESOLUTION_REL times the velocity energy is numerical noise and is
    left as NaN.  Rows with exactly zero range are kept as they are:
    normalized_profile already marks them invalid, and leaving them untouched
    keeps the map identical to the unguarded one whenever no row is masked.
    """
    Ts = np.asarray(Ts, dtype=int)
    common_time = np.linspace(source_time[0], source_time[-1], int(common_time_points))
    scale_map = np.full((len(Ts), len(common_time)), np.nan)
    energy = velocity_energy_per_component(sums)
    floor = CLOSURE_RESOLUTION_REL * energy
    eps = np.full(len(Ts), np.nan)        # range / velocity energy, for the log
    masked = np.zeros(len(Ts), dtype=bool)
    for i, T in enumerate(Ts):
        _, D = compute_metrics_for_T(*sums, int(T))
        row = interpolate_centered(D, int(T), source_time, common_time, dt)
        spread = temporal_range(row)
        eps[i] = spread / energy
        masked[i] = 0.0 < spread <= floor
        if not masked[i]:
            scale_map[i] = row
        if progress and ((i + 1) % 30 == 0 or i + 1 == len(Ts)):
            print(f"  closure scale map: {i + 1}/{len(Ts)}")
    if progress:
        positive = eps[np.isfinite(eps) & (eps > 0.0)]
        resolved = positive[positive > CLOSURE_RESOLUTION_REL]
        line = (f"  numerical floor: {int(masked.sum())}/{len(Ts)} scales masked; "
                f"smallest resolved range/velocity energy = "
                f"{resolved.min() if resolved.size else np.nan:.3g}")
        if masked.any():
            line += f"; largest masked = {eps[masked].max():.3g}"
        print(line)
    return common_time, scale_map


def globally_scaled_map(scale_map):
    """Display scaling of D(t,T): one global factor, 99% quantile -> 4."""
    q99 = float(np.nanquantile(scale_map, 0.99))
    return 4.0 * scale_map / max(q99, 1e-30)


# =============================================================================
# Temporal organization z_T(t)
# =============================================================================
def normalized_profile(row):
    """z_T = min([(D - med D) / (1.4826 MAD D)]_+, 20) on the scale's own support.

    The closure map is first interpolated onto the common time grid; the
    median and MAD are then taken over the finite entries of this row.  A row
    whose MAD vanishes carries no temporal organization and is marked invalid
    (all NaN), which keeps the map exactly invariant under D -> a D + b, a > 0.
    """
    row = np.asarray(row, dtype=float)
    valid = row[np.isfinite(row)]
    if valid.size == 0:
        return np.full_like(row, np.nan)
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    if not mad > 0.0:
        return np.full_like(row, np.nan)
    z = (row - med) / (MAD_TO_SIGMA * mad)
    return np.clip(z, 0.0, PROFILE_CAP)


def profile_correlation(a, b):
    """Pearson correlation of two profiles on their common finite support.

    Undefined (NaN) when the common support is shorter than
    MIN_COMMON_SUPPORT samples or either profile is constant on it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    common = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(common) < MIN_COMMON_SUPPORT:
        return np.nan
    x = a[common] - np.mean(a[common])
    y = b[common] - np.mean(b[common])
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= 1e-14:
        return np.nan
    return float(np.dot(x, y) / den)


# =============================================================================
# Reorganization rate S(T)
# =============================================================================
def smooth_on_log_grid(Ts, raw):
    """Interpolate raw S onto a uniform ln T grid and smooth it there.

    S(T) is defined on one contiguous range of scales.  Scales at the ends of
    the scan may be undefined (at the largest T too few windows remain for a
    common support); such ends are simply outside the range.  An undefined
    scale inside the range would have to be bridged by interpolation, so it
    is treated as an error rather than filled in.  Outside the range the
    smoothed S is NaN, not extrapolated.
    """
    Ts = np.asarray(Ts, dtype=float)
    raw = np.asarray(raw, dtype=float)
    finite = np.flatnonzero(np.isfinite(raw))
    if finite.size < 3:
        raise RuntimeError("Too few scales have valid neighbouring-scale correlations.")
    first, last = finite[0], finite[-1]
    if finite.size != last - first + 1:
        raise RuntimeError("S(T) is undefined at a scale inside the scanned range.")

    log_grid = np.linspace(np.log(Ts[first]), np.log(Ts[last]), LOG_GRID_SIZE)
    interp = np.interp(log_grid, np.log(Ts[first:last + 1]), raw[first:last + 1])
    sigma_grid = LOG_T_SMOOTH_WIDTH / float(log_grid[1] - log_grid[0])
    smooth = gaussian_filter1d(interp, sigma_grid, mode="nearest")

    smooth_actual = np.full(len(Ts), np.nan)
    smooth_actual[first:last + 1] = np.interp(np.log(Ts[first:last + 1]), log_grid, smooth)
    return {
        "raw": raw,
        "log_grid": log_grid,
        "smooth": smooth,
        "smooth_actual": smooth_actual,
    }


def compute_similarity(Ts, scale_map):
    """S(T): median correlation of z_T with its +/-1, +/-2 neighbours on this grid."""
    Ts = np.asarray(Ts, dtype=int)
    profiles = np.asarray([normalized_profile(row) for row in scale_map])
    raw = np.full(len(Ts), np.nan)
    for i in range(len(Ts)):
        neighbours = [j for off in range(1, SIMILARITY_NEIGHBOURS + 1)
                      for j in (i - off, i + off) if 0 <= j < len(Ts)]
        corr = [profile_correlation(profiles[i], profiles[j]) for j in neighbours]
        corr = [c for c in corr if np.isfinite(c)]
        if corr:
            raw[i] = np.median(corr)
    diag = smooth_on_log_grid(Ts, raw)
    diag["normalized"] = profiles
    return diag


# =============================================================================
# Nested log-T grids
# =============================================================================
def log_integer_grid(t_min, t_max, points_per_decade=LOG_POINTS_PER_DECADE_FINE):
    """Approximately uniform integer grid in log T, including both endpoints."""
    t_min, t_max = int(t_min), int(t_max)
    n = int(np.ceil(np.log10(t_max / t_min) * float(points_per_decade))) + 1
    vals = np.unique(np.rint(np.exp(np.linspace(np.log(t_min), np.log(t_max), n))).astype(int))
    vals = vals[(vals >= t_min) & (vals <= t_max)]
    return np.unique(np.r_[t_min, vals, t_max]).astype(int)


def nested_subgrid(fine, stride):
    """Every stride-th entry of the fine grid, always keeping the last one."""
    fine = np.asarray(fine, dtype=int)
    idx = np.arange(0, len(fine), int(stride))
    if idx[-1] != len(fine) - 1:
        idx = np.r_[idx, len(fine) - 1]
    return fine[idx]


def make_nested_log_grids(t_min, t_max):
    fine = log_integer_grid(t_min, t_max)
    return {
        "fine": fine,
        "baseline": nested_subgrid(fine, BASELINE_STRIDE),
        "coarse": nested_subgrid(fine, COARSE_STRIDE),
    }


# =============================================================================
# Reorganization events
# =============================================================================
def _interp_pattern(log_target, logTs, Z):
    """Normalized profile at an arbitrary ln T, by linear interpolation in ln T."""
    if log_target <= logTs[0]:
        return Z[0].copy()
    if log_target >= logTs[-1]:
        return Z[-1].copy()
    j = int(np.searchsorted(logTs, log_target))
    w = (log_target - logTs[j - 1]) / (logTs[j] - logTs[j - 1])
    a, b = Z[j - 1], Z[j]
    out = np.full_like(a, np.nan, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    out[both] = (1.0 - w) * a[both] + w * b[both]
    out[np.isfinite(a) & ~np.isfinite(b)] = a[np.isfinite(a) & ~np.isfinite(b)]
    out[~np.isfinite(a) & np.isfinite(b)] = b[~np.isfinite(a) & np.isfinite(b)]
    return out


def _shoulder_organization(log_center, logTs, Z):
    """Representative organization of a plateau: median profile over its shoulder."""
    q = np.linspace(log_center - SHOULDER_HALF_WIDTH,
                    log_center + SHOULDER_HALF_WIDTH, SHOULDER_SAMPLES)
    patterns = np.asarray([_interp_pattern(float(x), logTs, Z) for x in q])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(patterns, axis=0)


def _half_recovery_T(logg, S, i_min, i_right):
    """First T on the recovery branch where S regains RECOVERY_FRACTION of the dip."""
    target = S[i_min] + RECOVERY_FRACTION * (S[i_right] - S[i_min])
    branch = np.arange(i_min + 1, i_right + 1)
    j = int(branch[S[branch] >= target][0])
    j0 = j - 1
    if S[j] == S[j0]:
        return float(np.exp(logg[j]))
    w = float(np.clip((target - S[j0]) / (S[j] - S[j0]), 0.0, 1.0))
    return float(np.exp(logg[j0] + w * (logg[j] - logg[j0])))


def _pre_dip_recovery_T(Ts, S_actual, T_min, T_right, S_left):
    """Scanned T on the recovery branch whose S is closest to the pre-dip level."""
    idx = np.where((Ts > T_min) & (Ts <= T_right))[0]
    if len(idx) == 0:
        return int(np.rint(T_right))
    return int(Ts[idx[np.argmin(np.abs(S_actual[idx] - S_left))]])


def enclosure_prominence(S, i_min):
    """Height of the lower of the two barriers enclosing the dip at i_min.

    From the minimum, walk outwards on each side until S first drops below
    S(i_min) or the scan boundary is reached; the barrier is the maximum met on
    the way.  Unlike the nearest-shoulder depth, this ignores micro-maxima
    inside the basin.  For an interior minimum with deeper minima on both sides
    it equals the one-dimensional H0 persistence; a boundary-truncated basin
    gets the finite barrier available before the boundary.

    If ||S' - S||_inf <= eta < prominence / 2, S' has a local minimum between
    the same barriers with prominence >= prominence - 2 eta, lying where
    S <= S(i_min) + 2 eta.
    """
    s0 = S[i_min]
    barriers = []
    for side in (S[i_min::-1], S[i_min:]):
        lower = np.nonzero(side < s0)[0]
        stop = lower[0] if lower.size else len(side)
        barriers.append(float(np.max(side[:stop])))
    return min(barriers) - float(s0)


def find_scale_events(Ts, diag):
    """All complete decrease-recovery dips of the smoothed S(T) on one grid."""
    Ts = np.asarray(Ts, dtype=int)
    logg = np.asarray(diag["log_grid"], dtype=float)
    S = np.asarray(diag["smooth"], dtype=float)
    Z = np.asarray(diag["normalized"], dtype=float)
    logTs = np.log(Ts.astype(float))

    minima, _ = find_peaks(-S)
    maxima, _ = find_peaks(S)
    events = []
    for im in minima:
        left, right = maxima[maxima < im], maxima[maxima > im]
        if len(left) == 0 or len(right) == 0:
            continue
        il, ir, im = int(left[-1]), int(right[0]), int(im)

        # Net change of organization across the dip (reported, not used).
        zL = _shoulder_organization(float(logg[il]), logTs, Z)
        zR = _shoulder_organization(float(logg[ir]), logTs, Z)
        c = profile_correlation(zL, zR)

        T_min, T_right = float(np.exp(logg[im])), float(np.exp(logg[ir]))
        events.append({
            "T_left": float(np.exp(logg[il])),
            "T_min": T_min,
            "T_right": T_right,
            "T_half": _half_recovery_T(logg, S, im, ir),
            "T_pre_recovery": _pre_dip_recovery_T(
                Ts, diag["smooth_actual"], T_min, T_right, S[il]),
            "S_left": float(S[il]),
            "S_min": float(S[im]),
            "S_right": float(S[ir]),
            "depth": float(min(S[il], S[ir]) - S[im]),
            "prominence": enclosure_prominence(S, im),
            "d_org": float(1.0 - c) if np.isfinite(c) else np.nan,
            "i_left": il,
            "i_min": im,
            "i_right": ir,
        })
    return events


def _matching_dip(events, ev):
    """Dip on another grid that resolves the same reorganization as `ev`.

    Two dips match when each minimum lies inside the other's
    [T_left, T_right]; the reverse condition is what requires the other grid
    to have a dip there at all, since neighbouring intervals tile the range.
    Among several candidates the one closest in ln T_min is taken.
    """
    T = float(ev["T_min"])
    hits = [e for e in events
            if e["T_left"] <= T <= e["T_right"]
            and ev["T_left"] <= e["T_min"] <= ev["T_right"]]
    if not hits:
        return None
    return min(hits, key=lambda e: abs(np.log(e["T_min"]) - np.log(T)))


def persistent_baseline_events(events_fine, events_baseline, events_coarse):
    """Baseline dips that are matched on both the fine and the coarse grid."""
    persistent = []
    for b in events_baseline:
        f = _matching_dip(events_fine, b)
        c = _matching_dip(events_coarse, b)
        if f is not None and c is not None:
            persistent.append({**b, "fine_support": f, "coarse_support": c})
    return persistent



def detect_closure_events(beta, source_time, T, dt):
    """Detect closure-event peaks with one dimensionless rule for all systems.

    The signal is robustly normalized by its median and 95th percentile.
    Local maxima are then retained with a fixed normalized prominence and a
    minimum separation proportional to the selected window length ``T``.
    Nearby peaks that belong to the same closure episode are merged using a
    second scale-free rule: if consecutive candidates are separated by less
    than ``CLOSURE_EVENT_MERGE_FRACTION * T``, only the stronger of the two is
    kept. Event times are reported at window centres, so the ``T/2``
    correction is applied exactly once here.

    Parameters
    ----------
    beta : array_like
        Closure-defect trace indexed by window start.
    source_time : array_like
        Time attached to the window-start samples of ``beta``.
    T : int
        Selected window size in samples.
    dt : float
        Sampling interval.

    Returns
    -------
    peak_idx : ndarray of int
        Indices of retained peaks in ``beta``.
    event_time : ndarray of float
        Midpoint-corrected event times.
    beta_normalized : ndarray of float
        Robustly normalized trace used by the detector.
    """
    beta = np.asarray(beta, dtype=float)
    source_time = np.asarray(source_time, dtype=float)
    T = int(T)
    dt = float(dt)

    if beta.ndim != 1:
        raise ValueError("beta must be one-dimensional.")
    if source_time.ndim != 1 or len(source_time) < len(beta):
        raise ValueError("source_time must be one-dimensional and cover beta.")
    if T < 1 or dt <= 0.0:
        raise ValueError("T must be positive and dt must be > 0.")

    med = float(np.nanmedian(beta))
    q95 = float(np.nanquantile(beta, 0.95))
    scale = max(q95 - med, 1e-12)
    beta_normalized = (beta - med) / scale

    min_distance = max(
        1,
        int(round(CLOSURE_EVENT_MIN_DISTANCE_FRACTION * T)),
    )
    peak_idx, _ = find_peaks(
        beta_normalized,
        prominence=CLOSURE_EVENT_PROMINENCE,
        distance=min_distance,
    )

    # Peaks below the median background are not closure events.
    peak_idx = peak_idx[beta_normalized[peak_idx] > 0.0]

    # Merge neighbouring peaks that belong to the same closure episode.
    merge_gap = max(
        1,
        int(round(CLOSURE_EVENT_MERGE_FRACTION * T)),
    )
    if len(peak_idx) > 1:
        merged = [int(peak_idx[0])]
        for idx in map(int, peak_idx[1:]):
            prev = merged[-1]
            if idx - prev < merge_gap:
                if beta_normalized[idx] > beta_normalized[prev]:
                    merged[-1] = idx
            else:
                merged.append(idx)
        peak_idx = np.asarray(merged, dtype=int)
    else:
        peak_idx = np.asarray(peak_idx, dtype=int)

    event_time = (
        source_time[:len(beta)][peak_idx]
        + 0.5 * T * dt
    )
    return peak_idx, event_time, beta_normalized


# =============================================================================
# Characteristic scales
# =============================================================================
def characteristic_scales(persistent_events):
    """One scale per retained event, in order of increasing T.

    The first event is located where S recovers to its pre-dip plateau; every
    later event at the half-recovery point of its own dip.
    """
    if not persistent_events:
        return []
    scales = [int(persistent_events[0]["T_pre_recovery"])]
    scales += [int(np.rint(ev["T_half"])) for ev in persistent_events[1:]]
    return scales


def analyze_timescales(sums, source_time, dt, t_min, t_max, max_windows=2, progress=False):
    """Full analysis from window sums to characteristic scales.

    Returns a dictionary with
        selected            first max_windows characteristic scales [T1, T2, ...]
        selected_events     the events they represent
        hierarchy           all characteristic scales (one per retained event)
        persistent_events   all retained events with their attributes
        T_values, scale_map, diag            baseline grid (S(T) panel)
        display_T_values, display_scale_map  fine grid (heat map)
        common_time         time grid shared by the maps
        grids               fine / baseline / coarse maps, S(T) and dips
    """
    r = sums[2].shape[0]
    n = sums[2].shape[1] - 1
    source_time = np.asarray(source_time, dtype=float)
    if source_time.ndim != 1 or len(source_time) != n:
        raise ValueError("source_time must be one-dimensional and match the reduced trajectory length.")
    if not np.all(np.diff(source_time) > 0.0):
        raise ValueError("source_time must be strictly increasing.")
    if int(max_windows) < 1:
        raise ValueError("max_windows must be at least 1.")
    t_min, t_max = int(t_min), min(int(t_max), n - 1)
    if t_min <= r:
        # Through-origin least squares can interpolate windows with T <= r.
        raise ValueError(f"t_min={t_min} must exceed the rank r={r}.")

    Ts_dict = make_nested_log_grids(t_min, t_max)
    common_time, fine_map = make_scale_map(
        sums, Ts_dict["fine"], source_time, dt, progress=progress)
    if not any(temporal_range(row) > 0.0 for row in fine_map):
        raise RuntimeError(
            "No numerically resolved temporal closure organization was found "
            "at any scanned scale.")

    grids = {}
    for key in ("fine", "baseline", "coarse"):
        Ts = Ts_dict[key]
        lookup = {int(T): i for i, T in enumerate(Ts_dict["fine"])}
        smap = fine_map[[lookup[int(T)] for T in Ts]]
        diag = compute_similarity(Ts, smap)
        grids[key] = {
            "T_values": Ts,
            "scale_map": smap,
            "diag": diag,
            "events": find_scale_events(Ts, diag),
        }

    persistent = persistent_baseline_events(
        grids["fine"]["events"], grids["baseline"]["events"], grids["coarse"]["events"])
    if not persistent:
        raise RuntimeError("No cross-resolution persistent reorganization event was found.")
    hierarchy = characteristic_scales(persistent)

    baseline = grids["baseline"]
    return {
        "selected": hierarchy[:int(max_windows)],
        "selected_events": persistent[:int(max_windows)],
        "hierarchy": hierarchy,
        "persistent_events": persistent,
        "T_values": baseline["T_values"],
        "scale_map": baseline["scale_map"],
        "diag": baseline["diag"],
        "display_T_values": grids["fine"]["T_values"],
        "display_scale_map": grids["fine"]["scale_map"],
        "common_time": common_time,
        "grids": grids,
        "t_min": t_min,
        "t_max": t_max,
    }
