"""Hierarchical two-rule synthetic signal for Fig. 1(c).

This is the multiscale extension of the mechanism sketched in Fig. 1(a).
Only TWO local dynamical rules are used, both on the same circular support and
with the same rotation direction:

    A1 = omega1 J     (slow)
    A2 = omega2 J     (fast)

Fast organization
-----------------
Within every FAST_PERIOD the trajectory switches between A1 and A2, so a short
observation window resolves individual A1/A2 mixing events.

Slow organization
-----------------
Every BLOCK_DURATION the duty fraction of A1 changes: one block spends
DUTY_A1_BLOCK_1 of each fast period in A1, the next DUTY_A1_BLOCK_2.  After
coarse-graining over several fast cycles the two duty patterns act as
different effective rules, although the microscopic rules are still A1 and A2.

Analysis
--------
The observable is the state itself, x = (cos theta, sin theta).  Both rules
rotate about the origin, so the origin is their common fixed point and serves
as the anchor of the local linear closure: no centring is applied.  The rank
follows the shared rule (cumulative singular values >= 98%), which keeps both
coordinates, r = 2.  Closure field, S(T), reorganization events and
characteristic scales all come from the shared closure_pipeline.
"""

import numpy as np

import closure_pipeline as cp

DT = 1e-3
T_TOTAL = 30.0

# Same qualitative rules as panel (a): slow A1 and fast A2, same direction.
BASE_OMEGA = 2.0 * np.pi
OMEGA_1 = 0.40 * BASE_OMEGA   # A1: slow
OMEGA_2 = 1.60 * BASE_OMEGA   # A2: fast

# Hierarchical temporal organization.
FAST_PERIOD = 0.50
BLOCK_DURATION = 6.0
DUTY_A1_BLOCK_1 = 0.75
DUTY_A1_BLOCK_2 = 0.25

# Observation-scale scan in samples.  The shared pipeline masks scales whose
# closure field varies in time by no more than its numerical-resolution floor.
T_SCAN_MIN = 100
MAX_SELECTED_SCALES = 2


def make_signal():
    """Rotation on the unit circle with a hierarchically switched frequency."""
    n = int(round(T_TOTAL / DT))
    t = np.arange(n) * DT

    # Slow blocks alternate the A1 duty fraction within the same fast cycle.
    block_index = np.floor(t / BLOCK_DURATION).astype(int)
    duty_a1 = np.where(block_index % 2 == 0, DUTY_A1_BLOCK_1, DUTY_A1_BLOCK_2)

    fast_phase = np.mod(t, FAST_PERIOD) / FAST_PERIOD
    in_A1 = fast_phase < duty_a1
    omega = np.where(in_A1, OMEGA_1, OMEGA_2)

    theta = np.empty(n)
    theta[0] = 0.0
    theta[1:] = np.cumsum(omega[:-1]) * DT

    X = np.vstack([np.cos(theta), np.sin(theta)])
    Xdot = np.vstack([-np.sin(theta) * omega, np.cos(theta) * omega])

    # Rule labels and boundaries are figure annotations only and do not enter
    # the closure analysis or characteristic-scale selection.
    return dict(
        t=t,
        X=X,
        Xdot=Xdot,
        omega=omega,
        in_A1=in_A1,
        duty_a1=duty_a1,
        fast_cycle_boundaries=np.arange(FAST_PERIOD, T_TOTAL, FAST_PERIOD),
        block_boundaries=np.arange(BLOCK_DURATION, T_TOTAL, BLOCK_DURATION),
    )


def analyse(sig, t_min=T_SCAN_MIN):
    """Shared closure analysis of the synthetic signal."""
    H, Hdot = sig["X"], sig["Xdot"]

    # Coordinates are already measured from the anchor (the rotation centre).
    U, sv, Vh = np.linalg.svd(H, full_matrices=False)
    rank = cp.retained_rank_from_singular_values(sv)
    V = sv[:rank, None] * Vh[:rank]          # U_r^T H
    Vdot = U[:, :rank].T @ Hdot              # exact velocity, same projection

    sums = cp.prepare_window_sums(V, Vdot)
    scale = cp.analyze_timescales(
        sums,
        sig["t"],
        DT,
        t_min=int(t_min),
        t_max=V.shape[1] - 1,
        max_windows=MAX_SELECTED_SCALES,
        progress=True,
    )

    # Event-level beta traces use the same window-centre convention as the map.
    traces = []
    for T in scale["selected"]:
        beta, _ = cp.compute_metrics_for_T(*sums, int(T))
        traces.append((sig["t"][:len(beta)] + 0.5 * int(T) * DT, beta))

    return dict(
        rank=rank,
        Ts=scale["T_values"],                 # baseline log grid for S(T)
        scale_map=scale["scale_map"],
        display_Ts=scale["display_T_values"], # fine log grid for the heat map
        display_scale_map=scale["display_scale_map"],
        common_time=scale["common_time"],
        diag=scale["diag"],
        selected=scale["selected"],
        selected_events=scale["selected_events"],
        persistent_events=scale["persistent_events"],
        t_min=scale["t_min"],
        t_max=scale["t_max"],
        traces=traces,
    )


if __name__ == "__main__":
    res = analyse(make_signal())
    print("Fig. 1 synthetic analysis")
    print("  retained rank:", res["rank"])
    print("  selected T:", res["selected"])
    for j, ev in enumerate(res["persistent_events"], start=1):
        print(
            f"  event {j}: [{ev['T_left']:.1f}, {ev['T_min']:.1f}, {ev['T_right']:.1f}], "
            f"T_half={ev['T_half']:.1f}, T_pre={ev['T_pre_recovery']}, "
            f"depth={ev['depth']:.4f}, prominence={ev['prominence']:.4f}, "
            f"d_org={ev['d_org']:.3f}"
        )
