# -*- coding: utf-8 -*-
"""
01_generate_ks_raw_data.py

Generate ONLY the raw Kuramoto--Sivashinsky (KS) modal trajectory.

This script matches the data-generation part of the original KS code:
    N = 16
    nu = 0.028509
    dt = 0.01
    t_start = 0.0
    t_end = 200
    a0[0] = 0
    a0[1:] = 1.7
    t = np.linspace(t_start, t_end, n_steps)
    solve_ivp(..., max_step=dt)

No window selection, SVD, closure-defect calculation, or plotting is performed
in this file. Those steps are performed by:
    02_process_ks_figures_bcd.py

Outputs
-------
ks_raw_data.npz
    Raw modal trajectory a(t), time array t, and simulation parameters.

ks_raw_data.csv
    Human-readable CSV table with columns:
    t, a1, a2, ..., a16.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp


# ============================================================
# KS parameters from the original code / manuscript setting
# ============================================================

Fourier_modes = 16
nu = 0.028509

Delta_t = 0.01
TIME_START = 0.0
TIME_END = 200.0

N_TIME_STEPS = int((TIME_END - TIME_START) / Delta_t)

initial_state_a = np.zeros(Fourier_modes, dtype=np.float64)
initial_state_a[0] = 0.0
initial_state_a[1:] = 1.7


def make_ks_evolution_equations(Fourier_modes, nu):
    """
    Return the truncated Fourier-mode ODE system for the KS equation.

    The modal equations follow the same antisymmetric Fourier-subspace
    implementation as the original code.
    """
    def evolution_equations(t, a):
        _ = t
        a_dot = np.zeros(Fourier_modes, dtype=np.float64)

        for k in range(1, Fourier_modes + 1):
            nonlinear_term = 0.0

            for m in range(-Fourier_modes, Fourier_modes + 1):
                if m > 0:
                    a_m = a[m - 1]
                elif m < 0:
                    a_m = -a[-m - 1]
                else:
                    a_m = 0.0

                if 0 < k - m < Fourier_modes:
                    a_km = a[k - m - 1]
                elif -Fourier_modes < k - m < 0:
                    a_km = -a[m - k - 1]
                else:
                    a_km = 0.0

                nonlinear_term += a_m * a_km

            a_dot[k - 1] = (k ** 2 - nu * k ** 4) * a[k - 1] - k * nonlinear_term

        return a_dot

    return evolution_equations


def main():
    """Generate and save the raw KS modal trajectory."""
    t = np.linspace(TIME_START, TIME_END, N_TIME_STEPS)

    evolution_equations = make_ks_evolution_equations(Fourier_modes, nu)

    solution = solve_ivp(
        fun=evolution_equations,
        t_span=(TIME_START, TIME_END),
        y0=initial_state_a,
        t_eval=t,
        max_step=Delta_t
    )

    if not solution.success:
        raise RuntimeError(f"KS integration failed: {solution.message}")

    a = solution.y

    for old_file in ["ks_raw_data.npz", "ks_raw_data.csv"]:
        if os.path.exists(old_file):
            os.remove(old_file)

    np.savez(
        "ks_raw_data.npz",
        t=t,
        a=a,
        n_modes=Fourier_modes,
        mu=nu,
        dt=Delta_t,
        t_start=TIME_START,
        t_end=TIME_END,
        n_steps=N_TIME_STEPS,
        initial_state=initial_state_a
    )

    csv_data = np.column_stack([t, a.T])
    header = ",".join(["t"] + [f"a{i}" for i in range(1, Fourier_modes + 1)])
    np.savetxt(
        "ks_raw_data.csv",
        csv_data,
        delimiter=",",
        header=header,
        comments=""
    )

    print("Raw KS data generated.")
    print("Saved: ks_raw_data.npz")
    print("Saved: ks_raw_data.csv")
    print(f"a.shape = {a.shape}")
    print(f"t.shape = {t.shape}")
    print(f"nominal dt = {Delta_t}")
    print(f"actual grid step = {t[1] - t[0]:.15f}")


if __name__ == "__main__":
    main()
