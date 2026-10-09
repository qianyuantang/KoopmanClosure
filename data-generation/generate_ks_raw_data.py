"""Generate the raw Kuramoto--Sivashinsky modal trajectory used by Fig. 3.

The script reproduces the original 16-mode KS data-generation settings and
writes the resulting machine-readable and human-readable files to ``data/``.

No dimensional reduction, closure analysis, scale selection, or figure
generation is performed here.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

N_MODES = 16
NU = 0.028509

DELTA_T = 0.01
TIME_START = 0.0
TIME_END = 200.0

# Preserved exactly from the original data-generation script.
N_TIME_STEPS = int((TIME_END - TIME_START) / DELTA_T)

INITIAL_STATE = np.zeros(N_MODES, dtype=np.float64)
INITIAL_STATE[0] = 0.0
INITIAL_STATE[1:] = 1.7

NPZ_NAME = "ks_raw_data.npz"
CSV_NAME = "ks_raw_data.csv"


def make_ks_rhs(n_modes, nu):
    """Return the truncated antisymmetric Fourier-mode KS system."""

    def ks_rhs(t, a):
        _ = t
        a_dot = np.zeros(n_modes, dtype=np.float64)

        for k in range(1, n_modes + 1):
            nonlinear_term = 0.0

            for m in range(-n_modes, n_modes + 1):
                if m > 0:
                    a_m = a[m - 1]
                elif m < 0:
                    a_m = -a[-m - 1]
                else:
                    a_m = 0.0

                if 0 < k - m < n_modes:
                    a_km = a[k - m - 1]
                elif -n_modes < k - m < 0:
                    a_km = -a[m - k - 1]
                else:
                    a_km = 0.0

                nonlinear_term += a_m * a_km

            a_dot[k - 1] = (
                (k ** 2 - nu * k ** 4) * a[k - 1]
                - k * nonlinear_term
            )

        return a_dot

    return ks_rhs


def generate_trajectory():
    """Integrate the truncated KS modal system."""
    t = np.linspace(TIME_START, TIME_END, N_TIME_STEPS)
    ks_rhs = make_ks_rhs(N_MODES, NU)

    solution = solve_ivp(
        fun=ks_rhs,
        t_span=(TIME_START, TIME_END),
        y0=INITIAL_STATE,
        t_eval=t,
        max_step=DELTA_T,
    )

    if not solution.success:
        raise RuntimeError(f"KS integration failed: {solution.message}")

    return t, solution.y


def save_outputs(t, a):
    """Write the raw modal trajectory in NPZ and CSV form."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = DATA_DIR / NPZ_NAME
    csv_path = DATA_DIR / CSV_NAME

    np.savez(
        npz_path,
        t=t,
        a=a,
        n_modes=N_MODES,
        mu=NU,
        dt=DELTA_T,
        t_start=TIME_START,
        t_end=TIME_END,
        n_steps=N_TIME_STEPS,
        initial_state=INITIAL_STATE,
    )

    csv_data = np.column_stack([t, a.T])
    header = ",".join(["t"] + [f"a{i}" for i in range(1, N_MODES + 1)])
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=header,
        comments="",
    )

    return npz_path, csv_path


def main():
    t, a = generate_trajectory()
    npz_path, csv_path = save_outputs(t, a)

    print("Raw KS data generated.")
    print(f"Saved: {npz_path}")
    print(f"Saved: {csv_path}")
    print(f"a.shape = {a.shape}")
    print(f"t.shape = {t.shape}")
    print(f"nominal dt = {DELTA_T}")
    print(f"actual grid step = {t[1] - t[0]:.15f}")


if __name__ == "__main__":
    main()
