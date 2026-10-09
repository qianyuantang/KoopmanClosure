"""Generate the raw Lorenz trajectory used by the manuscript analyses.

The script reproduces the original Lorenz data-generation settings and writes
the resulting machine-readable and human-readable files to ``data/``.

No delay embedding, dimensional reduction, closure analysis, scale selection,
or figure generation is performed here.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import odeint


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

LORENZ_A = 10.0
LORENZ_C = 28.0
LORENZ_B = 8.0 / 3.0

INITIAL_STATE = [0.0, 1.0, 0.0]

TIME_START = 0.0
TIME_END = 240.0
N_SAMPLES = 240000

NPZ_NAME = "lorenz_raw_data.npz"
CSV_NAME = "lorenz_raw_data.csv"


def lorenz_rhs(state, t, lorenz_a, lorenz_c, lorenz_b):
    """Lorenz system."""
    x, y, z = state
    return [
        lorenz_a * (y - x),
        x * (lorenz_c - z) - y,
        x * y - lorenz_b * z,
    ]


def generate_trajectory():
    """Integrate the Lorenz system and evaluate its analytical velocity."""
    t = np.linspace(TIME_START, TIME_END, N_SAMPLES)

    solution = odeint(
        lorenz_rhs,
        INITIAL_STATE,
        t,
        args=(LORENZ_A, LORENZ_C, LORENZ_B),
    )

    x, y, z = solution.T

    dx_dt = LORENZ_A * (y - x)
    dy_dt = x * (LORENZ_C - z) - y
    dz_dt = x * y - LORENZ_B * z

    return t, x, y, z, dx_dt, dy_dt, dz_dt


def save_outputs(t, x, y, z, dx_dt, dy_dt, dz_dt):
    """Write the raw trajectory in NPZ and CSV form."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = DATA_DIR / NPZ_NAME
    csv_path = DATA_DIR / CSV_NAME

    np.savez(
        npz_path,
        t=t,
        x=x,
        y=y,
        z=z,
        dx_dt=dx_dt,
        dy_dt=dy_dt,
        dz_dt=dz_dt,
        sigma=LORENZ_A,
        rho=LORENZ_C,
        beta=LORENZ_B,
        initial_state=np.array(INITIAL_STATE, dtype=float),
        time_start=TIME_START,
        time_end=TIME_END,
        n_samples=N_SAMPLES,
        dt=t[1] - t[0],
    )

    csv_data = np.column_stack((t, x, y, z, dx_dt, dy_dt, dz_dt))
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header="t,x,y,z,dx_dt,dy_dt,dz_dt",
        comments="",
    )

    return npz_path, csv_path


def main():
    t, x, y, z, dx_dt, dy_dt, dz_dt = generate_trajectory()
    npz_path, csv_path = save_outputs(t, x, y, z, dx_dt, dy_dt, dz_dt)

    print("Raw Lorenz data generated.")
    print(f"Saved: {npz_path}")
    print(f"Saved: {csv_path}")
    print(f"n_samples = {N_SAMPLES}")
    print(f"dt = {t[1] - t[0]:.15f}")


if __name__ == "__main__":
    main()
