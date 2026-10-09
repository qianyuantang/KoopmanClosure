"""Generate the raw Rössler trajectory used by Supplementary Fig. S2.

The script reproduces the original Rössler data-generation settings, evaluates
the analytical derivatives, and writes the resulting files to ``data/``.

No delay embedding, dimensional reduction, closure analysis, scale selection,
or figure generation is performed here.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import odeint


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

ROSSLER_A = 0.2
ROSSLER_B = 0.2
ROSSLER_C = 5.7

INITIAL_STATE = [0.0, 1.0, 0.0]

TIME_START = 0.0
TIME_END = 100.0
N_TIME_POINTS = 10000

NPZ_NAME = "rossler_raw_data.npz"
CSV_NAME = "rossler_raw_data.csv"


def rossler_rhs(state, t, a, b, c):
    """Rössler system."""
    x, y, z = state

    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)

    return [x_dot, y_dot, z_dot]


def generate_trajectory():
    """Integrate the Rössler system and evaluate its analytical velocity."""
    t = np.linspace(TIME_START, TIME_END, N_TIME_POINTS)

    solution = odeint(
        rossler_rhs,
        INITIAL_STATE,
        t,
        args=(ROSSLER_A, ROSSLER_B, ROSSLER_C),
    )

    x, y, z = solution.T

    x_dot = -y - z
    y_dot = x + ROSSLER_A * y
    z_dot = ROSSLER_B + z * (x - ROSSLER_C)

    return t, solution, x, y, z, x_dot, y_dot, z_dot


def save_outputs(t, solution, x, y, z, x_dot, y_dot, z_dot):
    """Write the raw trajectory in NPZ and CSV form."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = DATA_DIR / NPZ_NAME
    csv_path = DATA_DIR / CSV_NAME

    np.savez(
        npz_path,
        t=t,
        state_trajectory_xyz=solution,
        solution=solution,
        x=x,
        y=y,
        z=z,
        x_dot=x_dot,
        y_dot=y_dot,
        z_dot=z_dot,
        dx_dt=x_dot,
        dy_dt=y_dot,
        dz_dt=z_dot,
        a=ROSSLER_A,
        b=ROSSLER_B,
        c=ROSSLER_C,
        initial_state=np.asarray(INITIAL_STATE, dtype=float),
        t_start=TIME_START,
        t_end=TIME_END,
        n_points=N_TIME_POINTS,
    )

    csv_data = np.column_stack((t, x, y, z, x_dot, y_dot, z_dot))
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header="t,x,y,z,dx_dt,dy_dt,dz_dt",
        comments="",
    )

    return npz_path, csv_path


def main():
    t, solution, x, y, z, x_dot, y_dot, z_dot = generate_trajectory()
    npz_path, csv_path = save_outputs(
        t, solution, x, y, z, x_dot, y_dot, z_dot
    )

    print("Raw Rössler data generated.")
    print(f"Saved: {npz_path}")
    print(f"Saved: {csv_path}")
    print(
        f"a = {ROSSLER_A}, b = {ROSSLER_B}, c = {ROSSLER_C}"
    )
    print(f"initial_state = {INITIAL_STATE}")
    print(f"t.shape = {t.shape}")
    print(f"solution.shape = {solution.shape}")


if __name__ == "__main__":
    main()
