# -*- coding: utf-8 -*-
"""
01_generate_rossler_raw_data.py

Generate only the raw Rossler-system trajectory and its analytical derivatives.

No Hankel construction, SVD, window selection, closure-defect calculation, or
figure generation is performed in this script.

Outputs
-------
rossler_raw_data.npz
rossler_raw_data.csv
"""

import os
import numpy as np
from scipy.integrate import odeint


# ============================================================
# Rossler system parameters
# ============================================================

a = 0.2
b = 0.2
c = 5.7

initial_state_xyz = [0.0, 1.0, 0.0]

time_start = 0.0
time_end = 100.0
n_time_points = 10000


def rossler_system(state, t, a, b, c):
    """Rossler system."""
    x, y, z = state

    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)

    return [x_dot, y_dot, z_dot]


def main():
    """Generate and save the raw Rossler trajectory."""
    t = np.linspace(time_start, time_end, n_time_points)

    state_trajectory_xyz = odeint(
        rossler_system,
        initial_state_xyz,
        t,
        args=(a, b, c)
    )

    x, y, z = state_trajectory_xyz.T

    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)

    for old_file in ["rossler_raw_data.npz", "rossler_raw_data.csv"]:
        if os.path.exists(old_file):
            os.remove(old_file)

    np.savez(
        "rossler_raw_data.npz",
        t=t,
        state_trajectory_xyz=state_trajectory_xyz,
        solution=state_trajectory_xyz,
        x=x,
        y=y,
        z=z,
        # Renamed keys.
        x_dot=x_dot,
        y_dot=y_dot,
        z_dot=z_dot,

        # Original compatibility keys.
        dx_dt=x_dot,
        dy_dt=y_dot,
        dz_dt=z_dot,
        a=a,
        b=b,
        c=c,
        initial_state=np.asarray(initial_state_xyz, dtype=float),
        t_start=time_start,
        t_end=time_end,
        n_points=n_time_points
    )

    csv_data = np.column_stack((t, x, y, z, x_dot, y_dot, z_dot))
    np.savetxt(
        "rossler_raw_data.csv",
        csv_data,
        delimiter=",",
        header="t,x,y,z,dx_dt,dy_dt,dz_dt",
        comments=""
    )

    print("Raw Rossler data generated.")
    print("Saved: rossler_raw_data.npz")
    print("Saved: rossler_raw_data.csv")
    print(f"a = {a}, b = {b}, c = {c}")
    print(f"initial_state = {initial_state_xyz}")
    print(f"t.shape = {t.shape}")
    print(f"solution.shape = {state_trajectory_xyz.shape}")


if __name__ == "__main__":
    main()
