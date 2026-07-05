# -*- coding: utf-8 -*-
"""
02_process_lorenz_figures_svg_only_v3.py

Read lorenz_raw_data.npz and generate ONLY SVG figures.

This script writes results to a NEW folder:
    lorenz_results_svg_only/

The folder is deleted and recreated at the beginning of every run, so old PNG,
PDF, and NPZ outputs cannot remain in this output directory.
"""

import os
import shutil
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")


RAW_DATA_FILE = "lorenz_raw_data.npz"

# Use a new output directory to avoid mixing with previous PNG/PDF outputs.
OUTPUT_DIR = "lorenz_results"

ANALYSIS_START_INDEX = 220000
ANALYSIS_END_INDEX = 240000
ANALYSIS_SEGMENT_LENGTH = ANALYSIS_END_INDEX - ANALYSIS_START_INDEX

# Original plotting-code setting:
#     embedding_dimension = 20000 - 4
# This produces H.shape = (15, 19996) for stacked x, y, z.
OBSERVABLE_WINDOW_LENGTH = 19996

WINDOW_T_SHORT = 340
WINDOW_T_LONG = 10000

SVD_CUMULATIVE_THRESHOLD = 0.99

PEAK_PROMINENCE_FACTOR = 0.2
PEAK_DISTANCE = 20

KOOPMAN_BETA_THRESHOLD = 1000.0
KOOPMAN_SEARCH_WINDOW_LENGTH = 30
KOOPMAN_MIN_GAP = 0.4

POINCARE_Z_SECTION = 27.0
POINCARE_EPSILON = 0.2
POINCARE_SHIFT_SAMPLES = 220


plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


def reset_output_dir():
    """Delete and recreate the SVG-only output folder."""
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_svg(fig, name):
    """Save one SVG figure only."""
    path = os.path.join(OUTPUT_DIR, f"{name}.svg")
    fig.savefig(path, format="svg", facecolor="white", transparent=False)
    print(f"Saved: {path}")


def paper_axes(ax):
    """Single-axis style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(direction="out", length=3.5, width=1.0, pad=3)


def style_axis(ax1, ax2):
    """Twin-axis style."""
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.0)
    ax1.spines["bottom"].set_linewidth(1.0)
    ax2.spines["right"].set_linewidth(1.0)
    ax1.tick_params(direction="out", length=3.5, width=1.0, pad=3)
    ax2.tick_params(direction="out", length=3.5, width=1.0, pad=3)


def load_raw_data():
    """Load raw Lorenz data."""
    if not os.path.exists(RAW_DATA_FILE):
        raise FileNotFoundError(
            f"{RAW_DATA_FILE} was not found. Run 01_generate_lorenz_raw_data_v3.py first."
        )
    return np.load(RAW_DATA_FILE)


def build_observable_matrix_H(data, embedding_dimension):
    """
    Build Hankel matrix using the original logic.

    For each coordinate:
        n_rows = len(row_data) - embedding_dimension + 1

    With a 20000-sample segment and embedding_dimension = 19995:
        n_rows = 6 per coordinate

    For x, y, z stacked:
        final H.shape = (18, 19995)
    """
    observable_matrix_blocks_H = []

    for i in range(data.shape[0]):
        row_data = data[i, :]
        n_rows = len(row_data) - embedding_dimension + 1

        observable_matrix_block_H = np.zeros((n_rows, embedding_dimension))
        for k in range(n_rows):
            observable_matrix_block_H[k] = row_data[k:k + embedding_dimension]

        observable_matrix_blocks_H.append(observable_matrix_block_H)

    return np.vstack(observable_matrix_blocks_H)


def detect_all_peaks_and_valleys(signal):
    """Detect upper peaks and lower valleys."""
    prom = PEAK_PROMINENCE_FACTOR * (np.max(signal) - np.min(signal))
    peaks_up, _ = find_peaks(signal, prominence=prom, distance=PEAK_DISTANCE)
    peaks_down, _ = find_peaks(-signal, prominence=prom, distance=PEAK_DISTANCE)
    extrema = np.sort(np.concatenate([peaks_up, peaks_down]))

    print(f"Upper peaks: {len(peaks_up)}")
    print(f"Lower valleys: {len(peaks_down)}")
    print(f"All extrema: {len(extrema)}")

    return extrema


def select_window_from_extrema(extrema):
    """Use max peak-valley interval as the intermediate window."""
    if len(extrema) < 2:
        raise ValueError("Not enough extrema to select the window.")

    intervals = np.diff(extrema).astype(int)
    return int(np.max(intervals))


def compute_reduced_coordinates_V(observable_matrix_H, observable_velocity_matrix_H_dot):
    """Apply SVD and obtain reduced coordinates."""
    print("Running SVD...")

    U, s, Vh = np.linalg.svd(observable_matrix_H, full_matrices=False)

    total_sum = np.sum(s)
    cumulative_sum = np.cumsum(s)
    retained_rank_r = int(np.argmax(cumulative_sum >= SVD_CUMULATIVE_THRESHOLD * total_sum) + 1)

    left_singular_vectors_U_r = U[:, :retained_rank_r]
    singular_values_r = s[:retained_rank_r]
    right_singular_vectors_Vh_r = Vh[:retained_rank_r, :]

    reduced_coordinates_V = np.diag(singular_values_r) @ right_singular_vectors_Vh_r
    reduced_velocity_V_dot = left_singular_vectors_U_r.T @ observable_velocity_matrix_H_dot

    print(f"Retained singular values: {retained_rank_r}")
    return reduced_coordinates_V, reduced_velocity_V_dot


def prepare_window_sums_for_A_t_and_R_t(reduced_coordinates_V, reduced_velocity_V_dot):
    """Precompute cumulative matrices for sliding-window fits."""
    p, _ = reduced_coordinates_V.shape

    V_V_transpose_per_time = np.einsum("ik,jk->kij", reduced_coordinates_V, reduced_coordinates_V)
    V_dot_V_transpose_per_time = np.einsum("ik,jk->kij", reduced_velocity_V_dot, reduced_coordinates_V)

    cumulative_V_V_transpose = np.concatenate([np.zeros((1, p, p)), np.cumsum(V_V_transpose_per_time, axis=0)], axis=0)
    cumulative_V_dot_V_transpose = np.concatenate([np.zeros((1, p, p)), np.cumsum(V_dot_V_transpose_per_time, axis=0)], axis=0)
    cumulative_V_dot_squared_norm = np.concatenate([np.zeros((p, 1)), np.cumsum(reduced_velocity_V_dot ** 2, axis=1)], axis=1)

    return cumulative_V_V_transpose, cumulative_V_dot_V_transpose, cumulative_V_dot_squared_norm


def compute_closure_defect_beta_for_window_T(cumulative_V_V_transpose, cumulative_V_dot_V_transpose, cumulative_V_dot_squared_norm, window_length_T):
    """Compute local Koopman closure defect beta for one window length."""
    n = cumulative_V_dot_squared_norm.shape[1] - 1
    closure_defect_beta_values = np.zeros(n - window_length_T)

    for i in range(0, n - window_length_T):
        start, end = i, i + window_length_T

        V_t_V_t_transpose = cumulative_V_V_transpose[end] - cumulative_V_V_transpose[start]
        V_dot_t_V_t_transpose = cumulative_V_dot_V_transpose[end] - cumulative_V_dot_V_transpose[start]
        V_dot_t_squared_norm_sum = cumulative_V_dot_squared_norm[:, end] - cumulative_V_dot_squared_norm[:, start]

        A_t = V_dot_t_V_t_transpose @ np.linalg.pinv(V_t_V_t_transpose)

        R_t_row_squared_norms = (
            V_dot_t_squared_norm_sum
            - 2.0 * np.einsum("ij,ij->i", A_t, V_dot_t_V_t_transpose)
            + np.einsum("ij,jk,ik->i", A_t, V_t_V_t_transpose, A_t)
        )

        closure_defect_beta_values[i] = np.mean(np.sqrt(np.maximum(R_t_row_squared_norms, 0.0)))

    return closure_defect_beta_values


def get_koopman_partition_points(closure_defect_beta_raw, t, x, seg_start, window_length_T):
    """Extract Koopman-based candidate partition points."""
    local_times = []
    local_values = []
    dt = t[1] - t[0]

    for i in range(0, len(closure_defect_beta_raw) - KOOPMAN_SEARCH_WINDOW_LENGTH + 1, KOOPMAN_SEARCH_WINDOW_LENGTH):
        window = closure_defect_beta_raw[i:i + KOOPMAN_SEARCH_WINDOW_LENGTH]
        j = int(np.argmax(window))
        peak_idx = i + j
        peak_value = closure_defect_beta_raw[peak_idx]
        peak_time = t[seg_start + peak_idx] + window_length_T * dt / 2.0

        if peak_value > KOOPMAN_BETA_THRESHOLD:
            if len(local_times) == 0 or peak_time - local_times[-1] >= KOOPMAN_MIN_GAP:
                local_times.append(peak_time)
                local_values.append(peak_value)
            elif peak_value > local_values[-1]:
                local_times[-1] = peak_time
                local_values[-1] = peak_value

    local_times = np.array(local_times)

    if len(local_times) == 0:
        return np.array([]), np.array([])

    point_idx = np.array([np.argmin(np.abs(t - ti)) for ti in local_times], dtype=int)
    return local_times, x[point_idx]


def get_poincare_partition_points(x, y, z, t, seg_start, seg_len):
    """Extract Poincare-section-based reference points."""
    _ = y

    X_seg = x[seg_start:seg_start + seg_len]
    Z_seg = z[seg_start:seg_start + seg_len]

    crossing_idx = np.where(np.abs(Z_seg - POINCARE_Z_SECTION) < POINCARE_EPSILON)[0]
    if len(crossing_idx) < 2:
        return np.array([]), np.array([])

    def symbol_mapping(u):
        if u > 1e-8:
            return "+"
        if u < -1e-8:
            return "-"
        return "0"

    symbols = [symbol_mapping(u) for u in X_seg[crossing_idx]]

    change_points = []
    prev_symbol = symbols[0]

    for i in range(1, len(symbols)):
        curr_symbol = symbols[i]
        if curr_symbol in ["+", "-"] and prev_symbol in ["+", "-"] and curr_symbol != prev_symbol:
            change_points.append(i)
        prev_symbol = curr_symbol

    if len(change_points) == 0:
        return np.array([]), np.array([])

    point_idx = crossing_idx[np.array(change_points)] - POINCARE_SHIFT_SAMPLES
    valid = (point_idx >= 0) & (point_idx < seg_len)
    point_idx = point_idx[valid]

    return t[seg_start + point_idx], X_seg[point_idx]


def plot_window_selection(extrema, selected_window_T):
    """Figure 1: peak-valley interval distribution."""
    intervals = np.diff(extrema).astype(int)
    vals, counts = np.unique(intervals, return_counts=True)

    fig = plt.figure(figsize=(10, 2), dpi=300, facecolor="white")
    ax = fig.add_axes([0.14, 0.38, 0.80, 0.50], facecolor="white")

    ax.bar(vals, counts, width=0.9, edgecolor="black", linewidth=0.6, alpha=0.85, color="black")
    ax.axvline(selected_window_T, linestyle="--", linewidth=1.1, color="black")

    ymax = max(counts.max(), 1)
    ax.set_ylim(0, ymax * 1.20)

    ax.set_xlabel(r"Peak-valley interval $\Delta k$ (samples)", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.tick_params(axis="both", labelcolor="black", labelsize=20)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    paper_axes(ax)
    save_svg(fig, "fig1_window_selection")
    plt.close(fig)


def plot_three_windows(closure_defect_beta_by_T, t, observable_matrix_H, seg_start, window_lengths_T):
    """Figure 2: beta under different window lengths."""
    fig = plt.figure(figsize=(8, 6), dpi=300, facecolor="white")

    left, axis_width, axis_height = 0.105, 0.765, 0.205
    bottoms = [0.755, 0.445, 0.125]
    label_size, tick_size = 20, 20

    for idx, window_length_T in enumerate(window_lengths_T):
        ax1 = fig.add_axes([left, bottoms[idx], axis_width, axis_height], facecolor="white")
        ax2 = ax1.twinx()

        closure_defect_beta_plot = closure_defect_beta_by_T[window_length_T] / 100.0
        t_beta = t[seg_start:seg_start + len(closure_defect_beta_plot)]

        t_x = t[seg_start:seg_start + OBSERVABLE_WINDOW_LENGTH]
        x_trace = observable_matrix_H[1, :]

        ax1.plot(t_beta, closure_defect_beta_plot, linestyle="-", label=r"$\beta$", color="#C42238", alpha=0.8, linewidth=1.2)
        ax2.plot(t_x, x_trace, linestyle="-", label=r"$x$", color="#034A29", alpha=0.8, linewidth=1.2)

        ax1.set_ylabel(r"$\beta$", fontsize=label_size, labelpad=6)
        ax2.set_ylabel(r"$x$", fontsize=label_size, labelpad=8)
        ax1.yaxis.set_label_coords(-0.065, 0.5)
        ax2.yaxis.set_label_coords(1.075, 0.5)

        ax1.tick_params(axis="both", labelcolor="black", labelsize=tick_size)
        ax2.tick_params(axis="both", labelcolor="black", labelsize=tick_size)
        ax2.tick_params(axis="x", bottom=False, labelbottom=False)

        ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))

        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
        ax1.margins(x=0.01, y=0.06)
        ax2.margins(x=0.01, y=0.06)

        if idx < len(window_lengths_T) - 1:
            ax1.tick_params(axis="x", labelbottom=False)
        else:
            ax1.set_xlabel(r"$t$", fontsize=label_size, labelpad=5)
            ax1.xaxis.set_label_coords(0.5, -0.35)

        style_axis(ax1, ax2)

    save_svg(fig, "fig2_multi_window_results")
    plt.close(fig)


def plot_selected_window_comparison(closure_defect_beta_by_T, selected_window_T, t, x, y, z, observable_matrix_H, seg_start):
    """Figure 3: selected-window beta with two sets of partition points."""
    closure_defect_beta_raw = closure_defect_beta_by_T[selected_window_T]
    beta_plot = closure_defect_beta_raw / 100.0

    t_beta = t[seg_start:seg_start + len(beta_plot)]
    t_x = t[seg_start:seg_start + OBSERVABLE_WINDOW_LENGTH]
    x_trace = observable_matrix_H[1, :]

    koopman_t, koopman_x = get_koopman_partition_points(closure_defect_beta_raw, t, x, seg_start, selected_window_T)
    poincare_t, poincare_x = get_poincare_partition_points(x, y, z, t, seg_start, len(x_trace))

    fig = plt.figure(figsize=(10, 2.8), dpi=300, facecolor="white")
    ax1 = fig.add_axes([0.105, 0.32, 0.765, 0.56], facecolor="white")
    ax2 = ax1.twinx()

    line_beta, = ax1.plot(t_beta, beta_plot, linestyle="-", color="#C42238", alpha=0.8, linewidth=1.2, label=r"$\beta$")
    line_x, = ax2.plot(t_x, x_trace, linestyle="-", color="#034A29", alpha=0.8, linewidth=1.2, label=r"$x$")

    handles = [line_x, line_beta]

    if len(poincare_t) > 0:
        poincare_marker = ax2.scatter(
            poincare_t,
            poincare_x,
            color="#6A5ACD",
            marker="o",
            s=38,
            edgecolors="black",
            linewidth=0.6,
            zorder=10,
            label="Poincare-based",
        )
        handles.append(poincare_marker)

    if len(koopman_t) > 0:
        koopman_marker = ax2.scatter(
            koopman_t,
            koopman_x,
            color="#FF8C00",
            marker="o",
            s=38,
            edgecolors="black",
            linewidth=0.6,
            zorder=10,
            label="Koopman-based",
        )
        handles.append(koopman_marker)

    ax1.set_xlabel(r"$t$", fontsize=20, labelpad=8)
    ax1.set_ylabel(r"$\beta$", fontsize=20, labelpad=4)
    ax2.set_ylabel(r"$x$", fontsize=20, labelpad=4)

    ax1.xaxis.set_label_coords(0.5, -0.35)
    ax1.yaxis.set_label_coords(-0.060, 0.5)
    ax2.yaxis.set_label_coords(1.065, 0.5)

    ax1.tick_params(axis="both", labelcolor="black", labelsize=20)
    ax2.tick_params(axis="both", labelcolor="black", labelsize=20)
    ax2.tick_params(axis="x", bottom=False, labelbottom=False)

    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))

    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax1.margins(x=0.01, y=0.06)
    ax2.margins(x=0.01, y=0.06)

    style_axis(ax1, ax2)

    ax1.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=11,
        ncol=4,
        columnspacing=0.8,
        labelspacing=0.4,
        handlelength=1.2,
        handletextpad=0.5,
    )

    save_svg(fig, "fig3_selected_window_comparison")
    plt.close(fig)


def main():
    reset_output_dir()

    raw = load_raw_data()

    t = raw["t"]
    x = raw["x"]
    y = raw["y"]
    z = raw["z"]
    dx_dt = raw["dx_dt"]
    dy_dt = raw["dy_dt"]
    dz_dt = raw["dz_dt"]

    X = x[ANALYSIS_START_INDEX:ANALYSIS_END_INDEX]
    Y = y[ANALYSIS_START_INDEX:ANALYSIS_END_INDEX]
    Z = z[ANALYSIS_START_INDEX:ANALYSIS_END_INDEX]

    dX = dx_dt[ANALYSIS_START_INDEX:ANALYSIS_END_INDEX]
    dY = dy_dt[ANALYSIS_START_INDEX:ANALYSIS_END_INDEX]
    dZ = dz_dt[ANALYSIS_START_INDEX:ANALYSIS_END_INDEX]

    observed_coordinate_matrix = np.vstack((X, Y, Z))
    observed_velocity_matrix = np.vstack((dX, dY, dZ))

    print("Building Hankel matrices...")
    observable_matrix_H = build_observable_matrix_H(observed_coordinate_matrix, OBSERVABLE_WINDOW_LENGTH)
    observable_velocity_matrix_H_dot = build_observable_matrix_H(observed_velocity_matrix, OBSERVABLE_WINDOW_LENGTH)
    print("Hankel shape:", observable_matrix_H.shape)

    extrema = detect_all_peaks_and_valleys(observable_matrix_H[0, :])
    selected_window_T = select_window_from_extrema(extrema)

    window_lengths_T = [WINDOW_T_SHORT, selected_window_T, WINDOW_T_LONG]
    print(f"Selected window T = {selected_window_T}")
    print(f"Windows for Fig. 2 = {window_lengths_T}")

    plot_window_selection(extrema, selected_window_T)

    reduced_coordinates_V, reduced_velocity_V_dot = compute_reduced_coordinates_V(observable_matrix_H, observable_velocity_matrix_H_dot)

    print("Preparing window matrices...")
    cumulative_V_V_transpose, cumulative_V_dot_V_transpose, cumulative_V_dot_squared_norm = prepare_window_sums_for_A_t_and_R_t(reduced_coordinates_V, reduced_velocity_V_dot)

    closure_defect_beta_by_T = {}
    for window_length_T in window_lengths_T:
        print(f"Computing beta for T = {window_length_T} ...")
        closure_defect_beta_by_T[window_length_T] = compute_closure_defect_beta_for_window_T(cumulative_V_V_transpose, cumulative_V_dot_V_transpose, cumulative_V_dot_squared_norm, window_length_T)

    plot_three_windows(closure_defect_beta_by_T, t, observable_matrix_H, ANALYSIS_START_INDEX, window_lengths_T)
    plot_selected_window_comparison(closure_defect_beta_by_T, selected_window_T, t, x, y, z, observable_matrix_H, ANALYSIS_START_INDEX)

    print("Finished. Only the following SVG files were generated:")
    for name in sorted(os.listdir(OUTPUT_DIR)):
        print("  ", os.path.join(OUTPUT_DIR, name))


if __name__ == "__main__":
    main()
