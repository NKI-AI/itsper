from io import BytesIO

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from itsper.io import get_logger

logger = get_logger(__name__)


def calculate_qar(vs, vt, d_s, d_t) -> tuple[float, float]:
    epsilon_over_stroma = (2 * vs) / d_s - 2 * vs
    epsilon_under_stroma = (2 * vs * (1 - d_s)) / (2 - d_s)
    epsilon_over_tumor = (2 * vt) / d_t - 2 * vt
    epsilon_under_tumor = (2 * vt * (1 - d_t)) / (2 - d_t)
    itsp_high = (vs + epsilon_over_stroma) / ((vs + epsilon_over_stroma) + (vt - epsilon_under_tumor))
    itsp_low = (vs - epsilon_under_stroma) / ((vs - epsilon_under_stroma) + (vt + epsilon_over_tumor))
    return itsp_high, itsp_low


def plot_random_colored_circle(ax, tumor_volume, stroma_volume, resolution=500, cluster_size=10):
    total_volume = tumor_volume + stroma_volume
    if total_volume == 0:
        raise ValueError("Tumor and stroma volumes must sum to a positive value.")

    stroma_ratio = stroma_volume / total_volume

    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xv, yv = np.meshgrid(x, y)

    mask = xv ** 2 + yv ** 2 <= 1 ** 2

    noise = np.random.rand(resolution, resolution)
    smoothed_noise = gaussian_filter(noise, sigma=cluster_size)

    threshold = np.percentile(smoothed_noise[mask], (1 - stroma_ratio) * 100)
    stroma_region = (smoothed_noise > threshold) & mask
    tumor_region = (smoothed_noise <= threshold) & mask

    colors = np.ones((resolution, resolution, 3))
    colors[tumor_region] = [1, 0, 0]  # Red for tumor
    colors[stroma_region] = [0, 1, 0]  # Green for stroma

    ax.imshow(colors, extent=(-1, 1, -1, 1))
    ax.text(0.35, 0.95, f"Stroma: {stroma_ratio*100:.2f}%", fontsize=10, transform=ax.transAxes, color='black')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", "box")
    ax.set_title("Varying tumor and stroma volumes")
    ax.axis("off")

    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Tumor'),
        Patch(facecolor='green', edgecolor='black', label='Stroma')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False)


def create_frames(num_frames, Vs_values, Vt_values, QAR_grid, dice_stroma, dice_tumor):
    tumor_sizes = np.linspace(1, 0.1, num_frames)
    stroma_sizes = np.linspace(0.1, 1, num_frames)
    frames = []
    trajectory_points = []

    for i in range(num_frames):
        fig = plt.figure(figsize=(12, 10))
        # Add title
        fig.suptitle("Quantification ambiguity range (QAR) as a function of tumor and stroma volumes", fontsize=16)

        # Plot 2D randomly colored circle with clusters
        ax_2d = fig.add_subplot(121)
        plot_random_colored_circle(ax_2d, tumor_sizes[i], stroma_sizes[i], cluster_size=10)

        # Plot 3D QAR surface
        ax_3d = fig.add_subplot(122, projection="3d")
        current_vs = stroma_sizes[i] * 1000
        current_vt = tumor_sizes[i] * 1000
        high, low = calculate_qar(current_vs, current_vt, dice_stroma, dice_tumor)
        current_qar = (high - low) * 100
        plot_qar_surface(ax_3d, Vs_values, Vt_values, QAR_grid, dice_stroma, dice_tumor, current_vs, current_vt, current_qar, trajectory_points)

        # Save the frame to a buffer instead of a file
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=80, pad_inches=0.0003)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))
        buffer.close()
        plt.close(fig)

    return frames


def qar_surface_plotter(dice_stroma, dice_tumor, output_path):
    logger.info("QAR surface plotting initiated...")
    logger.info(f"Dice score for stroma: {dice_stroma}, Dice score for tumor: {dice_tumor}")
    num_frames = 50
    vs_values = np.linspace(100, 1000, 50)
    vt_values = np.linspace(100, 1000, 50)
    vs_grid, vt_grid = np.meshgrid(vs_values, vt_values)
    high, low = calculate_qar(vs_grid, vt_grid, dice_stroma, dice_tumor)

    frames = create_frames(num_frames, vs_grid, vt_grid, (high - low)*100, dice_stroma, dice_tumor)
    gif_path = f"{output_path}/qar_variation.gif"
    imageio.mimsave(gif_path, frames, duration=600, loop=0)
    logger.info(f"QAR surface plotting completed. GIF saved at {gif_path}")
    return gif_path


def plot_qar_surface(ax, vs_grid, vt_grid, qar_grid,d_s, d_t, vs, vt, current_qar, trajectory_points):
    ax.plot_surface(vs_grid, vt_grid, qar_grid, cmap="coolwarm", alpha=1)
    ax.text2D(0.04, 0.96, f"Dice stroma: {d_s}, Dice tumor: {d_t}",
              transform=ax.transAxes, fontsize=10, color="black")
    ax.set_xlabel("Tumor Volume (Vt)", fontsize=10, labelpad=8)
    ax.set_ylabel("Stroma Volume (Vs)", fontsize=10, labelpad=8)
    ax.set_zlabel("QAR (%)", fontsize=10, labelpad=8)
    ax.set_title("Model ambiguity with changing volumes", fontsize=12)

    ax.view_init(elev=45, azim=220)

    trajectory_points.append((vs, vt, current_qar))

    if len(trajectory_points) > 1:
        trajectory_vs, trajectory_vt, trajectory_qar = zip(*trajectory_points)
        ax.plot(trajectory_vs, trajectory_vt, trajectory_qar, color="black", linewidth=1, alpha=1, zorder=10)


def simulate_segmentation_errors(output_path):
    logger.info("Simulating segmentation errors...")
    v_values = np.linspace(1, 100, 50)
    d_values = np.linspace(0.1, 0.99, 50)
    v_grid, d_grid = np.meshgrid(v_values, d_values)

    epsilon_over = (2 * v_grid / d_grid) - 2 * v_grid
    epsilon_under = (2 * v_grid * (1 - d_grid)) / (2 - d_grid)

    epsilon_over_normalized = epsilon_over / np.max(epsilon_over)
    epsilon_under_normalized = epsilon_under / np.max(epsilon_under)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Dependence of segmentation error on Volume and Dice Score", fontsize=16)

    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(v_grid, d_grid, epsilon_over_normalized, cmap="plasma", edgecolor=None, alpha=0.8)
    ax1.set_xlabel("Volume (V)")
    ax1.set_ylabel("Dice Score (D)")
    ax1.set_zlabel(r"$\epsilon_{\text{over}}$", fontsize=12)
    ax1.set_title(r"Segmentation Error $\epsilon_{\text{over}}$ vs. V and D")
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(v_grid, d_grid, epsilon_under_normalized, cmap="inferno", edgecolor=None, alpha=0.8)
    ax2.set_xlabel("Volume (V)")
    ax2.set_ylabel("Dice Score (D)")
    ax2.set_zlabel(r"$\epsilon_{\text{under}}$", fontsize=12)
    ax2.set_title(r"Segmentation Error $\epsilon_{\text{under}}$ vs. V and D")
    ax2.view_init(elev=30, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.subplots_adjust(wspace=0.4)
    plt.savefig(output_path / "vd_vs_error.png", dpi=300)
    plt.close(fig)
    logger.info(f"Segmentation error simulation completed. Image saved at {output_path/'vd_vs_error.png'}")
