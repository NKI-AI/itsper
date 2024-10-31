from io import BytesIO

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from itsper.io import get_logger

logger = get_logger(__name__)


def calculate_qar(Vs, Vt, D_s, D_t):
    epsilon_over_stroma = (2 * Vs) / D_s - 2 * Vs
    epsilon_under_stroma = (2 * Vs * (1 - D_s)) / (2 - D_s)
    epsilon_over_tumor = (2 * Vt) / D_t - 2 * Vt
    epsilon_under_tumor = (2 * Vt * (1 - D_t)) / (2 - D_t)
    itsp_high = (Vs + epsilon_over_stroma) / ((Vs + epsilon_over_stroma) + (Vt - epsilon_under_tumor))
    itsp_low = (Vs - epsilon_under_stroma) / ((Vs - epsilon_under_stroma) + (Vt + epsilon_over_tumor))
    return itsp_high - itsp_low


def plot_concentric_circles(ax, tumor_size, stroma_size):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", "box")
    ax.set_title("Tumor (Red) & Stroma (Green) Volumes")
    ax.axis("off")
    # Draw circles with smaller one on top
    if stroma_size < tumor_size:
        ax.add_patch(Circle((0, 0), tumor_size, color="red"))
        ax.add_patch(Circle((0, 0), stroma_size, color="green"))
    else:
        ax.add_patch(Circle((0, 0), stroma_size, color="green"))
        ax.add_patch(Circle((0, 0), tumor_size, color="red"))


def plot_qar_surface(ax, vs_grid, vt_grid, qar_grid, vs, vt, current_qar, trajectory_points):
    ax.plot_surface(vs_grid, vt_grid, qar_grid, cmap="coolwarm", alpha=1)
    ax.set_xlabel("Stroma Volume (Vs)", fontsize=10, labelpad=8)
    ax.set_ylabel("Tumor Volume (Vt)", fontsize=10, labelpad=8)
    ax.set_zlabel("QAR", fontsize=10, labelpad=8, rotation=90)
    ax.set_title("Behaviour of QAR with changing volumes", fontsize=12)
    ax.view_init(elev=30, azim=220)
    # Add the current point to the trajectory
    trajectory_points.append((vs, vt, current_qar))

    # Plot the trajectory as a line connecting all points in the list
    if len(trajectory_points) > 1:
        trajectory_vs, trajectory_vt, trajectory_qar = zip(*trajectory_points)
        ax.plot(trajectory_vs, trajectory_vt, trajectory_qar, color="black", linewidth=1, alpha=1, zorder=10)


def create_frames(num_frames, Vs_grid, Vt_grid, QAR_grid, dice_stroma, dice_tumor):
    tumor_sizes = np.linspace(1, 0.2, num_frames)
    stroma_sizes = np.linspace(0.2, 1, num_frames)
    frames = []
    tragetory_points = []

    for i in range(num_frames):
        fig = plt.figure(figsize=(12, 10))
        # Add title
        fig.suptitle("QAR as a function of Tumor and Stroma Volumes", fontsize=16)

        # Plot 2D concentric circles
        ax_2d = fig.add_subplot(121)
        plot_concentric_circles(ax_2d, tumor_sizes[i], stroma_sizes[i])

        # Plot 3D QAR surface
        ax_3d = fig.add_subplot(122, projection="3d")
        current_vs = stroma_sizes[i] * 1000
        current_vt = tumor_sizes[i] * 1000
        current_qar = calculate_qar(current_vs, current_vt, dice_stroma, dice_tumor)
        plot_qar_surface(ax_3d, Vs_grid, Vt_grid, QAR_grid, current_vs, current_vt, current_qar, tragetory_points)

        # Save the frame to a buffer instead of a file
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=80, pad_inches=0.003)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))
        buffer.close()
        plt.close(fig)

    return frames


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


def qar_surface_plotter(dice_stroma, dice_tumor, output_path):
    logger.info("QAR surface plotting initiated...")
    logger.info(f"Dice score for stroma: {dice_stroma}, Dice score for tumor: {dice_tumor}")
    num_frames = 50
    vs_values = np.linspace(100, 1000, 50)
    vt_values = np.linspace(100, 1000, 50)
    vs_grid, vt_grid = np.meshgrid(vs_values, vt_values)
    qar_grid = calculate_qar(vs_grid, vt_grid, dice_stroma, dice_tumor)

    frames = create_frames(num_frames, vs_grid, vt_grid, qar_grid, dice_stroma, dice_tumor)
    gif_path = f"{output_path}/qar_variation.gif"
    imageio.mimsave(gif_path, frames, duration=0.1, loop=0)
    logger.info(f"QAR surface plotting completed. GIF saved at {gif_path}")
    return gif_path
