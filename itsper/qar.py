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

# For given volumes of tumor and stroma, calculate the QAR for different values of dice scores.
def calculate_qar_for_different_dice_scores():
    vs = 500
    vt = 500
    dice_scores = np.linspace(0.1, 0.99, 50)
    itsp_high_values = []
    itsp_low_values = []
    
    # Use the same Dice score for both stroma and tumor for simplicity
    for dice in dice_scores:
        itsp_high, itsp_low = calculate_qar(vs, vt, dice, dice)
        itsp_high_values.append(itsp_high)
        itsp_low_values.append(itsp_low)
    
    return dice_scores, itsp_high_values, itsp_low_values

# Plot the QAR for different values of dice scores.
def plot_qar_for_different_dice_scores(dice_stroma, dice_tumor, output_path):
    # Get data for plotting
    dice_scores, itsp_high_values, itsp_low_values = calculate_qar_for_different_dice_scores()
    
    # Calculate the QAR (ambiguity) as the absolute difference between high and low ITSP values
    qar_values = np.abs(np.array(itsp_high_values) - np.array(itsp_low_values)) * 100  # Convert to percentage
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First subplot: ITSP high and low values vs Dice score
    ax1.plot(dice_scores, itsp_high_values, 'b-', label='ITSP High')
    ax1.plot(dice_scores, itsp_low_values, 'r-', label='ITSP Low')
    ax1.fill_between(dice_scores, itsp_high_values, itsp_low_values, alpha=0.2, color='gray', label='Ambiguity Range')
    
    # Mark the current dice scores if provided
    if dice_stroma == dice_tumor:
        current_dice = dice_stroma
        current_high, current_low = calculate_qar(500, 500, current_dice, current_dice)
        ax1.plot(current_dice, current_high, 'bo', markersize=8)
        ax1.plot(current_dice, current_low, 'ro', markersize=8)
        ax1.axvline(x=current_dice, color='green', linestyle='--', alpha=0.5)
    
    ax1.set_title('ITSP Bounds vs Dice Score')
    ax1.set_xlabel('Dice Score')
    ax1.set_ylabel('ITSP Value')
    ax1.set_xlim(0.1, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Second subplot: QAR (ambiguity) vs Dice score
    ax2.plot(dice_scores, qar_values, 'g-', linewidth=2)
    
    # Mark the current dice score if provided
    if dice_stroma == dice_tumor:
        current_qar = np.abs(current_high - current_low) * 100
        ax2.plot(current_dice, current_qar, 'go', markersize=8)
        ax2.axvline(x=current_dice, color='green', linestyle='--', alpha=0.5)
    
    ax2.set_title('Quantification Ambiguity Range (QAR) vs Dice Score')
    ax2.set_xlabel('Dice Score')
    ax2.set_ylabel('QAR (%)')
    ax2.set_xlim(0.1, 1.0)
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Diminishing Ambiguity in ITSP Quantification with Improving Dice Score', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    # Save the figure
    plt.savefig(output_path / "qar_for_different_dice_scores.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_random_colored_circle(ax, tumor_volume, stroma_volume, resolution=500, cluster_size=10):
    total_volume = tumor_volume + stroma_volume
    if total_volume == 0:
        raise ValueError("Tumor and stroma volumes must sum to a positive value.")

    stroma_ratio = stroma_volume / total_volume

    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xv, yv = np.meshgrid(x, y)

    mask = xv**2 + yv**2 <= 1**2

    noise = np.random.rand(resolution, resolution)
    smoothed_noise = gaussian_filter(noise, sigma=cluster_size)

    threshold = np.percentile(smoothed_noise[mask], (1 - stroma_ratio) * 100)
    stroma_region = (smoothed_noise > threshold) & mask
    tumor_region = (smoothed_noise <= threshold) & mask

    colors = np.ones((resolution, resolution, 3))
    colors[tumor_region] = [1, 0, 0]  # Red for tumor
    colors[stroma_region] = [0, 1, 0]  # Green for stroma

    ax.imshow(colors, extent=(-1, 1, -1, 1))
    ax.text(0.35, 0.95, f"Stroma: {stroma_ratio*100:.2f}%", fontsize=10, transform=ax.transAxes, color="black")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", "box")
    ax.set_title("Varying tumor and stroma volumes")
    ax.axis("off")

    legend_elements = [
        Patch(facecolor="red", edgecolor="black", label="Tumor"),
        Patch(facecolor="green", edgecolor="black", label="Stroma"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False)


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
        plot_qar_surface(
            ax_3d,
            Vs_values,
            Vt_values,
            QAR_grid,
            dice_stroma,
            dice_tumor,
            current_vs,
            current_vt,
            current_qar,
            trajectory_points,
        )

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

    frames = create_frames(num_frames, vs_grid, vt_grid, (high - low) * 100, dice_stroma, dice_tumor)
    gif_path = f"{output_path}/qar_variation.gif"
    imageio.mimsave(gif_path, frames, duration=600, loop=0)
    logger.info(f"QAR surface plotting completed. GIF saved at {gif_path}")
    return gif_path


def plot_qar_surface(ax, vs_grid, vt_grid, qar_grid, d_s, d_t, vs, vt, current_qar, trajectory_points):
    ax.plot_surface(vs_grid, vt_grid, qar_grid, cmap="coolwarm", alpha=1)
    ax.text2D(0.04, 0.96, f"Dice stroma: {d_s}, Dice tumor: {d_t}", transform=ax.transAxes, fontsize=10, color="black")
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

def plot_qar_3d(tumor_dice, stroma_dice, output_path):
    """
    Create a 3D plot showing how QAR (itsp_high - itsp_low) varies with both tumor and stroma Dice scores.
    
    Args:
        tumor_dice: Dice score for tumor segmentation
        stroma_dice: Dice score for stroma segmentation
        output_path: Path to save the output figure
    """
    # Create a grid of Dice scores
    dice_values = np.linspace(0.1, 0.99, 30)  # Using 30 points for better performance
    dice_stroma_grid, dice_tumor_grid = np.meshgrid(dice_values, dice_values)
    
    # Initialize QAR grid
    qar_grid = np.zeros_like(dice_stroma_grid)
    
    # Fixed volumes for this visualization
    vs = 500
    vt = 500
    
    # Calculate QAR for each combination of Dice scores
    for i in range(len(dice_values)):
        for j in range(len(dice_values)):
            d_s = dice_stroma_grid[i, j]
            d_t = dice_tumor_grid[i, j]
            itsp_high, itsp_low = calculate_qar(vs, vt, d_s, d_t)
            qar_grid[i, j] = np.abs(itsp_high - itsp_low) * 100  # Convert to percentage
    
    # Find the indices in dice_values that are closest to the given tumor_dice and stroma_dice
    stroma_idx = np.abs(dice_values - stroma_dice).argmin()
    tumor_idx = np.abs(dice_values - tumor_dice).argmin()
    
    # Get the QAR value at these indices
    specific_qar = qar_grid[tumor_idx, stroma_idx]

    # Create the 3D plot with larger figure size
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(
        dice_stroma_grid, 
        dice_tumor_grid, 
        qar_grid, 
        cmap='viridis',
        linewidth=0, 
        antialiased=True,
        alpha=0.8
    )
    
    # Add a prominent scatter point for the specific dice scores
    ax.scatter(
        stroma_dice, 
        tumor_dice, 
        specific_qar, 
        color='red', 
        marker='o', 
        s=200,  # Larger size
        edgecolor='black',
        linewidth=2,
        label=f'Dice Scores: Stroma={stroma_dice:.2f}, Tumor={tumor_dice:.2f}',
        alpha=1
    )
    
    # Add a text annotation near the point
    ax.text(
        stroma_dice + 0.03, 
        tumor_dice + 0.03, 
        specific_qar + 5, 
        f'QAR: {specific_qar:.1f}%', 
        color='black',
        fontweight='bold',
        fontsize=12
    )
    
    # Add a color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=10)
    cbar.set_label('QAR (%)', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Stroma Dice Score', fontsize=12)
    ax.set_ylabel('Tumor Dice Score', fontsize=12)
    ax.set_zlabel('QAR (%)', fontsize=12)
    ax.set_title('Quantification Ambiguity Range (QAR) vs Dice Scores', fontsize=16)
    
    # Set axis limits
    ax.set_xlim(0.1, 0.99)
    ax.set_ylim(0.1, 0.99)
    
    # Add grid lines
    ax.grid(True, alpha=0.3)
    
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save the figure
    plt.savefig(output_path / "qar_3d_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path / "qar_3d_plot.png"
