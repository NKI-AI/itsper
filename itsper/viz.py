from pathlib import Path
from typing import Any, Generator, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL
from numpy._typing import NDArray
from PIL import Image, ImageDraw

from itsper.types import ItsperAnnotationTypes, ItsperClassIndices

# TODO: Make this configurable
TUMOR_PATCH = mpatches.Patch(color="red", label="Tumor")
STROMA_PATCH = mpatches.Patch(color="green", label="Stroma")
OTHER_PATCH = mpatches.Patch(color="yellow", label="Others")


def plot_2d(
    image: PIL.Image.Image,
    mask: npt.NDArray[np.int_] | None = None,
    mask_colors: dict[int, str] | None = None,
    mask_alpha: int = 70,
) -> PIL.Image.Image:
    """
    Plotting utility to overlay masks and geometries (Points, Polygons) on top of the image.

    Parameters
    ----------
    image : PIL.Image
    mask : np.ndarray
        Integer array
    mask_colors : dict
        A dictionary mapping the integer value of the mask to a PIL color value.
    mask_alpha : int
        Value between 0-100 defining the transparency of the overlays

    Returns
    -------
    PIL.Image
    """
    image = image.convert("RGBA")

    if mask is not None:
        if mask_colors is None:
            raise ValueError("mask_colors must be defined if mask is defined.")
        # Get unique values
        unique_vals = sorted(list(np.unique(mask)))
        for idx in unique_vals:
            if idx == 0:
                continue
            color = PIL.ImageColor.getcolor(mask_colors[idx], "RGBA")
            curr_mask = PIL.Image.fromarray(((mask == idx)[..., np.newaxis] * color).astype(np.uint8), mode="RGBA")
            alpha_channel = PIL.Image.fromarray(
                ((mask == idx) * int(mask_alpha * 255 / 100)).astype(np.uint8), mode="L"
            )
            curr_mask.putalpha(alpha_channel)
            image = PIL.Image.alpha_composite(image.copy(), curr_mask.copy()).copy()

    return image.convert("RGB")


def colorize(image: Image) -> Image:
    image_array = np.array(image)

    # Define your class index to RGBA color map as arrays for vectorized replacement
    # Including 255 for the alpha channel to indicate full opacity
    color_map = {
        0: np.array([0, 0, 0, 255]),  # Background (example)
        1: np.array([0, 255, 0, 255]),  # Stroma - Green
        2: np.array([255, 0, 0, 255]),  # Tumor - Red
        3: np.array([255, 255, 0, 255]),  # Others - Yellow
    }

    # Prepare an empty array for the colored image (4 channels for RGBA)
    colored_array = np.zeros((*image_array.shape, 4), dtype=np.uint8)

    # Apply the color map
    for class_index, color in color_map.items():
        mask = image_array == class_index
        # Use numpy broadcasting to set the color and alpha
        colored_array[mask] = color

    # Convert the NumPy array back to a PIL image in RGBA mode
    colored_image = Image.fromarray(colored_array, mode="RGBA")
    return colored_image


def render_visualization(
    image_dataset: Generator[dict[str, Any], int, None],
    prediction_dataset: Generator[dict[str, Any], int, None],
    tile_size: tuple[int, int],
    wsi_background: Image,
    prediction_background: Image,
) -> tuple[Image, Image]:
    for wsi_sample, prediction_sample in zip(image_dataset, prediction_dataset):
        coords = np.array(wsi_sample["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        roi = prediction_sample["annotation_data"]["roi"]
        prediction_tile = assign_index_to_pixels(prediction_sample["image"], roi=roi)
        prediction_tile = np.where(prediction_tile == 0, 255, prediction_tile)
        wsi_tile = np.asarray(wsi_sample["image"]) * roi[:, :, np.newaxis]
        wsi_tile = np.where(wsi_tile == 0, 255, wsi_tile)
        prediction_sample_viz = plot_2d(
            Image.fromarray(prediction_tile),
            mask=prediction_tile * roi,
            mask_colors={1: "green", 2: "red", 3: "yellow"},
        )
        prediction_background.paste(prediction_sample_viz, box)
        wsi_background.paste(Image.fromarray(wsi_tile.astype(np.uint8)), box)
        wsi_drawer = ImageDraw.Draw(wsi_background)
        prediction_drawer = ImageDraw.Draw(prediction_background)
        xy = []
        if wsi_sample["annotations"] and prediction_sample["annotations"] is not None:
            for polygon in wsi_sample["annotations"]:
                x, y = polygon.exterior.xy
                for x_coord, y_coord in zip(x, y):
                    xy.append(x_coord + wsi_sample["coordinates"][0])
                    xy.append(y_coord + wsi_sample["coordinates"][1])
                wsi_drawer.polygon(xy, outline="black")
                prediction_drawer.polygon(xy, outline="black")
                xy = []
    return wsi_background, prediction_background


def plot_visualization(
    original_image: Image,
    prediction_image: Image,
    inference_file: Path,
    human_itsp_score: Optional[float],
    ai_itsp_score: float,
    output_path: Path,
    slide_id: str,
    vizualization_type: ItsperAnnotationTypes,
) -> None:
    if vizualization_type == ItsperAnnotationTypes.MI_REGION:
        extra_space = 1000
    else:
        extra_space = 1200
    # Create a new image with space for the two images and additional text
    viz_image = Image.new(
        "RGBA",
        (original_image.size[0] + prediction_image.size[0] + extra_space, original_image.size[1] + 10),
        (255, 255, 255, 255),
    )

    # Paste the original and prediction images into the new image
    viz_image.paste(original_image, (0, 0))
    viz_image.paste(prediction_image, (original_image.size[0] + 150, 0))

    # Convert to RGB (removing alpha channel)
    viz_image = viz_image.convert("RGB")

    # Save the visualization image
    output_file_path = output_path / inference_file.parent.name / f"{slide_id}.png"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    plt.imshow(viz_image)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.gca().set_axis_off()
    font_size = max(7, min(plt.xlim()[1], plt.xlim()[0]) // 20)
    if human_itsp_score:
        plt.text(50, 0, f"Human score:{human_itsp_score} %", fontsize=font_size)
    plt.text(int(plt.xlim()[1] / 2), 0, f"AI score:{ai_itsp_score} %", fontsize=font_size)
    plt.legend(handles=[TUMOR_PATCH, STROMA_PATCH, OTHER_PATCH], fontsize=font_size, loc="upper right")
    plt.savefig(output_file_path, dpi=1000)
    plt.close()


def crop_image(
    wsi_background: Image, prediction_background: Image, setup_dictionary: dict[str, Any]
) -> tuple[Image, Image]:
    x0, y0, x1, y1 = setup_dictionary["scaled_annotation_bounds"]
    if setup_dictionary["annotation_type"] == ItsperAnnotationTypes.MI_REGION:
        draw_fov_wsi = ImageDraw.Draw(wsi_background)
        draw_fov_wsi.ellipse([(x0, y0), (x1, y1)], fill=None, outline="black", width=20)
        draw_fov_pred = ImageDraw.Draw(prediction_background)
        draw_fov_pred.ellipse([(x0, y0), (x1, y1)], fill=None, outline="black", width=20)
    original_tumor_bed = wsi_background.crop((x0, y0, x1, y1))
    prediciton_tumor_bed = prediction_background.crop((x0, y0, x1, y1))
    return original_tumor_bed, prediciton_tumor_bed


def assign_index_to_pixels(image: Image, roi: Optional[NDArray[np.float_]] = None) -> NDArray[np.uint8]:
    """
    Convert RGB pixel values to indices based on target colors.
    """
    valid_indices = [ItsperClassIndices.STROMA.value, ItsperClassIndices.TUMOR.value, ItsperClassIndices.OTHERS.value]
    image = np.array(image)
    # Assuming image is a NumPy array of shape (height, width, 4) and roi of shape (height, width)
    target_colors = np.array([[0, 128, 0, 255], [255, 0, 0, 255], [255, 255, 0, 255]])  # green  # red  # yellow

    # Calculate the difference between each pixel and the target colors
    diff = np.linalg.norm(image[:, :, None, :] - target_colors[None, None, :, :], axis=3)

    # Find the index of the minimum difference for each pixel
    indexed_image = np.argmin(diff, axis=2) + 1  # Add 1 to match your 1-based indexing

    # Check for any pixel that doesn't match any target color exactly and raise an error
    if not np.all(np.isin(indexed_image, valid_indices)):
        raise ValueError("Unknown color detected in the image.")

    if roi is not None:
        # Apply the ROI mask
        indexed_image = indexed_image * roi

    indexed_image = indexed_image.astype(np.uint8)

    return indexed_image  # type: ignore[no-any-return]
