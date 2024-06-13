from pathlib import Path
from typing import Any, Generator, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL
from PIL import Image, ImageDraw

from itsper.types import ItsperAnnotationTypes
from itsper.utils import check_if_roi_is_present

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


def paste_masked_tile_and_draw_polygons(
    image_canvas: Image, sample: dict[str, Any], tile_size: tuple[int, int]
) -> None:
    original_tile = np.asarray(sample["image"]).astype(np.uint8)
    roi = check_if_roi_is_present(sample)
    if roi is not None:
        roi_expanded = np.expand_dims(roi, axis=-1)
        roi_expanded = np.repeat(roi_expanded, 3, axis=2)
        tile = original_tile[:, :, :3] * roi_expanded
        tile[roi_expanded == 0] = 255
        tile = Image.fromarray(tile.astype(np.uint8))
    else:
        tile = Image.fromarray(original_tile)
    coords = np.array(sample["coordinates"])
    box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
    image_canvas.paste(tile, box)
    original_drawer = ImageDraw.Draw(image_canvas)
    xy = []
    if sample["annotations"] is not None:
        for polygon in sample["annotations"]:
            x, y = polygon.exterior.xy
            for x_coord, y_coord in zip(x, y):
                xy.append(x_coord + sample["coordinates"][0])
                xy.append(y_coord + sample["coordinates"][1])
            original_drawer.polygon(xy, outline="black")
            xy = []


def plot_tb_vizualization(
    original_image: Image,
    prediction_image: Image,
    inference_file: Path,
    human_itsp_score: Optional[float],
    ai_itsp_score: float,
    output_path: Path,
    slide_id: str,
) -> None:
    # TODO: Print ITSP scores on the images.
    viz_image = Image.new(
        "RGBA",
        (original_image.size[0] + prediction_image.size[0] + 5, original_image.size[1] + 30),
        (255, 255, 255, 255),
    )
    viz_image.paste(original_image, (0, 0))
    viz_image.paste(prediction_image, (original_image.size[0] + 5, 0))

    viz_image.convert("RGB")
    viz_image.save(str(output_path) + "/" + inference_file.parent.name + "/" + slide_id + ".png")


def plot_mi_visualization(
    wsi_viz: Image,
    pred_viz: Image,
    setup_dictionary: dict[str, Any],
    ai_itsp_score: float,
    output_path: Path,
    images_path: Path,
    slide_id: str,
    human_itsp_score: Optional[float] = None,
) -> None:
    x0, y0, x1, y1 = setup_dictionary["scaled_annotation_bounds"]
    if setup_dictionary["annotation_type"] == ItsperAnnotationTypes.MI_REGION:
        draw_fov = ImageDraw.Draw(wsi_viz)
        draw_fov.ellipse([(x0, y0), (x1, y1)], fill=None, outline="black", width=20)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(40)
    fig.set_figwidth(100)
    ax1.imshow(wsi_viz.crop((x0, y0, x1, y1)))
    if human_itsp_score:
        ax1.text(50, 20, f"Human score:{human_itsp_score} %", fontsize=70, backgroundcolor="white")
    ax2.imshow(pred_viz.crop((x0, y0, x1, y1)))
    ax2.text(50, 20, f"AI score:{ai_itsp_score} %", fontsize=70, backgroundcolor="white")
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_yticks([])
    ax2.set_yticks([])
    fig.legend(handles=[TUMOR_PATCH, STROMA_PATCH, OTHER_PATCH], fontsize=70, loc="upper right")
    plt.savefig(f"{output_path}/{images_path.name}/{slide_id}.png", dpi=300)
    plt.close(fig)


def render_tumor_bed(
    image_dataset: Generator[dict[str, Any], int, None], image_canvas: Image, tile_size: tuple[int, int]
) -> None:
    for image_tile in image_dataset:
        paste_masked_tile_and_draw_polygons(image_canvas, image_tile, tile_size)


def crop_image(
    wsi_background: Image, prediction_background: Image, setup_dictionary: dict[str, Any]
) -> tuple[Image, Image]:
    (x0, y0), (width, height) = setup_dictionary["scaled_annotations"].bounding_box
    original_tumor_bed = wsi_background.crop((x0, y0, (x0 + width), (y0 + height)))
    prediciton_tumor_bed = prediction_background.crop(
        (
            x0 - setup_dictionary["scaled_offset"][0],
            y0 - setup_dictionary["scaled_offset"][1],
            (x0 - setup_dictionary["scaled_offset"][0] + width),
            (y0 - setup_dictionary["scaled_offset"][1] + height),
        )
    )
    return original_tumor_bed, prediciton_tumor_bed
