from typing import Generator

import numpy as np
from PIL import Image, ImageDraw
from itsper.utils import check_if_roi_is_present


def colorize(image: Image):
    image_array = np.array(image)

    # Define your class index to RGBA color map as arrays for vectorized replacement
    # Including 255 for the alpha channel to indicate full opacity
    color_map = {
        0: np.array([0, 0, 0, 255]),  # Background (example)
        1: np.array([0, 255, 0, 255]),  # Stroma - Green
        2: np.array([255, 0, 0, 255]),  # Tumor - Red
        3: np.array([255, 255, 0, 255])  # Others - Yellow
    }

    # Prepare an empty array for the colored image (4 channels for RGBA)
    colored_array = np.zeros((*image_array.shape, 4), dtype=np.uint8)

    # Apply the color map
    for class_index, color in color_map.items():
        mask = (image_array == class_index)
        # Use numpy broadcasting to set the color and alpha
        colored_array[mask] = color

    # Convert the NumPy array back to a PIL image in RGBA mode
    colored_image = Image.fromarray(colored_array, mode='RGBA')
    return colored_image


def paste_masked_tile_and_draw_polygons(image_canvas, sample, tile_size):
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


def visualize(original_image, prediction_image, tiff_file, output_path, slide_id):
    viz_image = Image.new(
        "RGBA",
        (original_image.size[0] + prediction_image.size[0] + 5, original_image.size[1] + 30),
        (255, 255, 255, 255),
    )
    viz_image.paste(original_image, (0, 0))
    viz_image.paste(prediction_image, (original_image.size[0] + 5, 0))

    viz_image.convert("RGB")
    viz_image.save(str(output_path) + "/" + tiff_file.parent.name + "/" + slide_id + ".png")


def render_tumor_bed(image_dataset: Generator, image_canvas: Image, tile_size: tuple) -> None:
    for image_tile in image_dataset:
        paste_masked_tile_and_draw_polygons(image_canvas, image_tile, tile_size)


def crop_image(setup_dictionary) -> tuple[Image, Image]:
    (x0, y0), (width, height) = setup_dictionary["scaled_annotations"].bounding_box
    original_tumor_bed = setup_dictionary["original_image_canvas"].crop((x0, y0, (x0 + width), (y0 + height)))
    prediciton_tumor_bed = setup_dictionary["prediction_image_canvas"].crop(
        (
            x0 - setup_dictionary["scaled_offset"][0],
            y0 - setup_dictionary["scaled_offset"][1],
            (x0 - setup_dictionary["scaled_offset"][0] + width),
            (y0 - setup_dictionary["scaled_offset"][1] + height),
        )
    )
    return original_tumor_bed, prediciton_tumor_bed
