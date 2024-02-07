from typing import Generator

import numpy as np
from PIL import Image, ImageDraw


def paste_masked_tile_and_draw_polygons(image_canvas, sample, tile_size):
    original_tile = np.asarray(sample["image"]).astype(np.uint8)
    roi = sample["annotation_data"]["roi"]
    roi_expanded = np.expand_dims(roi, axis=-1)
    roi_expanded = np.repeat(roi_expanded, 3, axis=2)
    masked_tile = original_tile[:, :, :3] * roi_expanded
    masked_tile[roi_expanded == 0] = 255
    masked_tile = Image.fromarray(masked_tile.astype(np.uint8))
    coords = np.array(sample["coordinates"])
    box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
    image_canvas.paste(masked_tile, box)
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
