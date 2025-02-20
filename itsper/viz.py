from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

remap_colors = {0: (255, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 255, 0)}


def colorize(image: Image) -> Image:
    image_array = np.array(image)

    if len(image_array.shape) > 2:
        image_array = image_array[:, :, 0]  # Take first channel if multi-channel

    # Prepare an empty array for the colored image (4 channels for RGBA)
    colored_array = np.zeros((*image_array.shape, 4), dtype=np.uint8)

    # Apply the color map
    for class_index, color in remap_colors.items():
        mask = image_array == class_index
        if mask.any():  # Only apply if this class exists in the tile
            colored_array[mask, 0] = color[0]  # R
            colored_array[mask, 1] = color[1]  # G
            colored_array[mask, 2] = color[2]  # B
            # Set alpha to 64 only for non-background classes (class_index > 0)
            if class_index > 0:
                colored_array[mask, 3] = 64  # Semi-transparent for actual classes
            else:
                colored_array[mask, 3] = 0  # Fully transparent for background

    # Convert the NumPy array back to a PIL image in RGBA mode
    colored_image = Image.fromarray(colored_array, mode="RGBA")
    return colored_image


def create_canvas(dataset):
    tile_size = dataset.tile_size
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
    for sample in dataset:
        coordinates = sample.coordinates
        x0, y0 = coordinates
        x1, y1 = x0 + tile_size[0], y0 + tile_size[1]
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x1), max(max_y, y1)
    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
    return canvas, min_x, min_y


def render_visualization(
    dataset, inference_dataset, output_path: Path, slide_id: str, human_itsp_score: float, ai_itsp_score: float
):
    image_canvas, min_x, min_y = create_canvas(dataset)
    # Fill the canvas with white to make it fully opaque
    white_fill = Image.new("RGBA", image_canvas.size, (255, 255, 255, 255))
    image_canvas = Image.alpha_composite(image_canvas, white_fill)

    prediction_canvas = image_canvas.copy()
    canvas_min_x, canvas_min_y, canvas_max_x, canvas_max_y = float("inf"), float("inf"), float("-inf"), float("-inf")

    for image_sample, inference_sample in zip(dataset, inference_dataset):
        x_offset, y_offset = int(image_sample.coordinates[0] - min_x), int(image_sample.coordinates[1] - min_y)
        image_array = inference_sample.image.numpy()
        mask = inference_sample.annotations.rois.to_mask().numpy()

        mask = mask[..., np.newaxis].astype(np.uint8)  # Add channel dimension
        mask = np.repeat(mask, image_array.shape[-1], axis=-1)  # Repeat for all channels

        image_array = image_array * mask
        tile = image_sample.image.numpy() * mask
        tile = np.where(tile == 0, 255, tile)
        tile = Image.fromarray(tile)

        colored_image = colorize(image_array)
        box = (x_offset, y_offset, x_offset + inference_dataset.tile_size[0], y_offset + inference_dataset.tile_size[1])

        image_canvas.paste(tile, box)
        prediction_canvas.paste(tile, box)

        prediction_canvas.paste(colored_image, box, colored_image)
        if image_sample.annotations and inference_sample.annotations is not None:
            for polygon in image_sample.annotations.rois.get_geometries():
                x, y = polygon.to_shapely().exterior.xy
                for x_coord, y_coord in zip(x, y):
                    x_point = x_coord + x_offset
                    y_point = y_coord + y_offset
                    canvas_min_x = min(canvas_min_x, x_point)
                    canvas_max_x = max(canvas_max_x, x_point)
                    canvas_min_y = min(canvas_min_y, y_point)
                    canvas_max_y = max(canvas_max_y, y_point)

    image_canvas = image_canvas.convert("RGB")
    prediction_canvas = prediction_canvas.convert("RGB")
    image_canvas = image_canvas.crop((canvas_min_x, canvas_min_y, canvas_max_x, canvas_max_y))
    prediction_canvas = prediction_canvas.crop((canvas_min_x, canvas_min_y, canvas_max_x, canvas_max_y))

    extra_height = 70  # Space for text
    viz_image = Image.new("RGB", (image_canvas.size[0] * 2, image_canvas.size[1] + extra_height), (255, 255, 255))
    viz_image.paste(image_canvas, (0, 0))
    viz_image.paste(prediction_canvas, (image_canvas.size[0], 0))

    drawer = ImageDraw.Draw(viz_image)
    human_text = f"Human ITSP: {int(human_itsp_score)}%"
    ai_text = f"AI ITSP: {int(ai_itsp_score)}%"

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 50)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
        except OSError:
            font = ImageFont.load_default()
    drawer.text((50, image_canvas.size[1] + 10), human_text, fill="black", font=font)
    drawer.text((image_canvas.size[0] + 200, image_canvas.size[1] + 10), ai_text, fill="black", font=font)

    output_file_path = output_path / f"{slide_id}.png"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    viz_image.save(output_file_path, dpi=(300, 300))
