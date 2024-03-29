from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
from dlup import SlideImage
from dlup._image import Resampling
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledWsiDataset
from dlup.data.transforms import ConvertAnnotationsToMask
from dlup.tiling import TilingMode
from numpy.typing import NDArray
from PIL import Image

from itsper.annotations import offset_and_scale_tumorbed
from itsper.io import get_logger
from itsper.utils import (
    check_if_roi_is_present,
    get_list_of_files,
    make_csv_entries,
    make_directories_if_needed,
    verify_folders,
)
from itsper.viz import colorize, crop_image, paste_masked_tile_and_draw_polygons, render_tumor_bed, visualize

Image.MAX_IMAGE_PIXELS = 1000000 * 1000000
COLOR_MAP = {"green": 1, "red": 2, "yellow": 3}

logger = get_logger(__name__)


def get_class_pixels(sample: dict[str, Any]) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    Obtain the pixels corresponding to each class and the region of interest.
    """
    curr_mask = np.asarray(sample["indexed_image"])
    stroma_mask = (curr_mask == 1).astype(np.uint8)
    tumor_mask = (curr_mask == 2).astype(np.uint8)
    other_mask = (curr_mask == 3).astype(np.uint8)
    return stroma_mask, tumor_mask, other_mask


def assign_index_to_pixels(image: Image, roi: Optional[NDArray[np.float_]] = None) -> NDArray[np.uint8]:
    """
    Convert RGB pixel values to indices based on target colors.
    """
    valid_indices = [1, 2, 3]
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


def setup(image_path: Path, annotation_path: Optional[list[Path] | None], target_mpp: float) -> dict[str, Any]:
    """
    This function creates a dictionary object containing all the components necessary for the computation of ITSP.
    If there is an annotation file, it will also create Image objects for rendering neat visualizations.

    Parameters
    ----------
    image_path: Path
        Path to the image folders

    annotation_path: Optional[Path]
        Path to annotation files

    target_mpp: float
        The microns per pixel at which the images need to be rendered.
    """
    slide_image = SlideImage.from_file_path(image_path)

    scaling = slide_image.get_scaling(target_mpp)
    scaled_offset, scaled_bounds = slide_image.get_scaled_slide_bounds(scaling)
    scaled_image_size = slide_image.get_scaled_size(scaling)

    if len(annotation_path) > 0:
        annotations = WsiAnnotations.from_geojson(annotation_path, scaling=1.0)
        offset_annotations = offset_and_scale_tumorbed(slide_image, annotations)

        roi_name = annotations.available_labels[0].label
        transform = ConvertAnnotationsToMask(roi_name=roi_name, index_map={roi_name: 1})

        scaled_annotations = WsiAnnotations.from_geojson(annotation_path, scaling=scaling)
    else:
        offset_annotations = None
        scaled_annotations = None
        transform = None
        annotations = None

    original_image_canvas = Image.new("RGBA", tuple(scaled_image_size), (255, 255, 255, 255))
    prediction_output_canvas = Image.new("RGBA", tuple(scaled_image_size), (255, 255, 255, 255))

    output_dict = {
        "slide_image": slide_image,
        "scaling": scaling,
        "scaled_offset": scaled_offset,
        "scaled_bounds": scaled_bounds,
        "scaled_image_size": scaled_image_size,
        "annotations": annotations,
        "offset_annotations": offset_annotations,
        "scaled_annotations": scaled_annotations,
        "original_image_canvas": original_image_canvas,
        "prediction_image_canvas": prediction_output_canvas,
        "transform": transform,
    }
    return output_dict


def compute_itsp_and_render_visualization(
    image_dataset: Generator[dict[str, Any], int, None],
    image_canvas: Image,
    tile_size: tuple[int, int],
    render_images: bool = True,
) -> float:
    """ """
    total_tumor = 0
    total_stroma = 0
    total_others = 0
    for sample in image_dataset:
        roi = check_if_roi_is_present(sample)
        if sample["image"].mode == "L":
            sample["image"] = colorize(sample["image"])
        indexed_image = assign_index_to_pixels(sample["image"], roi=roi)
        sample["indexed_image"] = indexed_image
        stroma_mask, tumor_mask, other_mask = get_class_pixels(sample)
        total_tumor += tumor_mask.sum()
        total_stroma += stroma_mask.sum()
        total_others += other_mask.sum()
        if render_images:
            paste_masked_tile_and_draw_polygons(image_canvas, sample, tile_size)

    itsp = (total_stroma * 100) / (total_stroma + total_tumor)
    itsp = round(itsp, 2)
    return itsp


def itsp_computer(
    output_path: Path,
    images_path: Path,
    inference_path: Path,
    path_to_anotation_files: Optional[Path],
    image_format: str,
    target_mpp: float,
    tile_size: tuple[int, int],
    render_images: bool = True,
) -> None:
    if verify_folders(path=images_path, image_format=image_format):
        logger.info(f"Looking into slides from {images_path.name}")

        image_files, annotation_files, tiff_files = get_list_of_files(
            images_path, image_format, path_to_anotation_files, inference_path
        )

        if len(tiff_files) > 0:
            for tiff_file in tiff_files:
                slide_id = tiff_file.stem
                image_path = [image_file for image_file in image_files if image_file.stem == slide_id][0]
                if annotation_files is not None:
                    annotation_path = [
                        annotation_file
                        for annotation_file in annotation_files
                        if annotation_file.parent.name == slide_id
                    ]
                    if len(annotation_path) > 0:
                        logger.info(f"Generating visualizations for: {slide_id}")
                        make_directories_if_needed(folder=images_path, output_path=output_path)
                else:
                    logger.info(f"{slide_id} has no tumorbed annotation.")
                    annotation_path = None

                setup_dictionary = setup(image_path, annotation_path, target_mpp)

                image_dataset = TiledWsiDataset.from_standard_tiling(
                    image_path,
                    mpp=target_mpp,
                    tile_size=tile_size,
                    tile_overlap=(0, 0),
                    crop=False,
                    annotations=setup_dictionary["annotations"],
                    mask=setup_dictionary["annotations"],
                    mask_threshold=0.0,
                    transform=setup_dictionary["transform"],
                    tile_mode=TilingMode.overflow,
                    backend="OPENSLIDE",
                )
                try:
                    prediction_slide_dataset = TiledWsiDataset.from_standard_tiling(
                        tiff_file,
                        mpp=target_mpp,
                        tile_size=tile_size,
                        tile_overlap=(0, 0),
                        crop=False,
                        annotations=setup_dictionary["offset_annotations"],
                        mask=setup_dictionary["offset_annotations"],
                        mask_threshold=0.0,
                        transform=setup_dictionary["transform"],
                        tile_mode=TilingMode.overflow,
                        interpolator=Resampling.NEAREST,
                        backend="OPENSLIDE",
                    )
                except Exception as error:
                    logger.info(f"OPENSLIDE fails because: {error}. Attempting with TIFFFILE...")
                    prediction_slide_dataset = TiledWsiDataset.from_standard_tiling(
                        tiff_file,
                        mpp=target_mpp,
                        tile_size=tile_size,
                        tile_overlap=(0, 0),
                        crop=False,
                        annotations=setup_dictionary["offset_annotations"],
                        mask=setup_dictionary["offset_annotations"],
                        mask_threshold=0.0,
                        transform=setup_dictionary["transform"],
                        tile_mode=TilingMode.overflow,
                        interpolator=Resampling.NEAREST,
                        backend="TIFFFILE",
                    )

                itsp = compute_itsp_and_render_visualization(
                    prediction_slide_dataset,
                    setup_dictionary["prediction_image_canvas"],
                    tile_size=tile_size,
                    render_images=render_images,
                )
                logger.info(f"The ITSP for image: {slide_id} is: {itsp}%")

                if len(annotation_path) > 0:
                    render_tumor_bed(image_dataset, setup_dictionary["original_image_canvas"], tile_size=tile_size)
                    original_tumor_bed, prediction_tumor_bed = crop_image(setup_dictionary)
                    visualize(
                        original_tumor_bed,
                        prediction_tumor_bed,
                        tiff_file=tiff_file,
                        output_path=output_path,
                        slide_id=slide_id,
                    )
                make_csv_entries(tiff_file=tiff_file, output_path=output_path, slide_id=slide_id, itsp=itsp)
        else:
            logger.info(f"Skipping folder {images_path.name} because there are not predictions.")
