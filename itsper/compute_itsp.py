from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
from dlup import SlideImage
from dlup._image import Resampling
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledWsiDataset
from dlup.data.transforms import ConvertAnnotationsToMask
from dlup.tiling import TilingMode
from dlup.viz.plotting import plot_2d
from numpy.typing import NDArray
from PIL import Image

from itsper.annotations import offset_and_scale_tumorbed, get_most_invasive_region
from itsper.types import ItsperAnnotationTypes, ITSPScoringSheetHeaders
from itsper.io import get_logger
from itsper.utils import (
    check_if_roi_is_present,
    get_list_of_files,
    make_csv_entries,
    make_directories_if_needed, check_integrity_of_files,
)
from itsper.viz import colorize, plot_mi_visualization, render_tumor_bed, plot_tb_vizualization, crop_image

# TODO: Feels hacky. Need to find a better way to handle this.
Image.MAX_IMAGE_PIXELS = 1000000 * 1000000
COLOR_MAP = {"green": 1, "red": 2, "yellow": 3}

logger = get_logger(__name__)


def get_class_pixels(sample: dict[str, Any]) -> tuple[
    NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    Obtain the pixels corresponding to each class and the region of interest.
    """
    curr_mask = np.asarray(sample["image"])
    stroma_mask = (curr_mask == 1).astype(np.uint8)
    tumor_mask = (curr_mask == 2).astype(np.uint8)
    other_mask = (curr_mask == 3).astype(np.uint8)
    roi = sample["annotation_data"]["roi"]
    return curr_mask, stroma_mask, tumor_mask, other_mask, roi


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


def setup(image_path: Path, annotation_path: Optional[list[Path] | None], native_mpp_for_inference: float) -> dict[
    str, Any]:
    """
    This function creates a dictionary object containing all the components necessary for the computation of ITSP.
    If there is an annotation file, it will also create Image objects for rendering neat visualizations.

    Parameters
    ----------
    image_path: Path
        Path to the image folders

    annotation_path: Optional[Path]
        Path to annotation files

    native_mpp_for_inference: float
        The microns per pixel at which the images need to be rendered.
    """
    offset_annotations: WsiAnnotations | None = None

    slide_image = SlideImage.from_file_path(image_path)
    scaling = slide_image.get_scaling(native_mpp_for_inference)
    scaled_offset, scaled_bounds = slide_image.get_scaled_slide_bounds(scaling)
    scaled_image_size = slide_image.get_scaled_size(scaling)
    annotations = WsiAnnotations.from_geojson(annotation_path, scaling=1.0)
    scaled_annotation = WsiAnnotations.from_geojson(annotation_path, scaling=scaling)
    available_labels = annotations.available_labels
    for a_class in available_labels:
        if a_class.label == ItsperAnnotationTypes.TUMORBED:
            a_type = a_class.label
            offset_annotations = offset_and_scale_tumorbed(annotations, slide_image, native_mpp_for_inference)
        elif a_class.label == ItsperAnnotationTypes.MI_REGION:
            a_type = a_class.label
            annotations, offset_annotations = get_most_invasive_region(annotations, slide_image,
                                                                       native_mpp_for_inference)
    scaled_annotation_bounds = annotations.read_region((0, 0), scaling=scaling, size=scaled_image_size)[
        0].bounds

    roi_name = annotations.available_labels[0].label
    transform = ConvertAnnotationsToMask(roi_name=roi_name, index_map={roi_name: 1})

    setup_dict = {
        "slide_image": slide_image,
        "scaling": scaling,
        "scaled_offset": scaled_offset,
        "scaled_bounds": scaled_bounds,
        "scaled_image_size": scaled_image_size,
        "annotations": annotations,
        "offset_annotations": offset_annotations,
        "scaled_annotations": scaled_annotation,
        "scaled_annotation_bounds": scaled_annotation_bounds,
        "transform": transform,
        "annotation_type": a_type,
    }
    return setup_dict


def get_itsp_score(image_dataset: Generator[dict[str, Any], int, None]) -> float:
    total_tumor = 0
    total_stroma = 0
    total_others = 0
    for sample in image_dataset:
        roi = check_if_roi_is_present(sample)
        if sample["image"].mode == "L":
            sample["image"] = colorize(sample["image"])
        indexed_image = assign_index_to_pixels(sample["image"], roi=roi)
        sample["image"] = indexed_image
        _, stroma_compartment, tumor_compartment, other_compartment, roi = get_class_pixels(sample)
        total_tumor += (tumor_compartment * roi).sum()
        total_stroma += (stroma_compartment * roi).sum()
        total_others += (other_compartment * roi).sum()

    itsp = (total_stroma * 100) / (total_stroma + total_tumor)
    itsp = round(itsp, 2)
    return itsp


def render_mi_visualizations(image_dataset: Generator[dict[str, Any], int, None],
                             prediction_dataset: Generator[dict[str, Any], int, None],
                             tile_size: (int, int),
                             wsi_background: Image,
                             prediction_background: Image) -> (Image, Image):
    for wsi_sample, prediction_sample in zip(image_dataset, prediction_dataset):
        coords = np.array(wsi_sample["coordinates"])
        box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
        roi = prediction_sample["annotation_data"]["roi"]
        prediction_tile = assign_index_to_pixels(prediction_sample["image"], roi=roi)
        prediction_tile = np.where(prediction_tile == 0, 255, prediction_tile)
        wsi_tile = np.asarray(wsi_sample["image"]) * roi[:, :, np.newaxis]
        wsi_tile = np.where(wsi_tile == 0, 255, wsi_tile)
        prediction_sample_viz = plot_2d(Image.fromarray(prediction_tile), mask=prediction_tile * roi,
                                        mask_colors={1: "green", 2: "red", 3: "yellow"})
        prediction_background.paste(prediction_sample_viz, box)
        wsi_background.paste(Image.fromarray(wsi_tile.astype(np.uint8)), box)
    return wsi_background, prediction_background


def itsp_computer(
        output_path: Path,
        images_path: Path,
        inference_path: Path,
        annotations_path: Optional[Path],
        native_mpp_for_inference: float,
        tile_size: tuple[int, int],
        render_images: bool = True,
) -> None:
    logger.info(f"Looking into slides from {images_path.name}")
    logger.info(f"Loading the ITSP scoring sheet...")
    try:
        itsp_scoring_sheet = pd.read_csv(f"{str(annotations_path)}/itsp.txt",
                                         delimiter="\t",
                                         header=None,
                                         usecols=[ITSPScoringSheetHeaders.SLIDE_ID.value,
                                                  ITSPScoringSheetHeaders.ITSP_SCORE.value])
    except FileNotFoundError as error:
        logger.info(f"{error}: Slidescore scoring sheet for ITSP not found to render human scores.")
        itsp_scoring_sheet = None

    folder_dictionary = check_integrity_of_files(images_path, inference_path, annotations_path)
    image_files, annotation_files, inference_files = get_list_of_files(folder_dictionary)
    for inference_file in inference_files:
        slide_id = inference_file.stem
        wsi_path = [image_file for image_file in image_files if image_file.stem == slide_id][0]
        slide_annotation_path = [
            annotation_file
            for annotation_file in annotation_files
            if annotation_file.parent.name == slide_id
        ]

        if len(slide_annotation_path) > 0:
            make_directories_if_needed(folder=images_path, output_path=output_path)
        else:
            logger.info(f"{slide_id} has no annotation. Skipping it.")
            continue

        setup_dictionary = setup(wsi_path, slide_annotation_path, native_mpp_for_inference)

        image_dataset = TiledWsiDataset.from_standard_tiling(
            wsi_path,
            mpp=native_mpp_for_inference,
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
                inference_file,
                mpp=native_mpp_for_inference,
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
                inference_file,
                mpp=native_mpp_for_inference,
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
        logger.info(f"Generating visualizations for: {slide_id}")
        itsp = get_itsp_score(prediction_slide_dataset)
        logger.info(f"The ITSP for image: {slide_id} is: {itsp}%")
        if render_images:
            slide_image = image_dataset.slide_image
            scaled_region_view = slide_image.get_scaled_view(slide_image.get_scaling(native_mpp_for_inference))
            wsi_background = Image.new("RGBA", scaled_region_view.size, (255, 255, 255, 255))
            prediction_background = Image.new("RGBA", scaled_region_view.size, (255, 255, 255, 255))
            if itsp_scoring_sheet is not None:
                human_itsp_score = itsp_scoring_sheet.loc[itsp_scoring_sheet[ITSPScoringSheetHeaders.SLIDE_ID.value] == slide_id, ITSPScoringSheetHeaders.ITSP_SCORE.value].item()
            else:
                logger.info("No human scores found. So, not rendering of human scores.")
                human_itsp_score = None
            if setup_dictionary["annotation_type"] == ItsperAnnotationTypes.MI_REGION:
                wsi_viz, pred_viz = render_mi_visualizations(image_dataset, prediction_slide_dataset, tile_size,
                                                              wsi_background, prediction_background)
                plot_mi_visualization(wsi_viz, pred_viz, setup_dictionary, itsp, output_path, images_path, slide_id, human_itsp_score=human_itsp_score)
            elif setup_dictionary["annotation_type"] == ItsperAnnotationTypes.TUMORBED:
                render_tumor_bed(prediction_slide_dataset, image_canvas=prediction_background, tile_size=tile_size)
                render_tumor_bed(image_dataset, image_canvas=wsi_background, tile_size=tile_size)
                original_tumor_bed, prediction_tumor_bed = crop_image(wsi_background, prediction_background, setup_dictionary)
                plot_tb_vizualization(
                    original_tumor_bed,
                    prediction_tumor_bed,
                    human_itsp_score=human_itsp_score,
                    ai_itsp_score=itsp,
                    inference_file=inference_file,
                    output_path=output_path,
                    slide_id=slide_id,
                )

        make_csv_entries(inference_file=inference_file, output_path=output_path, slide_id=slide_id, human_itsp=human_itsp_score, ai_itsp=itsp)
