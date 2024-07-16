from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
from dlup import SlideImage
from dlup._image import Resampling
from dlup.annotations import WsiAnnotations
from dlup.backends import ImageBackend
from dlup.data.dataset import TiledWsiDataset
from dlup.data.transforms import ConvertAnnotationsToMask
from dlup.tiling import TilingMode
from numpy.typing import NDArray

from itsper.annotations import get_most_invasive_region, offset_and_scale_tumorbed
from itsper.io import get_logger
from itsper.types import ItsperAnnotationTypes, ITSPScoringSheetHeaders
from itsper.utils import (
    check_if_roi_is_present,
    check_integrity_of_files,
    get_list_of_files,
    make_csv_entries,
    make_directories_if_needed,
)
from itsper.viz import assign_index_to_pixels, colorize, crop_image, plot_visualization, render_visualization

logger = get_logger(__name__)


def get_class_pixels(
    sample: dict[str, Any]
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    Obtain the pixels corresponding to each class and the region of interest.
    """
    curr_mask = np.asarray(sample["image"])
    stroma_mask = (curr_mask == 1).astype(np.uint8)
    tumor_mask = (curr_mask == 2).astype(np.uint8)
    other_mask = (curr_mask == 3).astype(np.uint8)
    roi = sample["annotation_data"]["roi"]
    return curr_mask, stroma_mask, tumor_mask, other_mask, roi


def setup(image_path: Path, annotation_path: Path, native_mpp_for_inference: float) -> dict[str, Any]:
    """
    This function creates a dictionary object containing all the components necessary for the computation of ITSP.
    If there is an annotation file, it will also create Image objects for rendering neat visualizations.

    Parameters
    ----------
    image_path: Path
        Path to the image folders

    annotation_path: Path
        Path to annotation files

    native_mpp_for_inference: float
        The microns per pixel at which the images need to be rendered.
    """
    offset_annotations: WsiAnnotations | None = None

    slide_image = SlideImage.from_file_path(image_path, internal_handler="pil")
    scaling = slide_image.get_scaling(native_mpp_for_inference)
    scaled_wsi_size = slide_image.get_scaled_size(scaling)
    annotations = WsiAnnotations.from_geojson(annotation_path)
    available_labels = annotations.available_classes
    for a_class in available_labels:
        if a_class.label == ItsperAnnotationTypes.TUMORBED:
            a_type = a_class.label
            offset_annotations = offset_and_scale_tumorbed(annotations, slide_image, native_mpp_for_inference)
        elif a_class.label == ItsperAnnotationTypes.MI_REGION:
            a_type = a_class.label
            annotations, offset_annotations = get_most_invasive_region(
                annotations, slide_image, native_mpp_for_inference
            )
    scaled_annotation_bounds = annotations.read_region((0, 0), scaling=scaling, size=scaled_wsi_size)[0].bounds

    roi_name = annotations.available_classes[0].label
    transform = ConvertAnnotationsToMask(roi_name=roi_name, index_map={roi_name: 1})

    setup_dict = {
        "slide_image": slide_image,
        "scaling": scaling,
        "scaled_wsi_size": scaled_wsi_size,
        "annotations": annotations,
        "offset_annotations": offset_annotations,
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
        if sample["image"].mode == "L" or "p":
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
    logger.info("Loading the ITSP scoring sheet...")
    human_itsp_score = None
    try:
        itsp_scoring_sheet = pd.read_csv(
            f"{str(annotations_path)}/itsp.txt",
            delimiter="\t",
            header=None,
            usecols=[ITSPScoringSheetHeaders.SLIDE_ID.value, ITSPScoringSheetHeaders.ITSP_SCORE.value],
        )
    except FileNotFoundError as error:
        logger.info(f"{error}: Slidescore scoring sheet for ITSP not found to render human scores.")
        itsp_scoring_sheet = None

    folder_dictionary = check_integrity_of_files(images_path, inference_path, annotations_path)
    image_files, annotation_files, inference_files = get_list_of_files(folder_dictionary)
    for inference_file in inference_files:
        slide_id = inference_file.stem
        wsi_path = [image_file for image_file in image_files if image_file.stem == slide_id][0]
        slide_annotation_path = []
        if annotation_files is not None:
            slide_annotation_path = [
                annotation_file for annotation_file in annotation_files if annotation_file.parent.name == slide_id
            ]
            if len(slide_annotation_path) > 0:
                make_directories_if_needed(folder=images_path, output_path=output_path)
            else:
                logger.info(f"{slide_id} has no annotation. Skipping it.")
                continue

        setup_dictionary = setup(wsi_path, slide_annotation_path[0], native_mpp_for_inference)

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
            backend=ImageBackend.PYVIPS,
            internal_handler="vips",
        )
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
            backend=ImageBackend.TIFFFILE,
            internal_handler="vips",
        )
        logger.info(f"Generating visualizations for: {slide_id}")
        itsp = get_itsp_score(prediction_slide_dataset)
        logger.info(f"The ITSP for image: {slide_id} is: {itsp}%")
        if render_images:
            wsi_viz, pred_viz = render_visualization(
                image_dataset, prediction_slide_dataset, tile_size, setup_dictionary
            )
            if itsp_scoring_sheet is not None:
                human_itsp_score = itsp_scoring_sheet.loc[
                    itsp_scoring_sheet[ITSPScoringSheetHeaders.SLIDE_ID.value] == slide_id,
                    ITSPScoringSheetHeaders.ITSP_SCORE.value,
                ].item()
            else:
                logger.info("No human scores found. So, not rendering human scores.")
            plot_visualization(
                wsi_viz,
                pred_viz,
                human_itsp_score=human_itsp_score,
                ai_itsp_score=itsp,
                inference_file=inference_file,
                output_path=output_path,
                slide_id=slide_id,
                vizualization_type=setup_dictionary["annotation_type"],
            )

        make_csv_entries(
            inference_file=inference_file,
            output_path=output_path,
            slide_id=slide_id,
            human_itsp=human_itsp_score,
            ai_itsp=itsp,
        )
