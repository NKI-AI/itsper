from pathlib import Path
from typing import Any

import numpy as np
from dlup import SlideImage
from dlup._image import Resampling
from dlup.annotations import WsiAnnotations
from dlup.backends import ImageBackend
from dlup.data.dataset import RegionFromWsiDatasetSample, TiledWsiDataset
from dlup.data.transforms import ConvertAnnotationsToMask
from dlup.tiling import TilingMode
from numpy.typing import NDArray

from itsper.annotations import get_most_invasive_region, offset_and_scale_tumorbed
from itsper.data_manager import get_paired_data, open_db_session, summarize_database
from itsper.io import get_logger
from itsper.types import ItsperAnnotationTypes
from itsper.utils import check_if_roi_is_present, make_csv_entries, make_directories_if_needed
from itsper.viz import assign_index_to_pixels, colorize, plot_visualization, render_visualization
from itsper.qar import calculate_qar

logger = get_logger(__name__)


def get_class_pixels(
    sample: RegionFromWsiDatasetSample,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    Obtain the pixels corresponding to each class and the region of interest.
    """
    curr_mask = np.asarray(sample["image"])
    stroma_mask = (curr_mask == 1).astype(np.uint8)
    tumor_mask = (curr_mask == 2).astype(np.uint8)
    other_mask = (curr_mask == 3).astype(np.uint8)
    roi = sample["annotation_data"]["roi"]  # type: ignore
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
    kwargs = {}
    a_type: ItsperAnnotationTypes | None = None
    offset_annotations: WsiAnnotations | None = None
    if (
        image_path.name == "TCGA-OL-A5RY-01Z-00-DX1.AE4E9D74-FC1C-4C1E-AE6D-5DF38899BBA6.svs"
        or image_path.name == "TCGA-OL-A5RW-01Z-00-DX1.E16DE8EE-31AF-4EAF-A85F-DB3E3E2C3BFF.svs"
    ):
        kwargs["overwrite_mpp"] = (0.25, 0.25)
    slide_image = SlideImage.from_file_path(image_path, internal_handler="pil", **kwargs)  # type: ignore
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


def get_itsp_score(image_dataset: TiledWsiDataset) -> tuple[float, float, float, float]:
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
    return itsp, total_tumor, total_stroma, total_others


def itsp_computer(
    manifest_path: Path,
    images_root: Path,
    annotations_root: Path,
    inference_root: Path,
    output_path: Path,
    render_images: int = 1,
) -> None:
    session = open_db_session(manifest_path)
    summarize_database(session)
    dice_scores, data_rubric = get_paired_data(session)

    for index, image, inference_image, annotation, itsp_score in data_rubric:
        kwargs = {}
        slide_id = Path(image.filename).stem
        wsi_path = images_root / image.filename
        slide_annotation_path = annotations_root / annotation.filename
        inference_file = inference_root / inference_image.filename
        native_mpp_for_inference = float(inference_image.mpp)
        tile_size = (int(inference_image.tile_size), int(inference_image.tile_size))

        make_directories_if_needed(output_path=output_path)
        setup_dictionary = setup(wsi_path, slide_annotation_path, native_mpp_for_inference)

        if image.overwrite_mpp is not None:
            kwargs["overwrite_mpp"] = (image.overwrite_mpp, image.overwrite_mpp)

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
            **kwargs,  # type: ignore
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
        logger.info(f"Computing ITSP for case: {index}")
        itsp, total_tumor, total_stroma, total_others = get_itsp_score(prediction_slide_dataset)
        if dice_scores is not None:
            itsp_high, itsp_low = calculate_qar(total_stroma, total_tumor, dice_scores.stroma_dice, dice_scores.tumor_dice)
        human_itsp_score: float | None = itsp_score.score if itsp_score is not None else None
        logger.info(f"| AI: {round(itsp)}%  |  Human: {human_itsp_score}%")
        if render_images:
            logger.info("Rendering visualization...")
            wsi_viz, pred_viz = render_visualization(
                image_dataset, prediction_slide_dataset, tile_size, setup_dictionary
            )
            plot_visualization(
                wsi_viz,
                pred_viz,
                human_itsp_score=human_itsp_score,
                ai_itsp_score=itsp,
                inference_file=inference_file,
                output_path=output_path,
                slide_id=slide_id,
                vizualization_type=setup_dictionary["annotation_type"],
                qar=(itsp_high*100, itsp_low*100),
            )

        make_csv_entries(
            inference_file=inference_file,
            output_path=output_path,
            slide_id=slide_id,
            human_itsp=human_itsp_score,
            ai_itsp=itsp,
            total_tumor=total_tumor,
            total_stroma=total_stroma,
            total_others=total_others,
        )
