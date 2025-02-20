from pathlib import Path

import numpy as np
from aifo.dlup import SlideImage
from aifo.dlup.annotations import SlideAnnotations
from aifo.dlup.data.dataset import SlideDataset
from aifo.dlup.tiling import GridOrder, TilingMode
from numpy.typing import NDArray

from itsper.annotations import get_most_invasive_region, offset_and_scale_tumorbed
from itsper.data_manager import get_paired_data, open_db_session, summarize_database
from itsper.io import get_logger
from itsper.types import ItsperAnnotationTypes
from itsper.utils import check_if_roi_is_present, make_csv_entries, make_directories_if_needed
from itsper.viz import render_visualization

logger = get_logger(__name__)


def get_tissue_compartments(
    image_array,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    Obtain the pixels corresponding to each class and the region of interest.
    """
    # Take first channel if image is multi-channel
    if len(image_array.shape) > 2:
        image_array = image_array[..., 0]

    stroma_mask = (image_array == 1).astype(np.uint8)
    tumor_mask = (image_array == 2).astype(np.uint8)
    other_mask = (image_array == 3).astype(np.uint8)
    return stroma_mask, tumor_mask, other_mask


def get_relative_scaling(
    slide_image: SlideImage, native_mpp_at_inference: float
) -> tuple[float, tuple[int, int], tuple[float, float]]:
    scaling = slide_image.get_scaling(native_mpp_at_inference)
    return scaling


def get_itsp_score(image_dataset: SlideDataset) -> tuple[float, float, float, float]:
    total_tumor = 0
    total_stroma = 0
    total_others = 0
    for sample in image_dataset:
        image_array = np.asarray(sample.image).astype(np.uint8)
        roi = check_if_roi_is_present(sample)
        stroma_compartment, tumor_compartment, other_compartment = get_tissue_compartments(image_array)
        total_tumor += (tumor_compartment * roi).sum()
        total_stroma += (stroma_compartment * roi).sum()
        total_others += (other_compartment * roi).sum()

    itsp = (total_stroma * 100) / (total_stroma + total_tumor)
    itsp = round(itsp, 2)
    return itsp, total_tumor, total_stroma, total_others


def get_image_dataset(
    wsi_path: Path, annotation_path: Path, native_mpp_for_inference: float, tile_size: tuple[int, int], **kwargs
) -> SlideDataset:
    annotations = SlideAnnotations.from_geojson(annotation_path)
    slide_image = SlideImage.from_file_path(wsi_path, **kwargs)
    available_labels = annotations.available_classes
    for a_class in available_labels:
        if a_class == ItsperAnnotationTypes.TUMORBED:
            polygon_annotations = offset_and_scale_tumorbed(annotations, slide_image, native_mpp_for_inference)
        elif a_class == ItsperAnnotationTypes.MI_REGION:
            polygon_annotations = get_most_invasive_region(annotations, slide_image.mpp)

    image_dataset = SlideDataset.from_standard_tiling(
        wsi_path,
        mpp=native_mpp_for_inference,
        tile_size=tile_size,
        tile_overlap=(0, 0),
        crop=False,
        annotations=polygon_annotations,
        mask=polygon_annotations,
        tile_mode=TilingMode.overflow,
        grid_order=GridOrder.C,
        **kwargs,  # type: ignore
    )
    return image_dataset


def get_inference_dataset(
    wsi_path: Path,
    inference_file: Path,
    annotation_path: Path,
    native_mpp_for_inference: float,
    tile_size: tuple[int, int],
    **kwargs,
) -> SlideDataset:
    annotations = SlideAnnotations.from_geojson(annotation_path)
    slide_image = SlideImage.from_file_path(wsi_path, **kwargs)
    available_labels = annotations.available_classes
    for a_class in available_labels:
        if a_class == ItsperAnnotationTypes.TUMORBED:
            annotations = offset_and_scale_tumorbed(annotations, slide_image, native_mpp_for_inference)
        elif a_class == ItsperAnnotationTypes.MI_REGION:
            polygon_annotations = get_most_invasive_region(annotations, slide_image.mpp)
    scaling = get_relative_scaling(slide_image, native_mpp_for_inference)
    polygon_annotations.scale(scaling)
    polygon_annotations.rebuild_rtree()
    prediction_slide_dataset = SlideDataset.from_standard_tiling(
        inference_file,
        tile_size=tile_size,
        tile_overlap=(0, 0),
        mpp=native_mpp_for_inference,
        mask=polygon_annotations,
        annotations=polygon_annotations,
        crop=False,
        tile_mode=TilingMode.overflow,
        grid_order=GridOrder.C,
        interpolator="NEAREST",
        **kwargs,
    )
    return prediction_slide_dataset


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
    data_rubric = get_paired_data(session)

    for index, image, inference_image, annotation, itsp_score in data_rubric:
        kwargs = {}
        slide_id = Path(image.filename).stem
        wsi_path = images_root / image.filename
        slide_annotation_path = annotations_root / annotation.filename
        inference_file = inference_root / inference_image.filename
        native_mpp_for_inference = inference_image.mpp
        tile_size = (int(inference_image.tile_size), int(inference_image.tile_size))

        make_directories_if_needed(output_path=output_path)

        if image.overwrite_mpp is not None:
            kwargs["overwrite_mpp"] = (image.overwrite_mpp, image.overwrite_mpp)
        image_dataset = get_image_dataset(
            wsi_path, slide_annotation_path, native_mpp_for_inference, tile_size, **kwargs
        )
        try:
            prediction_slide_dataset = get_inference_dataset(
                wsi_path, inference_file, slide_annotation_path, native_mpp_for_inference, tile_size, **kwargs
            )
        except FileNotFoundError as e:
            logger.error(e)
            continue
        logger.info(f"Computing ITSP for case: {index}")
        itsp, total_tumor, total_stroma, total_others = get_itsp_score(prediction_slide_dataset)
        human_itsp_score: float | None = itsp_score.score if itsp_score is not None else None
        logger.info(f"| AI: {round(itsp)}%  |  Human: {int(human_itsp_score)}%")
        if render_images:
            logger.info("Rendering visualization...")
            render_visualization(
                image_dataset,
                prediction_slide_dataset,
                output_path=output_path,
                slide_id=slide_id,
                human_itsp_score=human_itsp_score,
                ai_itsp_score=itsp,
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
