from dlup import SlideImage
from dlup.annotations import AnnotationClass, AnnotationType
from dlup.annotations import Polygon as DlupPolygon
from dlup.annotations import WsiAnnotations
from shapely import Point
from shapely.affinity import affine_transform, translate
from typing import Type

from itsper.types import ItsperAnnotationTypes


def offset_and_scale_tumorbed(
    annotations: WsiAnnotations, slide_image: SlideImage, native_mpp_for_inference: float = 1.0
) -> WsiAnnotations:
    """
    Apply an appropriate scaling and translation to the tumorbed annotations.

    We take the following steps:

    1. Read the tumorbed annotations from slidescore at base level.
    2. Translate the annotations to remove the slide offset at the base level.
    3. Rescale the annotations to microns per pixel (mpp) set during inference time
    so that it matches with the base resolution of prediction tiff images.

    And return the rescaled, corrected annotations using WsiAnnotations from dlup.

    Returns
    -------
    offset_annotations: WsiAnnotations
        Annotations on the WSI rescaled to match the tiff image resolution.
    """
    single_annotations: list[DlupPolygon] = []
    single_annotation_tags: list[Type[AnnotationClass]] = []

    # We assume that there is only one class in the annotations.
    annotation_class = AnnotationClass
    annotation_class.annotation_type = AnnotationType.POLYGON
    annotation_class.label = ItsperAnnotationTypes.TUMORBED

    slide_offset, _ = slide_image.slide_bounds
    scaling_to_native_mpp_at_inference = slide_image.get_scaling(native_mpp_for_inference)
    transformation_matrix = [scaling_to_native_mpp_at_inference, 0, 0, scaling_to_native_mpp_at_inference, 0, 0]
    polygons = annotations.read_region((0, 0), scaling=1.0, size=slide_image.size)
    for polygon in polygons:
        translated_polygon = translate(polygon, -slide_offset[0], -slide_offset[1])
        transformed_polygon = affine_transform(translated_polygon, transformation_matrix)
        single_annotations.append(DlupPolygon(transformed_polygon, a_cls=annotation_class))
        single_annotation_tags.append(annotation_class)
    offset_annotations = WsiAnnotations(layers=single_annotations, tags=single_annotation_tags)
    return offset_annotations


def get_most_invasive_region(
    annotations: WsiAnnotations, slide_image: SlideImage, native_mpp_for_inference: float = 1.0
) -> tuple[WsiAnnotations, WsiAnnotations]:
    if len(annotations.available_classes) == 0:
        raise ValueError(f"No most invasive regions found for {slide_image.identifier}!.")
    if len(annotations.available_classes) > 1:
        raise ValueError(f"More than one most invasive regions found in the annotations for {slide_image.identifier}.")

    most_invasive_region: list[DlupPolygon] = []
    offset_most_invasive_region: list[DlupPolygon] = []

    annotation_class = AnnotationClass
    annotation_class.annotation_type = AnnotationType.POLYGON
    annotation_class.label = ItsperAnnotationTypes.MI_REGION

    slide_offset, _ = slide_image.slide_bounds
    scaling_to_native_mpp_at_inference = slide_image.get_scaling(native_mpp_for_inference)
    transformation_matrix = [scaling_to_native_mpp_at_inference, 0, 0, scaling_to_native_mpp_at_inference, 0, 0]

    mi = annotations.read_region((0, 0), scaling=1.0, size=slide_image.size)

    if annotations.available_classes[0].annotation_type == AnnotationType.POINT:
        center_x = mi[0].x
        center_y = mi[0].y
    else:
        raise ValueError(f"Most invasive region is not a point annotation for {slide_image.identifier}.")

    center = Point(center_x, center_y)
    # We fix the radius of the circle to 1.05mm following the Leiden protocol.
    radius = 1.05 * 10e-3 / (slide_image.mpp * 10e-6)
    # Create a circle polygon from the center point with the calculated radius
    most_invasive_region.append(DlupPolygon(center.buffer(radius), annotation_class))

    translated_most_invasive_region = translate(center.buffer(radius), -slide_offset[0], -slide_offset[1])
    transformed_most_invasive_region = affine_transform(translated_most_invasive_region, transformation_matrix)
    offset_most_invasive_region.append(DlupPolygon(transformed_most_invasive_region, annotation_class))
    return WsiAnnotations(most_invasive_region, [annotation_class]), WsiAnnotations(
        offset_most_invasive_region, [annotation_class]
    )

