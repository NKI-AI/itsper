from typing import Type

from aifo.dlup import SlideImage
from aifo.dlup.annotations import Polygon as DlupPolygon
from aifo.dlup.annotations import SlideAnnotations
from shapely import Point
from shapely.affinity import affine_transform, translate
import pdb
import copy

from itsper.types import ItsperAnnotationTypes


def offset_and_scale_tumorbed(
    annotations: SlideAnnotations, slide_image: SlideImage, native_mpp_for_inference: float = 1.0
) -> SlideAnnotations:
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
    annotations: SlideAnnotations, base_mpp: float) -> SlideAnnotations:
    """
    Get the most invasive region from the annotations.

    We take the following steps:

    1. Scale the annotations to the base resolution of the .
    2. Set the offset of the annotations.
    """
    if len(annotations.available_classes) == 0:
        raise ValueError(f"No most invasive regions found!.")
    if len(annotations.available_classes) > 1:
        raise ValueError(f"More than one most invasive regions found in the annotations.")

    most_invasive_region_base: SlideAnnotations = SlideAnnotations()
    index_map = {ItsperAnnotationTypes.MI_REGION: 1}

    if annotations.points is not None:
        center_x = annotations.points[0].x
        center_y = annotations.points[0].y
    else:
        raise ValueError(f"Most invasive region is not a point annotation.")

    center = Point(center_x, center_y)
    # We fix the radius of the circle to 1.05mm following the Leiden protocol.
    radius = 1.05 * 10e-3 / (base_mpp * 10e-6)
    # Create a circle polygon from the center point with the calculated radius
    most_invasive_region_base.add_polygon(DlupPolygon.from_shapely(center.buffer(radius)))
    most_invasive_region_base.polygons[0].label = ItsperAnnotationTypes.MI_REGION.value
    most_invasive_region_base.reindex_polygons(index_map)
    roi_ = most_invasive_region_base.polygons[0].to_shapely()
    roi = DlupPolygon.from_shapely(roi_)
    roi.label = ItsperAnnotationTypes.MI_REGION.value
    roi.index = 1
    most_invasive_region_base.add_roi(roi)
    most_invasive_region_base.rebuild_rtree()
    return most_invasive_region_base

def scale_and_offset_annotations(annotations: SlideAnnotations, scaling: float, slide_offset: tuple[float, float]) -> SlideAnnotations:
    # Create a new instance
    scaled_annotations = SlideAnnotations()
    
    # Copy each polygon
    for polygon in annotations.polygons:
        copy_polygon = polygon.clone()
        scaled_annotations.add_polygon(copy_polygon)

    
    # Copy points if any
    for point in annotations.points:
        copy_point = point.clone()
        scaled_annotations.add_point(copy_point)
    
    # Copy boxes if any
    for box in annotations.boxes:
        copy_box = box.clone()
        scaled_annotations.add_box(copy_box)
    
    # Copy ROIs if any
    for roi in annotations.rois:
        copy_roi = roi.clone()
        scaled_annotations.add_roi(copy_roi)
    
    # Apply scaling and offset
    scaled_annotations.scale(scaling)
    scaled_annotations.set_offset(slide_offset)
    scaled_annotations.rebuild_rtree()

    return scaled_annotations
