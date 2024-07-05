from typing import Any, Dict, List, Union

from dlup import SlideImage
from dlup.annotations import AnnotationClass, AnnotationType, Polygon, WsiAnnotations
from shapely import Point
from shapely.affinity import affine_transform, translate
from shapely.geometry import mapping

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
    single_annotations: list[Polygon] = []
    single_annotation_tags: list[AnnotationClass] = []
    annotation_class = annotations.available_classes
    slide_offset, _ = slide_image.slide_bounds
    scaling_to_native_mpp_at_inference = slide_image.get_scaling(native_mpp_for_inference)
    transformation_matrix = [scaling_to_native_mpp_at_inference, 0, 0, scaling_to_native_mpp_at_inference, 0, 0]
    polygons = annotations.read_region((0, 0), scaling=native_mpp_for_inference, size=slide_image.size)
    for ann_class, polygon in zip(annotation_class, polygons):
        translated_polygon = translate(polygon, -slide_offset[0], -slide_offset[1])
        transformed_polygon = affine_transform(translated_polygon, transformation_matrix)
        single_annotations.append(Polygon(transformed_polygon, ann_class))
        single_annotation_tags.append(ann_class)
    offset_annotations = WsiAnnotations(layers=single_annotations, tags=single_annotation_tags)
    return offset_annotations


def get_most_invasive_region(
    annotations: WsiAnnotations, slide_image: SlideImage, native_mpp_for_inference: float = 1.0
) -> tuple[WsiAnnotations, WsiAnnotations]:
    if len(annotations.available_classes) == 0:
        raise ValueError(f"No most invasive regions found for {slide_image.identifier}!.")
    if len(annotations.available_classes) > 1:
        raise ValueError(f"More than one most invasive regions found in the annotations for {slide_image.identifier}.")

    most_invasive_region: list[Polygon] = []
    offset_most_invasive_region: list[Polygon] = []

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
    most_invasive_region.append(Polygon(center.buffer(radius), annotation_class))

    translated_most_invasive_region = translate(center.buffer(radius), -slide_offset[0], -slide_offset[1])
    transformed_most_invasive_region = affine_transform(translated_most_invasive_region, transformation_matrix)
    offset_most_invasive_region.append(Polygon(transformed_most_invasive_region, annotation_class))
    return WsiAnnotations(most_invasive_region, [annotation_class]), WsiAnnotations(
        offset_most_invasive_region, [annotation_class]
    )


def to_geojson_format(list_of_points: list[Polygon], label: str) -> dict[str, Any]:
    """
    Convert a given list of annotations into the GeoJSON standard.

    Parameters
    ----------
    list_of_points: list
        A list containing annotation shapes or coordinates.
    label: str
        The string identifying the annotation class.
    """

    feature_collection = {
        "type": "FeatureCollection",
        "features": [],
    }

    features: List[Any] = []
    properties: Dict[str, Union[str, Dict[str, str | None]]] = {
        "classification": {"name": label, "color": None},
    }
    if len(list_of_points) == 0:
        feature_collection["features"] = []
    else:
        for data in list_of_points:
            geometry = mapping(data)
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": geometry,
                }
            )
        feature_collection["features"] = features
    return feature_collection
