import json
from pathlib import Path
from typing import Any, Dict, List, Union

from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from shapely.affinity import affine_transform, translate
from shapely.geometry import mapping


def offset_and_scale_tumorbed(
    slide_image: SlideImage, annotations: WsiAnnotations, native_mpp_at_inference: float = 1.0
) -> WsiAnnotations:
    """
    Apply an appropriate scaling and translation to the tumorbed annotations.

    We take the following steps:

    1. Read the tumorbed annotations from slidescore at base level.
    2. Translate the annotations to remove the slide offset at the base level.
    3. Rescale the annotations to microns per pixel (mpp) set during inference time
    so that it matches with the base resolution of prediction tiff images.

    And return the rescaled, corrected annotations using WsiAnnotations from dlup.

    Parameters
    ----------
    slide_image: SlideImage
        The pathology slide

    annotations: WsiAnnotations
        The full annotations at the base resolution

    native_mpp_at_inference: float
        The base resolution of the inference file.

    Returns
    -------
    offset_annotations: WsiAnnotations
        Annotations on the WSI rescaled to match the tiff image resolution.
    """
    slide_offset, _ = slide_image.slide_bounds
    scaling_to_native_mpp_at_inference = slide_image.get_scaling(native_mpp_at_inference)
    polygons = annotations.read_region((0, 0), scaling=native_mpp_at_inference, size=slide_image.size)
    translated_polygons = []
    affine_transform_polygons = []
    for polygon in polygons:
        translated_polygons.append(translate(polygon, -slide_offset[0], -slide_offset[1]))
    transformation_matrix = [scaling_to_native_mpp_at_inference, 0, 0, scaling_to_native_mpp_at_inference, 0, 0]
    for polygon in translated_polygons:
        affine_transform_polygons.append(affine_transform(polygon, transformation_matrix))
    final_polygons = to_geojson_format(affine_transform_polygons, label="Tumorbed")
    slide_name = slide_image.identifier.split("/")[-1].split(".mrxs")[0]
    with open(f"{slide_name}_Tumorbed.json", "w") as file:
        json.dump(final_polygons, file, indent=2)
    offset_annotations = WsiAnnotations.from_geojson(f"{slide_name}_Tumorbed.json")
    Path(f"{slide_name}_Tumorbed.json").unlink()
    return offset_annotations


def to_geojson_format(list_of_points: list, label: str) -> dict[str, Any]:
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
    properties: Dict[str, Union[str, Dict[str, str]]] = {
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
