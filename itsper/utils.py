import csv
from pathlib import Path
from typing import Any, Optional

from dlup.annotations import WsiAnnotations

from itsper.types import ItsperAnnotationExtensions, ItsperInferenceExtensions, ItsperWsiExtensions


def make_csv(output_path: Path) -> None:
    csv_file_path = output_path / Path("slide_details.csv")
    if not csv_file_path.is_file():
        with open(csv_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Folder Name",
                    "Image ID",
                    "ITSP_HUMAN",
                    "ITSP_AI",
                    "Total stroma pixels",
                    "Total tumor pixels",
                    "Total other pixels",
                ]
            )  # Writing the header


def check_if_roi_is_present(sample: dict[str, Any]) -> WsiAnnotations | None:
    if sample.get("annotation_data", None) is not None:
        roi = sample["annotation_data"]["roi"]
    else:
        roi = None
    return roi


def make_csv_entries(
    inference_file: Path,
    output_path: Path,
    slide_id: str,
    human_itsp: Optional[float],
    ai_itsp: float,
    total_tumor: float,
    total_stroma: float,
    total_others: float,
) -> None:
    slide_details = [
        inference_file.parent.name,
        slide_id,
        str(human_itsp),
        str(ai_itsp),
        str(total_tumor),
        str(total_stroma),
        str(total_others),
    ]

    csv_file_path = output_path / Path("slide_details.csv")
    with open(csv_file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(slide_details)


def make_directories_if_needed(folder: Path, output_path: Path) -> None:
    if not (output_path / folder.name).is_dir():
        # Make the directory if it doesn't exist in the output path
        (output_path / folder.name).mkdir()


def get_list_of_files(folder_dictionary: dict[str, Any]) -> tuple[list[Path], list[Path] | None, list[Path]]:
    images_path = folder_dictionary["images_path"]
    image_format = folder_dictionary["image_format"].value
    inference_folder = folder_dictionary["inference_path"]
    inference_format = folder_dictionary["inference_image_format"].value
    annotations_folder = folder_dictionary.get("annotations_path", None)
    annotations_format = folder_dictionary.get("annotation_format", None)

    paths_to_images = images_path.glob(f"*{ItsperWsiExtensions(image_format).value}")
    image_files = [x for x in paths_to_images if x.is_file()]

    if annotations_folder is not None:
        paths_to_annotation = annotations_folder.glob(f"**/*{ItsperAnnotationExtensions(annotations_format).value}")
        annotation_files = [x for x in paths_to_annotation if x.is_file()]
    else:
        annotation_files = None

    paths_to_inference = inference_folder.glob(f"**/*.{ItsperInferenceExtensions(inference_format).value}")
    inference_files = [x for x in paths_to_inference if x.is_file()]

    return image_files, annotation_files, inference_files


def get_image_format(images_path: Path) -> ItsperWsiExtensions:
    # Create a dictionary to store the files for each format in the enum
    images_dict = {format: list(images_path.glob(f"**/*{format.value}")) for format in ItsperWsiExtensions}
    # Filter out the formats where no files were found
    valid_images = {format: files for format, files in images_dict.items() if files}
    if len(valid_images) > 1:
        # More than one image format found, raise an error
        raise ValueError(f"Multiple image formats found: {', '.join([format.name for format in valid_images])}")
    elif len(valid_images) == 0:
        # No image formats found, raise an error
        raise ValueError("No recognized image formats found.")
    else:
        # Exactly one image format found, return it
        return next(iter(valid_images))


def get_annotation_format(path_to_annotation_files: Path) -> ItsperAnnotationExtensions:
    # Create a dictionary to store the files for each format in the enum
    annotation_dict = {
        format: list(path_to_annotation_files.glob(f"**/*{format.value}")) for format in ItsperAnnotationExtensions
    }
    # Filter out the formats where no files were found
    valid_annotations = {format: files for format, files in annotation_dict.items() if files}
    if len(valid_annotations) > 1:
        # More than one annotation format found, raise an error
        raise ValueError(
            f"Multiple annotation formats found: {', '.join([format.name for format in valid_annotations])}"
        )
    elif len(valid_annotations) == 0:
        # No annotation formats found, raise an error
        raise ValueError("No recognized annotation formats found.")
    else:
        # Exactly one annotation format found, return it
        return next(iter(valid_annotations))


def get_inference_image_format(inference_path: Path) -> ItsperInferenceExtensions:
    # Create a dictionary to store the files for each format in the enum
    inference_dict = {format: list(inference_path.glob(f"**/*{format.value}")) for format in ItsperInferenceExtensions}
    # Filter out the formats where no files were found
    valid_inference = {format: files for format, files in inference_dict.items() if files}
    if len(valid_inference) > 1:
        # More than one inference format found, raise an error
        raise ValueError(f"Multiple inference formats found: {', '.join([format.name for format in valid_inference])}")
    elif len(valid_inference) == 0:
        # No inference formats found, raise an error
        raise ValueError("No recognized inference formats found.")
    else:
        # Exactly one inference format found, return it
        return next(iter(valid_inference))


def check_integrity_of_files(
    images_path: Path, inference_path: Path, annotations_path: Optional[Path]
) -> dict[str, Any]:
    image_format = get_image_format(images_path)
    if annotations_path is not None:
        annotation_format = get_annotation_format(annotations_path)
    else:
        annotation_format = None
    inference_image_format = get_inference_image_format(inference_path)

    folder_dictionary = {
        "images_path": images_path,
        "annotations_path": annotations_path,
        "inference_path": inference_path,
        "image_format": image_format,
        "annotation_format": annotation_format,
        "inference_image_format": inference_image_format,
    }
    return folder_dictionary
