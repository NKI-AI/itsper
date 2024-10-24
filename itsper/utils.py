import csv
from pathlib import Path
from typing import Any, Optional

from dlup.annotations import WsiAnnotations


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
