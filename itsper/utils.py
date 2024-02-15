import csv
from pathlib import Path


def make_csv(output_path: Path) -> None:
    csv_file_path = output_path / Path("slide_details.csv")
    if not csv_file_path.is_file():
        with open(csv_file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Folder Name", "Image ID", "ITSP_AI"])  # Writing the header


def make_csv_entries(tiff_file: Path, output_path: Path, slide_id: str, itsp: float) -> None:
    slide_details = [tiff_file.parent.name, slide_id, str(itsp)]

    csv_file_path = output_path / Path("slide_details.csv")
    with open(csv_file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(slide_details)


def verify_folders(path: Path) -> bool:
    file_list = list(path.glob(f'*.mrxs'))
    if len(file_list) > 0:
        return True
    else:
        raise RuntimeError(f"No mrxs images found in {path}")


def make_directories_if_needed(folder: Path, output_path: Path) -> None:
    if not (output_path / folder.name).is_dir():
        # Make the directory if it doesn't exist in the output path
        (output_path / folder.name).mkdir()


def get_list_of_files(
    image_folder: Path, annotations_folder: Path, tiff_folder: Path
) -> tuple[list[Path], list[Path], list[Path]]:
    paths_to_images = image_folder.glob("**/*.mrxs")
    image_files = [x for x in paths_to_images if x.is_file()]

    paths_to_annotation = annotations_folder.glob("**/*.json")
    annotation_files = [x for x in paths_to_annotation if x.is_file()]

    paths_to_inference = tiff_folder.glob("**/*.tiff")
    tiff_files = [x for x in paths_to_inference if x.is_file()]

    return image_files, annotation_files, tiff_files