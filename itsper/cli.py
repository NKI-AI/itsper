import argparse
from pathlib import Path

from itsper.compute_itsp import itsp_computer
from itsper.io import display_launch_graphic


def main() -> None:
    display_launch_graphic()
    parser = argparse.ArgumentParser(description="Compute ITSP over tissue segmentation masks")
    parser.add_argument("--manifest-path", required=True, help="Path to the itsp manifest file")
    parser.add_argument("--render-images", required=False, help="Render images", default=1)
    parser.add_argument("--output-path", required=True, help="Path to save the outputs")
    parser.add_argument("--images-root", required=True, help="Path to the images root")
    parser.add_argument("--annotations-root", required=True, help="Path to the annotations root")
    parser.add_argument("--inference-root", required=True, help="Path to the inference root")

    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    render_images = args.render_images
    output_path = Path(args.output_path)
    images_root = Path(args.images_root)
    annotations_root = Path(args.annotations_root)
    inference_root = Path(args.inference_root)

    itsp_computer(manifest_path, images_root, annotations_root, inference_root, output_path, render_images)


if __name__ == "__main__":
    main()
