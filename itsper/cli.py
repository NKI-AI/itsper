import argparse
from pathlib import Path

from itsper.compute_itsp import itsp_computer


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ITSP over tissue segmentation masks")
    parser.add_argument("--output-path", required=True, help="Path to save output images")
    parser.add_argument("--images-path", required=True, help="Path to input images")
    parser.add_argument("--inference-path", required=True, help="Path to inference data")
    parser.add_argument("--inference-tile-size", required=True, help="The tile size used during inference")
    parser.add_argument("--inference-mpp", required=True, help="The native mpp of inference data")
    parser.add_argument("--annotation-path", required=False, help="Path to annotation data")

    args = parser.parse_args()

    # Convert string paths to Path objects
    output_path = Path(args.output_path)
    images_path = Path(args.images_path)
    inference_path = Path(args.inference_path)
    if args.annotation_path is not None:
        annotation_path = Path(args.annotation_path)
    else:
        annotation_path = None
    # This stands for the mpp used during inference time. It is the base resolution of the inference images.
    native_mpp_during_inference = float(args.inference_mpp)
    tile_size_for_inference = (int(args.inference_tile_size), int(args.inference_tile_size))

    itsp_computer(
        output_path,
        images_path,
        inference_path,
        annotation_path,
        native_mpp_during_inference,
        tile_size_for_inference,
    )


if __name__ == "__main__":
    main()
