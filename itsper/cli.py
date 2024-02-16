import argparse
from pathlib import Path

from itsper.compute_itsp import itsp_computer


def main():
    parser = argparse.ArgumentParser(description="Compute ITSP over segmentation masks")
    parser.add_argument("--output-path", required=True, help="Path to save output images")
    parser.add_argument("--images-path", required=True, help="Path to input images")
    parser.add_argument("--image-format", required=True, help="Extention of original WSIs")
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
    native_mpp_for_inference = float(args.inference_mpp)
    tile_size_for_inference = (int(args.inference_tile_size), int(args.inference_tile_size))
    image_format = args.image_format

    itsp_computer(
        output_path, images_path, inference_path, annotation_path, image_format, native_mpp_for_inference, tile_size_for_inference
    )


if __name__ == "__main__":
    main()
