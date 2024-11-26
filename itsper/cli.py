import argparse
from pathlib import Path

from itsper.compute_itsp import itsp_computer
from itsper.io import display_launch_graphic
from itsper.qar import qar_surface_plotter, simulate_segmentation_errors


def main() -> None:
    display_launch_graphic()

    parser = argparse.ArgumentParser(description="Compute ITSP or run QAR simulation over tissue segmentation masks")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Subparser for ITSP generation
    itsp_parser = subparsers.add_parser("itsp", help="Generate ITSP from tissue segmentation masks")
    itsp_parser.add_argument("--manifest-path", required=True, help="Path to the ITSP manifest file")
    itsp_parser.add_argument("--render-images", required=False, help="Render images", default=1, type=int)
    itsp_parser.add_argument("--output-path", required=True, help="Path to save the ITSP outputs")
    itsp_parser.add_argument("--images-root", required=True, help="Path to the images root")
    itsp_parser.add_argument("--annotations-root", required=True, help="Path to the annotations root")
    itsp_parser.add_argument("--inference-root", required=True, help="Path to the inference root")

    qar_parser = subparsers.add_parser("qar", help="Run Quantification Ambiguity Range (QAR) simulation")
    qar_parser.add_argument("--output-path", required=True, help="Path to save the QAR outputs")
    qar_parser.add_argument(
        "--simulation-type",
        required=True,
        choices=["simulate-segmentation-errors", "plot-qar-surface"],
        help="Type of QAR simulation: simulate segmentation errors or plot the QAR surface",
    )

    qar_parser.add_argument(
        "--tumor-dice", type=float, help="Dice coefficient for tumor region (required for plotting QAR surface)"
    )
    qar_parser.add_argument(
        "--stroma-dice", type=float, help="Dice coefficient for stroma region (required for plotting QAR surface)"
    )

    args = parser.parse_args()

    if args.mode == "itsp":
        # Set up paths and parameters based on mode
        manifest_path = Path(args.manifest_path)
        output_path = Path(args.output_path)
        images_root = Path(args.images_root)
        annotations_root = Path(args.annotations_root)
        inference_root = Path(args.inference_root)
        render_images = args.render_images
        itsp_computer(manifest_path, images_root, annotations_root, inference_root, output_path, render_images)

    elif args.mode == "simulate_qar":
        output_path = Path(args.output_path)
        if args.simulation_type == "simulate-segmentation-errors":
            simulate_segmentation_errors(output_path)

        elif args.simulation_type == "plot-qar-surface":
            if args.tumor_dice is None or args.stroma_dice is None:
                parser.error("tumor_dice and stroma_dice are required for plotting the QAR surface")
            qar_surface_plotter(args.tumor_dice, args.stroma_dice, output_path)


if __name__ == "__main__":
    main()
