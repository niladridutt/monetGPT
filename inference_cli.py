#!/usr/bin/env python3
"""
MonetGPT Inference CLI

Unified command-line interface for LLM-based image editing inference.
"""
import argparse
import sys
import os
from inference.core import process_single_image, batch_process_images
import glob


def create_parser():
    """Create argument parser for inference CLI."""
    parser = argparse.ArgumentParser(description="MonetGPT Inference CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single image processing
    single_parser = subparsers.add_parser("single", help="Process single image")
    single_parser.add_argument("image", help="Input image path")
    single_parser.add_argument("--output", help="Output base path (without extension)")
    single_parser.add_argument(
        "--style",
        choices=["balanced", "vibrant", "retro"],
        default="balanced",
        help="Editing style",
    )
    single_parser.add_argument(
        "--inference-config",
        default="configs/inference_config.yaml",
        help="Inference config file",
    )
    single_parser.add_argument(
        "--pipeline-config",
        default="configs/pipeline_config.yaml",
        help="Pipeline config file",
    )

    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Process multiple images")
    batch_parser.add_argument("images", help="Input image directory")
    batch_parser.add_argument(
        "--output-dir", default="./inference_output", help="Output directory"
    )
    batch_parser.add_argument(
        "--style",
        choices=["balanced", "vibrant", "retro"],
        default="balanced",
        help="Editing style",
    )
    batch_parser.add_argument(
        "--inference-config",
        default="configs/inference_config.yaml",
        help="Inference config file",
    )
    batch_parser.add_argument(
        "--pipeline-config",
        default="configs/pipeline_config.yaml",
        help="Pipeline config file",
    )
    batch_parser.add_argument("--suffix", default="", help="Suffix for output files")
    batch_parser.add_argument(
        "--replace", action="store_true", help="Replace existing files"
    )

    return parser


def handle_single_command(args):
    """Handle single image processing command."""
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return False

    # Determine output path
    if args.output:
        output_base_path = args.output
    else:
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        if args.style != "balanced":
            image_name += f"_{args.style}"
        output_base_path = f"./inference_output/{image_name}"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

    try:
        print(f"Processing: {args.image}")
        final_image_path, adjustments, reasoning = process_single_image(
            args.image, output_base_path, args.style
        )

        print(f"Successfully processed: {args.image}")
        print(f"Output saved to: {final_image_path}")
        return True

    except Exception as e:
        print(f"Error processing {args.image}: {e}")
        return False


def handle_batch_command(args):
    """Handle batch processing command."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Filter existing images
    all_images = list(glob.glob(args.images + "/*"))
    valid_images = [
        img
        for img in all_images
        if img.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ]

    print(f"Processing {len(valid_images)} images...")

    try:
        results = batch_process_images(valid_images, args.output_dir, args.style)

        successful = sum(1 for result in results if result[0] is not None)
        print(f"Successfully processed {successful}/{len(valid_images)} images")
        return successful > 0

    except Exception as e:
        print(f"Error in batch processing: {e}")
        return False


def main():
    """Main CLI function."""
    parser = create_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == "single":
        success = handle_single_command(args)
    elif args.command == "batch":
        success = handle_batch_command(args)
    else:
        parser.print_help()
        return

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
