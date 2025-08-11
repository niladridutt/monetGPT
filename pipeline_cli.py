#!/usr/bin/env python3
"""
Main CLI for image editing pipeline operations.
"""
import argparse
import sys
from pipeline import (
    ImageEditingPipeline,
    PuzzleImageGenerator,
    BatchProcessor,
    load_pipeline_config,
)


def main():
    parser = argparse.ArgumentParser(description="MonetGPT Image Editing Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single edit command
    edit_parser = subparsers.add_parser("edit", help="Execute single image edit")
    edit_parser.add_argument("config_path", help="Path to JSON config file")
    edit_parser.add_argument("src_image", help="Source image path")
    edit_parser.add_argument("output_path", help="Output image path")
    edit_parser.add_argument(
        "--config", default="configs/pipeline_config.yaml", help="Pipeline config file"
    )

    # Puzzle image generation
    puzzle_parser = subparsers.add_parser("puzzle", help="Generate puzzle images")
    puzzle_parser.add_argument(
        "puzzle_type", choices=["1", "2", "3", "all"], help="Puzzle type to generate"
    )
    puzzle_parser.add_argument(
        "--config", default="configs/pipeline_config.yaml", help="Pipeline config file"
    )

    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Execute batch processing")
    batch_parser.add_argument(
        "batch_type", choices=["predictions", "monetgpt"], help="Batch processing type"
    )
    batch_parser.add_argument(
        "--target-editor", default="a", help="Target editor identifier"
    )
    batch_parser.add_argument(
        "--config", default="configs/pipeline_config.yaml", help="Pipeline config file"
    )

    args = parser.parse_args()

    if args.command == "edit":
        print(
            f"Executing edit: {args.config_path} -> {args.src_image} -> {args.output_path}"
        )
        pipeline = ImageEditingPipeline(args.config)
        pipeline.execute_edit(args.config_path, args.src_image, args.output_path)
        print("Edit completed successfully!")

    elif args.command == "puzzle":
        print(f"Generating puzzle images for puzzle {args.puzzle_type}...")
        generator = PuzzleImageGenerator(args.config)

        if args.puzzle_type == "1":
            generator.generate_puzzle1_images()
        elif args.puzzle_type == "2":
            generator.generate_puzzle2_images()
        elif args.puzzle_type == "3":
            generator.generate_puzzle3_images()
        elif args.puzzle_type == "all":
            generator.generate_all_puzzles()

        print("Puzzle image generation completed!")

    elif args.command == "batch":
        print(f"Executing batch processing: {args.batch_type}")
        processor = BatchProcessor(args.config)

        if args.batch_type == "predictions":
            processor.execute_batch_predictions(args.target_editor)
        elif args.batch_type == "monetgpt":
            processor.execute_monetgpt_predictions(args.target_editor)

        print("Batch processing completed!")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
