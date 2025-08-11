#!/usr/bin/env python3
"""
Main CLI for dataset operations.
"""
import argparse
import sys
from dataset import (
    create_all_datasets,
    generate_all_puzzles,
    query_puzzle1,
    query_puzzle2,
    query_puzzle3,
)


def main():
    parser = argparse.ArgumentParser(description="MonetGPT Dataset CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate puzzle configs command (Step 1)
    generate_parser = subparsers.add_parser(
        "generate", help="Generate puzzle config files (operation parameters)"
    )
    generate_parser.add_argument(
        "--config", default="configs/dataset_config.yaml", help="Config file path"
    )

    # Query LLM command (Step 2)
    query_parser = subparsers.add_parser(
        "query", help="Query LLM to add reasoning to puzzle configs"
    )
    query_parser.add_argument(
        "puzzle_type", choices=["1", "2", "3"], help="Puzzle type"
    )
    query_parser.add_argument(
        "--range_a", type=int, help="Start index", default=0
    )
    query_parser.add_argument(
        "--range_b", type=int, help="End index", default=-1
    )
    query_parser.add_argument(
        "--config", default="configs/dataset_config.yaml", help="Config file path"
    )

    # Create datasets command (Step 3)
    create_parser = subparsers.add_parser(
        "create", help="Create LLM training datasets from configs and reasoning"
    )
    create_parser.add_argument(
        "--config", default="configs/dataset_config.yaml", help="Config file path"
    )

    args = parser.parse_args()

    if args.command == "generate":
        print("Step 1: Generating puzzle config files (operation parameters)...")
        generate_all_puzzles(args.config)
        print("✓ Config files generated. Next: run pipeline cli to generate the puzzle images.")

    elif args.command == "query":
        print(
            f"Step 2: Querying LLM for puzzle {args.puzzle_type} reasoning (range {args.range_a}-{args.range_b})..."
        )
        if args.puzzle_type == "1":
            query_puzzle1(args.range_a, args.range_b, args.config)
        elif args.puzzle_type == "2":
            query_puzzle2(args.range_a, args.range_b, args.config)
        elif args.puzzle_type == "3":
            query_puzzle3(args.range_a, args.range_b, args.config)
        print(
            "✓ LLM reasoning added. Next: run 'create' to generate training datasets."
        )

    elif args.command == "create":
        print("Step 3: Creating LLM training datasets from configs and reasoning...")
        create_all_datasets(args.config)
        print("✓ Training datasets created in JSON format.")

    else:
        parser.print_help()
        print("\nWorkflow:")
        print("1. generate  - Create puzzle config files with operation parameters")
        print("2. query     - Add LLM reasoning to configs")
        print("3. create    - Convert to training dataset JSON format")
        sys.exit(1)


if __name__ == "__main__":
    main()
