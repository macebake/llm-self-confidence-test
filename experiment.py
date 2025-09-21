#!/usr/bin/env python3
"""
Metacognitive Accuracy Experiment

Simple, clean interface for running the experiment.
"""

import argparse

from src.experiment.experiment_core import run_experiment
from src.utils.utils import parse_puzzle_ids


def main():
    parser = argparse.ArgumentParser(description="Run metacognitive accuracy experiment")
    parser.add_argument(
        "--puzzles", type=str, default="1", help="Puzzle IDs (e.g., '1' or '1-5' or '1,3,5')"
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Iterations per condition (default: 1)"
    )
    parser.add_argument("--save", action="store_true", help="Save results to results.jsonl")
    parser.add_argument("--puzzle-dir", type=str, default="puzzles", help="Puzzle directory to use (default: puzzles)")

    args = parser.parse_args()

    puzzle_ids = parse_puzzle_ids(args.puzzles)

    print("=== Metacognitive Accuracy Experiment ===")
    print(f"Puzzles: {puzzle_ids}, Iterations: {args.iterations}, Directory: {args.puzzle_dir}")

    results = run_experiment(puzzle_ids, args.iterations, args.puzzle_dir, args.save)

    if args.save:
        if results:
            print(f"\nSaved {len(results)} results to results.jsonl (written incrementally)")
        else:
            print(f"\nNo results to save")
    elif results:
        print(f"\nGenerated {len(results)} results (use --save to persist)")

    print("Experiment complete!")


if __name__ == "__main__":
    main()