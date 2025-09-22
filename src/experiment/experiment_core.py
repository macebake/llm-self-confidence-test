"""
Core experiment logic for the metacognitive accuracy experiment.
"""

import os
import time
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from .runner import ExperimentRunner
from ..utils.utils import load_puzzle, save_results, append_result
from .validation import ResultValidator

load_dotenv()


def run_experiment(puzzle_ids: List[int], iterations: int = 1, puzzle_dir: str = "puzzles", save_incremental: bool = False, model: str = "gpt-4o") -> List[Dict]:
    """Run experiment on specified puzzles"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    validator = ResultValidator(client, model)
    runner = ExperimentRunner(client, validator, model)

    conditions = ["single-shot", "confidence-pre", "confidence-post", "control"]
    all_results = []

    # Clear results file if saving incrementally
    if save_incremental:
        with open("results.jsonl", "w") as f:
            pass  # Clear the file

    for puzzle_id in puzzle_ids:
        puzzle = load_puzzle(puzzle_id, puzzle_dir)
        if not puzzle:
            print(f"Puzzle {puzzle_id} not found in {puzzle_dir}, skipping")
            continue

        print(
            f"\nPuzzle {puzzle_id}: {puzzle['target_sequence']} ({len(puzzle['constraints'])} constraints)"
        )

        for condition in conditions:
            print(f"  {condition}:", end=" ")

            for iteration in range(1, iterations + 1):
                if condition == "single-shot":
                    confidence, solution = runner.run_single_shot(
                        puzzle["constraints"], puzzle["target_sequence"]
                    )
                elif condition == "confidence-pre":
                    confidence, solution = runner.run_confidence_pre(
                        puzzle["constraints"], puzzle["target_sequence"]
                    )
                elif condition == "confidence-post":
                    confidence, solution = runner.run_confidence_post(
                        puzzle["constraints"], puzzle["target_sequence"]
                    )
                else:  # control
                    confidence, solution = runner.run_control(
                        puzzle["constraints"], puzzle["target_sequence"]
                    )

                # Validate solution
                correct = False
                if solution:
                    correct = validator.validate_solution(
                        solution, puzzle["target_sequence"], puzzle["constraints"]
                    )

                result = {
                    "puzzle_id": puzzle["puzzle_id"],
                    "condition": condition,
                    "iteration": iteration,
                    "correct": correct,
                    "confidence": confidence,
                    "proposed_solution": solution,
                    "target_solution": puzzle["target_sequence"],
                    "model": model,
                }

                all_results.append(result)

                # Save immediately if incremental saving is enabled
                if save_incremental:
                    append_result(result)

                # Progress indicator
                if correct:
                    print("✓", end="")
                else:
                    print("✗", end="")

                time.sleep(0.5)  # Rate limiting

            print()

        # Summary for this puzzle
        puzzle_results = [r for r in all_results if r["puzzle_id"] == puzzle_id]
        for condition in conditions:
            condition_results = [r for r in puzzle_results if r["condition"] == condition]
            correct_count = sum(1 for r in condition_results if r["correct"])
            avg_conf = (
                sum(r["confidence"] for r in condition_results if r["confidence"])
                / len([r for r in condition_results if r["confidence"]])
                if any(r["confidence"] for r in condition_results)
                else None
            )
            conf_label = "conf" if len(condition_results) == 1 else "avg conf"
            print(
                f"    {condition}: {correct_count}/{len(condition_results)} correct"
                + (f", {conf_label}: {avg_conf:.1f}" if avg_conf else "")
            )

    return all_results