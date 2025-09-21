"""
Utility functions for the metacognitive accuracy experiment.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple


def load_puzzle(puzzle_id: int, puzzle_dir: str = "puzzles") -> Optional[Dict]:
    """Load a specific puzzle from file"""
    puzzle_file = f"{puzzle_dir}/puzzle_{puzzle_id:03d}.json"
    if not os.path.exists(puzzle_file):
        return None

    with open(puzzle_file) as f:
        return json.load(f)


def save_results(results: List[Dict], filename: str = "results.jsonl") -> None:
    """Save results to JSONL file"""
    with open(filename, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def append_result(result: Dict, filename: str = "results.jsonl") -> None:
    """Append single result to JSONL file"""
    with open(filename, "a") as f:
        f.write(json.dumps(result) + "\n")


def parse_puzzle_ids(puzzle_arg: str) -> List[int]:
    """Parse puzzle ID argument into list of integers"""
    puzzle_ids = []
    if "-" in puzzle_arg:
        start, end = map(int, puzzle_arg.split("-"))
        puzzle_ids = list(range(start, end + 1))
    elif "," in puzzle_arg:
        puzzle_ids = [int(x.strip()) for x in puzzle_arg.split(",")]
    else:
        puzzle_ids = [int(puzzle_arg)]
    return puzzle_ids


def format_constraints(constraints: List[str]) -> str:
    """Format constraints as numbered list"""
    return "\n".join([f"{i + 1}. {c}" for i, c in enumerate(constraints)])


def prepare_letters(target_sequence: str) -> str:
    """Shuffle letters from target sequence"""
    import random

    letters_to_use = list(set(target_sequence))
    random.shuffle(letters_to_use)
    return "".join(letters_to_use)


def extract_confidence_and_solution(response_text: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract confidence score and solution from formatted model response"""
    # Extract confidence (handle both formats: CONFIDENCE: X or <number>)
    confidence_match = re.search(
        r"(?:CONFIDENCE:\s*<?)(\d+(?:\.\d+)?)>?", response_text, re.IGNORECASE
    )
    confidence = float(confidence_match.group(1)) if confidence_match else None

    # Handle percentage format
    if confidence and confidence > 10:
        confidence = confidence / 10

    # Extract solution (handle both formats: SOLUTION: X or <answer>)
    solution_match = re.search(r"(?:SOLUTION:\s*<?)([A-L]+)>?", response_text, re.IGNORECASE)
    solution = solution_match.group(1).upper() if solution_match else None

    return confidence, solution