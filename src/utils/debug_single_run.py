#!/usr/bin/env python3
"""
Debug script to show full conversation history for one iteration
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from ..experiment.prompts import PromptTemplates
from .utils import extract_confidence_and_solution, format_constraints, prepare_letters

load_dotenv()


def debug_single_run(puzzle_dir="puzzles"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load puzzle 1
    with open(f"{puzzle_dir}/puzzle_001.json") as f:
        puzzle = json.load(f)

    print(f"=== DEBUG: Single Run of Puzzle {puzzle['puzzle_id']} ===")
    print(f"Target sequence: {puzzle['target_sequence']}")
    print(f"Constraints: {puzzle['constraints']}")
    print()

    # Prepare shuffled letters (using same method as experiment.py)
    shuffled_letters = prepare_letters(puzzle["target_sequence"])

    print(f"Letters given to model (shuffled): {shuffled_letters}")
    print(f"Correct target: {puzzle['target_sequence']}")
    print()

    log = {
        "puzzle_id": puzzle["puzzle_id"],
        "target_sequence": puzzle["target_sequence"],
        "shuffled_letters": shuffled_letters,
        "constraints": puzzle["constraints"],
        "conversations": {},
        "timestamp": datetime.now().isoformat(),
    }

    # Test single-shot condition
    print("=" * 60)
    print("SINGLE-SHOT CONDITION")
    print("=" * 60)

    constraints_text = format_constraints(puzzle["constraints"])
    puzzle_prompt = PromptTemplates.get_base_puzzle(constraints_text, shuffled_letters)
    prompt = PromptTemplates.get_single_shot(puzzle_prompt)

    print("PROMPT SENT:")
    print(prompt)
    print("\n" + "-" * 40)

    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=100, timeout=30
    )

    response_text = response.choices[0].message.content
    print("MODEL RESPONSE:")
    print(response_text)
    print()

    log["conversations"]["single_shot"] = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text},
        ]
    }

    # Parse response (using same function as experiment.py)
    confidence, solution = extract_confidence_and_solution(response_text)

    print("PARSED:")
    print(f"  Confidence: {confidence}")
    print(f"  Solution: {solution}")
    print(f"  Correct: {solution == puzzle['target_sequence']}")
    print()

    # Test confidence-pre condition
    print("=" * 60)
    print("CONFIDENCE-PRE CONDITION")
    print("=" * 60)

    # Turn 1: Confidence
    confidence_prompt = PromptTemplates.get_confidence_pre(puzzle_prompt)

    print("TURN 1 - CONFIDENCE PROMPT:")
    print(confidence_prompt)
    print("\n" + "-" * 40)

    confidence_response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": confidence_prompt}], max_tokens=20
    )

    confidence_text = confidence_response.choices[0].message.content
    print("TURN 1 - MODEL RESPONSE:")
    print(confidence_text)
    print()

    # Turn 2: Solution
    solution_prompt = PromptTemplates.get_solution_request()

    print("TURN 2 - SOLUTION PROMPT:")
    print(solution_prompt)
    print("\n" + "-" * 40)

    solution_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": confidence_prompt},
            {"role": "assistant", "content": confidence_text},
            {"role": "user", "content": solution_prompt},
        ],
        max_tokens=50,
    )

    solution_text = solution_response.choices[0].message.content
    print("TURN 2 - MODEL RESPONSE:")
    print(solution_text)
    print()

    log["conversations"]["confidence_pre"] = {
        "messages": [
            {"role": "user", "content": confidence_prompt},
            {"role": "assistant", "content": confidence_text},
            {"role": "user", "content": solution_prompt},
            {"role": "assistant", "content": solution_text},
        ]
    }

    # Parse responses (using same extraction function from utils)
    confidence2, _ = extract_confidence_and_solution(confidence_text)
    _, solution2 = extract_confidence_and_solution(solution_text)

    print("PARSED:")
    print(f"  Confidence: {confidence2}")
    print(f"  Solution: {solution2}")
    print(f"  Correct: {solution2 == puzzle['target_sequence']}")
    print()

    # Test control condition
    print("=" * 60)
    print("CONTROL CONDITION")
    print("=" * 60)

    control_prompt = PromptTemplates.get_control(puzzle_prompt)

    print("CONTROL PROMPT:")
    print(control_prompt)
    print("\n" + "-" * 40)

    control_response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": control_prompt}], max_tokens=50, timeout=30
    )

    control_text = control_response.choices[0].message.content
    print("CONTROL RESPONSE:")
    print(control_text)
    print()

    log["conversations"]["control"] = {
        "messages": [
            {"role": "user", "content": control_prompt},
            {"role": "assistant", "content": control_text},
        ]
    }

    # Parse control response (should have no confidence)
    confidence3, solution3 = extract_confidence_and_solution(control_text)

    # If no formatted solution, try to extract raw answer
    if not solution3:
        import re
        letter_match = re.search(r'\b([A-L]{4,5})\b', control_text.upper())
        if letter_match:
            solution3 = letter_match.group(1)

    print("PARSED:")
    print(f"  Confidence: {confidence3} (should be None)")
    print(f"  Solution: {solution3}")
    print(f"  Correct: {solution3 == puzzle['target_sequence']}")
    print()

    # Save full log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"debug_conversation_{timestamp}.json"

    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Full conversation log saved to: {log_file}")

    return log


if __name__ == "__main__":
    debug_single_run()
