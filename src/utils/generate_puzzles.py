#!/usr/bin/env python3
"""
Puzzle Generation Script

Generates the 100 puzzles for the metacognitive accuracy experiment.
Run this first to create puzzles that can be inspected before running experiments.
"""

import argparse
import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
# No import needed - PuzzleGenerator defined below
from openai import OpenAI

load_dotenv()


class PuzzleGenerator:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.letters = list("ABCDEFGHIJKL")

    def generate_sequence(self, length: int = 5) -> str:
        """Generate a random sequence of 5 unique letters"""
        import random
        selected_letters = random.sample(self.letters, length)
        return "".join(selected_letters)

    def generate_constraints(self, target_sequence: str) -> list[str]:
        """Use GPT to generate logical constraints for the target sequence"""
        prompt = f"""Given this target sequence: {target_sequence}, generate 4-5 logical constraints that would uniquely determine this arrangement.

Use these types of constraints as inspiration but create novel combinations:
- Ordering: "X must come before Y"
- Adjacency: "X cannot be adjacent to Y"
- Position: "X must be in an odd/even position"
- Distance: "X must be exactly N positions from Y"

Format as a numbered list. Ensure constraints are sufficient to uniquely determine the sequence."""

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    timeout=30
                )

                return self._parse_constraints(response.choices[0].message.content)

            except Exception as e:
                print(f"Error generating constraints (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        return []

    def _parse_constraints(self, response_text: str) -> list[str]:
        """Parse numbered constraint list from GPT response"""
        constraints = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                constraint = line.split('.', 1)[-1].strip()
                if constraint:
                    constraints.append(constraint)
        return constraints

    def create_puzzle(self, puzzle_id: int, length: int = 5) -> dict:
        """Create a complete puzzle with sequence and constraints"""
        target_sequence = self.generate_sequence(length)
        constraints = self.generate_constraints(target_sequence)

        if not constraints:
            return None

        return {
            "puzzle_id": puzzle_id,
            "target_sequence": target_sequence,
            "constraints": constraints,
            "generated_by": "gpt-4o",
            "timestamp": datetime.now().isoformat()
        }


def generate_puzzles(num_puzzles: int = 100, puzzle_dir: str = "puzzles", length: int = 5):
    """Generate puzzles and save them to files"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    generator = PuzzleGenerator(client)

    # Create puzzles directory if it doesn't exist
    os.makedirs(puzzle_dir, exist_ok=True)

    print(f"Generating {num_puzzles} puzzles...")
    puzzles = []
    failed_puzzles = 0

    for i in range(1, num_puzzles + 1):
        puzzle_file = f"{puzzle_dir}/puzzle_{i:03d}.json"

        # Skip if already exists (resume functionality)
        if os.path.exists(puzzle_file):
            print(f"Puzzle {i}/{num_puzzles}... SKIP (already exists)")
            try:
                with open(puzzle_file) as f:
                    puzzle = json.load(f)
                    puzzles.append(puzzle)
            except:
                print(f"  Warning: Could not read existing puzzle {i}, will regenerate")
                os.remove(puzzle_file)
            else:
                continue

        print(f"Generating puzzle {i}/{num_puzzles}...", end=" ")

        try:
            puzzle = generator.create_puzzle(i, length)

            if not puzzle["constraints"]:
                print("FAILED (no constraints)")
                failed_puzzles += 1
                continue

            # Save puzzle to file
            with open(puzzle_file, "w") as f:
                json.dump(puzzle, f, indent=2)

            puzzles.append(puzzle)
            print("SUCCESS")

        except Exception as e:
            print(f"ERROR: {e}")
            failed_puzzles += 1
            continue

        time.sleep(0.5)  # Rate limiting

    print("\n=== Generation Complete ===")
    print(f"Successfully generated: {len(puzzles)} puzzles")
    print(f"Failed: {failed_puzzles} puzzles")
    print(f"Puzzles saved to: {puzzle_dir}/")

    # Create summary file
    summary = {
        "total_generated": len(puzzles),
        "failed": failed_puzzles,
        "generated_at": datetime.now().isoformat(),
        "puzzles": [
            {
                "id": p["puzzle_id"],
                "sequence": p["target_sequence"],
                "constraint_count": len(p["constraints"]),
            }
            for p in puzzles
        ],
    }

    with open(f"{puzzle_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {puzzle_dir}/summary.json")

    return puzzles


def inspect_puzzles(puzzle_dir: str = "puzzles"):
    """Show a preview of generated puzzles"""
    try:
        with open(f"{puzzle_dir}/summary.json") as f:
            summary = json.load(f)

        print(f"\n=== Puzzle Overview ({puzzle_dir}) ===")
        print(f"Total puzzles: {summary['total_generated']}")
        print(f"Generated at: {summary['generated_at']}")
        print()

        print("Sample puzzles:")
        for i, puzzle_info in enumerate(summary["puzzles"][:5]):  # Show first 5
            puzzle_file = f"{puzzle_dir}/puzzle_{puzzle_info['id']:03d}.json"
            with open(puzzle_file) as f:
                puzzle = json.load(f)

            print(f"\nPuzzle {puzzle['puzzle_id']}:")
            print(f"  Target: {puzzle['target_sequence']}")
            print(f"  Constraints ({len(puzzle['constraints'])}):")
            for j, constraint in enumerate(puzzle["constraints"][:3], 1):  # Show first 3
                print(f"    {j}. {constraint}")
            if len(puzzle["constraints"]) > 3:
                print(f"    ... and {len(puzzle['constraints']) - 3} more")

    except FileNotFoundError:
        print("No puzzles found. Run with --generate first.")


def main():
    parser = argparse.ArgumentParser(description="Generate puzzles for metacognitive experiment")
    parser.add_argument("--generate", action="store_true", help="Generate new puzzles")
    parser.add_argument("--inspect", action="store_true", help="Inspect existing puzzles")
    parser.add_argument("--count", type=int, default=100, help="Number of puzzles to generate")
    parser.add_argument("--dir", type=str, default="puzzles", help="Directory to save puzzles in")
    parser.add_argument("--length", type=int, default=5, help="Length of puzzle sequences")

    args = parser.parse_args()

    if args.generate:
        generate_puzzles(args.count, args.dir, args.length)
    elif args.inspect:
        inspect_puzzles(args.dir)
    else:
        # Default behavior - check if puzzles exist, if not generate them
        if os.path.exists(f"{args.dir}/summary.json"):
            print(
                f"Puzzles already exist in {args.dir}/. Use --inspect to view them or --generate to create new ones."
            )
            inspect_puzzles(args.dir)
        else:
            print(f"No puzzles found in {args.dir}/. Generating...")
            generate_puzzles(args.count, args.dir, args.length)


if __name__ == "__main__":
    main()
