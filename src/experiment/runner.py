"""
Experiment runner for the metacognitive accuracy experiment.
"""

from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from .prompts import PromptTemplates
from ..utils.utils import extract_confidence_and_solution, format_constraints, prepare_letters
from .validation import ResultValidator


class ExperimentRunner:
    def __init__(self, openai_client: OpenAI, validator: ResultValidator):
        self.client = openai_client
        self.validator = validator

    def _make_api_call(self, messages: List[Dict], max_tokens: int = 100) -> Optional[str]:
        """Make API call with error handling"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=max_tokens, timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call error: {e}")
            return None

    def _create_base_puzzle(self, constraints: List[str], shuffled_letters: str) -> str:
        """Create base puzzle description"""
        constraints_text = format_constraints(constraints)
        return PromptTemplates.get_base_puzzle(constraints_text, shuffled_letters)

    def run_single_shot(
        self, constraints: List[str], target_sequence: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Condition A: single-shot"""
        shuffled_letters = prepare_letters(target_sequence)
        puzzle = self._create_base_puzzle(constraints, shuffled_letters)
        prompt = PromptTemplates.get_single_shot(puzzle)

        response_text = self._make_api_call([{"role": "user", "content": prompt}])
        if response_text:
            return extract_confidence_and_solution(response_text)
        return None, None

    def run_confidence_pre(
        self, constraints: List[str], target_sequence: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Condition B: confidence-pre"""
        shuffled_letters = prepare_letters(target_sequence)
        puzzle = self._create_base_puzzle(constraints, shuffled_letters)

        # Turn 1: Get confidence
        confidence_prompt = PromptTemplates.get_confidence_pre(puzzle)

        confidence_text = self._make_api_call(
            [{"role": "user", "content": confidence_prompt}], max_tokens=20
        )
        if not confidence_text:
            return None, None

        confidence, _ = extract_confidence_and_solution(confidence_text)

        # Turn 2: Get solution
        solution_prompt = PromptTemplates.get_solution_request()

        solution_text = self._make_api_call(
            [
                {"role": "user", "content": confidence_prompt},
                {"role": "assistant", "content": confidence_text},
                {"role": "user", "content": solution_prompt},
            ],
            max_tokens=50,
        )

        if solution_text:
            _, solution = extract_confidence_and_solution(solution_text)
            return confidence, solution

        return confidence, None

    def run_confidence_post(
        self, constraints: List[str], target_sequence: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Condition C: confidence-post"""
        shuffled_letters = prepare_letters(target_sequence)
        puzzle = self._create_base_puzzle(constraints, shuffled_letters)

        # Turn 1: Get solution
        solution_prompt = PromptTemplates.get_solution_only(puzzle)

        solution_text = self._make_api_call(
            [{"role": "user", "content": solution_prompt}], max_tokens=50
        )
        if not solution_text:
            return None, None

        _, solution = extract_confidence_and_solution(solution_text)

        # Turn 2: Get confidence
        confidence_prompt = PromptTemplates.get_post_confidence_request()

        confidence_text = self._make_api_call(
            [
                {"role": "user", "content": solution_prompt},
                {"role": "assistant", "content": solution_text},
                {"role": "user", "content": confidence_prompt},
            ],
            max_tokens=20,
        )

        if confidence_text:
            confidence, _ = extract_confidence_and_solution(confidence_text)
            return confidence, solution

        return None, solution

    def run_control(
        self, constraints: List[str], target_sequence: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Condition D: control (no confidence language at all)"""
        shuffled_letters = prepare_letters(target_sequence)
        puzzle = self._create_base_puzzle(constraints, shuffled_letters)
        prompt = PromptTemplates.get_control(puzzle)

        response_text = self._make_api_call([{"role": "user", "content": prompt}])
        if response_text:
            # For control, we'll try to extract solution directly from response
            # No confidence should be present, but we'll parse anyway
            confidence, solution = extract_confidence_and_solution(response_text)

            # If no formatted solution found, try to extract the raw answer
            if not solution:
                # Look for letter sequences in the response
                import re
                letter_match = re.search(r'\b([A-L]{4,5})\b', response_text.upper())
                if letter_match:
                    solution = letter_match.group(1)

            return None, solution  # Control never has confidence
        return None, None