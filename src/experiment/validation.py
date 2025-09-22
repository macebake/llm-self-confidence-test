"""
Result validation module for the metacognitive accuracy experiment.
"""

from typing import List
from openai import OpenAI


class ResultValidator:
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o"):
        self.client = openai_client
        self.model = model

    def primary_check(self, proposed: str, target: str) -> bool:
        """Primary validation: exact match with target sequence"""
        return proposed.strip().upper() == target.strip().upper()

    def secondary_check(self, proposed: str, constraints: List[str], target: str) -> bool:
        """Secondary validation: GPT reasoning check if primary fails"""
        constraints_text = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(constraints)])

        prompt = f"""Check if this solution satisfies the constraints. Think step by step.

Constraints:
{constraints_text}

Proposed solution: {proposed}
Correct solution example: {target}

Does the proposed solution satisfy all constraints? Answer: Yes/No"""

        try:
            # GPT-5 has different parameters and needs reasoning_effort for speed
            if self.model.startswith("gpt-5"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"reasoning_effort": "minimal"},  # Fast responses for validation
                    timeout=120
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=100, timeout=60
                )

            result = response.choices[0].message.content.strip().lower()
            return "yes" in result and "no" not in result

        except Exception as e:
            print(f"Error in secondary validation: {e}")
            return False

    def validate_solution(self, proposed: str, target: str, constraints: List[str]) -> bool:
        """Two-tier validation system"""
        if self.primary_check(proposed, target):
            return True
        return self.secondary_check(proposed, constraints, target)