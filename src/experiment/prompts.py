"""
Prompt templates for the metacognitive accuracy experiment.
"""


class PromptTemplates:
    """All prompts in one place for easy editing"""

    PROMPTS = {
        "base_puzzle": """You are solving a constraint satisfaction puzzle with the letters {shuffled_letters}.

Rules:
{constraints}

Solve this puzzle by reasoning step by step internally.
Check each candidate arrangement carefully against all rules.
Do not show your reasoning.""",

        "confidence_request": "Rate your confidence you can solve this correctly. Respond with exactly this format:\n\nCONFIDENCE: <number from 1-10, decimals allowed>",

        "solution_request": "Now provide the solution sequence. Respond with exactly this format:\n\nSOLUTION: <your answer>",

        "post_confidence_request": "Rate your confidence that you solved it correctly. Respond with exactly this format:\n\nCONFIDENCE: <number from 1-10, decimals allowed>",

        "single_shot": """{puzzle}

Output only the valid sequence(s).

Respond with EXACTLY this format (no additional text):

CONFIDENCE: <number from 1-10, decimals allowed>
SOLUTION: <your answer>""",

        "confidence_pre": """{puzzle}

{confidence_request}""",

        "solution_only": """{puzzle}

Output only the valid sequence(s).

Respond with EXACTLY this format:

SOLUTION: <your answer>""",

        "control": """{puzzle}

Output only the valid sequence(s).""",
    }

    @classmethod
    def get_base_puzzle(cls, constraints_text: str, shuffled_letters: str) -> str:
        """Create base puzzle description"""
        return cls.PROMPTS["base_puzzle"].format(
            shuffled_letters=shuffled_letters, constraints=constraints_text
        )

    @classmethod
    def get_single_shot(cls, puzzle: str) -> str:
        """Single-shot: confidence + solution in one turn"""
        return cls.PROMPTS["single_shot"].format(puzzle=puzzle)

    @classmethod
    def get_confidence_pre(cls, puzzle: str) -> str:
        """Confidence-pre: ask for confidence first"""
        return cls.PROMPTS["confidence_pre"].format(
            puzzle=puzzle, confidence_request=cls.PROMPTS["confidence_request"]
        )

    @classmethod
    def get_solution_request(cls) -> str:
        """Follow-up solution request"""
        return cls.PROMPTS["solution_request"]

    @classmethod
    def get_solution_only(cls, puzzle: str) -> str:
        """Solution-only prompt (for confidence-post first turn)"""
        return cls.PROMPTS["solution_only"].format(puzzle=puzzle)

    @classmethod
    def get_post_confidence_request(cls) -> str:
        """Post-solution confidence request"""
        return cls.PROMPTS["post_confidence_request"]

    @classmethod
    def get_control(cls, puzzle: str) -> str:
        """Control condition: just solve, no confidence language"""
        return cls.PROMPTS["control"].format(puzzle=puzzle)