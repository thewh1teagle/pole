"""Prompt mutation strategies."""

from typing import Protocol
from .constants import DEFAULT_NUM_VARIATIONS, DEFAULT_MUTATION_STRATEGIES


class PromptMutator(Protocol):
    """Interface for prompt mutation strategies."""

    def mutate(self, prompt: str, context: dict) -> list[str]:
        """
        Generate variations of the given prompt.

        Args:
            prompt: The current prompt string
            context: Dictionary with metadata (iteration, loss, etc.)

        Returns:
            List of mutated prompt candidates
        """
        ...


class DefaultMutator:
    """
    Simple deterministic prompt mutator.

    Applies small, deterministic string edits to generate variations.
    No randomness - same prompt always produces same variations.
    """

    def __init__(self, num_variations: int = DEFAULT_NUM_VARIATIONS):
        self.num_variations = num_variations

    def mutate(self, prompt: str, context: dict) -> list[str]:
        variations = []

        # Strategy 1: Add clarifying instructions
        variations.append(f"{prompt}\n\n{DEFAULT_MUTATION_STRATEGIES[0]}")

        # Strategy 2: Add output format constraint
        variations.append(f"{prompt}\n\n{DEFAULT_MUTATION_STRATEGIES[1]}")

        # Strategy 3: Rephrase with emphasis
        variations.append(f"Task: {prompt}\n\n{DEFAULT_MUTATION_STRATEGIES[2]}")

        # Strategy 4: Add step-by-step instruction
        variations.append(f"{prompt}\n\n{DEFAULT_MUTATION_STRATEGIES[3]}")

        # Strategy 5: Simplify
        simplified = prompt.split('.')[0] if '.' in prompt else prompt
        variations.append(simplified.strip())

        return variations[:self.num_variations]
