"""Simple deterministic prompt mutator."""

from ..constants.config import DEFAULT_NUM_VARIATIONS
from ..constants.mutation_strategies import DEFAULT_MUTATION_STRATEGIES


class SimpleMutator:
    """
    Simple deterministic prompt mutator.

    Applies small, deterministic string edits to generate variations.
    No randomness - same prompt always produces same variations.

    This is the default mutator when no custom mutator is provided.
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


# Alias for backward compatibility
DefaultMutator = SimpleMutator
