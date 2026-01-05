"""Base protocol for prompt mutators."""

from typing import Protocol


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
