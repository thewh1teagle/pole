"""Agent for generating prompt variations."""

from typing import Callable
from ..constants.config import MIN_VARIATION_LENGTH, VARIATION_GENERATION_MULTIPLIER
from ..constants.prompts import MUTATOR_INSTRUCTION_TEMPLATE


class MutatorAgent:
    """Agent that generates prompt variations using an LLM."""

    def __init__(self, mutator_fn: Callable[[str], str], num_variations: int):
        """
        Args:
            mutator_fn: Callable that takes an instruction and returns LLM response
            num_variations: Number of variations to generate
        """
        self.mutator_fn = mutator_fn
        self.num_variations = num_variations

    def generate_variations(self, prompt: str, context: dict) -> list[str]:
        """
        Generate prompt variations.

        Args:
            prompt: Current prompt to improve
            context: Dictionary with iteration, loss, etc.

        Returns:
            List of prompt variations
        """
        instruction = self._build_instruction(prompt, context)
        response = self.mutator_fn(instruction)
        return self._parse_response(response)

    def _build_instruction(self, prompt: str, context: dict) -> str:
        """Build the instruction for the mutator agent."""
        current_loss = context.get("current_loss", None)
        best_loss = context.get("best_loss", None)

        loss_info = ""
        if current_loss is not None and best_loss is not None:
            if current_loss > best_loss:
                loss_info = f"\nCurrent error rate: {current_loss:.3f} (best: {best_loss:.3f}) - Need significant improvement!"
            else:
                loss_info = f"\nCurrent error rate: {current_loss:.3f}"

        return MUTATOR_INSTRUCTION_TEMPLATE.format(
            num_variations=self.num_variations,
            prompt=prompt,
            loss_info=loss_info
        )

    def _parse_response(self, response: str) -> list[str]:
        """Parse the LLM response to extract prompt variations."""
        parts = response.split('---')
        variations = []

        for part in parts:
            part = part.strip()
            if part and len(part) > MIN_VARIATION_LENGTH:
                variations.append(part)

        return variations[:self.num_variations * VARIATION_GENERATION_MULTIPLIER]
