"""Agent-based prompt mutation with verification."""

from typing import Callable, Optional
from ..constants.config import DEFAULT_NUM_VARIATIONS
from ..constants.prompts import FALLBACK_VARIATION
from ..agents.mutator_agent import MutatorAgent
from ..agents.verifier_agent import VerifierAgent


class AgentMutator:
    """
    Agent-based mutator that uses an LLM to generate prompt variations,
    with another agent to verify they make sense.

    Architecture:
    1. Mutator Agent: Generates creative prompt variations
    2. Verifier Agent: Filters out nonsensical/garbage variations

    This is a headless implementation - user provides the actual LLM callables.
    """

    def __init__(
        self,
        mutator_fn: Callable[[str], str],
        verifier_fn: Optional[Callable[[str], str]] = None,
        num_variations: int = DEFAULT_NUM_VARIATIONS,
        on_warning: Optional[Callable[[str], None]] = None
    ):
        """
        Args:
            mutator_fn: Callable that takes an instruction string and returns
                       the LLM's response as a string.
            verifier_fn: Optional callable that takes a verification question string
                        and returns the LLM's yes/no answer as a string.
                        If None, skips verification.
            num_variations: Number of variations to generate per mutation
            on_warning: Optional callback for warnings (e.g., lambda msg: print(msg))
        """
        self.mutator = MutatorAgent(mutator_fn, num_variations)
        self.verifier = VerifierAgent(verifier_fn, on_warning) if verifier_fn else None
        self.num_variations = num_variations
        self.on_warning = on_warning

    def mutate(self, prompt: str, context: dict) -> list[str]:
        """
        Generate variations using mutator agent, then verify with verifier agent.

        Args:
            prompt: Current prompt to improve
            context: Dictionary with iteration, loss, etc.

        Returns:
            List of verified prompt variations
        """
        raw_variations = self.mutator.generate_variations(prompt, context)

        if self.verifier is None:
            return raw_variations[:self.num_variations]

        verified = []
        for variation in raw_variations:
            if self.verifier.verify(prompt, variation):
                verified.append(variation)

        # Fallback if all rejected
        if not verified:
            if self.on_warning:
                self.on_warning("⚠️  Verifier rejected all variations, using fallback")
            verified = [FALLBACK_VARIATION.format(prompt=prompt)]

        return verified[:self.num_variations]
