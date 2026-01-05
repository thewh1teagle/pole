"""Agent for verifying prompt variations."""

from typing import Callable, Optional
from ..constants.config import MIN_VIABLE_PROMPT_LENGTH
from ..constants.prompts import VERIFIER_INSTRUCTION_TEMPLATE


class VerifierAgent:
    """Agent that verifies if prompt variations are valid."""

    def __init__(
        self,
        verifier_fn: Callable[[str], str],
        on_warning: Optional[Callable[[str], None]] = None
    ):
        """
        Args:
            verifier_fn: Callable that takes a verification question and returns answer
            on_warning: Optional callback for warnings
        """
        self.verifier_fn = verifier_fn
        self.on_warning = on_warning

    def verify(self, original: str, variation: str) -> bool:
        """
        Verify if a variation is valid.

        Args:
            original: Original prompt
            variation: Proposed variation

        Returns:
            True if variation is valid, False otherwise
        """
        if not variation or len(variation) < MIN_VIABLE_PROMPT_LENGTH:
            return False

        if variation == original:
            return False

        verification_question = VERIFIER_INSTRUCTION_TEMPLATE.format(
            original=original,
            variation=variation
        )

        try:
            response = self.verifier_fn(verification_question)
            answer = response.strip().upper()
            return "YES" in answer
        except Exception as e:
            if self.on_warning:
                self.on_warning(f"⚠️  Verifier error: {e}")
            return True  # Accept on error to avoid blocking
