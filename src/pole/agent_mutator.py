"""Agent-based prompt mutation with verification."""

from typing import Callable, Optional
from .constants import (
    DEFAULT_NUM_VARIATIONS,
    MIN_VARIATION_LENGTH,
    MIN_VIABLE_PROMPT_LENGTH,
    VARIATION_GENERATION_MULTIPLIER,
)


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
        self.mutator_fn = mutator_fn
        self.verifier_fn = verifier_fn
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
        instruction = self._build_mutator_instruction(prompt, context)
        raw_variations = self._generate_variations(instruction)

        verified = []
        for variation in raw_variations:
            if self._verify_variation(prompt, variation):
                verified.append(variation)

        # Fallback if all rejected
        if not verified:
            if self.on_warning:
                self.on_warning("⚠️  Verifier rejected all variations, using fallback")
            verified = [f"{prompt}\n\nBe more precise."]

        return verified[:self.num_variations]

    def _build_mutator_instruction(self, prompt: str, context: dict) -> str:
        iteration = context.get("iteration", 0)
        current_loss = context.get("current_loss", None)
        best_loss = context.get("best_loss", None)

        loss_info = ""
        if current_loss is not None and best_loss is not None:
            if current_loss > best_loss:
                loss_info = f"\nCurrent error rate: {current_loss:.3f} (best: {best_loss:.3f}) - Need significant improvement!"
            else:
                loss_info = f"\nCurrent error rate: {current_loss:.3f}"

        instruction = f"""You are a prompt engineering expert. Your task is to improve this prompt by generating {self.num_variations} better variations.

CURRENT PROMPT (preserve the core task and format):
\"\"\"
{prompt}
\"\"\"{loss_info}

TASK: Generate {self.num_variations} COMPLETE, IMPROVED versions of the above prompt.

Requirements:
- Keep the EXACT same task/goal
- Keep the same output format requirements
- Add helpful clarifications or examples
- Improve wording for clarity
- Make small, focused improvements (don't change the whole approach)

Output format: Write each complete prompt on its own, separated by blank lines.
Do NOT add labels like "Variation 1:" or explanations.
Output ONLY the {self.num_variations} complete prompts."""

        return instruction

    def _generate_variations(self, instruction: str) -> list[str]:
        response = self.mutator_fn(instruction)

        # Split by double newlines to get distinct prompts
        variations = []
        parts = response.split('\n\n')

        for part in parts:
            part = part.strip()

            if not part or len(part) < MIN_VARIATION_LENGTH:
                continue

            # Remove common prefixes
            for prefix in ['VARIATION:', 'VARIATION', '1.', '2.', '3.', '1)', '2)', '3)',
                          'Prompt:', 'Prompt 1:', 'Prompt 2:', 'Prompt 3:',
                          '**Variation', '*Variation', 'Example:']:
                if part.startswith(prefix):
                    newline_idx = part.find('\n')
                    if newline_idx != -1:
                        part = part[newline_idx+1:].strip()
                    else:
                        part = part[len(prefix):].strip()
                    break

            # Clean markdown formatting
            part = part.replace('**', '').replace('*', '').replace('```', '')

            if part and len(part) > MIN_VARIATION_LENGTH:
                variations.append(part)

        # Fallback: split by single newlines if nothing found
        if not variations:
            lines = response.split('\n')
            current = []
            for line in lines:
                line = line.strip()
                if not line:
                    if current:
                        var = '\n'.join(current).strip()
                        if len(var) > 50:
                            variations.append(var)
                        current = []
                else:
                    current.append(line)
            if current:
                var = '\n'.join(current).strip()
                if len(var) > 50:
                    variations.append(var)

        return variations[:self.num_variations * VARIATION_GENERATION_MULTIPLIER]

    def _verify_variation(self, original: str, variation: str) -> bool:
        """
        Use verifier agent to check if variation makes sense.
        Returns True if variation is good, False if garbage.
        """
        if self.verifier_fn is None:
            return True

        if not variation or len(variation) < MIN_VIABLE_PROMPT_LENGTH:
            return False

        if variation == original:
            return False

        verification_question = f"""You are a quality control agent. Verify if this prompt variation is valid and makes sense.

Original Prompt:
"{original}"

Proposed Variation:
"{variation}"

Is this variation:
1. A valid prompt (not gibberish)?
2. Related to the original task?
3. Potentially useful?

Answer ONLY with: YES or NO"""

        try:
            response = self.verifier_fn(verification_question)
            answer = response.strip().upper()
            return "YES" in answer
        except Exception as e:
            if self.on_warning:
                self.on_warning(f"⚠️  Verifier error: {e}")
            return True  # Accept on error to avoid blocking
