"""Agent-based prompt mutation with verification."""

from typing import Callable, Optional


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
        num_variations: int = 3
    ):
        """
        Args:
            mutator_fn: Callable that takes an instruction string and returns 
                       the LLM's response as a string.
                       Example: lambda instruction: ollama.chat(...)["message"]["content"]
            
            verifier_fn: Optional callable that takes a verification question string
                        and returns the LLM's yes/no answer as a string.
                        Example: lambda question: ollama.chat(...)["message"]["content"]
                        If None, skips verification.
            
            num_variations: Number of variations to generate per mutation
        """
        self.mutator_fn = mutator_fn
        self.verifier_fn = verifier_fn
        self.num_variations = num_variations
    
    def mutate(self, prompt: str, context: dict) -> list[str]:
        """
        Generate variations using mutator agent, then verify with verifier agent.
        
        Args:
            prompt: Current prompt to improve
            context: Dictionary with iteration, loss, etc.
            
        Returns:
            List of verified prompt variations
        """
        # Build instruction for mutator agent
        instruction = self._build_mutator_instruction(prompt, context)
        
        # Generate variations with mutator agent
        raw_variations = self._generate_variations(instruction)
        
        # Verify each variation
        verified = []
        for variation in raw_variations:
            if self._verify_variation(prompt, variation):
                verified.append(variation)
        
        # Fallback if all rejected
        if not verified:
            print("⚠️  Verifier rejected all variations, using fallback")
            verified = [f"{prompt}\n\nBe more precise."]
        
        return verified[:self.num_variations]
    
    def _build_mutator_instruction(self, prompt: str, context: dict) -> str:
        """Build the instruction for the mutator agent."""
        iteration = context.get("iteration", 0)
        current_loss = context.get("current_loss", None)
        best_loss = context.get("best_loss", None)
        
        loss_info = ""
        if current_loss is not None and best_loss is not None:
            if current_loss > best_loss:
                loss_info = f"\n\nCurrent performance is worse. Loss: {current_loss:.3f} (best: {best_loss:.3f})"
            else:
                loss_info = f"\n\nCurrent performance. Loss: {current_loss:.3f}"
        
        instruction = f"""You are a prompt engineering expert. Your task is to generate {self.num_variations} improved variations of this prompt.

Current Prompt:
"{prompt}"

Iteration: {iteration}{loss_info}

Generate {self.num_variations} variations that are:
1. Clearer and more specific
2. More likely to produce accurate results
3. Different from each other (explore different strategies)

Guidelines:
- Keep the core task the same
- Add clarifying instructions
- Improve structure and formatting
- Make it more actionable

Output ONLY the {self.num_variations} variations, one per line, starting with "VARIATION:"
Do not include explanations or numbering."""
        
        return instruction
    
    def _generate_variations(self, instruction: str) -> list[str]:
        """Call the mutator agent to generate variations."""
        response = self.mutator_fn(instruction)
        
        # Parse variations from response
        variations = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('VARIATION:'):
                variation = line.replace('VARIATION:', '').strip()
                if variation:
                    variations.append(variation)
        
        # Fallback: treat each non-empty line as a variation
        if not variations:
            variations = [
                line.strip() 
                for line in response.split('\n') 
                if line.strip() and len(line.strip()) > 10
            ]
        
        return variations[:self.num_variations * 2]  # Generate extra for filtering
    
    def _verify_variation(self, original: str, variation: str) -> bool:
        """
        Use verifier agent to check if variation makes sense.
        
        Returns True if variation is good, False if garbage.
        """
        # Skip verification if no verifier
        if self.verifier_fn is None:
            return True
        
        # Basic sanity checks first
        if not variation or len(variation) < 5:
            return False
        
        if variation == original:
            return False
        
        # Build verification question
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
        
        # Call verifier agent
        try:
            response = self.verifier_fn(verification_question)
            answer = response.strip().upper()
            return "YES" in answer
        except Exception as e:
            print(f"⚠️  Verifier error: {e}")
            return True  # Accept on error to avoid blocking

