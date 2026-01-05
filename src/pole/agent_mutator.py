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
                loss_info = f"\nCurrent error rate: {current_loss:.3f} (best: {best_loss:.3f}) - Need significant improvement!"
            else:
                loss_info = f"\nCurrent error rate: {current_loss:.3f}"
        
        instruction = f"""You are a prompt engineering expert. Your task is to improve this prompt by generating {self.num_variations} better variations.

CURRENT PROMPT (preserve the core task and format):
\"\"\"
{prompt}
\"\"\"
{loss_info}

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
        """Call the mutator agent to generate variations."""
        response = self.mutator_fn(instruction)
        
        # Split by double newlines to get distinct prompts
        variations = []
        parts = response.split('\n\n')
        
        for part in parts:
            part = part.strip()
            
            # Skip empty or too short
            if not part or len(part) < 20:
                continue
            
            # Remove common prefixes
            for prefix in ['VARIATION:', 'VARIATION', '1.', '2.', '3.', '1)', '2)', '3)', 
                          'Prompt:', 'Prompt 1:', 'Prompt 2:', 'Prompt 3:', 
                          '**Variation', '*Variation', 'Example:']:
                if part.startswith(prefix):
                    # Find end of prefix line
                    newline_idx = part.find('\n')
                    if newline_idx != -1:
                        part = part[newline_idx+1:].strip()
                    else:
                        part = part[len(prefix):].strip()
                    break
            
            # Clean markdown formatting
            part = part.replace('**', '').replace('*', '').replace('```', '')
            
            if part and len(part) > 20:
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

