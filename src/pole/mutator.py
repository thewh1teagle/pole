"""Prompt mutation strategies."""

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


class DefaultMutator:
    """
    Simple deterministic prompt mutator.
    
    Applies small, deterministic string edits to generate variations.
    No randomness - same prompt always produces same variations.
    """
    
    def __init__(self, num_variations: int = 3):
        """
        Args:
            num_variations: Number of variations to generate per mutation
        """
        self.num_variations = num_variations
    
    def mutate(self, prompt: str, context: dict) -> list[str]:
        """Generate deterministic variations of the prompt."""
        variations = []
        
        # Strategy 1: Add clarifying instructions
        variations.append(f"{prompt}\n\nBe precise and concise.")
        
        # Strategy 2: Add output format constraint
        variations.append(f"{prompt}\n\nOutput format: plain text, no explanations.")
        
        # Strategy 3: Rephrase with emphasis
        variations.append(f"Task: {prompt}\n\nFocus on accuracy.")
        
        # Strategy 4: Add step-by-step instruction
        variations.append(f"{prompt}\n\nThink step by step before responding.")
        
        # Strategy 5: Simplify
        simplified = prompt.split('.')[0] if '.' in prompt else prompt
        variations.append(simplified.strip())
        
        # Return up to num_variations
        return variations[:self.num_variations]

