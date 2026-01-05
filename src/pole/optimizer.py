"""Core optimization loop."""

from dataclasses import dataclass
from typing import Callable, Optional

from .mutator import PromptMutator, DefaultMutator
from .history import OptimizationHistory


@dataclass
class OptimizationResult:
    """Result of the optimization process."""
    best_prompt: str
    best_loss: float
    iterations: int
    converged: bool
    convergence_reason: str


class PromptOptimizer:
    """
    Optimizes prompts via iterative mutation and evaluation.
    
    The user provides:
    - model: A callable that takes (prompt, input_text) and returns output
    - loss_fn: A function that computes loss(predicted, expected) -> float
    
    The optimizer handles:
    - Iterative loop
    - Prompt mutation
    - Convergence detection
    - History tracking
    """
    
    def __init__(
        self,
        model: Callable[[str, str], str],
        loss_fn: Callable[[str, str], float],
        mutator: Optional[PromptMutator] = None,
        max_iterations: int = 20,
        patience: int = 5,
        loss_threshold: Optional[float] = None,
        top_k: int = 5,
        output_dir: str = "output/"
    ):
        """
        Args:
            model: Callable that takes (prompt, input_text) and returns output
            loss_fn: Function computing loss(predicted, expected) -> float
            mutator: Custom prompt mutator (uses DefaultMutator if None)
            max_iterations: Maximum number of optimization iterations
            patience: Stop if no improvement for this many iterations
            loss_threshold: Stop if loss drops below this value (optional)
            top_k: Number of best prompts to track
            output_dir: Directory to save optimization history
        """
        self.model = model
        self.loss_fn = loss_fn
        self.mutator = mutator or DefaultMutator()
        self.max_iterations = max_iterations
        self.patience = patience
        self.loss_threshold = loss_threshold
        self.output_dir = output_dir
        
        self.history = OptimizationHistory(top_k=top_k)
    
    def _evaluate_prompt(
        self,
        prompt: str,
        test_cases: list[tuple[str, str]]
    ) -> float:
        """
        Evaluate a prompt on test cases and return average loss.
        
        Args:
            prompt: The prompt to evaluate
            test_cases: List of (input, expected) pairs
            
        Returns:
            Average loss across all test cases
        """
        total_loss = 0.0
        
        for input_text, expected in test_cases:
            # Get model output
            predicted = self.model(prompt, input_text)
            
            # Compute loss
            loss = self.loss_fn(predicted, expected)
            total_loss += loss
        
        return total_loss / len(test_cases)
    
    def optimize(
        self,
        initial_prompt: str,
        test_cases: list[tuple[str, str]]
    ) -> OptimizationResult:
        """
        Run the optimization loop.
        
        Args:
            initial_prompt: Starting prompt
            test_cases: List of (input, expected) pairs for evaluation
            
        Returns:
            OptimizationResult with best prompt and metadata
        """
        current_prompt = initial_prompt
        current_loss = self._evaluate_prompt(current_prompt, test_cases)
        
        # Track initial state
        self.history.add(0, current_prompt, current_loss)
        
        best_prompt = current_prompt
        best_loss = current_loss
        patience_counter = 0
        
        for iteration in range(1, self.max_iterations + 1):
            # Generate variations
            context = {
                "iteration": iteration,
                "current_loss": current_loss,
                "best_loss": best_loss
            }
            variations = self.mutator.mutate(current_prompt, context)
            
            # Evaluate each variation
            improved = False
            for variant in variations:
                variant_loss = self._evaluate_prompt(variant, test_cases)
                self.history.add(iteration, variant, variant_loss)
                
                # Check if this is better
                if variant_loss < best_loss:
                    best_loss = variant_loss
                    best_prompt = variant
                    current_prompt = variant
                    current_loss = variant_loss
                    improved = True
                    patience_counter = 0
                    break
            
            if not improved:
                patience_counter += 1
            
            # Check convergence conditions
            if self.loss_threshold is not None and best_loss < self.loss_threshold:
                self.history.save(self.output_dir)
                return OptimizationResult(
                    best_prompt=best_prompt,
                    best_loss=best_loss,
                    iterations=iteration,
                    converged=True,
                    convergence_reason="loss_threshold"
                )
            
            if patience_counter >= self.patience:
                self.history.save(self.output_dir)
                return OptimizationResult(
                    best_prompt=best_prompt,
                    best_loss=best_loss,
                    iterations=iteration,
                    converged=True,
                    convergence_reason="patience"
                )
        
        # Max iterations reached
        self.history.save(self.output_dir)
        return OptimizationResult(
            best_prompt=best_prompt,
            best_loss=best_loss,
            iterations=self.max_iterations,
            converged=True,
            convergence_reason="max_iterations"
        )

