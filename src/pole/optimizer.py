"""Core optimization loop."""

from dataclasses import dataclass
from typing import Callable, Optional

from .mutator import PromptMutator, DefaultMutator
from .history import OptimizationHistory
from .reporter import ProgressReporter
from .constants.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_PATIENCE,
    DEFAULT_TOP_K,
    DEFAULT_OUTPUT_DIR,
    PROMPT_PREVIEW_LENGTH,
)


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
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        patience: int = DEFAULT_PATIENCE,
        loss_threshold: Optional[float] = None,
        top_k: int = DEFAULT_TOP_K,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        reporter: Optional[ProgressReporter] = None
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
            reporter: Optional progress reporter (silent if None)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.mutator = mutator or DefaultMutator()
        self.max_iterations = max_iterations
        self.patience = patience
        self.loss_threshold = loss_threshold
        self.output_dir = output_dir
        self.reporter = reporter

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
        if self.reporter:
            self.reporter.on_evaluating_test_cases(len(test_cases))

        total_loss = 0.0

        for input_text, expected in test_cases:
            try:
                predicted = self.model(prompt, input_text)
                loss = self.loss_fn(predicted, expected)
                total_loss += loss
            except Exception as e:
                raise RuntimeError(
                    f"Error during evaluation:\n"
                    f"  Input: {input_text[:100]}...\n"
                    f"  Error: {str(e)}\n"
                    f"Make sure your model() and loss_fn() functions handle errors properly."
                ) from e

        avg_loss = total_loss / len(test_cases)
        return avg_loss

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
        if not test_cases:
            raise ValueError("test_cases cannot be empty")

        if not initial_prompt or not initial_prompt.strip():
            raise ValueError("initial_prompt cannot be empty")

        current_prompt = initial_prompt
        current_loss = self._evaluate_prompt(current_prompt, test_cases)

        self.history.add(0, current_prompt, current_loss)

        if self.reporter:
            self.reporter.on_initial_evaluation(current_loss)

        best_prompt = current_prompt
        best_loss = current_loss
        patience_counter = 0

        for iteration in range(1, self.max_iterations + 1):
            if self.reporter:
                self.reporter.on_iteration_start(
                    iteration, self.max_iterations, best_loss, patience_counter, self.patience
                )

            # Generate variations
            context = {
                "iteration": iteration,
                "current_loss": current_loss,
                "best_loss": best_loss
            }
            variations = self.mutator.mutate(current_prompt, context)

            if self.reporter:
                self.reporter.on_generating_variations(len(variations))

            # Evaluate all variations before selecting
            variation_losses = []
            for i, variant in enumerate(variations, 1):
                if self.reporter:
                    preview = variant[:PROMPT_PREVIEW_LENGTH].replace('\n', ' ')
                    if len(variant) > PROMPT_PREVIEW_LENGTH:
                        preview += "..."
                    self.reporter.on_evaluating_variation(i, len(variations), preview)

                variant_loss = self._evaluate_prompt(variant, test_cases)
                variation_losses.append((variant, variant_loss))
                self.history.add(iteration, variant, variant_loss)

                is_best = variant_loss < best_loss
                if self.reporter:
                    self.reporter.on_evaluation_complete(variant_loss, is_best)

            # Select best variation from this iteration
            best_variant, best_variant_loss = min(variation_losses, key=lambda x: x[1])

            improved = False
            if best_variant_loss < best_loss:
                best_loss = best_variant_loss
                best_prompt = best_variant
                current_prompt = best_variant
                current_loss = best_variant_loss
                improved = True
                patience_counter = 0
            else:
                patience_counter += 1

            if self.reporter:
                self.reporter.on_iteration_end(improved, patience_counter, self.patience)

            # Check convergence conditions
            if self.loss_threshold is not None and best_loss < self.loss_threshold:
                if self.reporter:
                    self.reporter.on_convergence("loss_threshold", best_loss, iteration)
                self.history.save(self.output_dir)
                return OptimizationResult(
                    best_prompt=best_prompt,
                    best_loss=best_loss,
                    iterations=iteration,
                    converged=True,
                    convergence_reason="loss_threshold"
                )

            if patience_counter >= self.patience:
                if self.reporter:
                    self.reporter.on_convergence("patience", best_loss, iteration)
                self.history.save(self.output_dir)
                return OptimizationResult(
                    best_prompt=best_prompt,
                    best_loss=best_loss,
                    iterations=iteration,
                    converged=True,
                    convergence_reason="patience"
                )

        # Max iterations reached
        if self.reporter:
            self.reporter.on_convergence("max_iterations", best_loss, self.max_iterations)
        self.history.save(self.output_dir)
        return OptimizationResult(
            best_prompt=best_prompt,
            best_loss=best_loss,
            iterations=self.max_iterations,
            converged=True,
            convergence_reason="max_iterations"
        )
