"""Progress reporting for optimization."""

from typing import Protocol


class ProgressReporter(Protocol):
    """Protocol for reporting optimization progress."""

    def on_iteration_start(self, iteration: int, max_iterations: int, best_loss: float, patience_counter: int, patience: int):
        """Called at the start of each iteration."""
        ...

    def on_generating_variations(self, num_variations: int):
        """Called when starting to generate variations."""
        ...

    def on_evaluating_variation(self, variation_num: int, total_variations: int, prompt_preview: str):
        """Called when starting to evaluate a variation."""
        ...

    def on_evaluating_test_cases(self, num_test_cases: int):
        """Called when starting to evaluate test cases."""
        ...

    def on_evaluation_complete(self, loss: float, is_best: bool):
        """Called after evaluating a variation."""
        ...

    def on_iteration_end(self, improved: bool, patience_counter: int, patience: int):
        """Called at the end of each iteration."""
        ...

    def on_convergence(self, reason: str, best_loss: float, iterations: int):
        """Called when optimization converges."""
        ...

    def on_initial_evaluation(self, loss: float):
        """Called after evaluating the initial prompt."""
        ...


class ConsoleReporter:
    """Simple console-based progress reporter."""

    def on_iteration_start(self, iteration: int, max_iterations: int, best_loss: float, patience_counter: int, patience: int):
        print(f"\n{'='*60}", flush=True)
        print(f"Iteration {iteration}/{max_iterations}", flush=True)
        print(f"Current best loss: {best_loss:.4f} | Patience: {patience_counter}/{patience}", flush=True)

    def on_generating_variations(self, num_variations: int):
        print(f"Generating prompt variations... done ({num_variations} variations)", flush=True)

    def on_evaluating_variation(self, variation_num: int, total_variations: int, prompt_preview: str):
        print(f"  Variation {variation_num}/{total_variations}:", flush=True)
        print(f"    Prompt: \"{prompt_preview}\"", flush=True)

    def on_evaluating_test_cases(self, num_test_cases: int):
        print(f"  Evaluating on {num_test_cases} test cases...", end=" ", flush=True)

    def on_evaluation_complete(self, loss: float, is_best: bool):
        improvement = "✓ NEW BEST!" if is_best else ""
        print(f"done (avg loss: {loss:.4f})", flush=True)
        if is_best:
            print(f"    → Final loss: {loss:.4f} {improvement}", flush=True)
        else:
            print(f"    → Final loss: {loss:.4f}", flush=True)

    def on_iteration_end(self, improved: bool, patience_counter: int, patience: int):
        if not improved:
            print(f"  No improvement this iteration (patience: {patience_counter}/{patience})", flush=True)

    def on_convergence(self, reason: str, best_loss: float, iterations: int):
        if reason == "loss_threshold":
            print(f"\n✓ Converged: loss {best_loss:.4f} below threshold", flush=True)
        elif reason == "patience":
            print(f"\n✓ Converged: No improvement for {iterations} iterations", flush=True)
        elif reason == "max_iterations":
            print(f"\n✓ Max iterations ({iterations}) reached", flush=True)

    def on_initial_evaluation(self, loss: float):
        print(f"\nIteration 0 [Initial]: loss = {loss:.4f}", flush=True)
