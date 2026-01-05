"""Optimization history tracking."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PromptCheckpoint:
    """A single prompt evaluation checkpoint."""
    iteration: int
    prompt: str
    loss: float

    def to_dict(self) -> dict:
        return asdict(self)


class OptimizationHistory:
    """Tracks the full history of prompt optimization."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.checkpoints: list[PromptCheckpoint] = []
        self.best_checkpoint: Optional[PromptCheckpoint] = None

    def add(self, iteration: int, prompt: str, loss: float):
        checkpoint = PromptCheckpoint(iteration, prompt, loss)
        self.checkpoints.append(checkpoint)

        if self.best_checkpoint is None or loss < self.best_checkpoint.loss:
            self.best_checkpoint = checkpoint

    def get_top_k(self) -> list[PromptCheckpoint]:
        """Get the top K prompts by loss (lower is better)."""
        sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c.loss)
        return sorted_checkpoints[:self.top_k]

    def save(self, output_dir: str):
        """Save history to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        history_data = {
            "best": self.best_checkpoint.to_dict() if self.best_checkpoint else None,
            "top_k": [c.to_dict() for c in self.get_top_k()],
            "all_iterations": [c.to_dict() for c in self.checkpoints]
        }

        with open(output_path / "history.json", "w") as f:
            json.dump(history_data, f, indent=2)
