"""
pole - Prompt Optimization via Loop Evaluation

A minimal library for optimizing prompts through iterative evaluation.
No LLM dependencies - bring your own model.
"""

from .optimizer import PromptOptimizer, OptimizationResult
from .mutator import PromptMutator, DefaultMutator
from .history import OptimizationHistory, PromptCheckpoint
from .agent_mutator import AgentMutator

__version__ = "0.1.0"

__all__ = [
    "PromptOptimizer",
    "OptimizationResult",
    "PromptMutator",
    "DefaultMutator",
    "OptimizationHistory",
    "PromptCheckpoint",
    "AgentMutator",
]
