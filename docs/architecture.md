# pole - Architecture Overview

**Prompt Optimization via Loop Evaluation**

A minimal, headless library for optimizing prompts through iterative evaluation.

## Core Principles

1. **Headless & Unopinionated** - Zero LLM dependencies, user provides all callables
2. **Loss-Driven** - User's loss function is the source of truth
3. **Clean Separation** - Prompt representation, optimizer, mutator, evaluation are independent
4. **History Mandatory** - All iterations tracked and saved

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       PromptOptimizer                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Loop: mutate → evaluate → select → converge?       │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────┬──────────────┬──────────────┬───────────────┘
               │              │              │
               ▼              ▼              ▼
        PromptMutator    model(prompt,  OptimizationHistory
        (generates       input) → str   (tracks best,
         variations)                     saves JSON)
               │
               ▼
        loss_fn(predicted, expected) → float
```

## Components

### 1. PromptOptimizer

**Location**: `src/pole/optimizer.py`

The main optimization loop. Handles:
- Iterative mutation and evaluation
- Convergence detection (max_iterations, patience, loss_threshold)
- History tracking
- Result aggregation

**User provides**:
- `model: Callable[[str, str], str]` - Takes (prompt, input), returns output
- `loss_fn: Callable[[str, str], float]` - Takes (predicted, expected), returns loss

```python
optimizer = PromptOptimizer(
    model=my_model_fn,
    loss_fn=my_loss_fn,
    max_iterations=20,
    patience=5
)
```

### 2. PromptMutator (Protocol)

**Location**: `src/pole/mutator.py`

Interface for generating prompt variations:

```python
class PromptMutator(Protocol):
    def mutate(self, prompt: str, context: dict) -> list[str]:
        ...
```

**Implementations**:

#### DefaultMutator
- Deterministic string transformations
- No external dependencies
- 5 built-in strategies (add instructions, rephrase, simplify, etc.)

```python
mutator = DefaultMutator(num_variations=3)
```

#### AgentMutator
- LLM-powered mutation
- Optional verifier agent for quality control
- User provides LLM callables (headless)

```python
mutator = AgentMutator(
    mutator_fn=lambda instruction: llm_call(instruction),
    verifier_fn=lambda question: llm_call(question),
    num_variations=3
)
```

### 3. OptimizationHistory

**Location**: `src/pole/history.py`

Tracks all iterations:
- Best prompt + loss
- Top-K checkpoints
- Full iteration history
- Saves to JSON

### 4. User-Provided Components

#### Model Function
```python
def my_model(prompt: str, input_text: str) -> str:
    # Call your LLM (OpenAI, Ollama, Anthropic, etc.)
    return llm_output
```

#### Loss Function
```python
def my_loss(predicted: str, expected: str) -> float:
    # Compute error (smaller = better)
    return error_score
```

## Data Flow

```
1. Initial Prompt
   ↓
2. Mutator.mutate(prompt, context) → [variation1, variation2, ...]
   ↓
3. For each variation:
     For each test_case:
       model(variation, test_input) → predicted
       loss_fn(predicted, expected) → loss
     average_loss = mean(losses)
   ↓
4. Select best variation (lowest loss)
   ↓
5. Check convergence:
   - Loss < threshold?
   - No improvement for N iterations?
   - Max iterations reached?
   ↓
6. If not converged: Go to step 2
   If converged: Return best prompt
```

## Convergence Strategies

The optimizer stops when **any** of these conditions are met:

1. **max_iterations**: Hard limit on number of iterations
2. **patience**: Stop if no improvement for N iterations
3. **loss_threshold**: Stop if loss drops below threshold (optional)

## Extension Points

### Custom Mutator

```python
class MyMutator:
    def mutate(self, prompt: str, context: dict) -> list[str]:
        # context contains: iteration, current_loss, best_loss
        # Generate variations based on your strategy
        return [variation1, variation2, ...]

optimizer = PromptOptimizer(mutator=MyMutator())
```

### Custom Loss Function

```python
def domain_specific_loss(predicted: str, expected: str) -> float:
    # Your domain-specific metric
    # - Edit distance
    # - BLEU score
    # - Exact match
    # - Phoneme error rate
    # - Custom business metric
    return score

optimizer = PromptOptimizer(loss_fn=domain_specific_loss)
```

## File Structure

```
pole/
├── src/pole/
│   ├── __init__.py           # Public API
│   ├── optimizer.py          # Main optimization loop
│   ├── mutator.py            # DefaultMutator + Protocol
│   ├── agent_mutator.py      # AgentMutator (LLM-powered)
│   └── history.py            # History tracking
├── examples/
│   ├── basic_example.py      # No dependencies
│   └── sentiment/
│       ├── optimize_sentiment.py       # With Ollama
│       └── optimize_with_agent.py      # With AgentMutator
├── docs/
│   └── agent_mutator.md      # Agent-based mutation guide
└── README.md
```

## Design Decisions

### Why headless?

Users should control:
- Which LLM to use
- How to call it
- What error metrics matter
- Cost vs performance tradeoffs

### Why loss-driven?

The optimizer doesn't "understand" prompts. It:
- Tries variations (mutator's job)
- Measures quality (loss function's job)
- Keeps what works (optimizer's job)

This is simple, transparent, and extensible.

### Why track history?

- Debugging: See what was tried
- Analysis: Understand optimization trajectory
- Checkpointing: Resume or rollback
- Inspection: Human review of variations

## Performance Characteristics

| Aspect | Details |
|--------|---------|
| Time complexity | O(iterations × variations × test_cases × model_time) |
| Memory | Minimal (stores only history, not model weights) |
| LLM calls | (variations per iteration) × iterations × test_cases |
| Determinism | DefaultMutator: deterministic<br>AgentMutator: depends on LLM |

## Future Extensions (Not Implemented)

- Parallel evaluation of variations
- Adaptive mutators (learn from history)
- Multi-objective optimization
- Prompt interpolation/crossover
- Bayesian optimization strategies
- Caching of model outputs

## See Also

- [Agent Mutator Guide](agent_mutator.md)
- [Examples](../examples/README.md)

