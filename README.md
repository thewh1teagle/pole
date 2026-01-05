# pole

**P**rompt **O**ptimization via **L**oop **E**valuation

A minimal Python library for optimizing prompts through iterative evaluation and mutation.

## Features

- 🔄 **Iterative prompt optimization** - Automatically improve prompts via mutation + evaluation
- 🎯 **Loss-driven convergence** - User defines the truth (loss function), library handles the loop
- 🔌 **Bring your own LLM** - No dependencies, works with any model (OpenAI, Ollama, Anthropic, etc.)
- 📊 **Full history tracking** - Saves best prompts and iteration history to JSON
- 🛠️ **Customizable mutators** - Built-in strategies or plug in your own

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from pole import PromptOptimizer

# 1. Define your model (any LLM)
def my_model(prompt: str, input_text: str) -> str:
    # Call your LLM here (OpenAI, Ollama, etc.)
    return llm_output

# 2. Define your loss function
def loss_fn(predicted: str, expected: str) -> float:
    # Return a number (smaller = better)
    # Use industry-standard metrics:
    # - jiwer.cer() for character/phoneme comparison
    # - jiwer.wer() for word comparison
    # - Custom business metrics
    return compute_error(predicted, expected)

# 3. Create optimizer
optimizer = PromptOptimizer(
    model=my_model,
    loss_fn=loss_fn,
    max_iterations=20,
    patience=5
)

# 4. Optimize!
result = optimizer.optimize(
    initial_prompt="Convert input to output.",
    test_cases=[
        ("input1", "expected1"),
        ("input2", "expected2"),
    ]
)

print(result.best_prompt)
print(result.best_loss)
```

## Examples

See `examples/` directory:
- **`basic_example.py`** - Minimal placeholder example (no dependencies)
- **`sentiment/`** - Sentiment classification with Ollama (real optimization)
- **`g2p/`** - Hebrew→IPA phoneme conversion (domain-specific, advanced)

### Quick Demo

```bash
# 1. Install dependencies
uv sync
uv pip install ollama jiwer

# 2. Run basic example (no LLM needed)
uv run examples/basic_example.py

# 3. Run sentiment optimization (requires Ollama)
ollama pull gemma3:270m
uv run examples/sentiment/optimize_sentiment.py

# 4. Test G2P with mock data (no fine-tuned model needed)
uv run examples/g2p/test_mock.py
```

## Architecture

Clean separation of concerns:

- **`PromptOptimizer`** - Main optimization loop and convergence logic
- **`PromptMutator`** - Strategies for generating prompt variations
- **`OptimizationHistory`** - Tracking and persistence of results

## Design Principles

1. **User defines truth** - Loss function is authoritative
2. **No magic** - Clear, readable code with no hidden complexity
3. **No dependencies** - Pure Python, bring your own LLM
4. **History is mandatory** - All iterations saved for analysis

## License

MIT

