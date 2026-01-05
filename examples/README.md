# Examples

This directory contains examples showing how to use `pole` for prompt optimization.

## Quick Start Examples

### 1. Basic Example (No Dependencies)
**`basic_example.py`** - Minimal example with placeholder model

- No external dependencies
- Shows the core API
- Good starting point for understanding the library

```bash
python examples/basic_example.py
```

### 2. Sentiment Classification (Ollama)
**`sentiment/`** - Real optimization with local LLM

- Uses Ollama (free, runs locally)
- Small dataset (7 examples)
- ~30 seconds to run
- Shows measurable accuracy improvement

```bash
# Setup
ollama pull llama3.2:1b
pip install ollama

# Run
python examples/sentiment/optimize_sentiment.py
```

### 3. G2P (Grapheme-to-Phoneme)
**`g2p/`** - Advanced Hebrew phoneme conversion

- Real-world use case
- Larger dataset
- Custom loss function (phoneme error rate)
- Shows domain-specific optimization

```bash
python examples/g2p/optimize_g2p.py
```

## Example Comparison

| Example | Model | Dataset Size | Runtime | Complexity |
|---------|-------|--------------|---------|------------|
| basic_example | Placeholder | 3 | <1s | ⭐ Beginner |
| sentiment | Ollama | 7 | ~30s | ⭐⭐ Intermediate |
| g2p | Ollama | 100+ | ~5min | ⭐⭐⭐ Advanced |

## Learning Path

1. **Start with `basic_example.py`** - Understand the API
2. **Try `sentiment/`** - See real optimization in action
3. **Explore `g2p/`** - Apply to your domain

## Common Patterns

All examples follow the same structure:

```python
from pole import PromptOptimizer

# 1. Define your model wrapper
def my_model(prompt: str, input_text: str) -> str:
    return llm_call(prompt, input_text)

# 2. Define your loss function
def loss_fn(predicted: str, expected: str) -> float:
    return compute_error(predicted, expected)

# 3. Create optimizer and run
optimizer = PromptOptimizer(model=my_model, loss_fn=loss_fn)
result = optimizer.optimize(initial_prompt="...", test_cases=[...])
```

## Tips

- **Start small**: Test with 5-10 examples first
- **Use fast models**: Smaller models = faster iteration
- **Monitor history**: Check `output/history.json` after each run
- **Adjust patience**: Lower for quick tests, higher for thorough optimization

