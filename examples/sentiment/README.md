# Sentiment Classification Example

A simple example showing prompt optimization for sentiment analysis using a local Ollama model.

## Setup

```bash
# Install Ollama
# https://ollama.ai

# Pull a small model (1B parameters, ~1GB)
ollama pull llama3.2:1b

# Install Python package
pip install ollama
```

## Run

```bash
python examples/sentiment/optimize_sentiment.py
```

## What it does

1. Starts with a basic prompt: `"Classify the sentiment as positive, negative, or neutral."`
2. Generates variations of the prompt
3. Tests each variation on a small dataset (7 examples)
4. Keeps the best performing prompt
5. Stops when accuracy stops improving

## Expected Output

```
Sentiment Classification Prompt Optimization
======================================================================

Dataset size: 7 examples
Model: llama3.2:1b (via Ollama)

Starting optimization...

======================================================================
OPTIMIZATION COMPLETE
======================================================================

Best Prompt:
  "Classify the sentiment as positive, negative, or neutral.
   Output format: plain text, no explanations."

Accuracy: 100.0%
Iterations: 3
Converged: True (patience)

History saved to: output/sentiment/history.json
```

## Customization

### Use a different model

```python
# Change the model in sentiment_model():
model="llama3.2:3b",  # Larger model
model="gemma2:2b",    # Different family
```

### Use a different dataset

```python
DATASET = [
    ("Your text here", "positive"),
    ("Another example", "negative"),
    # ...
]
```

### Use a different LLM backend

Replace the `sentiment_model` function with any LLM:

```python
# HuggingFace Transformers
from transformers import pipeline
classifier = pipeline("text-classification")

# OpenAI
import openai
response = openai.chat.completions.create(...)

# Anthropic
import anthropic
response = anthropic.messages.create(...)
```

