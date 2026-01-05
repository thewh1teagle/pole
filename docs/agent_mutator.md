# Agent-Based Prompt Mutation

This document explains how to use the **AgentMutator** - a mutator that uses LLM agents to generate and verify prompt variations.

## Architecture

The `AgentMutator` uses two agents:

1. **Mutator Agent**: Generates creative prompt variations
2. **Verifier Agent**: Filters out nonsensical or garbage variations

Both agents are **user-provided callables** - the library is headless and unopinionated.

## Basic Usage

```python
from pole import PromptOptimizer, AgentMutator

# 1. Define your mutator function (any LLM)
def mutator_fn(instruction: str) -> str:
    """
    Takes an instruction, returns LLM's response.
    
    The instruction will ask the LLM to generate prompt variations.
    """
    # Your LLM call here (OpenAI, Anthropic, Ollama, etc.)
    response = your_llm.chat(instruction)
    return response

# 2. Define your verifier function (optional)
def verifier_fn(question: str) -> str:
    """
    Takes a yes/no question, returns LLM's answer.
    
    The question will ask if a variation is valid.
    """
    response = your_llm.chat(question)
    return response

# 3. Create the agent mutator
mutator = AgentMutator(
    mutator_fn=mutator_fn,
    verifier_fn=verifier_fn,  # Optional, set to None to skip
    num_variations=3
)

# 4. Use it with the optimizer
optimizer = PromptOptimizer(
    model=your_model,
    loss_fn=your_loss,
    mutator=mutator  # ← Agent-based mutator
)

result = optimizer.optimize(initial_prompt="...", test_cases=[...])
```

## Complete Example (Ollama)

```python
from pole import AgentMutator
import ollama

def create_ollama_mutator(model: str = "gemma3:270m"):
    """Create agent mutator using Ollama."""
    
    def mutator_fn(instruction: str) -> str:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a prompt engineering expert."},
                {"role": "user", "content": instruction}
            ],
            options={"temperature": 0.8, "num_predict": 500}
        )
        return response["message"]["content"]
    
    def verifier_fn(question: str) -> str:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a quality control expert."},
                {"role": "user", "content": question}
            ],
            options={"temperature": 0.1, "num_predict": 10}
        )
        return response["message"]["content"]
    
    return AgentMutator(
        mutator_fn=mutator_fn,
        verifier_fn=verifier_fn,
        num_variations=3
    )

# Use it
mutator = create_ollama_mutator()
optimizer = PromptOptimizer(model=my_model, loss_fn=my_loss, mutator=mutator)
```

## Using Different LLMs

### OpenAI

```python
import openai

def mutator_fn(instruction: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": instruction}],
        temperature=0.8
    )
    return response.choices[0].message.content

mutator = AgentMutator(mutator_fn=mutator_fn, num_variations=3)
```

### Anthropic

```python
import anthropic

def mutator_fn(instruction: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": instruction}],
        max_tokens=500
    )
    return response.content[0].text

mutator = AgentMutator(mutator_fn=mutator_fn, num_variations=3)
```

## How It Works

### 1. Mutator Agent

The library generates an instruction like:

```
You are a prompt engineering expert. Generate 3 improved variations of the following prompt.

ORIGINAL PROMPT:
"Classify the sentiment as positive, negative, or neutral."

TASK: Create 3 complete, ready-to-use prompt variations...
```

Your `mutator_fn` calls the LLM with this instruction and returns the response.

### 2. Verifier Agent (Optional)

For each generated variation, the library creates a verification question:

```
You are a quality control agent. Verify if this prompt variation is valid.

Original Prompt: "..."
Proposed Variation: "..."

Is this variation:
1. A valid prompt (not gibberish)?
2. Related to the original task?
3. Potentially useful?

Answer ONLY with: YES or NO
```

Your `verifier_fn` calls the LLM and returns "YES" or "NO".

### 3. Selection

Only variations that pass verification are kept. If all are rejected, a fallback is used.

## Benefits Over DefaultMutator

| Feature | DefaultMutator | AgentMutator |
|---------|---------------|--------------|
| Creativity | Fixed templates | LLM-powered |
| Domain adaptation | Generic | Context-aware |
| Quality control | None | Built-in verifier |
| Setup | Zero config | Requires LLM |
| Cost | Free | LLM API costs |

## Tips

1. **Use a creative temperature** (0.7-0.9) for the mutator agent
2. **Use a low temperature** (0.1-0.2) for the verifier agent
3. **Skip verification** (set `verifier_fn=None`) for faster iteration
4. **Monitor costs** - each iteration makes multiple LLM calls
5. **Start with DefaultMutator** - only use agents if needed

## Debugging

To see what the agents generate:

```python
mutator = AgentMutator(mutator_fn, verifier_fn, num_variations=3)

# Test manually
instruction = mutator._build_mutator_instruction("Test prompt", {"iteration": 1})
print("Instruction:", instruction)

response = mutator_fn(instruction)
print("Agent response:", response)

variations = mutator._generate_variations(instruction)
print("Parsed variations:", variations)
```

## See Also

- `examples/sentiment/optimize_with_agent.py` - Complete working example
- `src/pole/mutator.py` - DefaultMutator (simpler alternative)

