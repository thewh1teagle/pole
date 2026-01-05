# Plan 003: Solving Prompt Diversity Problem

**Issue**: Optimizer gets stuck in local minima, generating only similar variations
**Constraint**: Cannot use temperature parameter (user provides LLM as-is)
**Solution**: Agent-based diversity mechanisms

---

## Problem Analysis

### Current Behavior (Greedy Hill Climbing)
```python
# Iteration 1
best_prompt = "Classify sentiment. Be concise."
variations = mutator.mutate(best_prompt, context)
# → ["Classify sentiment. Be precise.", "Classify sentiment. One word.", ...]

# Iteration 2
best_prompt = "Classify sentiment. One word."  # Best from iteration 1
variations = mutator.mutate(best_prompt, context)
# → ["Classify sentiment. Single word.", "Classify the sentiment. One word only.", ...]
```

**Result**: Stuck in "Classify sentiment [modifier]" space forever.

### Why This Happens
1. **Mutator only sees current best**: No memory of exploration history
2. **Small mutations**: LLM naturally makes small, safe changes
3. **No diversity pressure**: Nothing encourages radical departures
4. **Greedy selection**: Always continues from single best prompt

---

## Solution: Agent-Based Diversity Strategies

### Strategy 1: **Diversity Agent** (RECOMMENDED)

Add a specialized agent that generates diverse prompts by analyzing the history.

#### Architecture
```python
class DiversityAgent:
    """Generates diverse prompts by analyzing what's been tried."""

    def generate_diverse_prompt(
        self,
        current_best: str,
        history: list[str],  # All prompts tried so far
        task_description: str
    ) -> str:
        """Generate a radically different prompt."""
```

#### Agent Instruction (Prompt)
```
You are a diversity agent for prompt optimization.

CURRENT BEST PROMPT:
{current_best}

PROMPTS ALREADY TRIED:
{history}

TASK:
{task_description}

Your job: Generate a COMPLETELY DIFFERENT prompt that:
1. Achieves the same goal (task)
2. Uses a DIFFERENT approach/structure than all tried prompts
3. Is radically different, not just word changes

Examples of GOOD diversity:

If tried prompts are all like "Classify X as Y":
- Try: "You are an expert. Analyze X and output Y."
- Try: "Read X. Respond with only: Y options."
- Try: "Task: Determine Y from X. Output format: single word."

If tried prompts are imperative ("Classify..."):
- Try: "You will analyze..." (future tense)
- Try: "I need you to..." (first person)
- Try: "This is a classification task..." (descriptive)

Generate ONE completely different prompt:
```

#### Integration
```python
class PromptOptimizer:
    def __init__(self, ..., diversity_agent=None, diversity_frequency=3):
        self.diversity_agent = diversity_agent
        self.diversity_frequency = diversity_frequency  # Every N iterations

    def optimize(self, ...):
        for iteration in range(1, max_iterations + 1):
            # Regular mutations
            variations = self.mutator.mutate(current_prompt, context)

            # Every N iterations, inject diverse prompt
            if iteration % self.diversity_frequency == 0 and self.diversity_agent:
                diverse_prompt = self.diversity_agent.generate_diverse_prompt(
                    current_best=best_prompt,
                    history=self.history.get_all_prompts(),
                    task_description=initial_prompt  # Original task description
                )
                variations.append(diverse_prompt)

            # Evaluate all variations (including diverse one)
            ...
```

#### Benefits
- ✅ Works with any LLM (no temperature needed)
- ✅ Uses history to avoid repetition
- ✅ Explicit diversity instruction
- ✅ Controlled via frequency parameter

---

### Strategy 2: **Beam Search** (Track Multiple Candidates)

Instead of greedy (single best), track top-K prompts.

#### Architecture
```python
class PromptOptimizer:
    def __init__(self, ..., beam_width=3):
        self.beam_width = beam_width  # Track top-3 prompts

    def optimize(self, ...):
        # Track multiple candidates
        beam = [(initial_prompt, initial_loss)]

        for iteration in range(1, max_iterations + 1):
            candidates = []

            # Generate variations from ALL beam prompts
            for prompt, loss in beam:
                variations = self.mutator.mutate(prompt, context)
                for v in variations:
                    v_loss = self._evaluate_prompt(v, test_cases)
                    candidates.append((v, v_loss))

            # Keep top-K candidates
            beam = sorted(candidates, key=lambda x: x[1])[:self.beam_width]
            best_prompt, best_loss = beam[0]

            ...
```

#### Benefits
- ✅ Explores multiple paths simultaneously
- ✅ Less likely to get stuck in local minima
- ✅ No new agents needed

#### Drawbacks
- ❌ More expensive (K × variations evaluations)
- ❌ Still doesn't guarantee radical diversity

---

### Strategy 3: **Random Restart Agent**

Periodically start fresh with a completely new prompt.

#### Architecture
```python
class RandomRestartAgent:
    """Generates fresh prompts for the same task."""

    def generate_fresh_prompt(self, task_description: str, example_io: list) -> str:
        """Generate a new prompt from scratch."""
```

#### Agent Instruction
```
You are generating a prompt for this task:

TASK: {task_description}

EXAMPLE INPUT/OUTPUT:
Input: "I love this movie!"
Expected: "positive"

Generate a prompt that would make an LLM solve this task.
Be creative - try different styles:
- Instructional ("Classify...")
- Role-play ("You are an expert...")
- Few-shot (show examples)
- Chain-of-thought ("Think step by step...")
- Format-focused ("Output only one word...")

Generate ONE creative prompt:
```

#### Integration
```python
class PromptOptimizer:
    def optimize(self, ...):
        patience_counter = 0

        for iteration in range(1, max_iterations + 1):
            # Regular optimization
            variations = self.mutator.mutate(current_prompt, context)

            # If stuck (no improvement for N iterations), random restart
            if patience_counter >= patience // 2:  # Halfway to convergence
                restart_prompt = self.restart_agent.generate_fresh_prompt(
                    task_description=initial_prompt,
                    example_io=test_cases[:3]
                )
                variations.append(restart_prompt)

            ...
```

---

### Strategy 4: **Diversity via Multi-Agent Ensemble**

Use multiple mutator agents with different "personalities."

#### Architecture
```python
class EnsembleMutator:
    """Multiple mutator agents with different styles."""

    def __init__(self, mutator_fn, num_variations=3):
        self.conservative = MutatorAgent(mutator_fn, style="conservative")
        self.creative = MutatorAgent(mutator_fn, style="creative")
        self.radical = MutatorAgent(mutator_fn, style="radical")

    def mutate(self, prompt, context):
        variations = []

        # Conservative: small changes
        variations += self.conservative.mutate(prompt, context)[:1]

        # Creative: moderate changes
        variations += self.creative.mutate(prompt, context)[:1]

        # Radical: big changes
        variations += self.radical.mutate(prompt, context)[:1]

        return variations
```

#### Agent Instructions
```python
# Conservative mutator
"Make SMALL improvements. Change only 1-2 words. Keep structure."

# Creative mutator
"Make MODERATE changes. Try different wording, but same approach."

# Radical mutator
"Make BIG changes. Try completely different prompt structure or style."
```

---

## Recommended Implementation

### Phase 1: **Add Diversity Agent** (Simplest, Most Effective)

1. **Create `DiversityAgent` class** in `agents/diversity_agent.py`
2. **Add diversity prompt** to `constants/prompts.py`
3. **Integrate into optimizer** with `diversity_frequency` parameter
4. **Test**: Verify it generates truly different prompts

### Phase 2: **Add Beam Search** (Better Exploration)

1. **Add `beam_width` parameter** to `PromptOptimizer`
2. **Modify optimization loop** to track multiple candidates
3. **Update reporter** to show beam status
4. **Test**: Verify performance vs greedy

### Phase 3: **Add Random Restart** (Escape Local Minima)

1. **Create `RandomRestartAgent`** in `agents/restart_agent.py`
2. **Trigger on plateau** (no improvement for N/2 iterations)
3. **Test**: Verify it helps when stuck

---

## Code Example: Diversity Agent

### `agents/diversity_agent.py`
```python
"""Agent for generating diverse prompt variations."""

from typing import Callable
from ..constants.prompts import DIVERSITY_INSTRUCTION_TEMPLATE


class DiversityAgent:
    """Generates diverse prompts by analyzing optimization history."""

    def __init__(self, diversity_fn: Callable[[str], str]):
        """
        Args:
            diversity_fn: LLM callable for generating diverse prompts
        """
        self.diversity_fn = diversity_fn

    def generate_diverse_prompt(
        self,
        current_best: str,
        history: list[str],
        task_description: str
    ) -> str:
        """
        Generate a radically different prompt.

        Args:
            current_best: Current best prompt
            history: All prompts tried so far (for diversity)
            task_description: Original task description

        Returns:
            A diverse prompt
        """
        # Deduplicate and format history
        unique_history = list(dict.fromkeys(history))[-10:]  # Last 10 unique
        history_str = "\n".join(f"- {p}" for p in unique_history)

        instruction = DIVERSITY_INSTRUCTION_TEMPLATE.format(
            current_best=current_best,
            history=history_str,
            task_description=task_description
        )

        diverse_prompt = self.diversity_fn(instruction)
        return diverse_prompt.strip()
```

### `constants/prompts.py` (add this)
```python
DIVERSITY_INSTRUCTION_TEMPLATE = """You are a diversity agent for prompt optimization.

CURRENT BEST PROMPT:
{current_best}

PROMPTS ALREADY TRIED:
{history}

ORIGINAL TASK:
{task_description}

Your job: Generate a COMPLETELY DIFFERENT prompt that solves the same task.

Requirements:
1. Achieve the same goal as the original task
2. Use a DIFFERENT structure/approach than all tried prompts
3. Be radically different, not just word swaps

Diversity strategies:
- If all tried prompts are imperative ("Classify..."): Try role-play ("You are...")
- If all are short: Try detailed instructions
- If all are formal: Try conversational
- If all are direct: Try chain-of-thought
- If all are single-sentence: Try multi-step instructions

Generate ONE completely different prompt (raw prompt only, no explanation):"""
```

### Integration in `optimizer.py`
```python
class PromptOptimizer:
    def __init__(
        self,
        ...,
        diversity_agent: Optional['DiversityAgent'] = None,
        diversity_frequency: int = 3
    ):
        """
        Args:
            ...
            diversity_agent: Optional agent for generating diverse prompts
            diversity_frequency: Inject diverse prompt every N iterations
        """
        ...
        self.diversity_agent = diversity_agent
        self.diversity_frequency = diversity_frequency

    def optimize(self, initial_prompt, test_cases):
        ...
        for iteration in range(1, self.max_iterations + 1):
            ...
            # Generate variations
            variations = self.mutator.mutate(current_prompt, context)

            # Periodically inject diverse prompt
            if (self.diversity_agent and
                iteration % self.diversity_frequency == 0):

                diverse_prompt = self.diversity_agent.generate_diverse_prompt(
                    current_best=best_prompt,
                    history=self.history.get_all_prompts(),
                    task_description=initial_prompt
                )
                variations.append(diverse_prompt)

                if self.reporter:
                    self.reporter.on_diversity_injection(diverse_prompt)

            # Evaluate all variations (including diverse one if added)
            ...
```

---

## Testing Plan

### Test 1: Diversity Detection
```python
# Run optimizer with diversity agent
result = optimizer.optimize(initial_prompt, test_cases)

# Check history for diversity
prompts = result.history.get_all_prompts()

# Manually verify:
# - Are there structurally different prompts?
# - Do they use different approaches (imperative vs role-play)?
# - Are there periodic "jumps" to different spaces?
```

### Test 2: Performance Comparison
```python
# Without diversity
result_greedy = optimizer_greedy.optimize(prompt, test_cases)

# With diversity
result_diverse = optimizer_diverse.optimize(prompt, test_cases)

# Compare:
# - Final loss (is diverse better or equal?)
# - Convergence speed (does diversity help or hurt?)
# - Prompt variety (more diverse in history?)
```

### Test 3: Local Minima Escape
```python
# Create a task where greedy gets stuck
# (e.g., sentiment with tricky edge cases)

# Verify:
# - Greedy converges to suboptimal
# - Diversity finds better solution
```

---

## Expected Outcomes

### With Diversity Agent:
```
Iteration 1-3: Regular mutations
  "Classify sentiment" → "Classify sentiment. Be concise." → ...

Iteration 3: DIVERSITY INJECTION
  "You are a sentiment analysis expert. Analyze the text and respond..."

Iteration 4-6: Regular mutations from best
  (Could be from diverse prompt if it was better)

Iteration 6: DIVERSITY INJECTION
  "Read the following text and determine its emotional tone..."
```

### Benefits:
- ✅ Explores multiple prompt spaces
- ✅ Less likely to get stuck
- ✅ No temperature control needed
- ✅ Controlled exploration (via frequency)
- ✅ Uses LLM intelligence for diversity

### Tradeoffs:
- ⚠️ Slightly more expensive (extra LLM calls for diversity)
- ⚠️ May slow convergence (exploring bad areas)
- ⚠️ Needs tuning (diversity_frequency parameter)

---

## Configuration Recommendations

```python
# Conservative (fast convergence, may get stuck)
optimizer = PromptOptimizer(
    ...,
    diversity_agent=None  # No diversity
)

# Balanced (recommended)
optimizer = PromptOptimizer(
    ...,
    diversity_agent=DiversityAgent(llm_fn),
    diversity_frequency=3  # Every 3 iterations
)

# Aggressive (maximum exploration)
optimizer = PromptOptimizer(
    ...,
    diversity_agent=DiversityAgent(llm_fn),
    diversity_frequency=2,  # Every 2 iterations
    beam_width=3  # Also use beam search
)
```

---

## Priority

**MEDIUM-HIGH**

This solves a real problem (local minima) but:
- ✅ System works without it (current results are good)
- ⚠️ May not always help (depends on task)
- ✅ Easy to implement (just another agent)
- ✅ Low risk (can be disabled via parameter)

**Recommendation**: Implement **Strategy 1 (Diversity Agent)** first as it's:
- Simple to add (follows existing agent pattern)
- Low cost (periodic, not every iteration)
- Proven approach (used in genetic algorithms, prompt optimization research)
- Works within our constraints (no temperature needed)

Then evaluate if we need beam search or random restarts.
