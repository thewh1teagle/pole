# Plan 004: Quick Win Diversity Solution

**Goal**: Add simple diversity mechanism with minimal code
**Constraint**: Keep it simple, no new agents, no complex logic
**Win**: Avoid local minima without adding complexity

---

## The Problem (Recap)

Current optimizer does greedy hill climbing:
```
"Classify sentiment" → "Classify sentiment. Be concise." → "Classify sentiment. One word."
```

Gets stuck in similar variations forever.

---

## Quick Win Solution: Mutate from History

**Key Insight**: Instead of always mutating from current best, occasionally mutate from a PREVIOUS good prompt.

### Simple Change to Optimizer

```python
class PromptOptimizer:
    def optimize(self, initial_prompt, test_cases):
        ...
        for iteration in range(1, max_iterations + 1):
            # QUICK WIN: Every N iterations, mutate from a random historical prompt
            if iteration % 3 == 0 and len(self.history.top_prompts) > 1:
                # Pick a random prompt from top-K history (not current best)
                historical_prompts = [p for p, _ in self.history.top_prompts if p != current_prompt]
                if historical_prompts:
                    import random
                    random_prompt = random.choice(historical_prompts)
                    historical_variations = self.mutator.mutate(random_prompt, context)
                    variations.extend(historical_variations[:1])  # Add 1 variation

            # Regular mutations from current best
            variations = self.mutator.mutate(current_prompt, context)

            # Rest of optimization logic...
```

### Why This Works

1. **Explores different branches**: Mutating from 2nd or 3rd best prompt explores different spaces
2. **No new dependencies**: Uses existing history tracking
3. **Minimal cost**: Only 1 extra variation every 3 iterations
4. **Simple**: 5 lines of code

### Example Behavior

```
Iteration 1:
  Initial: "Classify sentiment"
  Best: "Classify sentiment. Be concise." (loss: 0.3)
  2nd: "Determine sentiment" (loss: 0.35)
  3rd: "What is the sentiment?" (loss: 0.4)

Iteration 2:
  Mutate from: "Classify sentiment. Be concise." (current best)
  Variations: ["Classify sentiment precisely.", "Classify sentiment. One word."]

Iteration 3 (DIVERSITY):
  Mutate from: "Determine sentiment" (random historical)
  Extra variation: "Determine the sentiment. Output one word."
  Regular from: "Classify sentiment. One word." (current best)
  Variations: ["Classify sentiment. Single word.", ...]

Result: Explores BOTH "Classify..." and "Determine..." branches
```

---

## Even Simpler Alternative: Restart from Initial

**Simpler version**: Every N iterations, add a variation from the initial prompt.

```python
# Every 4 iterations, mutate from initial prompt
if iteration % 4 == 0:
    initial_variations = self.mutator.mutate(initial_prompt, context)
    variations.extend(initial_variations[:1])
```

**Why this works**:
- Initial prompt is maximally different from optimized prompts
- Forces exploration back to "starting point"
- Extremely simple (2 lines of code)

---

## Implementation: Simplest Possible

### Add to PromptOptimizer.__init__()
```python
def __init__(
    self,
    ...,
    diversity_frequency: int = 3,  # NEW: How often to inject diversity
):
    ...
    self.diversity_frequency = diversity_frequency
```

### Modify optimize() loop

**Option 1: Random from history**
```python
# After generating regular variations
variations = self.mutator.mutate(current_prompt, context)

# Every N iterations, add variation from random historical prompt
if (iteration % self.diversity_frequency == 0 and
    len(self.history.top_prompts) > 1):

    # Get all top prompts except current
    import random
    historical = [p for p, _ in self.history.top_prompts if p != best_prompt]

    if historical:
        random_prompt = random.choice(historical)
        historical_var = self.mutator.mutate(random_prompt, context)
        variations.append(historical_var[0])  # Add just one
```

**Option 2: Restart from initial** (SIMPLEST)
```python
# After generating regular variations
variations = self.mutator.mutate(current_prompt, context)

# Every N iterations, add variation from initial prompt
if iteration % self.diversity_frequency == 0 and iteration > 1:
    initial_var = self.mutator.mutate(initial_prompt, context)
    variations.append(initial_var[0])
```

---

## Pros & Cons

### Pros:
✅ **Extremely simple**: 3-5 lines of code
✅ **No new classes**: Uses existing components
✅ **No new parameters** (optional): Can hardcode frequency=3
✅ **Minimal cost**: 1 extra variation per N iterations
✅ **Works immediately**: No need to understand complex logic
✅ **Low risk**: If it doesn't help, minimal harm done

### Cons:
⚠️ **Less intelligent**: Not analyzing history, just random/restart
⚠️ **Less control**: Can't target specific diversity strategies
⚠️ **Limited diversity**: Still constrained by mutator's creativity

---

## Expected Impact

### Without diversity:
```
Iteration 1: "Classify sentiment" → "Classify. Be concise." (0.3)
Iteration 2: "Classify. Be concise." → "Classify. One word." (0.25)
Iteration 3: "Classify. One word." → "Classify. Single word." (0.25) [stuck]
Iteration 4: "Classify. Single word." → "Classify precisely." (0.25) [stuck]
```

### With quick win diversity (restart from initial):
```
Iteration 1: "Classify sentiment" → "Classify. Be concise." (0.3)
Iteration 2: "Classify. Be concise." → "Classify. One word." (0.25)
Iteration 3:
  - Regular: "Classify. One word." → "Classify. Single word." (0.25)
  - DIVERSITY: "Classify sentiment" → "Sentiment analysis task" (0.28)
Iteration 4: "Classify. One word." → "Determine sentiment" (0.22) [improvement!]
```

### Win:
- Breaks out of local minima
- Explores alternative formulations
- May find better prompts in different space

---

## Recommendation: Go with Option 2 (Restart from Initial)

**Why:**
1. **Simplest code**: 2 lines
2. **No randomness**: Deterministic, reproducible
3. **Clear semantics**: "Revisit starting point"
4. **Easy to explain**: "Every 3 iterations, try a variation of the original prompt"

### Implementation (Full Code)

```python
# In optimizer.py, in optimize() method, after generating variations:

# Generate regular variations from current best
context = {...}
variations = self.mutator.mutate(current_prompt, context)

# Quick win diversity: Every 3 iterations, add variation from initial prompt
if iteration % 3 == 0 and iteration > 1:
    initial_variation = self.mutator.mutate(initial_prompt, context)
    if initial_variation:
        variations.append(initial_variation[0])
        if self.reporter:
            self.reporter.on_diversity_injection_simple("initial restart")

# Continue with evaluation...
```

---

## Testing

### Quick Test
```python
# Run optimizer on challenging dataset
optimizer = PromptOptimizer(
    model=model,
    loss_fn=sentiment_loss,
    mutator=AgentMutator(...),
    max_iterations=10
)

result = optimizer.optimize(
    initial_prompt="Classify sentiment",
    test_cases=challenging_dataset
)

# Check history
history = result.history.get_all_prompts()

# Look for:
# 1. Are there prompts similar to initial prompt appearing later?
# 2. Do we see "jumps" back to simpler formulations?
# 3. Does final result improve?
```

---

## Cost Analysis

**Per optimization run (10 iterations, 3 variations/iteration):**

Without diversity:
- Total variations: 10 × 3 = 30

With diversity (every 3 iterations):
- Regular: 10 × 3 = 30
- Diversity: 3 extra (iterations 3, 6, 9)
- Total: 33 variations (+10%)

**Cost increase: ~10%**
**Potential benefit: Escape local minima, find better prompts**

Worth it!

---

## When to Use

**Use this quick win if:**
- You want diversity without complexity
- You're okay with 10% more evaluations
- You want something that "just works"
- You don't need fine-grained control

**Skip this if:**
- You need intelligent diversity (use full DiversityAgent from plan_003)
- 10% cost is too much
- Current greedy approach is working perfectly

---

## Upgrade Path

Start with this quick win, then if needed:

1. **Quick Win** (plan_004): Restart from initial (2 lines)
2. **Medium Win** (plan_003, Strategy 3): Random restart agent
3. **Full Solution** (plan_003, Strategy 1): Diversity agent with history analysis

Each level adds more intelligence and control, but also more complexity.

---

## Priority: HIGH (for quick win)

**Why:**
- ✅ 2 lines of code
- ✅ 10% cost increase
- ✅ Addresses real problem
- ✅ Zero complexity added
- ✅ Can implement in 5 minutes

**Next step**: Add the 2 lines to optimizer.py and test on challenging dataset.

---

## Final Code Snippet

```python
# In src/pole/optimizer.py, in optimize() method

for iteration in range(1, self.max_iterations + 1):
    ...
    # Generate variations
    context = {...}
    variations = self.mutator.mutate(current_prompt, context)

    # QUICK WIN: Every 3 iterations, add variation from initial prompt
    if iteration % 3 == 0 and iteration > 1:
        restart_vars = self.mutator.mutate(initial_prompt, context)
        if restart_vars:
            variations.append(restart_vars[0])

    # Continue with evaluation...
```

**That's it!** Simple, effective, low-risk diversity mechanism.
