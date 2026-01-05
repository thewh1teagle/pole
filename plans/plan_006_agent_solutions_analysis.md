# Plan 006: Agent-Based Solutions for Optimizer Flaws

**Goal**: Analyze which agents can solve the flaws identified in Plan 005
**Context**: Current agents (MutatorAgent, VerifierAgent) exist. Need to identify additional agents that address specific flaws.
**Approach**: Map each flaw to agent-based solutions

---

## Current Agent Architecture

### Existing Agents
1. **MutatorAgent**: Generates prompt variations using LLM
2. **VerifierAgent**: Validates that variations are sensible

### Proposed Agents (from Plan 003)
3. **DiversityAgent**: Generates diverse prompts by analyzing history
4. **RandomRestartAgent**: Generates fresh prompts when stuck

---

## Flaw-to-Agent Mapping

### Flaw #1: Diversity Mechanism Not Visible/Working

**Problem**: Can't verify if diversity injection is working, no logging

**Agent Solution**: **DiversityAgent** (from Plan 003)

**How it helps**:
- **Explicit agent**: Dedicated agent makes diversity visible and testable
- **History-aware**: Analyzes all tried prompts to generate truly different ones
- **Structured output**: Agent can log its reasoning/strategy
- **Better than current**: Current approach just mutates initial prompt (may generate similar prompts)

**Implementation**:
```python
class DiversityAgent:
    """Generates diverse prompts by analyzing optimization history."""
    
    def generate_diverse_prompt(
        self,
        current_best: str,
        history: list[tuple[str, float]],  # (prompt, loss) pairs
        task_description: str
    ) -> tuple[str, str]:  # Returns (prompt, strategy_used)
        """
        Generate a radically different prompt.
        Returns both the prompt and the strategy used (for logging).
        """
        # Analyze history to find patterns
        # Generate prompt using different strategy
        # Return prompt + strategy name for logging
```

**Benefits over current approach**:
- ✅ **Visible**: Agent can report what strategy it used
- ✅ **Intelligent**: Analyzes history, not just mutates initial
- ✅ **Loggable**: Can log "DiversityAgent used role-play strategy"
- ✅ **Testable**: Can unit test agent separately

**Addresses**: Flaw #1 (visibility), Flaw #5 (effectiveness)

---

### Flaw #2: Mutator Generates Repetitive Variations

**Problem**: Same prompts appearing multiple times, wasting evaluations

**Agent Solution**: **DeduplicationAgent** or **DeduplicationFilter**

**Option A: Deduplication Filter (Non-Agent)**
- Simple: Filter duplicates before returning variations
- Fast: No LLM call needed
- **Recommended**: Use this first (simpler, faster)

**Option B: DeduplicationAgent (Agent-Based)**
- **Semantic deduplication**: Detects semantically similar prompts, not just exact duplicates
- **LLM-powered**: Can identify "Classify sentiment" vs "Classify the sentiment" as duplicates
- **More intelligent**: Understands that "Be concise" and "Keep it short" are similar

**Implementation (Option A - Recommended)**:
```python
# In AgentMutator.mutate()
def mutate(self, prompt: str, context: dict) -> list[str]:
    raw_variations = self.mutator.generate_variations(prompt, context)
    
    # Deduplicate exact matches
    seen = set()
    unique_variations = []
    for v in raw_variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)
    
    return unique_variations[:self.num_variations]
```

**Implementation (Option B - Advanced)**:
```python
class DeduplicationAgent:
    """Agent that identifies semantically similar prompts."""
    
    def __init__(self, similarity_fn: Callable[[str, str], bool]):
        self.similarity_fn = similarity_fn  # LLM-based similarity check
    
    def filter_duplicates(self, variations: list[str]) -> list[str]:
        """Remove semantically similar variations."""
        unique = []
        for v in variations:
            is_duplicate = any(
                self.similarity_fn(v, existing) 
                for existing in unique
            )
            if not is_duplicate:
                unique.append(v)
        return unique
```

**Recommendation**: Start with Option A (simple deduplication), upgrade to Option B if needed.

**Addresses**: Flaw #2 (repetitive variations)

---

### Flaw #3: Patience Too Low for Exploration

**Problem**: Quits too early, doesn't use all iterations

**Agent Solution**: **AdaptivePatienceAgent** or **ExplorationAgent**

**Option A: Configuration Fix (Non-Agent)**
- Simple: Increase patience parameter
- **Recommended**: Do this first (1 minute fix)

**Option B: AdaptivePatienceAgent (Agent-Based)**
- **Intelligent**: Analyzes optimization progress to adjust patience
- **Context-aware**: More patience when diversity is active, less when making progress
- **Learning**: Learns from history when to give up vs. keep trying

**Implementation (Option A - Recommended)**:
```python
# In optimize_sentiment.py
optimizer = PromptOptimizer(
    ...
    patience=10,  # Increased from 5
    ...
)
```

**Implementation (Option B - Advanced)**:
```python
class AdaptivePatienceAgent:
    """Agent that adjusts patience based on optimization progress."""
    
    def should_continue(
        self,
        patience_counter: int,
        current_patience: int,
        history: OptimizationHistory,
        diversity_active: bool
    ) -> tuple[bool, str]:
        """
        Returns (should_continue, reason)
        """
        # If diversity just injected, reset patience
        if diversity_active and patience_counter > current_patience // 2:
            return True, "diversity_injection_reset"
        
        # If making slow progress, extend patience
        recent_improvements = self._count_recent_improvements(history)
        if recent_improvements > 0:
            return True, "slow_but_steady_progress"
        
        # If completely stuck, give up
        return False, "no_progress"
```

**Recommendation**: Start with Option A, consider Option B for advanced use cases.

**Addresses**: Flaw #3 (patience too low)

---

### Flaw #4: No Forced Exploration When Stuck

**Problem**: No mechanism to try radical mutations when stuck

**Agent Solution**: **ExplorationAgent** or **DesperationAgent**

**How it helps**:
- **Triggered on plateau**: Activates when patience_counter > threshold
- **Radical mutations**: Generates completely different prompts
- **Multiple strategies**: Tries different approaches (role-play, chain-of-thought, etc.)
- **History-aware**: Avoids repeating failed strategies

**Implementation**:
```python
class ExplorationAgent:
    """Agent that forces exploration when optimizer is stuck."""
    
    def __init__(self, exploration_fn: Callable[[str], str]):
        self.exploration_fn = exploration_fn
        self.failed_strategies = set()
    
    def generate_exploration_prompts(
        self,
        current_best: str,
        history: list[tuple[str, float]],
        task_description: str,
        num_prompts: int = 3
    ) -> list[tuple[str, str]]:
        """
        Generate radical exploration prompts.
        Returns list of (prompt, strategy) tuples.
        """
        strategies = [
            "role_play",      # "You are a sentiment analyst..."
            "chain_of_thought", # "First, identify key words..."
            "examples",        # "Examples: 'I love it' → positive..."
            "structured",      # "Format: [SENTIMENT: ...]"
            "conversational"  # "What's the sentiment? Tell me..."
        ]
        
        # Filter out failed strategies
        available = [s for s in strategies if s not in self.failed_strategies]
        
        exploration_prompts = []
        for strategy in available[:num_prompts]:
            prompt = self._generate_with_strategy(
                strategy, current_best, history, task_description
            )
            exploration_prompts.append((prompt, strategy))
        
        return exploration_prompts
    
    def record_failure(self, strategy: str):
        """Record that a strategy failed to improve."""
        self.failed_strategies.add(strategy)
```

**Integration in Optimizer**:
```python
# In optimizer.py, in optimize() loop
if patience_counter >= self.patience // 2:  # Halfway to giving up
    if self.exploration_agent:
        exploration_prompts = self.exploration_agent.generate_exploration_prompts(
            current_best=best_prompt,
            history=self.history.get_all_prompts(),
            task_description=initial_prompt,
            num_prompts=2
        )
        for prompt, strategy in exploration_prompts:
            variations.append(prompt)
            if self.reporter:
                self.reporter.on_exploration_injection(prompt, strategy)
```

**Benefits**:
- ✅ **Forces exploration**: Doesn't give up easily
- ✅ **Multiple strategies**: Tries different approaches
- ✅ **Learning**: Avoids repeating failed strategies
- ✅ **Visible**: Logs which strategy was used

**Addresses**: Flaw #4 (no forced exploration)

---

### Flaw #5: Diversity Injection May Be Ineffective

**Problem**: Even if diversity runs, mutator might generate similar prompts

**Agent Solution**: **DiversityAgent** (same as Flaw #1)

**How DiversityAgent solves this**:
1. **History analysis**: Looks at ALL tried prompts, not just current best
2. **Pattern detection**: Identifies what's been tried (imperative, role-play, etc.)
3. **Strategy selection**: Chooses a different strategy explicitly
4. **Explicit instructions**: Tells LLM to be "radically different"

**Example**:
```
Current best: "Classify sentiment: positive, negative, or neutral. One word only."
History: ["Classify...", "Determine...", "Analyze..."] (all imperative)

DiversityAgent detects: All prompts are imperative, direct commands
DiversityAgent strategy: Try role-play approach
DiversityAgent generates: "You are a sentiment analyst. For each text, identify whether the sentiment is positive, negative, or neutral. Respond with only the sentiment word."
```

**Comparison**:
- **Current approach**: Mutate initial prompt → might generate "Classify sentiment..." (similar)
- **DiversityAgent**: Analyze history → detect pattern → choose different strategy → generate role-play prompt (different)

**Addresses**: Flaw #5 (diversity effectiveness)

---

### Flaw #6: No Feedback Loop for Failed Mutations

**Problem**: No learning from failures, repeats same mistakes

**Agent Solution**: **FeedbackAgent** or **LearningAgent**

**How it helps**:
- **Tracks failures**: Records which mutation patterns fail
- **Guides mutator**: Provides feedback to mutator about what to avoid
- **Pattern recognition**: Identifies that "Be concise" variations keep failing
- **Adaptive mutation**: Adjusts mutation strategy based on failures

**Implementation**:
```python
class FeedbackAgent:
    """Agent that learns from failed mutations and guides future mutations."""
    
    def __init__(self):
        self.failed_patterns = {}  # pattern -> failure_count
        self.successful_patterns = {}  # pattern -> success_count
    
    def analyze_failure(
        self,
        prompt: str,
        loss: float,
        best_loss: float
    ) -> dict:
        """Extract patterns from failed prompt."""
        patterns = {
            "has_be_concise": "be concise" in prompt.lower(),
            "has_one_word": "one word" in prompt.lower(),
            "is_imperative": prompt[0].isupper() and not prompt.startswith("You"),
            "is_role_play": prompt.lower().startswith("you are"),
            "length": len(prompt.split())
        }
        
        if loss > best_loss:
            # Failed - record patterns
            for pattern, value in patterns.items():
                key = f"{pattern}:{value}"
                self.failed_patterns[key] = self.failed_patterns.get(key, 0) + 1
        else:
            # Success - record patterns
            for pattern, value in patterns.items():
                key = f"{pattern}:{value}"
                self.successful_patterns[key] = self.successful_patterns.get(key, 0) + 1
        
        return patterns
    
    def get_mutation_guidance(self) -> str:
        """Generate guidance for mutator based on failures."""
        # Find most failed patterns
        top_failures = sorted(
            self.failed_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        guidance = "Avoid these patterns that have failed:\n"
        for pattern, count in top_failures:
            guidance += f"- {pattern} (failed {count} times)\n"
        
        # Find most successful patterns
        top_successes = sorted(
            self.successful_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        guidance += "\nPrefer these patterns that have succeeded:\n"
        for pattern, count in top_successes:
            guidance += f"- {pattern} (succeeded {count} times)\n"
        
        return guidance
```

**Integration**:
```python
# In optimizer.py, after evaluating variations
for variant, variant_loss in variation_losses:
    if variant_loss > best_loss:
        # Failed - get feedback
        guidance = self.feedback_agent.analyze_failure(
            variant, variant_loss, best_loss
        )
        # Store guidance for next iteration
        self.mutation_guidance = self.feedback_agent.get_mutation_guidance()

# In next iteration, pass guidance to mutator
context = {
    "iteration": iteration,
    "current_loss": current_loss,
    "best_loss": best_loss,
    "guidance": self.mutation_guidance  # NEW: Feedback from failures
}
variations = self.mutator.mutate(current_prompt, context)
```

**Benefits**:
- ✅ **Learning**: Adapts based on what works/fails
- ✅ **Efficiency**: Avoids repeating failed patterns
- ✅ **Intelligence**: Recognizes patterns, not just individual prompts

**Addresses**: Flaw #6 (no feedback loop)

---

## Recommended Agent Implementation Priority

### Phase 1: Quick Wins (Non-Agent Solutions)
1. ✅ **Deduplication Filter** (15 min) - Solves Flaw #2
2. ✅ **Increase Patience** (1 min) - Solves Flaw #3
3. ✅ **Add Diversity Logging** (5 min) - Improves Flaw #1 visibility

**Time**: 20 minutes
**Impact**: High (solves 2.5 flaws)

### Phase 2: Core Agent (DiversityAgent)
4. ✅ **DiversityAgent** (1-2 hours) - Solves Flaw #1, #5

**Time**: 1-2 hours
**Impact**: High (solves 2 flaws, improves exploration)

### Phase 3: Advanced Agents (If Needed)
5. ⚠️ **ExplorationAgent** (2-3 hours) - Solves Flaw #4
6. ⚠️ **FeedbackAgent** (2-3 hours) - Solves Flaw #6

**Time**: 4-6 hours
**Impact**: Medium (nice-to-have, but may not be necessary if Phase 1-2 work)

---

## Agent Comparison Matrix

| Agent | Solves Flaws | Complexity | Cost | Priority |
|-------|-------------|------------|------|----------|
| **DeduplicationFilter** | #2 | Low | None | P1 |
| **DiversityAgent** | #1, #5 | Medium | 1 LLM call/3 iter | P2 |
| **ExplorationAgent** | #4 | Medium | 2-3 LLM calls when stuck | P3 |
| **FeedbackAgent** | #6 | High | Ongoing analysis | P3 |
| **AdaptivePatienceAgent** | #3 | Medium | Logic only | P3 |

**Cost**: LLM calls per iteration
- DeduplicationFilter: 0 (no LLM)
- DiversityAgent: +1 call every 3 iterations
- ExplorationAgent: +2-3 calls when patience > threshold
- FeedbackAgent: 0 (analysis only, no LLM)
- AdaptivePatienceAgent: 0 (logic only)

---

## Implementation Plan

### Step 1: Quick Fixes (20 min)
```python
# 1. Add deduplication to AgentMutator
# 2. Increase patience in optimize_sentiment.py
# 3. Add diversity logging to optimizer
```

### Step 2: DiversityAgent (1-2 hours)
```python
# 1. Create agents/diversity_agent.py
# 2. Add DIVERSITY_INSTRUCTION_TEMPLATE to constants/prompts.py
# 3. Integrate into optimizer.py
# 4. Add on_diversity_injection to reporter
# 5. Test on challenging dataset
```

### Step 3: Evaluate (After Step 2)
- Run optimizer with DiversityAgent
- Check if it solves the issues
- If yes: Stop here
- If no: Consider ExplorationAgent

### Step 4: Advanced Agents (If Needed)
```python
# 1. Create agents/exploration_agent.py
# 2. Create agents/feedback_agent.py
# 3. Integrate into optimizer
# 4. Test and compare
```

---

## Expected Impact

### Before Agents:
- Flaw #1: ❌ No visibility
- Flaw #2: ❌ Duplicates waste evaluations
- Flaw #3: ❌ Quits too early
- Flaw #4: ❌ No forced exploration
- Flaw #5: ❌ Diversity ineffective
- Flaw #6: ❌ No learning

### After Phase 1 (Quick Fixes):
- Flaw #1: ⚠️ Visible but not intelligent
- Flaw #2: ✅ Solved (deduplication)
- Flaw #3: ✅ Solved (higher patience)
- Flaw #4: ❌ Still missing
- Flaw #5: ❌ Still ineffective
- Flaw #6: ❌ Still no learning

### After Phase 2 (DiversityAgent):
- Flaw #1: ✅ Solved (intelligent, visible)
- Flaw #2: ✅ Solved (deduplication)
- Flaw #3: ✅ Solved (higher patience)
- Flaw #4: ⚠️ Partially (DiversityAgent helps, but not forced)
- Flaw #5: ✅ Solved (intelligent diversity)
- Flaw #6: ❌ Still no learning

### After Phase 3 (All Agents):
- All flaws: ✅ Solved or significantly improved

---

## Conclusion

**Yes, agents can help solve these issues!**

**Recommended approach**:
1. **Start with quick fixes** (Phase 1) - 20 minutes, high impact
2. **Add DiversityAgent** (Phase 2) - 1-2 hours, solves core diversity issues
3. **Evaluate results** - See if more is needed
4. **Add advanced agents** (Phase 3) - Only if Phase 1-2 don't solve it

**Key insight**: DiversityAgent is the most important agent because it:
- Solves visibility (Flaw #1)
- Solves effectiveness (Flaw #5)
- Partially helps with exploration (Flaw #4)
- Is relatively simple to implement
- Follows existing agent pattern

**Next step**: Implement Phase 1 (quick fixes) + Phase 2 (DiversityAgent)

