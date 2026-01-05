# Pole Refactoring Plan

**Goal**: Transform pole into a production-ready, minimal library similar to HuggingFace Trainer with clean API, no noise, and user-controlled logging.

---

## 1. README Simplification

**Current Problem**: README is 105 lines with too much explanatory text that feels LLM-generated.

**Target**: ~50 lines, human-written, concise.

### Changes:
- Remove verbose feature descriptions (lines 9-13)
- Consolidate Installation + Quick Start into single section
- Remove "Design Principles" section (move to docs if needed)
- Remove detailed architecture section (exists in docs/architecture.md)
- Keep only: Title, 1-liner description, Install, Minimal example, Link to examples
- Make it feel like a real open-source project README (see: requests, flask, click)

### New Structure:
```markdown
# pole

Prompt optimization via iterative mutation and evaluation.

## Install
pip install -e .

## Usage
[One minimal example - 15 lines max]

## Examples
See examples/ directory

## License
MIT
```

---

## 2. Remove Unnecessary Files

**Target**: Keep only essential files for a minimal v0 library.

### Files to Remove:
- `docs/architecture.md` - Over-engineered for v0, can recreate if needed
- `docs/agent_mutator.md` - Keep AgentMutator code but remove docs for now
- `examples/README.md` - Too verbose (98 lines), examples should be self-documenting
- `examples/sentiment/README.md` - Redundant with example code comments
- `output/` directory - Should be git-ignored, not committed

### Files to Keep:
- Core library: `src/pole/*.py`
- Examples: `examples/basic_example.py`, `examples/sentiment/`, `examples/g2p/`
- Project config: `pyproject.toml`, `.gitignore`, `.python-version`
- Main README only

---

## 3. Remove Print Statements (Production Library Pattern)

**Current Problem**: Library has hardcoded `print()` statements everywhere (optimizer.py has ~15 print calls).

**Target**: Follow HuggingFace Trainer pattern:
- No prints unless user opts in
- Use structured logging or callbacks
- Silent by default

### Changes to `optimizer.py`:

**Current**:
```python
verbose: bool = True  # Default is noisy
if self.verbose:
    print(f"Iteration {iteration}...")
```

**New Pattern** (Option 1 - Callback based):
```python
class PromptOptimizer:
    def __init__(
        self,
        model,
        loss_fn,
        on_iteration: Optional[Callable] = None,  # User-provided callback
        on_evaluate: Optional[Callable] = None,
    ):
        self.on_iteration = on_iteration
        self.on_evaluate = on_evaluate

    def _log_iteration(self, iteration, loss):
        if self.on_iteration:
            self.on_iteration(iteration, loss)
```

**New Pattern** (Option 2 - Progress reporter):
```python
class ProgressReporter(Protocol):
    def on_iteration_start(self, iteration: int): ...
    def on_iteration_end(self, iteration: int, loss: float): ...

class ConsoleReporter:
    """Built-in console reporter (user can opt-in)"""
    def on_iteration_start(self, iteration: int):
        print(f"Iteration {iteration}...")

optimizer = PromptOptimizer(
    model=my_model,
    loss_fn=my_loss,
    reporter=ConsoleReporter()  # User explicitly enables logging
)
```

**Recommended**: Option 2 (cleaner, more extensible, follows Trainer pattern)

### Also remove prints from:
- `agent_mutator.py` (lines 66, 201) - use callbacks or logger
- Keep errors/warnings as exceptions, not prints

---

## 4. Create Constants File

**Current Problem**: Hardcoded values scattered across files.

**Target**: `src/pole/constants.py` with all magic numbers.

### What to extract:

From `optimizer.py`:
```python
# constants.py
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_PATIENCE = 5
DEFAULT_TOP_K = 5
DEFAULT_OUTPUT_DIR = "output/"
DEFAULT_NUM_VARIATIONS = 3
PROMPT_PREVIEW_LENGTH = 150
```

From `mutator.py`:
```python
DEFAULT_STRATEGIES = [
    "Be precise and concise.",
    "Output format: plain text, no explanations.",
    # etc...
]
```

From `agent_mutator.py`:
```python
MIN_VARIATION_LENGTH = 20
MIN_VIABLE_PROMPT_LENGTH = 5
VARIATION_GENERATION_MULTIPLIER = 2  # Generate 2x for filtering
```

---

## 5. Remove Unnecessary Docstrings

**Current Problem**: Over-documented for a minimal library. Every trivial method has long docstrings.

**Keep docstrings for**:
- Public API (PromptOptimizer.__init__, optimize())
- User-facing protocols (PromptMutator)
- Non-obvious logic

**Remove docstrings from**:
- Private methods (unless complex)
- Obvious data classes (PromptCheckpoint.to_dict())
- Simple helpers (_verify_variation if logic is clear)

Example:
```python
# REMOVE:
def to_dict(self) -> dict:
    """Convert checkpoint to dictionary."""
    return asdict(self)

# KEEP:
def to_dict(self) -> dict:
    return asdict(self)
```

**Guideline**: If the function name + type hints explain it, skip docstring.

---

## 6. Fix Examples

**Current Problems**:
1. Too many sys.path hacks (`sys.path.insert(0, ...)`)
2. Examples are overly verbose
3. No shared utilities (every example reimplements model wrapper)

### Changes:

**6.1. Remove sys.path hacks**
- Examples should import pole normally: `from pole import PromptOptimizer`
- User runs with: `uv run examples/sentiment/optimize_sentiment.py`
- Let uv/pip handle the import path

**6.2. Create shared example utilities**
```
examples/
├── _utils.py          # Shared helpers
├── basic_example.py
├── sentiment/
│   └── optimize_sentiment.py
└── g2p/
    ├── optimize_g2p.py
    └── base_prompt.py
```

`examples/_utils.py`:
```python
"""Shared utilities for examples."""

def create_ollama_model(model_name: str, temperature: float = 0.1):
    """Factory for Ollama model wrappers."""
    import ollama

    def model_fn(prompt: str, input_text: str) -> str:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            options={"temperature": temperature}
        )
        return response["message"]["content"].strip()

    return model_fn
```

**6.3. Update examples to use reporter**

```python
from pole import PromptOptimizer, ConsoleReporter

optimizer = PromptOptimizer(
    model=model_fn,
    loss_fn=loss_fn,
    reporter=ConsoleReporter()  # User opts into logging
)
```

---

## 7. Architecture Review & Potential Flaws

### 7.1. Core Concept Evaluation

**The Idea**: Optimize prompts by iteratively mutating them and selecting variants with lowest loss.

**Does it work?**
✅ **YES** - This is a valid approach with precedent:
- Similar to DSPy's prompt optimization
- Related to AutoPrompt, APE (Automatic Prompt Engineering)
- Analogous to genetic algorithms for discrete optimization

**Potential Issues**:

#### Issue 1: Local Minima
**Problem**: Greedy selection (take first improvement) can get stuck.

**Evidence in code** (optimizer.py:169-177):
```python
if variant_loss < best_loss:
    best_loss = variant_loss
    best_prompt = variant
    improved = True
    break  # ⚠️ Stops at first improvement
```

**Impact**: Might miss better variations later in the list.

**Fix**:
- Evaluate all variations before selecting (more LLM calls, but better)
- Or use probabilistic acceptance (simulated annealing style)

#### Issue 2: Mutation Quality
**Problem**: `DefaultMutator` is too simplistic - just appends generic strings.

**Evidence** (mutator.py:43-56):
```python
variations.append(f"{prompt}\n\nBe precise and concise.")
variations.append(f"{prompt}\n\nOutput format: plain text...")
```

**Impact**:
- These generic additions may not help domain-specific tasks
- Variations are too similar (low diversity)

**Fix**:
- Document that `DefaultMutator` is a toy example
- Encourage users to implement custom mutators or use `AgentMutator`
- Consider adding a `RandomMutator` with actual prompt rewriting

#### Issue 3: Test Set Overfitting
**Problem**: No train/val split - optimizing on the same test cases.

**Risk**: Prompt might overfit to test cases and not generalize.

**Fix**:
- Add validation_cases parameter
- Evaluate on train during optimization, report on validation
- Document this limitation for v0

#### Issue 4: Small Test Sets
**Problem**: Examples use tiny datasets (3-10 samples).

**Impact**: High variance in loss estimates, unstable optimization.

**Status**: This is acceptable for v0 examples, but should document:
> "Use 50+ test cases for reliable optimization in production"

#### Issue 5: No Exploitation of History
**Problem**: Mutator doesn't learn from past iterations.

**Current** (mutator.py:38):
```python
def mutate(self, prompt: str, context: dict) -> list[str]:
    # context has iteration, loss - but we don't use it!
```

**Opportunity**:
- Track what strategies worked before
- Use history to inform mutation direction
- This is a future enhancement, not v0 blocker

### 7.2. G2P Example Viability

**Specific Concern**: Can this actually work for phoneme conversion?

**Analysis**:
- G2P is a deterministic mapping task (Hebrew → IPA)
- Prompt optimization can help with:
  - Forcing consistent format
  - Reducing hallucination
  - Improving edge cases
- But limited by model quality (if model doesn't know Hebrew phonology, no prompt will fix it)

**Verdict**:
- ✅ Can work if base model is decent (fine-tuned model helps)
- ⚠️ Won't work with models that have no Hebrew knowledge
- 📝 Document model requirements

### 7.3. Production Readiness Gaps

**Missing for production use**:
1. ❌ No input validation (what if test_cases is empty?)
2. ❌ No error handling (what if model() crashes?)
3. ❌ No resume from checkpoint
4. ❌ No parallel evaluation (slow for large test sets)
5. ❌ No caching (re-evaluates same prompts)

**Recommendations**:
- Add input validation in v0 (cheap to add)
- Add error handling for model calls (critical)
- Other features: document as "future work"

---

## 8. Implementation Checklist

### Phase 1: Cleanup (No Behavior Changes)
- [ ] Remove docs/architecture.md
- [ ] Remove docs/agent_mutator.md
- [ ] Remove examples/README.md
- [ ] Remove examples/sentiment/README.md
- [ ] Add output/ to .gitignore, remove committed output files
- [ ] Create constants.py and extract magic numbers
- [ ] Remove unnecessary docstrings (keep public API docs)
- [ ] Simplify README to ~50 lines

### Phase 2: Logging Refactor (Breaking Change)
- [ ] Create ProgressReporter protocol
- [ ] Implement ConsoleReporter
- [ ] Remove all print() statements from optimizer.py
- [ ] Remove all print() statements from agent_mutator.py
- [ ] Update optimizer to accept optional reporter
- [ ] Default behavior: silent (no reporter)
- [ ] Update examples to use ConsoleReporter

### Phase 3: Examples Cleanup
- [ ] Create examples/_utils.py
- [ ] Add create_ollama_model() helper
- [ ] Remove sys.path hacks from all examples
- [ ] Update examples to use shared utilities
- [ ] Update examples to use ConsoleReporter
- [ ] Test all examples still work

### Phase 4: Robustness (Critical Fixes)
- [ ] Add input validation (test_cases not empty, etc.)
- [ ] Add try/except around model() calls with clear error messages
- [ ] Fix greedy selection (evaluate all variations before choosing)
- [ ] Add validation_cases parameter (optional for v0)
- [ ] Update examples with larger test sets where possible

### Phase 5: Documentation
- [ ] Update README with new API
- [ ] Add inline code examples for ConsoleReporter
- [ ] Document model requirements for G2P
- [ ] Add "Limitations" section to README
- [ ] Update examples with comments about test set size

---

## 9. Risk Assessment

### High Risk (Must Fix):
1. **Print statements in library** - Makes it unusable in production
2. **No error handling** - Library crashes are unacceptable
3. **Greedy selection** - Fundamental algorithm flaw

### Medium Risk (Should Fix):
1. **Test set overfitting** - Can add validation split easily
2. **Input validation** - Prevents cryptic errors
3. **Poor documentation** - Reduces adoption

### Low Risk (Future Work):
1. **No history learning** - Nice-to-have enhancement
2. **No parallel evaluation** - Performance optimization
3. **No caching** - Performance optimization

---

## 10. Success Metrics

**Before Refactor**:
- README: 105 lines
- Docs: 2 files, ~250 lines
- Core library: 565 lines (with prints)
- Example files: 4 files
- Public API: Noisy by default

**After Refactor**:
- README: ~50 lines ✅
- Docs: 0 files (clean) ✅
- Core library: ~600 lines (with reporter, constants, validation) ✅
- Example files: 3 examples + 1 utils ✅
- Public API: Silent by default, opt-in logging ✅

**Quality Gates**:
- ✅ No print() in src/pole/*.py
- ✅ All examples run successfully
- ✅ Constants file exists and is used
- ✅ README reads like a real project (not LLM-generated)
- ✅ Library doesn't crash on bad inputs

---

## 11. Post-Refactor TODO (Future Enhancements)

These are NOT part of this refactor but should be tracked:

1. **Logging Integration**: Add proper Python logging support
2. **Progress Bars**: Integrate with tqdm for visual progress
3. **Weights & Biases**: Add optional W&B reporter for experiment tracking
4. **Parallel Evaluation**: Use multiprocessing for test case evaluation
5. **Resume from Checkpoint**: Load history.json and continue optimization
6. **Adaptive Mutations**: Learn from history to guide mutation strategies
7. **Multi-objective Optimization**: Optimize for multiple metrics
8. **Prompt Interpolation**: Combine successful prompts

---

## 12. Implementation Strategy

**Recommended Order**:
1. Start with Phase 1 (cleanup) - safest, no breaking changes
2. Do Phase 4 (robustness) - fixes critical bugs
3. Do Phase 2 (logging) - this is the breaking change
4. Do Phase 3 (examples) - depends on new API
5. Do Phase 5 (docs) - final polish

**Estimated Effort**:
- Phase 1: 1 hour
- Phase 2: 2-3 hours
- Phase 3: 1-2 hours
- Phase 4: 2-3 hours
- Phase 5: 1 hour

**Total**: ~8-10 hours of focused work

---

## Conclusion

**Is the core idea sound?** ✅ YES
- Prompt optimization via mutation + loss is a valid approach
- Similar to established methods (DSPy, AutoPrompt)
- Works if base model is reasonable

**Will it work for the examples?** ✅ YES, with caveats
- **basic_example**: ✅ Toy example, will work
- **sentiment**: ✅ Should work with decent model
- **g2p**: ⚠️ Works only if model has Hebrew knowledge (fine-tuned model recommended)

**Major Flaws Found**:
1. ❌ Greedy selection (fixable)
2. ❌ No error handling (critical fix)
3. ❌ Noisy by default (architectural fix needed)
4. ⚠️ Test set overfitting (document for v0, fix later)
5. ⚠️ Weak default mutator (document, encourage custom mutators)

**Recommendation**:
Proceed with refactor. The core concept is solid, but needs production polish. Focus on making it a **minimal, silent, robust library** that users can build on.
