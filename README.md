# pole

Optimize prompts like you'd train a model, but without the weights.

Instead of gradient descent, use search. Define what "better" means (your loss function), and pole finds prompts that score lower. Plug in any mutation strategy: simple rules, LLM agents, RAG, evolutionary search, or scrape the web for prompt techniques.

## How it works

1. **You provide**: a model to test, a loss function, and test cases
2. **pole does**: mutate prompt → evaluate on test cases → keep what works → repeat
3. **You get**: the best prompt and full optimization history

Silent by default. Opt-in to progress logging.

## Examples

```bash
uv run examples/basic_example.py      # Minimal placeholder
uv run examples/sentiment/            # Sentiment with AgentMutator
uv run examples/g2p/                  # Hebrew→IPA (advanced)
```

See `examples/` for complete code.

## API

Three core pieces:
- `PromptOptimizer` - main loop with convergence logic
- `PromptMutator` - generates variations (or write your own)
- `ConsoleReporter` - optional logging

## License

MIT
