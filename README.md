# pole

Prompt optimization via iterative mutation and evaluation.

## Usage

See `examples/` directory for complete working examples:
- `basic_example.py` - Minimal example with placeholder model
- `sentiment/` - Sentiment classification with Ollama
- `g2p/` - Hebrew phoneme conversion (advanced)

Run with:
```bash
uv run examples/basic_example.py
```

## Features

- **No dependencies** - Bring your own LLM and metrics
- **Silent by default** - Opt-in to progress reporting
- **Loss-driven** - Evaluates all variations before selecting best
- **Full history** - Saves all iterations to JSON

## Architecture

Three core components:
- **PromptOptimizer** - Main loop with convergence logic
- **PromptMutator** - Generates variations (built-in or custom)
- **ProgressReporter** - Optional logging (silent by default)

## License

MIT
