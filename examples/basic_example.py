"""
Basic example: How to use pole with a placeholder LLM.

This example shows the minimal interface without any real LLM calls.
"""

from pole import PromptOptimizer


def my_placeholder_model(prompt: str, input_text: str) -> str:
    """
    Placeholder model - replace this with your actual LLM call.
    
    Examples:
    - OpenAI: openai.chat.completions.create(...)
    - Ollama: ollama.chat(model="...", messages=[...])
    - Anthropic: anthropic.messages.create(...)
    """
    # Simulate some output based on prompt quality
    if "precise" in prompt.lower() or "accurate" in prompt.lower():
        return f"High quality output for: {input_text}"
    return f"Basic output for: {input_text}"


def simple_loss_fn(predicted: str, expected: str) -> float:
    """
    Simple loss function - replace with your domain-specific metric.
    
    Examples:
    - Edit distance (Levenshtein)
    - BLEU score
    - Exact match (0.0 if same, 1.0 if different)
    - Custom phoneme error rate
    """
    # Simple character-level loss (smaller is better)
    if predicted == expected:
        return 0.0
    
    # Count mismatches
    max_len = max(len(predicted), len(expected))
    differences = sum(
        1 for i in range(max_len)
        if i >= len(predicted) or i >= len(expected) or predicted[i] != expected[i]
    )
    return differences / max_len if max_len > 0 else 1.0


def main():
    """Run basic optimization example."""
    
    # Create optimizer
    optimizer = PromptOptimizer(
        model=my_placeholder_model,
        loss_fn=simple_loss_fn,
        max_iterations=10,
        patience=3,
        top_k=5,
        output_dir="output/basic_example/"
    )
    
    # Define test cases
    test_cases = [
        ("hello", "High quality output for: hello"),
        ("world", "High quality output for: world"),
        ("test", "High quality output for: test"),
    ]
    
    # Run optimization
    print("Starting optimization...")
    result = optimizer.optimize(
        initial_prompt="Process the following input:",
        test_cases=test_cases
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    print(f"Best Prompt:\n  {result.best_prompt}")
    print(f"\nBest Loss: {result.best_loss:.4f}")
    print(f"Iterations: {result.iterations}")
    print(f"Converged: {result.converged} (reason: {result.convergence_reason})")
    print(f"\nHistory saved to: output/basic_example/history.json")


if __name__ == "__main__":
    main()

