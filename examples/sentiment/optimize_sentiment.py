"""Sentiment classification with AgentMutator (LLM-powered mutation)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from _utils import create_ollama_model, create_agent_mutator_funcs
from pole import PromptOptimizer, AgentMutator, ConsoleReporter


DATASET = [
    ("This movie was absolutely amazing!", "positive"),
    ("Worst experience ever, total waste of time.", "negative"),
    ("The product works fine, nothing special.", "neutral"),
    ("I love this so much! Best purchase ever!", "positive"),
    ("Terrible quality, very disappointed.", "negative"),
    ("It's okay, does what it says.", "neutral"),
    ("Incredible! Exceeded all my expectations!", "positive"),
]


def sentiment_loss(predicted: str, expected: str) -> float:
    """Returns 0.0 if correct, 1.0 if incorrect."""
    pred_clean = predicted.strip().lower()
    expected_clean = expected.strip().lower()

    if expected_clean in pred_clean:
        return 0.0

    variations = {
        "positive": ["pos", "good", "great"],
        "negative": ["neg", "bad"],
        "neutral": ["neut", "okay", "ok"]
    }

    if expected_clean in variations:
        for variant in variations[expected_clean]:
            if variant in pred_clean:
                return 0.0

    return 1.0


def main():
    print("="*70)
    print("Sentiment Optimization with AgentMutator")
    print("="*70)
    print(f"\nDataset size: {len(DATASET)} examples")
    print("Model: gemma3:270m")
    print("Mutator: gemma3:1b (LLM-powered)\n")

    try:
        model = create_ollama_model("gemma3:270m", temperature=0.1, num_predict=10)
        mutator_fn, verifier_fn = create_agent_mutator_funcs("gemma3:1b")
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure ollama is running and models are downloaded:")
        print("  ollama pull gemma3:270m")
        print("  ollama pull gemma3:1b")
        sys.exit(1)

    mutator = AgentMutator(
        mutator_fn=mutator_fn,
        verifier_fn=verifier_fn,
        num_variations=3,
        on_warning=lambda msg: print(msg)
    )

    optimizer = PromptOptimizer(
        model=model,
        loss_fn=sentiment_loss,
        mutator=mutator,
        max_iterations=5,
        patience=3,
        output_dir="output/agent_sentiment/",
        reporter=ConsoleReporter()
    )

    result = optimizer.optimize(
        initial_prompt="Classify the sentiment as positive, negative, or neutral.",
        test_cases=DATASET
    )

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest Prompt:")
    print(f"  \"{result.best_prompt}\"")
    print(f"\nAccuracy: {(1 - result.best_loss) * 100:.1f}%")
    print(f"Iterations: {result.iterations}")
    print(f"\nHistory saved to: output/agent_sentiment/history.json")


if __name__ == "__main__":
    main()
