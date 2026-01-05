"""Sentiment classification prompt optimization with Ollama."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from _utils import create_ollama_model
from pole import PromptOptimizer, ConsoleReporter


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

    # Accept common variations
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
    print("Sentiment Classification Prompt Optimization")
    print("="*70)
    print(f"\nDataset size: {len(DATASET)} examples")
    print("Model: gemma3:270m (via Ollama)")
    print("\nStarting optimization...\n")

    try:
        model = create_ollama_model("gemma3:270m", temperature=0.1, num_predict=10)
    except ImportError:
        print("ERROR: ollama not installed. Run: pip install ollama")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure ollama is running and model is downloaded:")
        print("  ollama pull gemma3:270m")
        sys.exit(1)

    optimizer = PromptOptimizer(
        model=model,
        loss_fn=sentiment_loss,
        max_iterations=8,
        patience=3,
        top_k=3,
        output_dir="output/sentiment/",
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
    print(f"Converged: {result.converged} ({result.convergence_reason})")
    print(f"\nHistory saved to: output/sentiment/history.json")

    print(f"\n{'='*70}")
    print("TOP 3 PROMPTS")
    print("="*70)
    top_prompts = optimizer.history.get_top_k()
    for i, checkpoint in enumerate(top_prompts[:3], 1):
        accuracy = (1 - checkpoint.loss) * 100
        print(f"\n{i}. Accuracy: {accuracy:.1f}% (Iteration {checkpoint.iteration})")
        print(f"   \"{checkpoint.prompt[:80]}...\"" if len(checkpoint.prompt) > 80
              else f"   \"{checkpoint.prompt}\"")


if __name__ == "__main__":
    main()
