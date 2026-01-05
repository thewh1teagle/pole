"""G2P (Grapheme-to-Phoneme) prompt optimization for Hebrew to IPA conversion."""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from _utils import create_ollama_model
from pole import PromptOptimizer, AgentMutator, ConsoleReporter
from base_prompt import BASE_PROMPT


def load_test_cases(csv_path: str, max_samples: int = 10) -> list[tuple[str, str]]:
    """Load test cases from CSV file."""
    test_cases = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_samples:
                    break
                test_cases.append((row['transcript'], row['phonemes']))
    except FileNotFoundError:
        print(f"ERROR: {csv_path} not found!")
        print("Download it with:")
        print("  wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/gemma3-pro/gt.csv")
        sys.exit(1)

    return test_cases


def phoneme_error_rate(predicted: str, expected: str) -> float:
    """Compute Character Error Rate using jiwer (ideal for phoneme comparison)."""
    try:
        import jiwer
        return jiwer.cer(expected, predicted)
    except ImportError:
        print("ERROR: jiwer not installed. Run: pip install jiwer")
        sys.exit(1)


def create_g2p_mutator():
    """Create agent-based mutator specialized for G2P tasks."""
    import ollama

    def mutator_fn(instruction: str) -> str:
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {"role": "system", "content": "You are an expert in Hebrew linguistics and IPA phoneme transcription."},
                {"role": "user", "content": instruction}
            ],
            options={"temperature": 0.9, "num_predict": 800}
        )
        return response["message"]["content"]

    def verifier_fn(question: str) -> str:
        response = ollama.chat(
            model="gemma3:1b",
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
        num_variations=3,
        on_warning=lambda msg: print(msg)
    )


def main():
    print("="*70)
    print("G2P (Hebrew → IPA) Prompt Optimization")
    print("="*70)

    csv_path = Path(__file__).parent / "gt_small.csv"
    print(f"\nLoading test cases from: {csv_path}")

    test_cases = load_test_cases(str(csv_path), max_samples=10)
    print(f"Loaded {len(test_cases)} test cases")

    print("\nSample test case:")
    print(f"  Input:    {test_cases[0][0]}")
    print(f"  Expected: {test_cases[0][1]}")

    print("\nStarting optimization with AgentMutator (gemma3:1b)...")
    print("(This will take several minutes)\n")

    try:
        model = create_ollama_model("gemma3:1b", temperature=0.1, num_predict=150)
        mutator = create_g2p_mutator()
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure ollama is running and model is downloaded:")
        print("  ollama pull gemma3:1b")
        sys.exit(1)

    optimizer = PromptOptimizer(
        model=model,
        loss_fn=phoneme_error_rate,
        mutator=mutator,
        max_iterations=20,
        patience=20,
        top_k=5,
        output_dir="output/g2p/",
        reporter=ConsoleReporter()
    )

    result = optimizer.optimize(
        initial_prompt=BASE_PROMPT,
        test_cases=test_cases
    )

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nInitial Prompt:")
    print(f'  """{BASE_PROMPT}"""')
    print(f"\nBest Prompt:")
    print(f'  """{result.best_prompt}"""')
    print(f"\nPhoneme Error Rate:")
    print(f"  Initial: {optimizer.history.checkpoints[0].loss:.3f}")
    print(f"  Best:    {result.best_loss:.3f}")
    print(f"  Improvement: {(optimizer.history.checkpoints[0].loss - result.best_loss) * 100:.1f}%")
    print(f"\nIterations: {result.iterations}")
    print(f"\nHistory saved to: output/g2p/history.json")

    print(f"\n{'='*70}")
    print("TOP 3 PROMPTS")
    print("="*70)
    top_prompts = optimizer.history.get_top_k()
    for i, checkpoint in enumerate(top_prompts[:3], 1):
        print(f"\n{i}. Error Rate: {checkpoint.loss:.3f} (Iteration {checkpoint.iteration})")
        print(f'   """{checkpoint.prompt}"""')


if __name__ == "__main__":
    main()
