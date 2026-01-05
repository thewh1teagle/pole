"""
G2P (Grapheme-to-Phoneme) Prompt Optimization

Optimizes a prompt for converting Hebrew text to IPA phonemes using a fine-tuned model.

Requirements:
    uv pip install ollama jiwer
    ollama pull gemma3:1b
    
Data:
    wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/gemma3-pro/gt.csv

Run:
    uv run examples/g2p/optimize_g2p.py
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pole import PromptOptimizer, AgentMutator
from base_prompt import BASE_PROMPT


def load_test_cases(csv_path: str, max_samples: int = 10) -> list[tuple[str, str]]:
    """
    Load test cases from CSV.
    
    Args:
        csv_path: Path to gt.csv file
        max_samples: Number of samples to use (small for fast iteration)
        
    Returns:
        List of (hebrew_text, expected_phonemes) pairs
    """
    test_cases = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_samples:
                    break
                transcript = row['transcript']
                phonemes = row['phonemes']
                test_cases.append((transcript, phonemes))
    except FileNotFoundError:
        print(f"ERROR: {csv_path} not found!")
        print("Download it with:")
        print("  wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/gemma3-pro/gt.csv")
        sys.exit(1)
    
    return test_cases


def g2p_model(prompt: str, input_text: str) -> str:
    """
    G2P model wrapper for Ollama.
    
    Args:
        prompt: System prompt
        input_text: Hebrew text to convert
        
    Returns:
        IPA phonemes string
    """
    try:
        import ollama
        
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 150,
            }
        )
        
        return response["message"]["content"].strip()
        
    except ImportError:
        print("ERROR: ollama not installed. Run: uv pip install ollama")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR calling model: {e}")
        print("Make sure ollama is running and model exists:")
        print("  ollama pull gemma3:1b")
        sys.exit(1)


def phoneme_error_rate(predicted: str, expected: str) -> float:
    """
    Compute phoneme error rate using Character Error Rate (CER) from jiwer.
    
    CER computes the edit distance at character level, which is ideal for
    phoneme comparison as phonemes are represented as characters.
    
    Args:
        predicted: Predicted IPA phonemes
        expected: Ground truth IPA phonemes
        
    Returns:
        Character Error Rate (0.0 = perfect, 1.0+ = very wrong)
    """
    try:
        import jiwer
        
        # Use Character Error Rate (perfect for phoneme comparison)
        cer = jiwer.cer(expected, predicted)
        return cer
        
    except ImportError:
        print("ERROR: jiwer not installed. Run: uv pip install jiwer")
        sys.exit(1)


def create_g2p_mutator() -> AgentMutator:
    """
    Create an agent-based mutator for G2P prompt optimization.
    
    The agent understands the G2P task and can generate relevant variations.
    """
    import ollama
    
    def mutator_fn(instruction: str) -> str:
        """Agent that generates G2P-specific prompt variations."""
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert in Hebrew linguistics and IPA phoneme transcription."
                },
                {"role": "user", "content": instruction}
            ],
            options={
                "temperature": 0.9,  # More creative for variations
                "num_predict": 800,
            }
        )
        return response["message"]["content"]
    
    def verifier_fn(question: str) -> str:
        """Agent that verifies prompt quality."""
        response = ollama.chat(
            model="gemma3:1b",
            messages=[
                {"role": "system", "content": "You are a quality control expert."},
                {"role": "user", "content": question}
            ],
            options={
                "temperature": 0.1,  # Very deterministic
                "num_predict": 10,
            }
        )
        return response["message"]["content"]
    
    return AgentMutator(
        mutator_fn=mutator_fn,
        verifier_fn=verifier_fn,
        num_variations=3
    )


def main():
    print("="*70)
    print("G2P (Hebrew → IPA) Prompt Optimization")
    print("="*70)
    
    # Load test cases
    csv_path = Path(__file__).parent / "gt_small.csv"  # Use small dataset for fast iteration
    print(f"\nLoading test cases from: {csv_path}")
    
    test_cases = load_test_cases(str(csv_path), max_samples=10)  # Will load all 3 rows
    print(f"Loaded {len(test_cases)} test cases")
    
    # Show sample
    print("\nSample test case:")
    print(f"  Input:    {test_cases[0][0]}")
    print(f"  Expected: {test_cases[0][1]}")
    
    print("\nStarting optimization with AgentMutator (gemma3:1b)...")
    print("(This will take several minutes - agent generates creative variations)")
    print()
    
    # Create agent-based mutator
    mutator = create_g2p_mutator()
    
    # Create optimizer with agent mutator
    optimizer = PromptOptimizer(
        model=g2p_model,
        loss_fn=phoneme_error_rate,
        mutator=mutator,  # Use agent-based mutations
        max_iterations=20,  # More iterations
        patience=20,  # No early stopping
        top_k=5,
        output_dir="output/g2p/",
        verbose=True  # Show progress
    )
    
    # Run optimization
    result = optimizer.optimize(
        initial_prompt=BASE_PROMPT,
        test_cases=test_cases
    )
    
    # Print results
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
    print(f"Converged: {result.converged} ({result.convergence_reason})")
    print(f"\nHistory saved to: output/g2p/history.json")
    
    # Show top 3 prompts
    print(f"\n{'='*70}")
    print("TOP 3 PROMPTS")
    print("="*70)
    top_prompts = optimizer.history.get_top_k()
    for i, checkpoint in enumerate(top_prompts[:3], 1):
        print(f"\n{i}. Error Rate: {checkpoint.loss:.3f} (Iteration {checkpoint.iteration})")
        print(f'   """{checkpoint.prompt}"""')


if __name__ == "__main__":
    main()

