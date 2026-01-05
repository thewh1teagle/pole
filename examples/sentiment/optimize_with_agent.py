"""
Agent-Based Sentiment Optimization

Uses an LLM agent to generate prompt mutations, with a verifier agent
to filter out garbage variations.

Requirements:
    uv pip install ollama
    ollama pull gemma3:270m

Run:
    uv run examples/sentiment/optimize_with_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pole import PromptOptimizer, AgentMutator


# Small sentiment dataset (text, expected_sentiment)
DATASET = [
    ("This movie was absolutely amazing!", "positive"),
    ("Worst experience ever, total waste of time.", "negative"),
    ("The product works fine, nothing special.", "neutral"),
    ("I love this so much! Best purchase ever!", "positive"),
    ("Terrible quality, very disappointed.", "negative"),
    ("It's okay, does what it says.", "neutral"),
    ("Incredible! Exceeded all my expectations!", "positive"),
]


def sentiment_model(prompt: str, input_text: str) -> str:
    """Wrapper for local Ollama model."""
    try:
        import ollama
        
        response = ollama.chat(
            model="gemma3:270m",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 10,
            }
        )
        
        output = response["message"]["content"].strip().lower()
        return output
        
    except ImportError:
        print("ERROR: ollama not installed. Run: uv pip install ollama")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR calling model: {e}")
        print("Make sure ollama is running and model is downloaded:")
        print("  ollama pull gemma3:270m")
        sys.exit(1)


def sentiment_loss(predicted: str, expected: str) -> float:
    """Simple accuracy-based loss."""
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


def create_mutator_agent() -> AgentMutator:
    """
    Create agent-based mutator with Ollama.
    
    This shows how to plug in your own LLM for mutation and verification.
    """
    import ollama
    
    def mutator_fn(instruction: str) -> str:
        """
        Mutator agent: generates creative prompt variations.
        
        User provides this callable - can be any LLM (OpenAI, Anthropic, etc.)
        """
        response = ollama.chat(
            model="gemma3:270m",
            messages=[
                {"role": "system", "content": "You are a prompt engineering expert."},
                {"role": "user", "content": instruction}
            ],
            options={
                "temperature": 0.8,  # More creative
                "num_predict": 500,
            }
        )
        return response["message"]["content"]
    
    def verifier_fn(question: str) -> str:
        """
        Verifier agent: checks if variations make sense.
        
        User provides this callable - can be same or different LLM.
        """
        response = ollama.chat(
            model="gemma3:270m",
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
    print("Agent-Based Prompt Optimization")
    print("="*70)
    print(f"\nDataset: {len(DATASET)} examples")
    print("Model: gemma3:270m")
    print("Mutator: Agent-based with verification")
    print("\nStarting optimization...\n")
    
    # Create agent-based mutator (user-provided LLM callables)
    mutator = create_mutator_agent()
    
    # Create optimizer with agent mutator
    optimizer = PromptOptimizer(
        model=sentiment_model,
        loss_fn=sentiment_loss,
        mutator=mutator,  # ← Agent-based mutator
        max_iterations=5,
        patience=2,
        top_k=3,
        output_dir="output/agent_sentiment/"
    )
    
    # Run optimization
    result = optimizer.optimize(
        initial_prompt="Classify the sentiment as positive, negative, or neutral.",
        test_cases=DATASET
    )
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nBest Prompt:")
    print(f"  \"{result.best_prompt}\"")
    print(f"\nAccuracy: {(1 - result.best_loss) * 100:.1f}%")
    print(f"Iterations: {result.iterations}")
    print(f"Converged: {result.converged} ({result.convergence_reason})")
    print(f"\nHistory saved to: output/agent_sentiment/history.json")
    
    # Show top 3
    print(f"\n{'='*70}")
    print("TOP 3 PROMPTS (Generated by Agent)")
    print("="*70)
    top_prompts = optimizer.history.get_top_k()
    for i, checkpoint in enumerate(top_prompts[:3], 1):
        accuracy = (1 - checkpoint.loss) * 100
        print(f"\n{i}. Accuracy: {accuracy:.1f}% (Iteration {checkpoint.iteration})")
        # Print full prompt since agent generates different ones
        print(f"   \"{checkpoint.prompt}\"")


if __name__ == "__main__":
    main()

