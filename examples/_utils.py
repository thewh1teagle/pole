"""Shared utilities for examples."""


def create_ollama_model(model_name: str, temperature: float = 0.1, num_predict: int = 100):
    """
    Factory for Ollama model wrappers.

    Args:
        model_name: Name of the Ollama model (e.g., "gemma3:270m")
        temperature: Sampling temperature (lower = more deterministic)
        num_predict: Maximum number of tokens to generate

    Returns:
        Callable that takes (prompt, input_text) and returns model output
    """
    import ollama

    def model_fn(prompt: str, input_text: str) -> str:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            options={
                "temperature": temperature,
                "num_predict": num_predict,
            }
        )
        return response["message"]["content"].strip()

    return model_fn


def create_agent_mutator_funcs(model_name: str = "gemma3:1b"):
    """
    Create mutator and verifier functions for AgentMutator.

    Args:
        model_name: Name of the Ollama model to use

    Returns:
        Tuple of (mutator_fn, verifier_fn)
    """
    import ollama

    def mutator_fn(instruction: str) -> str:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": instruction}],
            options={"temperature": 0.7, "num_predict": 500}
        )
        return response["message"]["content"]

    def verifier_fn(question: str) -> str:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": question}],
            options={"temperature": 0.1, "num_predict": 10}
        )
        return response["message"]["content"]

    return mutator_fn, verifier_fn
