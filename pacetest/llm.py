"""LLM helper function that calls a local Ollama server"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 500

def llm(prompt: str, temperature: float = 0.3, seed: int = DEFAULT_SEED, max_tokens: int = DEFAULT_MAX_TOKENS, model: str = MODEL) -> str:
    """Send a prompt to local Ollama, return the generated text.

    Args:
        prompt: The text to send to the model.
        temperature: 0.0 = deterministic, higher = more random.
        seed: Fixed seed for reproducibility.
        max_tokens: Maximum tokens in a response
        model: Ollama model name. Defaults to qwen3:8b.
    
    Returns:
        The generated text as a string.

    Raises:
        RuntimeError : If Ollama is not reachable or returns an error

    """

    payload = {
        "model" : model ,
        "prompt" : prompt,
        "stream" : False,
        "options" : {
            "temperature": temperature,
            "seed" : seed,
            "num_predict" : max_tokens,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout = 120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama at " + OLLAMA_URL + ". Is Ollama running? Try: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama call timed out after 120 seconds")

    data = response.json()

    return data["response"]



#Quick smoke test
if __name__ == "__main__":
    print(llm("What is a MacBook? Answer in one sentence."))