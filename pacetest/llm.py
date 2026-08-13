"""LLM helper function that calls a local Ollama server.

Week 11 addition: an optional hosted-API backend for the proposal's
cross-model validation sweep. The backend is selected by the environment
variable PACETEST_BACKEND and defaults to "ollama", so every run,
script, and test written before Week 11 behaves exactly as before unless
the variable is set explicitly.

    PACETEST_BACKEND=ollama     (default) local qwen3:8b, seeded, deterministic
    PACETEST_BACKEND=anthropic  hosted Messages API, NOT seed-deterministic
    PACETEST_BACKEND=openai     hosted chat-completions, best-effort seed only

PACETEST_TEMPERATURE, if set, overrides the temperature for every call on
every backend. It exists so that temperature can be swept as an
experimental variable without editing call sites. Unset by default, in
which case each backend keeps its own default (0.3 local, 0.0 hosted).
"""
import os

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
DEFAULT_SEED = 42


def active_backend() -> str:
    """Return the backend name currently selected by the environment."""
    return os.environ.get("PACETEST_BACKEND", "ollama").strip().lower()


def active_model() -> str:
    """Return the model identifier the active backend will actually call.

    The logger writes this into every log header, so a log always states
    which model produced it rather than assuming the local default.
    """
    b = active_backend()
    if b == "anthropic":
        from pacetest.llm_api import DEFAULT_API_MODEL
        return os.environ.get("PACETEST_API_MODEL", DEFAULT_API_MODEL)
    if b == "openai":
        from pacetest.llm_api import DEFAULT_OPENAI_MODEL
        return os.environ.get("PACETEST_API_MODEL", DEFAULT_OPENAI_MODEL)
    return MODEL


def active_temperature(default: float) -> float:
    """Return the temperature to use, honouring PACETEST_TEMPERATURE if set.

    Args:
        default: the temperature the caller would otherwise have used.

    Returns:
        The override if the environment variable is set and parses as a
        float, otherwise the caller's default.
    """
    raw = os.environ.get("PACETEST_TEMPERATURE", "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(
            f"PACETEST_TEMPERATURE={raw!r} is not a number."
        )


def is_deterministic_backend() -> bool:
    """True when the active backend honours a fixed seed.

    False for hosted APIs, which accept no seed parameter. Section 3.5's
    reproducibility argument applies only when this is True.
    """
    return active_backend() == "ollama"
# Bumped from 500 to 1500 on Week 6 Day 5 to accommodate GSM8K responses.
# qwen3:8b emits a <think>...</think> reasoning block before its visible
# output, and on word problems that block can be 300 to 800 tokens.
# The old cap of 500 was appropriate for toy tasks (short TOOL_CALL +
# ANSWER only) but truncated GSM8K responses to the empty string, which
# triggered a Week-3-style cascade documented in Section 4.7. The
# rewriter's per-call max_tokens overrides (600 for agent-prompt rewrites,
# 800 for tool-doc rewrites) are unchanged.
DEFAULT_MAX_TOKENS = 1500

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

    # Backend dispatch. Kept inside llm() so that every existing caller
    # (forward_pass, rewriter) picks the hosted backend up automatically
    # when the environment variable is set, with no import changes.
    backend = active_backend()
    if backend == "anthropic":
        from pacetest.llm_api import llm_anthropic
        return llm_anthropic(
            prompt,
            temperature=active_temperature(0.0),
            seed=None,                # no seed parameter on the Messages API
            max_tokens=max_tokens,
            model=None,               # resolved from PACETEST_API_MODEL
        )
    if backend == "openai":
        from pacetest.llm_api import llm_openai
        return llm_openai(
            prompt,
            temperature=active_temperature(0.0),
            seed=seed,                # best-effort seed, passed through
            max_tokens=max_tokens,
            model=None,
        )

    temperature = active_temperature(temperature)
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