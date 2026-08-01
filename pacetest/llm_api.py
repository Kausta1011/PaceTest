"""Hosted-API backend for the cross-model validation sweep (Week 11).

The proposal (Phase 5) specifies a 20 to 30 run cross-model check executed
"through the same experimental pipeline by swapping a single LLM-call
function". This module is that swap. It exposes `llm_anthropic`, which has
the same signature and return type as `pacetest.llm.llm`, so every caller
upstream (forward_pass, rewriter, the pacemakers, the loop) is untouched.

Why raw `requests` instead of the official SDK: the project already depends
on `requests` for Ollama, and adding no new dependency keeps the frozen
environment of Section 3.5 reproducible.

Three differences from the local backend are load-bearing for the thesis
and are recorded in the log header by `pacetest.logger`:

1.  No seed parameter. The Messages API accepts no seed, so the
    determinism guarantee of Section 3.5 does NOT hold for these runs.
    Temperature is pinned to 0.0 by default to get as close as the API
    allows, but repeat runs may differ. Cross-model runs are therefore
    reported as a single-shot robustness check, not as a replication.
2.  No `<think>` block. qwen3:8b emits a visible reasoning block that the
    1500-token budget was sized for (see pacetest/llm.py). Frontier models
    do not, so the same budget is generous rather than tight.
3.  Every call is metered. `USAGE` accumulates input and output tokens for
    the process so a run can be costed exactly rather than estimated.
"""
import os
import time

import requests

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_API_MODEL = "claude-haiku-4-5-20251001"

# Published rates in USD per million tokens, used only by the cost report.
# Verify against https://docs.claude.com/en/docs/about-claude/pricing before
# quoting any figure in the thesis; these are a convenience, not a source.
PRICE_PER_MTOK = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    # Sonnet 5 introductory rate, valid to 31 August 2026; standard rate
    # afterwards is 3.00 / 15.00. Re-check before quoting in the thesis.
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}

# Process-wide meter. A plain dict rather than a class because it is read
# once at the end of a run; `usage_report()` formats it.
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in the shell that runs "
            "the sweep:  export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    return key


def llm_anthropic(prompt: str, temperature: float = 0.0, seed: int = None,
                  max_tokens: int = 1500, model: str = None) -> str:
    """Drop-in replacement for `pacetest.llm.llm` backed by the Messages API.

    Args:
        prompt: The text to send.
        temperature: Pinned to 0.0 by callers for maximum determinism.
        seed: Accepted and IGNORED. Present only so the signature matches
            the Ollama backend; the Messages API has no seed parameter.
        max_tokens: Maximum tokens in the response.
        model: API model identifier. Defaults to DEFAULT_API_MODEL, or to
            the PACETEST_API_MODEL environment variable if set.

    Returns:
        The generated text, or "" if the model returned no text block. An
        empty string is a legitimate outcome the loop already handles via
        the sanity fallback of Section 3.2.7.

    Raises:
        RuntimeError: on missing key, or after _MAX_RETRIES failed attempts.
    """
    if model is None:
        model = os.environ.get("PACETEST_API_MODEL", DEFAULT_API_MODEL)

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": _api_key(),
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.post(API_URL, json=payload, headers=headers,
                                     timeout=180)
            # 429 is rate limiting, 5xx is transient on the provider side.
            # Both are worth retrying; 4xx of any other kind is a bug in the
            # request and retrying would just burn quota.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(_BACKOFF_BASE ** attempt)
                continue
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            time.sleep(_BACKOFF_BASE ** attempt)
            continue

        data = response.json()
        usage = data.get("usage", {})
        USAGE["calls"] += 1
        USAGE["input_tokens"] += usage.get("input_tokens", 0)
        USAGE["output_tokens"] += usage.get("output_tokens", 0)

        blocks = [b.get("text", "") for b in data.get("content", [])
                  if b.get("type") == "text"]
        return "".join(blocks)

    raise RuntimeError(
        f"Anthropic API call failed after {_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def usage_report(model: str = None) -> dict:
    """Return the process-wide call and token totals, with a cost estimate."""
    if model is None:
        model = os.environ.get("PACETEST_API_MODEL", DEFAULT_API_MODEL)
    rates = PRICE_PER_MTOK.get(model)
    cost = None
    if rates:
        cost = round(
            USAGE["input_tokens"] / 1e6 * rates["input"]
            + USAGE["output_tokens"] / 1e6 * rates["output"], 4)
    return {
        "model": model,
        "calls": USAGE["calls"],
        "input_tokens": USAGE["input_tokens"],
        "output_tokens": USAGE["output_tokens"],
        "estimated_cost_usd": cost,
    }


if __name__ == "__main__":
    # Smoke test: one cheap call, then print the meter.
    print(llm_anthropic("Reply with exactly the word: ok", max_tokens=16))
    print(usage_report())
